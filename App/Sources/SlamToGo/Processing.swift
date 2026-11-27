import Accelerate
import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import SlamCore
import UIKit
import simd

final class CancelToken: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var isCancelled: Bool { lock.withLock { flag } }
    func cancel() { lock.withLock { flag = true } }
}

/// Post-scan processing on the device, at full precision (no real-time limit).
///
/// Built like a game frame pipeline so the CPU, GPU and storage work in parallel and memory stays flat:
///  - a ring of `slotCount` preallocated frame slots guarded by two semaphores (no per-frame allocation),
///  - stage 1, GPU (Core Image): decode JPEG/PNG, 480x360 luma, LiDAR + confidence, RGB-aligned depth,
///  - stage 2, CPU (C++ core): IMU + frames into the sliding-window VLIO, strictly in order,
///  - stage 3, I/O: aligned-depth PNGs streamed to disk from pooled buffers on a small concurrent queue.
/// Then loop closure over the whole scan (place recognition + LiDAR-verified loops + pose graph).
final class RecordingProcessor: @unchecked Sendable {
    enum Failure: LocalizedError {
        case missing(String)
        var errorDescription: String? {
            switch self { case .missing(let what): return "Missing \(what)" }
        }
    }

    static let stateFileName = "processing.json"
    static func status(of dir: URL) -> String? {
        guard let data = try? Data(contentsOf: dir.appendingPathComponent(stateFileName)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return json["status"] as? String
    }

    private struct FrameRow {
        let index: Int
        let t, fx, fy, cx, cy, width: Double
    }

    private struct ImuRow {
        let t: Double
        let a: SIMD3<Double>
        let g: SIMD3<Double>
    }

    /// One pipeline slot: everything the VLIO core needs for a frame, in preallocated memory.
    private final class Slot {
        let gray: UnsafeMutablePointer<UInt8>
        let depth: UnsafeMutablePointer<Float>
        let conf: UnsafeMutablePointer<UInt8>
        var valid = false
        init(grayCount: Int, depthCount: Int) {
            gray = .allocate(capacity: grayCount)
            depth = .allocate(capacity: depthCount)
            conf = .allocate(capacity: depthCount)
        }
        deinit {
            gray.deallocate()
            depth.deallocate()
            conf.deallocate()
        }
    }

    private let dir: URL
    private let qos: DispatchQoS
    private let trackWidth = 480
    private let slotCount = 3
    /// One context for this background task, linear (no colour management) full-float precision.
    private let ctx = CIContext(options: [
        .workingColorSpace: NSNull(),
        .outputColorSpace: NSNull(),
        .workingFormat: NSNumber(value: CIFormat.RGBAf.rawValue),
        .cacheIntermediates: false,
    ])

    /// `qos`: .userInitiated while the user watches the progress (performance cores),
    /// .utility when the app is in the background.
    init(dir: URL, qos: DispatchQoS = .userInitiated) {
        self.dir = dir
        self.qos = qos
    }

    /// Called on memory warnings.
    func purgeCaches() { ctx.clearCaches() }

    func run(progress: @escaping (String, Double) -> Void, isCancelled: @escaping () -> Bool) throws {
        let started = Date()
        let frames = try loadFrames()
        let imu = try loadImu()
        guard let first = frames.first else { throw Failure.missing("RGB-D frames") }
        let fm = FileManager.default
        try fm.createDirectory(at: dir.appendingPathComponent("depth"), withIntermediateDirectories: true)

        guard let rgb0 = CIImage(contentsOf: file("rgb", first.index, "jpg")),
              let lid0 = linear(file("lidar", first.index, "png")) else { throw Failure.missing("first RGB-D frame") }
        let rgbW = Int(rgb0.extent.width), rgbH = Int(rgb0.extent.height)
        let tw = trackWidth, th = Int((Double(rgbH) * Double(tw) / Double(rgbW)).rounded())
        let dw = Int(lid0.extent.width), dh = Int(lid0.extent.height)
        let trackScale = Double(tw) / first.width

        // ---- ring buffer + pools (allocated once)
        let slots = (0..<slotCount).map { _ in Slot(grayCount: tw * th, depthCount: dw * dh) }
        let freeSlots = DispatchSemaphore(value: slotCount)
        let readySlots = DispatchSemaphore(value: 0)
        let alignedFloat = UnsafeMutablePointer<Float>.allocate(capacity: rgbW * rgbH)   // producer-owned scratch
        defer { alignedFloat.deallocate() }
        let writerCount = 3
        let mmPool = BufferPool(bytes: rgbW * rgbH * 2, preallocate: writerCount)
        let writeSlots = DispatchSemaphore(value: writerCount)
        let writer = DispatchQueue(label: "slamtogo.process.write", qos: qos, attributes: .concurrent)
        let writes = DispatchGroup()

        // ---- stage 1: GPU preparation (producer)
        let producer = DispatchQueue(label: "slamtogo.process.prepare", qos: qos)
        producer.async { [self] in
            let upsample = CIFilter.edgePreserveUpsample()       // CIFilter is not thread-safe: one per thread
            upsample.spatialSigma = 3
            upsample.lumaSigma = 0.15
            for (k, f) in frames.enumerated() {
                freeSlots.wait()
                let slot = slots[k % slotCount]
                slot.valid = false
                if !isCancelled() {
                    autoreleasepool {
                        prepare(f, slot: slot, tw: tw, th: th, dw: dw, dh: dh, rgbW: rgbW, rgbH: rgbH,
                                upsample: upsample, alignedFloat: alignedFloat, mmPool: mmPool,
                                writer: writer, writes: writes, writeSlots: writeSlots)
                    }
                }
                readySlots.signal()
            }
        }

        // ---- stage 2: VLIO core (consumer, this thread)
        var R_bc: [Double] = [0, -1, 0, -1, 0, 0, 0, 0, -1], p_bc: [Double] = [0, 0, 0]
        guard let core = slam_create(&R_bc, &p_bc, first.fx * trackScale) else { throw Failure.missing("VLIO core") }
        defer { slam_destroy(core) }
        slam_enable_loop_closure(core, 1)
        let header = "# timestamp tx ty tz qx qy qz qw  (camera->world, OpenCV camera, VLIO world z-up)\n"
        var framesOut = Data(header.utf8)
        framesOut.reserveCapacity(frames.count * 90)
        var kfOut = Data("# final keyframe poses (sliding-window smoothed), same convention\n".utf8)
        var kfBuf = [Double](repeating: 0, count: 13 * 64)
        var res = SlamFrameResult()
        var imuIndex = 0, keyframes = 0, restarts = 0
        var lastReported = -1.0
        var solveMs = 0.0

        for (k, f) in frames.enumerated() {
            readySlots.wait()
            let slot = slots[k % slotCount]
            defer { freeSlots.signal() }
            waitIfCritical(isCancelled)
            if isCancelled() || !slot.valid { continue }

            while imuIndex < imu.count && imu[imuIndex].t <= f.t + 0.02 {
                let s = imu[imuIndex]
                slam_add_imu(core, s.t, s.a.x, s.a.y, s.a.z, s.g.x, s.g.y, s.g.z)
                imuIndex += 1
            }
            let ok = slam_add_frame(core, f.t, slot.gray, Int32(tw), Int32(th), f.fx * trackScale, f.fy * trackScale,
                                    f.cx * trackScale, f.cy * trackScale, slot.depth, slot.conf, Int32(dw), Int32(dh), &res)
            if ok == 1 {
                Self.appendTum(&framesOut, t: res.t, R: res.R, p: res.p)
                if res.keyframe == 1 { keyframes += 1; solveMs += res.solve_ms }
                if res.reinitialized == 1 { restarts += 1 }
            }
            let n = Int(slam_take_finalized(core, &kfBuf, 64))
            for i in 0..<n { Self.appendTum(&kfOut, kfBuf, offset: 13 * i) }

            let fraction = Double(k + 1) / Double(frames.count)
            if fraction - lastReported >= 0.01 {
                progress("Depth alignment and visual-inertial optimisation", fraction)
                lastReported = fraction
            }
        }
        let n = Int(slam_window_keyframes(core, &kfBuf, 64))
        for i in 0..<n { Self.appendTum(&kfOut, kfBuf, offset: 13 * i) }
        writes.wait()
        if isCancelled() { return }

        // ---- loop closure: revisits verified with LiDAR 3D, pose graph, every frame corrected
        progress("Loop closure", 0.99)
        var loops = SlamLoopStats()
        slam_finish(core, &loops)
        func finalTum(keyframes: Int32, header: String) -> Data {
            let count = Int(slam_final_count(core, keyframes))
            var buf = [Double](repeating: 0, count: 13 * max(count, 1))
            let n = Int(slam_final_poses(core, keyframes, &buf, Int32(count)))
            var out = Data(header.utf8)
            out.reserveCapacity(header.utf8.count + n * 90)
            for i in 0..<n { Self.appendTum(&out, buf, offset: 13 * i) }
            return out
        }
        let slamFrames = finalTum(keyframes: 0,
                                  header: "# timestamp tx ty tz qx qy qz qw  (camera->world after loop closure, VLIO world z-up)\n")
        let slamKeyframes = finalTum(keyframes: 1, header: "# keyframes after loop closure, same convention\n")
        if isCancelled() { return }

        // ---- outputs: one write each (vlio_* = odometry only, slam_* = final, loop closed)
        try framesOut.write(to: dir.appendingPathComponent("vlio_frames_tum.txt"), options: .atomic)
        try kfOut.write(to: dir.appendingPathComponent("vlio_keyframes_tum.txt"), options: .atomic)
        try slamFrames.write(to: dir.appendingPathComponent("slam_frames_tum.txt"), options: .atomic)
        try slamKeyframes.write(to: dir.appendingPathComponent("slam_keyframes_tum.txt"), options: .atomic)
        let state: [String: Any] = [
            "status": "done",
            "version": 2,
            "loop_closure": [
                "candidates": Int(loops.candidates),
                "verified": Int(loops.verified),
                "accepted": Int(loops.accepted),
                "max_correction_m": loops.max_correction_m,
                "max_correction_deg": loops.max_correction_deg,
                "ms": loops.ms,
            ],
            "processed_at": ISO8601DateFormatter().string(from: Date()),
            "duration_s": Date().timeIntervalSince(started),
            "frames": frames.count,
            "keyframes": keyframes,
            "vlio_restarts": restarts,
            "solve_ms_mean": keyframes > 0 ? solveMs / Double(keyframes) : 0,
            "steps": ["depth_aligned", "vlio", "loop_closure"],
            "conditions": DeviceConditions.snapshot,
        ]
        let json = try JSONSerialization.data(withJSONObject: state, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: dir.appendingPathComponent(Self.stateFileName), options: .atomic)
        progress("Done", 1)
    }

    // MARK: Stage 1

    private func prepare(_ f: FrameRow, slot: Slot, tw: Int, th: Int, dw: Int, dh: Int, rgbW: Int, rgbH: Int,
                         upsample: CIFilter & CIEdgePreserveUpsample, alignedFloat: UnsafeMutablePointer<Float>,
                         mmPool: BufferPool, writer: DispatchQueue, writes: DispatchGroup, writeSlots: DispatchSemaphore) {
        guard let rgb = CIImage(contentsOf: file("rgb", f.index, "jpg")),
              let lidar = linear(file("lidar", f.index, "png")) else { return }
        let conf = linear(file("confidence", f.index, "png"))

        // Luma for tracking (Rec.601 weights), 2x2 box downscale by bilinear sampling at exact 0.5.
        let luma = rgb.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.299, y: 0.587, z: 0.114, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
        ]).transformed(by: CGAffineTransform(scaleX: CGFloat(tw) / rgb.extent.width, y: CGFloat(th) / rgb.extent.height))
        ctx.render(luma, toBitmap: slot.gray, rowBytes: tw, bounds: CGRect(x: 0, y: 0, width: tw, height: th),
                   format: .R8, colorSpace: nil)

        // Raw LiDAR as metres (PNG holds mm; Core Image normalises 16-bit to 0..1).
        ctx.render(lidar, toBitmap: slot.depth, rowBytes: dw * 4, bounds: lidar.extent, format: .Rf, colorSpace: nil)
        var toMetres: Float = 65.535
        vDSP_vsmul(slot.depth, 1, &toMetres, slot.depth, 1, vDSP_Length(dw * dh))     // SIMD (Accelerate)
        if let conf {
            ctx.render(conf, toBitmap: slot.conf, rowBytes: dw, bounds: conf.extent, format: .R8, colorSpace: nil)
        } else {
            slot.conf.initialize(repeating: 2, count: dw * dh)
        }
        slot.valid = true

        // RGB-aligned depth at the RGB size (skipped if already produced by an earlier run).
        let out = file("depth", f.index, "png")
        guard !FileManager.default.fileExists(atPath: out.path) else { return }
        upsample.inputImage = rgb
        upsample.smallImage = lidar
        guard let up = upsample.outputImage else { return }
        ctx.render(up, toBitmap: alignedFloat, rowBytes: rgbW * 4, bounds: CGRect(x: 0, y: 0, width: rgbW, height: rgbH),
                   format: .Rf, colorSpace: nil)
        writeSlots.wait()
        let mm = mmPool.take()
        let n = vDSP_Length(rgbW * rgbH)
        // normalised -> mm, clamp to [0, 65535], round to uint16: all SIMD (Accelerate).
        var scale: Float = 65535, lo: Float = 0, hi: Float = 65535
        vDSP_vsmul(alignedFloat, 1, &scale, alignedFloat, 1, n)
        vDSP_vclip(alignedFloat, 1, &lo, &hi, alignedFloat, 1, n)
        vDSP_vfixru16(alignedFloat, 1, mm.assumingMemoryBound(to: UInt16.self), 1, n)
        writes.enter()
        writer.async {
            Recorder.writePNG(mm, width: rgbW, height: rgbH, bits: 16, to: out)
            mmPool.give(mm)
            writeSlots.signal()
            writes.leave()
        }
    }

    // MARK: Helpers

    /// Apple: at `.critical` reduce or stop work. Full speed otherwise.
    private func waitIfCritical(_ isCancelled: () -> Bool) {
        while ProcessInfo.processInfo.thermalState == .critical && !isCancelled() {
            Thread.sleep(forTimeInterval: 1.0)
        }
    }

    private func file(_ folder: String, _ index: Int, _ ext: String) -> URL {
        dir.appendingPathComponent(folder).appendingPathComponent(String(format: "%06d.%@", index, ext))
    }

    /// Image without colour management: pixel values map linearly to 0..1.
    private func linear(_ url: URL) -> CIImage? {
        CIImage(contentsOf: url, options: [.colorSpace: NSNull()])
    }

    private func loadFrames() throws -> [FrameRow] {
        guard let text = try? String(contentsOf: dir.appendingPathComponent("frames.csv"), encoding: .utf8)
        else { throw Failure.missing("frames.csv") }
        var rows: [FrameRow] = []
        for line in text.split(separator: "\n").dropFirst() {
            let c = line.split(separator: ",")
            guard c.count >= 18, c[17] == "1", let idx = Int(c[0]), let t = Double(c[1]),
                  let fx = Double(c[10]), let fy = Double(c[11]), let cx = Double(c[12]), let cy = Double(c[13]),
                  let w = Double(c[14]) else { continue }
            rows.append(FrameRow(index: idx, t: t, fx: fx, fy: fy, cx: cx, cy: cy, width: w))
        }
        return rows
    }

    /// Gyro samples with the accelerometer interpolated to their timestamps.
    private func loadImu() throws -> [ImuRow] {
        func read(_ name: String) throws -> [(Double, SIMD3<Double>)] {
            guard let text = try? String(contentsOf: dir.appendingPathComponent(name), encoding: .utf8)
            else { throw Failure.missing(name) }
            return text.split(separator: "\n").dropFirst().compactMap { line in
                let c = line.split(separator: ",").compactMap { Double($0) }
                return c.count == 4 ? (c[0], SIMD3(c[1], c[2], c[3])) : nil
            }
        }
        let acc = try read("accel.csv"), gyr = try read("gyro.csv")
        guard acc.count > 1 else { return [] }
        var out: [ImuRow] = []
        out.reserveCapacity(gyr.count)
        var a = 0
        for (t, g) in gyr {
            while a + 2 < acc.count && acc[a + 1].0 < t { a += 1 }
            let (t0, a0) = acc[a], (t1, a1) = acc[a + 1]
            let w = t1 > t0 ? min(max((t - t0) / (t1 - t0), 0), 1) : 0
            out.append(ImuRow(t: t, a: a0 + (a1 - a0) * w, g: g))
        }
        return out
    }

    private static func appendTum(_ out: inout Data, t: Double, R: (Double, Double, Double, Double, Double, Double, Double, Double, Double),
                                  p: (Double, Double, Double)) {
        let m = simd_double3x3(rows: [SIMD3(R.0, R.1, R.2), SIMD3(R.3, R.4, R.5), SIMD3(R.6, R.7, R.8)])
        let q = simd_quatd(m)
        out.append(contentsOf: String(format: "%.6f %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n", t, p.0, p.1, p.2,
                                      q.imag.x, q.imag.y, q.imag.z, q.real).utf8)
    }

    private static func appendTum(_ out: inout Data, _ b: [Double], offset o: Int) {
        appendTum(&out, t: b[o], R: (b[o + 1], b[o + 2], b[o + 3], b[o + 4], b[o + 5], b[o + 6], b[o + 7], b[o + 8], b[o + 9]),
                  p: (b[o + 10], b[o + 11], b[o + 12]))
    }
}

/// Queue of recordings to process. Scanning has priority: nothing is processed while recording,
/// and a running job yields (and is resumed later) when a new scan starts.
@MainActor
final class ProcessingCenter: ObservableObject {
    static let shared = ProcessingCenter()

    struct Job: Equatable {
        let url: URL
        var step: String
        var fraction: Double
    }

    @Published private(set) var job: Job?
    @Published private(set) var pending: [URL] = []
    @Published private(set) var message: String?

    private var scanning = false
    private var inForeground = true
    private var token: CancelToken?
    private var processor: RecordingProcessor?
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    private let queue = DispatchQueue(label: "slamtogo.process", qos: .userInitiated)
    private var observers: [NSObjectProtocol] = []

    private init() {
        let nc = NotificationCenter.default
        observers.append(nc.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.processor?.purgeCaches() }
        })
        observers.append(nc.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.inForeground = false }
        })
        observers.append(nc.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.inForeground = true
                self?.startNext()
            }
        })
    }

    func enqueue(_ url: URL) {
        guard job?.url != url, !pending.contains(url) else { return }
        pending.append(url)
        startNext()
    }

    func fraction(for url: URL) -> Double? { job?.url == url ? job?.fraction : nil }
    func isQueued(_ url: URL) -> Bool { pending.contains(url) }

    /// Removes a recording from processing before it is deleted. `completion` runs (on the main actor)
    /// once nothing touches the folder any more: immediately, or after the running job has stopped.
    func discard(_ url: URL, completion: @escaping () -> Void) {
        pending.removeAll { $0 == url }
        if job?.url == url {
            discardCompletions[url, default: []].append(completion)
            token?.cancel()
        } else {
            completion()
        }
    }
    private var discardCompletions: [URL: [() -> Void]] = [:]

    /// The camera, GPU and CPU belong to the scan while recording.
    func setScanning(_ value: Bool) {
        scanning = value
        if value { token?.cancel() } else { startNext() }
    }

    private func startNext() {
        guard job == nil, !scanning, inForeground, !pending.isEmpty else { return }
        let url = pending.removeFirst()
        let token = CancelToken()
        let processor = RecordingProcessor(dir: url, qos: inForeground ? .userInitiated : .utility)
        self.token = token
        self.processor = processor
        job = Job(url: url, step: "Preparing", fraction: 0)
        message = nil
        // Lets the job finish if the user leaves the app; on expiry it is cancelled and resumed later.
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "Process scan") { [weak self] in
            MainActor.assumeIsolated {
                self?.token?.cancel()
                self?.endBackgroundTask()
            }
        }
        queue.async { [weak self] in
            var failure: Error?
            do {
                try processor.run(progress: { step, fraction in
                    DispatchQueue.main.async {
                        MainActor.assumeIsolated {
                            guard let self, self.job?.url == url else { return }
                            self.job?.step = step
                            self.job?.fraction = fraction
                        }
                    }
                }, isCancelled: { token.isCancelled })
            } catch {
                failure = error
            }
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    self?.finish(url: url, error: failure, cancelled: token.isCancelled)
                }
            }
        }
    }

    private func finish(url: URL, error: Error?, cancelled: Bool) {
        job = nil
        token = nil
        processor = nil
        if let completions = discardCompletions.removeValue(forKey: url) {
            completions.forEach { $0() }                                  // being deleted: do not resume
        } else if cancelled {
            if !pending.contains(url) { pending.insert(url, at: 0) }     // resume later
        } else if let error {
            message = "Processing failed: \(error.localizedDescription)"
        } else {
            message = "Scan processed"
        }
        endBackgroundTask()
        startNext()
    }

    private func endBackgroundTask() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }
}
