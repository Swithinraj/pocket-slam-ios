import ARKit
import CoreMotion
import Foundation
import SlamCore
import simd

/// Live status of our own VLIO core.
struct VlioStatus {
    var initialized = false
    var tracked = 0
    var landmarks = 0
    var keyframes = 0
    var reinits = 0
    var solveMs = 0.0
    var frameMs = 0.0
    var hz = 0.0
    var drift: Float = 0          // distance between VLIO and ARKit camera positions [m]
}

/// Runs the shared C++ VLIO core (camera + IMU + LiDAR, sliding-window optimisation) on device.
/// Thread model: IMU and frames are handed over from their capture queues; every call into the core
/// happens on `queue`, so the core itself needs no locking.
final class SlamEngine: @unchecked Sendable {
    /// Called on the main queue: status, new trajectory point (ARKit world x,z), tracked features
    /// (normalised image coordinates of the landscape sensor image).
    var onUpdate: ((VlioStatus, SIMD2<Float>?, [CGPoint]) -> Void)?

    private let queue = DispatchQueue(label: "slamtogo.vlio", qos: .userInitiated)
    private let lock = NSLock()
    private var handle: OpaquePointer?

    private let downscale = 4                 // 1920x1440 -> 480x360 tracking image
    private let minFrameInterval = 1.0 / 15
    private var busy = false                  // lock
    private var enabled = true                // lock
    private var lastFrameTime = 0.0           // lock
    private var lastImuTime = 0.0             // lock

    // Frame buffers allocated once and reused: only one frame is in flight (`busy`), so the AR queue
    // fills them only while the engine queue is idle.
    private var grayBuf: UnsafeMutableBufferPointer<UInt8>?
    private var depthBuf: UnsafeMutableBufferPointer<Float>?
    private var confBuf: UnsafeMutableBufferPointer<UInt8>?

    var isEnabled: Bool {
        get { lock.withLock { enabled } }
        set { lock.withLock { enabled = newValue } }
    }

    // IMU merge (IMU queue only)
    private var accelPrev: (t: Double, a: SIMD3<Double>)?
    private var accelLast: (t: Double, a: SIMD3<Double>)?

    // Engine-queue state
    private var keyframes = 0
    private var reinits = 0
    private var T_arkit_vlio: simd_double4x4?
    private var lastPoint: SIMD2<Float>?
    private var rateWindow: [Double] = []
    private var framesOut: FileHandle?
    private var keyframesOut: FileHandle?

    /// Body (device: x right, y up, z out of screen) <- camera (OpenCV axes, landscape sensor).
    private static let R_bc: [Double] = [0, -1, 0, -1, 0, 0, 0, 0, -1]

    deinit {
        if let handle { slam_destroy(handle) }
        grayBuf?.deallocate()
        depthBuf?.deallocate()
        confBuf?.deallocate()
    }

    // MARK: Input

    func ingest(accel d: CMAccelerometerData) {
        let g = -9.80665
        accelPrev = accelLast
        accelLast = (d.timestamp, SIMD3(d.acceleration.x * g, d.acceleration.y * g, d.acceleration.z * g))
    }

    func ingest(gyro d: CMGyroData) {
        guard let last = accelLast else { return }
        var a = last.a
        if let prev = accelPrev, last.t > prev.t {           // interpolate accel to the gyro timestamp
            let w = min(max((d.timestamp - prev.t) / (last.t - prev.t), 0), 1.5)
            a = prev.a + (last.a - prev.a) * w
        }
        let t = d.timestamp, w = d.rotationRate
        guard lock.withLock({ lastImuTime = t; return enabled }) else { return }
        queue.async { [self] in
            guard let handle else { return }
            slam_add_imu(handle, t, a.x, a.y, a.z, w.x, w.y, w.z)
        }
    }

    /// Called on the ARKit delegate queue for every frame; copies what it needs and returns quickly.
    func ingest(frame: ARFrame) {
        let t = frame.timestamp
        let accept: Bool = lock.withLock {
            guard enabled, !busy, t - lastFrameTime >= minFrameInterval else { return false }
            busy = true
            lastFrameTime = t
            return true
        }
        guard accept else { return }

        // Luma plane of the YCbCr camera image, downscaled into the reused buffer.
        let pb = frame.capturedImage
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        let w = CVPixelBufferGetWidthOfPlane(pb, 0), h = CVPixelBufferGetHeightOfPlane(pb, 0)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let tw = w / downscale, th = h / downscale
        let gray = Self.ensure(&grayBuf, count: tw * th)
        if let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) {
            slam_downsample_luma(base.assumingMemoryBound(to: UInt8.self), Int32(w), Int32(h), Int32(stride),
                                 Int32(downscale), gray.baseAddress)
        }
        CVPixelBufferUnlockBaseAddress(pb, .readOnly)

        // Raw LiDAR depth + confidence into reused buffers.
        var dw = 0, dh = 0, hasConf = false
        if let sd = frame.sceneDepth ?? frame.smoothedSceneDepth {
            dw = CVPixelBufferGetWidth(sd.depthMap)
            dh = CVPixelBufferGetHeight(sd.depthMap)
            Self.copy(sd.depthMap, into: UnsafeMutableRawPointer(Self.ensure(&depthBuf, count: dw * dh).baseAddress!),
                      bytesPerPixel: 4)
            if let c = sd.confidenceMap {
                Self.copy(c, into: UnsafeMutableRawPointer(Self.ensure(&confBuf, count: dw * dh).baseAddress!), bytesPerPixel: 1)
                hasConf = true
            }
        }

        let K = frame.camera.intrinsics
        let s = Double(downscale)
        let intr = (Double(K[0][0]) / s, Double(K[1][1]) / s, Double(K[2][0]) / s, Double(K[2][1]) / s)
        process(t: t, width: tw, height: th, intr: intr, dw: dw, dh: dh, hasConf: hasConf,
                arkit: frame.camera.transform, attempts: 0)
    }

    private func process(t: Double, width: Int, height: Int, intr: (Double, Double, Double, Double),
                         dw: Int, dh: Int, hasConf: Bool, arkit: simd_float4x4, attempts: Int) {
        queue.async { [self] in
            // The core needs IMU samples beyond t: wait briefly for CoreMotion if needed.
            if lock.withLock({ lastImuTime }) < t, attempts < 6 {
                queue.asyncAfter(deadline: .now() + 0.005) {
                    self.process(t: t, width: width, height: height, intr: intr, dw: dw, dh: dh, hasConf: hasConf,
                                 arkit: arkit, attempts: attempts + 1)
                }
                return
            }
            defer { lock.withLock { busy = false } }
            if handle == nil {
                var R = Self.R_bc, p: [Double] = [0, 0, 0]
                handle = slam_create(&R, &p, intr.0)
            }
            guard let handle else { return }

            var res = SlamFrameResult()
            let ok = slam_add_frame(handle, t, grayBuf?.baseAddress, Int32(width), Int32(height), intr.0, intr.1, intr.2,
                                    intr.3, dw > 0 ? depthBuf?.baseAddress : nil, hasConf ? confBuf?.baseAddress : nil,
                                    Int32(dw), Int32(dh), &res)
            publish(ok: ok == 1, res: res, arkit: arkit, handle: handle)
        }
    }

    // MARK: Output

    private func publish(ok: Bool, res: SlamFrameResult, arkit: simd_float4x4, handle: OpaquePointer) {
        var status = VlioStatus()
        status.tracked = Int(res.tracked)
        status.frameMs = res.total_ms
        var point: SIMD2<Float>?
        if ok {
            if res.keyframe == 1 { keyframes += 1 }
            let T_vlio = Self.pose(res)
            // Fix the VLIO world into the ARKit world at the first pose (both as OpenCV camera poses).
            let T_arkit = Self.arkitToCV(arkit)
            if T_arkit_vlio == nil || res.reinitialized == 1 { T_arkit_vlio = T_arkit * T_vlio.inverse }
            if res.reinitialized == 1 { reinits += 1 }
            let T = T_arkit_vlio! * T_vlio
            let pos = SIMD3<Float>(Float(T.columns.3.x), Float(T.columns.3.y), Float(T.columns.3.z))
            status.initialized = true
            status.landmarks = Int(res.landmarks)
            status.keyframes = keyframes
            status.reinits = reinits
            status.solveMs = res.solve_ms
            status.drift = simd_distance(pos, SIMD3(arkit.columns.3.x, arkit.columns.3.y, arkit.columns.3.z))
            if lastPoint == nil || simd_distance(lastPoint!, SIMD2(pos.x, pos.z)) > 0.03 {
                point = SIMD2(pos.x, pos.z)
                lastPoint = point
            }
            write(framesOut, t: res.t, T: T_vlio)
            var buf = [Double](repeating: 0, count: 13 * 32)
            let n = Int(slam_take_finalized(handle, &buf, 32))
            for i in 0..<n {
                let o = Array(buf[(13 * i)..<(13 * i + 13)])
                write(keyframesOut, t: o[0], R: Array(o[1...9]), p: Array(o[10...12]))
            }
        }
        rateWindow.append(res.t)
        rateWindow.removeAll { res.t - $0 > 1 }
        status.hz = Double(rateWindow.count)

        var uv = [Float](repeating: 0, count: 400)
        let nf = Int(slam_features(handle, &uv, 200))
        let features = (0..<nf).map { CGPoint(x: CGFloat(uv[2 * $0]) / 480, y: CGFloat(uv[2 * $0 + 1]) / 360) }
        DispatchQueue.main.async { [onUpdate] in onUpdate?(status, point, features) }
    }

    func reset() {
        queue.async { [self] in
            if let handle { slam_destroy(handle) }
            handle = nil
            keyframes = 0
            reinits = 0
            T_arkit_vlio = nil
            lastPoint = nil
        }
    }

    // MARK: Recording (vlio_frames_tum.txt / vlio_keyframes_tum.txt next to the ARKit data)

    /// Live preview output (only when the preview is enabled). The full-precision post-scan run writes
    /// vlio_frames_tum.txt / vlio_keyframes_tum.txt.
    func startRecording(to dir: URL) {
        guard isEnabled else { return }
        queue.async { [self] in
            let header = "# timestamp tx ty tz qx qy qz qw  (camera->world, OpenCV camera, VLIO world z-up)\n"
            framesOut = Self.open(dir.appendingPathComponent("vlio_live_frames_tum.txt"), header)
            keyframesOut = Self.open(dir.appendingPathComponent("vlio_live_keyframes_tum.txt"),
                                     "# final keyframe poses, same convention\n")
        }
    }

    func stopRecording() {
        queue.sync {
            if let handle {
                var buf = [Double](repeating: 0, count: 13 * 64)
                let n = Int(slam_take_finalized(handle, &buf, 64))
                for i in 0..<n {
                    let o = Array(buf[(13 * i)..<(13 * i + 13)])
                    write(keyframesOut, t: o[0], R: Array(o[1...9]), p: Array(o[10...12]))
                }
            }
            try? framesOut?.close()
            try? keyframesOut?.close()
            framesOut = nil
            keyframesOut = nil
        }
    }

    private static func open(_ url: URL, _ header: String) -> FileHandle? {
        FileManager.default.createFile(atPath: url.path, contents: Data(header.utf8))
        let h = try? FileHandle(forWritingTo: url)
        _ = try? h?.seekToEnd()
        return h
    }

    private func write(_ fh: FileHandle?, t: Double, T: simd_double4x4) {
        let R = [T.columns.0.x, T.columns.1.x, T.columns.2.x,
                 T.columns.0.y, T.columns.1.y, T.columns.2.y,
                 T.columns.0.z, T.columns.1.z, T.columns.2.z]
        write(fh, t: t, R: R, p: [T.columns.3.x, T.columns.3.y, T.columns.3.z])
    }

    private func write(_ fh: FileHandle?, t: Double, R: [Double], p: [Double]) {
        guard let fh else { return }
        let m = simd_double3x3(rows: [SIMD3(R[0], R[1], R[2]), SIMD3(R[3], R[4], R[5]), SIMD3(R[6], R[7], R[8])])
        let q = simd_quatd(m)
        let line = String(format: "%.6f %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n", t, p[0], p[1], p[2],
                          q.imag.x, q.imag.y, q.imag.z, q.real)
        try? fh.write(contentsOf: Data(line.utf8))
    }

    // MARK: Helpers

    private static func pose(_ r: SlamFrameResult) -> simd_double4x4 {
        let R = withUnsafeBytes(of: r.R) { Array($0.bindMemory(to: Double.self)) }
        let p = withUnsafeBytes(of: r.p) { Array($0.bindMemory(to: Double.self)) }
        return simd_double4x4(columns: (SIMD4(R[0], R[3], R[6], 0), SIMD4(R[1], R[4], R[7], 0),
                                        SIMD4(R[2], R[5], R[8], 0), SIMD4(p[0], p[1], p[2], 1)))
    }

    /// ARKit camera (x right, y up, z back) -> OpenCV camera (x right, y down, z forward), as double.
    private static func arkitToCV(_ m: simd_float4x4) -> simd_double4x4 {
        var d = simd_double4x4(columns: (SIMD4<Double>(m.columns.0), SIMD4<Double>(m.columns.1),
                                         SIMD4<Double>(m.columns.2), SIMD4<Double>(m.columns.3)))
        d.columns.1 = -d.columns.1
        d.columns.2 = -d.columns.2
        return d
    }

    /// Returns `buf`, (re)allocating only when the required size changes.
    private static func ensure<T>(_ buf: inout UnsafeMutableBufferPointer<T>?, count: Int) -> UnsafeMutableBufferPointer<T> {
        if let b = buf, b.count == count { return b }
        buf?.deallocate()
        let b = UnsafeMutableBufferPointer<T>.allocate(capacity: count)
        buf = b
        return b
    }

    private static func copy(_ buffer: CVPixelBuffer, into dst: UnsafeMutableRawPointer, bytesPerPixel: Int) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let packed = w * bytesPerPixel
        for y in 0..<h { memcpy(dst + y * packed, base + y * rowBytes, packed) }
    }
}
