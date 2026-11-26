import Accelerate
import ARKit
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct RecordingSettings {
    var imageRate: Int            // RGB-D frames saved per second; 0 = poses + IMU only
    var fullResolutionRGB: Bool   // RGB-D at 1920x1440 instead of 960x720
    var saveMesh: Bool            // export the LiDAR mesh as mesh.obj on stop
    var frameRate: Int = 60       // ARKit video format frame rate (for missed-frame accounting)
}

struct RecordingStats {
    var startedAt = Date()
    var frames = 0
    var images = 0
    var droppedImages = 0
    var missedFrames = 0          // ARKit frames never delivered (gaps in the camera stream)
    var imuSamples = 0
    var bytes: Int64 = 0
}

/// Thermal state and Low Power Mode, recorded with every scan (iOS lowers camera rates under both).
enum DeviceConditions {
    static var snapshot: [String: Any] {
        let thermal: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }
        return ["thermal_state": thermal, "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled]
    }
}

/// Fixed-size byte buffers reused across frames (no per-frame heap allocation in steady state).
final class BufferPool: @unchecked Sendable {
    private let lock = NSLock()
    private var free: [UnsafeMutableRawPointer] = []
    let bytes: Int

    init(bytes: Int, preallocate: Int) {
        self.bytes = bytes
        for _ in 0..<preallocate { free.append(.allocate(byteCount: bytes, alignment: 16)) }
    }
    deinit { free.forEach { $0.deallocate() } }

    func take() -> UnsafeMutableRawPointer {
        lock.withLock { free.popLast() } ?? .allocate(byteCount: bytes, alignment: 16)
    }
    func give(_ p: UnsafeMutableRawPointer) { lock.withLock { free.append(p) } }
}

/// Online capture stage: stores the raw sensor data only, as cheaply as possible.
/// Everything derived (RGB-aligned depth, VLIO trajectory, ...) is computed after the scan by
/// `RecordingProcessor`, so the phone does no avoidable work while scanning.
///
///     <Documents>/recordings/<yyyy-MM-dd_HH-mm-ss>/
///       meta.json             device, settings, conditions, conventions, counts
///       frames.csv            every ARKit frame: pose, intrinsics, exposure, tracking
///       trajectory_tum.txt    ARKit poses in TUM format
///       accel.csv gyro.csv mag.csv   raw IMU, same clock as frames
///       rgb/NNNNNN.jpg        camera images at `imageRate`
///       lidar/NNNNNN.png      raw ARKit LiDAR depth 256x192, uint16 mm
///       confidence/NNNNNN.png LiDAR confidence 0/1/2
///       mesh.obj              ARKit LiDAR mesh (world)
///     after processing: depth/NNNNNN.png (RGB-aligned, same size as rgb), vlio_*.txt, processing.json
final class Recorder: @unchecked Sendable {
    enum IMUStream { case accel, gyro, mag }
    private enum Stream: Hashable { case frames, tum, accel, gyro, mag }

    static var recordingsRoot: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("recordings", isDirectory: true)
    }

    private let lock = NSLock()
    private let io = DispatchQueue(label: "slamtogo.recorder.io", qos: .userInitiated)           // CSV streams (ordered)
    private let encoders = DispatchQueue(label: "slamtogo.recorder.encode", qos: .userInitiated,
                                         attributes: .concurrent)                                    // JPEG / PNG
    private let encodeGroup = DispatchGroup()
    private let ciContext = CIContext()                       // one context for the capture task (Apple guidance)
    private let maxImagesInFlight = 4
    private let flushBytes = 256 * 1024                       // batch small CSV writes
    private var rgbPool: CVPixelBufferPool?                   // ARKit queue only
    private var rgbPoolSize: CGSize = .zero
    private var lidarPool: BufferPool?                        // float32 depth copies
    private var confPool: BufferPool?                         // uint8 confidence copies
    private var mmPool: BufferPool?                           // uint16 mm conversion

    // Guarded by `lock`
    private var active = false
    private var settings = RecordingSettings(imageRate: 10, fullResolutionRGB: false, saveMesh: true)
    private var directory: URL?
    private var stats = RecordingStats()
    private var lastImageTimestamp: TimeInterval = -1
    private var imagesInFlight = 0
    private var firstTimestamp: TimeInterval?
    private var lastTimestamp: TimeInterval = 0
    private var intrinsics: simd_float3x3?
    private var imageResolution: CGSize = .zero
    private var deviceInfo: [String: Any] = [:]

    // Confined to `io`
    private var writers: [Stream: FileHandle] = [:]
    private var pending: [Stream: Data] = [:]

    var isRecording: Bool { lock.withLock { active } }
    var directoryURL: URL? { lock.withLock { active ? directory : nil } }
    var currentStats: RecordingStats { lock.withLock { stats } }

    // MARK: Start / stop

    func start(settings: RecordingSettings, deviceInfo: [String: Any]) throws {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let dir = Self.recordingsRoot.appendingPathComponent(formatter.string(from: Date()), isDirectory: true)
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        if settings.imageRate > 0 {
            for sub in ["rgb", "lidar", "confidence"] {
                try fm.createDirectory(at: dir.appendingPathComponent(sub), withIntermediateDirectories: true)
            }
        }

        let headers: [Stream: (String, String)] = [
            .frames: ("frames.csv", "frame,timestamp,tracking,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,width,height,exposure_s,image"),
            .tum: ("trajectory_tum.txt", "# timestamp tx ty tz qx qy qz qw  (ARKit world frame, camera-to-world)"),
            .accel: ("accel.csv", "timestamp,ax,ay,az"),
            .gyro: ("gyro.csv", "timestamp,gx,gy,gz"),
            .mag: ("mag.csv", "timestamp,mx,my,mz"),
        ]
        var handles: [Stream: FileHandle] = [:]
        for (stream, (name, header)) in headers {
            let url = dir.appendingPathComponent(name)
            fm.createFile(atPath: url.path, contents: Data((header + "\n").utf8))
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            handles[stream] = handle
        }

        io.sync {
            writers = handles
            pending = [:]
            for s in handles.keys { pending[s] = Data(capacity: flushBytes + 4096) }
        }
        lock.withLock {
            self.settings = settings
            self.directory = dir
            self.deviceInfo = deviceInfo
            stats = RecordingStats()
            lastImageTimestamp = -1
            imagesInFlight = 0
            firstTimestamp = nil
            intrinsics = nil
            active = true
        }
    }

    func stop(meshAnchors: [ARMeshAnchor], completion: @escaping (RecordingSummary?) -> Void) {
        lock.lock()
        guard active, let dir = directory else {
            lock.unlock()
            completion(nil)
            return
        }
        active = false
        let finalStats = stats
        let settings = self.settings
        let duration = firstTimestamp.map { lastTimestamp - $0 } ?? 0
        let K = intrinsics
        let resolution = imageResolution
        let device = deviceInfo
        lock.unlock()

        // Serial queue: every CSV write queued before this point is done when this block runs;
        // the group waits for the image encoders still running.
        io.async { [self] in
            encodeGroup.wait()
            for (stream, handle) in writers {
                if let data = pending[stream], !data.isEmpty { try? handle.write(contentsOf: data) }
                try? handle.close()
            }
            writers.removeAll()
            pending.removeAll()

            var mesh: (vertices: Int, faces: Int)?
            if settings.saveMesh, !meshAnchors.isEmpty {
                mesh = MeshExporter.writeOBJ(anchors: meshAnchors, to: dir.appendingPathComponent("mesh.obj"))
            }

            let scale = settings.fullResolutionRGB ? 1.0 : 0.5
            var meta: [String: Any] = [
                "format": "slamtogo-v2",
                "created": ISO8601DateFormatter().string(from: finalStats.startedAt),
                "device": device,
                "duration_s": duration,
                "frames": finalStats.frames,
                "rgbd_frames": finalStats.images,
                "rgbd_dropped": finalStats.droppedImages,
                "arkit_missed_frames": finalStats.missedFrames,
                "imu_samples": finalStats.imuSamples,
                "conditions_at_end": DeviceConditions.snapshot,
                "settings": [
                    "arkit_fps": settings.frameRate,
                    "rgbd_rate_hz": settings.imageRate,
                    "rgb_scale": scale,
                    "rgbd_width": Int((resolution.width * scale).rounded()),
                    "rgbd_height": Int((resolution.height * scale).rounded()),
                ],
                "conventions": [
                    "clock": "timestamps in seconds, iOS system uptime clock (shared by ARKit and CoreMotion)",
                    "world": "ARKit world frame: gravity-aligned, +y up, origin at session start",
                    "pose": "camera-to-world; camera axes +x right, +y up, -z forward (optical axis), sensor in landscape",
                    "intrinsics": "frames.csv fx fy cx cy are for width x height; multiply by rgb_scale for rgb/*.jpg",
                    "depth": "depth/*.png (written by on-device processing) uint16 millimetres (0 = invalid), pixel-aligned with rgb/*.jpg and the same size; RGB-guided edge-preserving upsample of the LiDAR depth",
                    "lidar": "lidar/*.png uint16 millimetres, raw ARKit sceneDepth 256x192, same field of view as rgb (intrinsics scale = 256 / width)",
                    "confidence": "confidence/*.png uint8 0 low, 1 medium, 2 high, 256x192 (matches lidar/)",
                    "accel": "m/s^2 specific force (device at rest face-up reads +9.81 on z), device axes x right y up z out of screen",
                    "gyro": "rad/s, device axes",
                    "mag": "microtesla, device axes, uncalibrated",
                ],
            ]
            if let K {
                meta["camera"] = [
                    "width": Int(resolution.width), "height": Int(resolution.height),
                    "fx": K[0][0], "fy": K[1][1], "cx": K[2][0], "cy": K[2][1],
                ]
            }
            if let mesh {
                meta["mesh"] = ["file": "mesh.obj", "vertices": mesh.vertices, "faces": mesh.faces]
            }
            if let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) {
                try? data.write(to: dir.appendingPathComponent("meta.json"))
            }

            let summary = RecordingSummary.load(from: dir)
            DispatchQueue.main.async { completion(summary) }
        }
    }

    // MARK: Ingest (ARKit queue)

    func ingest(frame: ARFrame) {
        lock.lock()
        guard active, let dir = directory else {
            lock.unlock()
            return
        }
        let index = stats.frames
        stats.frames += 1
        if firstTimestamp == nil {
            firstTimestamp = frame.timestamp
            intrinsics = frame.camera.intrinsics
            imageResolution = frame.camera.imageResolution
        } else {
            let gap = frame.timestamp - lastTimestamp
            let fps = Double(max(settings.frameRate, 1))
            if gap > 1.5 / fps { stats.missedFrames += Int((gap * fps).rounded()) - 1 }
        }
        lastTimestamp = frame.timestamp

        var saveImage = false
        if settings.imageRate > 0 {
            let interval = 1.0 / Double(settings.imageRate)
            if lastImageTimestamp < 0 || frame.timestamp - lastImageTimestamp >= interval * 0.95 {
                if imagesInFlight < maxImagesInFlight {
                    saveImage = true
                    imagesInFlight += 1
                    lastImageTimestamp = frame.timestamp
                    stats.images += 1
                } else {
                    stats.droppedImages += 1
                }
            }
        }
        let fullRes = settings.fullResolutionRGB
        lock.unlock()

        let camera = frame.camera
        let T = camera.transform
        let q = simd_quatf(T)
        let K = camera.intrinsics
        let tracking: Int
        switch camera.trackingState {
        case .normal: tracking = 2
        case .limited: tracking = 1
        case .notAvailable: tracking = 0
        }
        let t = frame.timestamp
        write(.frames, String(
            format: "%d,%.6f,%d,%.6f,%.6f,%.6f,%.7f,%.7f,%.7f,%.7f,%.4f,%.4f,%.4f,%.4f,%d,%d,%.6f,%d\n",
            index, t, tracking, T.columns.3.x, T.columns.3.y, T.columns.3.z,
            q.imag.x, q.imag.y, q.imag.z, q.real,
            K[0][0], K[1][1], K[2][0], K[2][1],
            Int(camera.imageResolution.width), Int(camera.imageResolution.height),
            camera.exposureDuration, saveImage ? 1 : 0))
        write(.tum, String(
            format: "%.6f %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n",
            t, T.columns.3.x, T.columns.3.y, T.columns.3.z, q.imag.x, q.imag.y, q.imag.z, q.real))

        guard saveImage else { return }
        // Copy everything out of ARKit's buffers now (holding them starves ARKit's pool -> skipped frames).
        let lidar = (frame.sceneDepth ?? frame.smoothedSceneDepth).flatMap { copyLidar($0) }
        guard let rgbCopy = copyRGB(frame.capturedImage, fullResolution: fullRes) else {
            lidar.map { releaseLidar($0) }
            lock.withLock { imagesInFlight -= 1; stats.images -= 1; stats.droppedImages += 1 }
            return
        }
        let name = String(format: "%06d", index)

        encodeGroup.enter()
        encoders.async { [self] in
            defer { encodeGroup.leave() }
            var bytes = 0
            let rgbURL = dir.appendingPathComponent("rgb/\(name).jpg")
            if let srgb = CGColorSpace(name: CGColorSpace.sRGB) {
                let opts = [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.85]
                try? ciContext.writeJPEGRepresentation(of: CIImage(cvPixelBuffer: rgbCopy), to: rgbURL, colorSpace: srgb,
                                                       options: opts)
                bytes += Self.fileSize(rgbURL)
            }
            if let l = lidar {
                // float metres -> uint16 mm, in a pooled buffer, streamed to the PNG file.
                let mm = mmPool!.take()
                Self.metresToMillimetres(l.depth.assumingMemoryBound(to: Float32.self),
                                         mm.assumingMemoryBound(to: UInt16.self), count: l.width * l.height)
                let lu = dir.appendingPathComponent("lidar/\(name).png")
                let cu = dir.appendingPathComponent("confidence/\(name).png")
                Self.writePNG(mm, width: l.width, height: l.height, bits: 16, to: lu)
                if let c = l.confidence { Self.writePNG(c, width: l.width, height: l.height, bits: 8, to: cu) }
                bytes += Self.fileSize(lu) + Self.fileSize(cu)
                mmPool!.give(mm)
                releaseLidar(l)
            }
            lock.withLock {
                imagesInFlight -= 1
                stats.bytes += Int64(bytes)
            }
        }
    }

    // MARK: Ingest (IMU queue)

    func ingestIMU(_ stream: IMUStream, t: TimeInterval, _ x: Double, _ y: Double, _ z: Double) {
        let isActive: Bool = lock.withLock {
            if active { stats.imuSamples += 1 }
            return active
        }
        guard isActive else { return }
        let line = String(format: "%.6f,%.7f,%.7f,%.7f\n", t, x, y, z)
        switch stream {
        case .accel: write(.accel, line)
        case .gyro: write(.gyro, line)
        case .mag: write(.mag, line)
        }
    }

    // MARK: Helpers

    /// Appends to an in-memory batch; flushes in large chunks (fewer, larger disk writes).
    private func write(_ stream: Stream, _ line: String) {
        io.async { [self] in
            guard let handle = writers[stream] else { return }   // closed: recording stopped
            pending[stream, default: Data()].append(contentsOf: line.utf8)
            if pending[stream]!.count >= flushBytes {
                try? handle.write(contentsOf: pending[stream]!)
                pending[stream]!.removeAll(keepingCapacity: true)
            }
            lock.withLock { stats.bytes += Int64(line.utf8.count) }
        }
    }

    private struct LidarCopy {
        let depth: UnsafeMutableRawPointer
        let confidence: UnsafeMutableRawPointer?
        let width: Int, height: Int
    }

    private func copyLidar(_ d: ARDepthData) -> LidarCopy? {
        let buf = d.depthMap
        let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
        if lidarPool == nil || lidarPool!.bytes != w * h * 4 {
            lidarPool = BufferPool(bytes: w * h * 4, preallocate: maxImagesInFlight)
            confPool = BufferPool(bytes: w * h, preallocate: maxImagesInFlight)
            mmPool = BufferPool(bytes: w * h * 2, preallocate: maxImagesInFlight)
        }
        let depth = lidarPool!.take()
        Self.copyPlane(buf, into: depth, bytesPerPixel: 4)
        var conf: UnsafeMutableRawPointer?
        if let c = d.confidenceMap {
            conf = confPool!.take()
            Self.copyPlane(c, into: conf!, bytesPerPixel: 1)
        }
        return LidarCopy(depth: depth, confidence: conf, width: w, height: h)
    }

    private func releaseLidar(_ l: LidarCopy) {
        lidarPool?.give(l.depth)
        if let c = l.confidence { confPool?.give(c) }
    }

    /// Renders (and downscales) ARKit's YCbCr camera image into a pooled BGRA buffer, synchronously.
    private func copyRGB(_ source: CVPixelBuffer, fullResolution: Bool) -> CVPixelBuffer? {
        var image = CIImage(cvPixelBuffer: source)
        if !fullResolution {
            image = image.transformed(by: CGAffineTransform(scaleX: 0.5, y: 0.5))
        }
        let size = CGSize(width: image.extent.width.rounded(), height: image.extent.height.rounded())
        if rgbPool == nil || rgbPoolSize != size {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: Int(size.width),
                kCVPixelBufferHeightKey as String: Int(size.height),
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            ]
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, [kCVPixelBufferPoolMinimumBufferCountKey as String: maxImagesInFlight + 2] as CFDictionary,
                                    attrs as CFDictionary, &pool)
            rgbPool = pool
            rgbPoolSize = size
        }
        guard let pool = rgbPool else { return nil }
        var out: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out) == kCVReturnSuccess, let out else { return nil }
        ciContext.render(image, to: out)
        return out
    }

    private static func copyPlane(_ buffer: CVPixelBuffer, into dst: UnsafeMutableRawPointer, bytesPerPixel: Int) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let packed = w * bytesPerPixel
        for y in 0..<h { memcpy(dst + y * packed, base + y * rowBytes, packed) }
    }

    /// Grayscale PNG streamed straight to `url` (8-bit, or 16-bit little-endian), no intermediate Data.
    static func writePNG(_ pixels: UnsafeRawPointer, width: Int, height: Int, bits: Int, to url: URL) {
        let size = width * height * bits / 8
        guard let provider = CGDataProvider(dataInfo: nil, data: pixels, size: size, releaseData: { _, _, _ in })
        else { return }
        var info = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        if bits == 16 { info.insert(.byteOrder16Little) }
        guard let image = CGImage(width: width, height: height, bitsPerComponent: bits, bitsPerPixel: bits,
                                  bytesPerRow: width * bits / 8, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: info,
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
    }

    /// Float metres -> uint16 millimetres with clamping and rounding (SIMD). ARKit depth is finite and >= 0
    /// (0 = no return). Works in place on the float buffer, which is a pooled copy owned by the caller.
    private static func metresToMillimetres(_ src: UnsafeMutablePointer<Float>, _ dst: UnsafeMutablePointer<UInt16>, count: Int) {
        let n = vDSP_Length(count)
        var k: Float = 1000, lo: Float = 0, hi: Float = 65535
        vDSP_vsmul(src, 1, &k, src, 1, n)
        vDSP_vclip(src, 1, &lo, &hi, src, 1, n)
        vDSP_vfixru16(src, 1, dst, 1, n)
    }

    private static func fileSize(_ url: URL) -> Int {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
    }
}
