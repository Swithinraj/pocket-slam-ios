import ARKit
import CoreMotion
import Foundation
import RealityKit

enum TrackingQuality {
    case good, limited, unavailable
}

/// Snapshot of live sensor state, published to the UI at ~10 Hz.
struct Telemetry {
    var quality: TrackingQuality = .unavailable
    var trackingText = "Starting…"
    var cameraHz: Double = 0
    var imuHz: Double = 0
    var depthSize = "—"
    var position = SIMD3<Float>(repeating: 0)
    var heading = SIMD2<Float>(0, -1)      // camera forward projected on the ground plane (x, z)
    var meshAnchors = 0
    var distance: Float = 0                // metres travelled while tracking was normal
}

/// UI-facing owner of the AR session (camera + LiDAR + ARKit VIO), the IMU streams and the recorder.
@MainActor
final class SensorSession: ObservableObject {
    @Published private(set) var telemetry = Telemetry()
    @Published private(set) var trajectory: [SIMD2<Float>] = []
    @Published private(set) var vlio = VlioStatus()
    @Published private(set) var vlioTrajectory: [SIMD2<Float>] = []
    @Published private(set) var features: [CGPoint] = []      // normalised sensor-image coordinates
    @Published var showMesh = true {
        didSet { applyMeshVisibility() }
    }
    /// Read before registering for changes (Apple guidance); drives the critical-temperature auto-save.
    @Published private(set) var thermal = ProcessInfo.processInfo.thermalState
    /// Capture frame rate of the ARKit video format (60 by default; 30 halves camera + tracking work).
    var captureFPS = 60
    /// Live VLIO preview while scanning (the full-precision VLIO always runs after the scan).
    var liveVlio = true {
        didSet { engine.isEnabled = liveVlio }
    }

    let arView: ARView
    let recorder = Recorder()
    let engine = SlamEngine()

    private let processor: FrameProcessor
    private let motion = CMMotionManager()
    private let imuQueue = OperationQueue()
    private let arQueue = DispatchQueue(label: "slamtogo.arkit", qos: .userInteractive)
    private var running = false
    private var thermalObserver: NSObjectProtocol?

    init() {
        arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        // Only visual effects are disabled; none of them affects the recorded sensor data.
        arView.renderOptions.insert([.disableDepthOfField, .disableMotionBlur, .disablePersonOcclusion, .disableHDR,
                                     .disableCameraGrain, .disableAREnvironmentLighting, .disableGroundingShadows])
        engine.isEnabled = liveVlio
        processor = FrameProcessor(recorder: recorder, engine: engine)
        imuQueue.name = "slamtogo.imu"
        imuQueue.maxConcurrentOperationCount = 1
        imuQueue.qualityOfService = .userInteractive

        processor.onTelemetry = { [weak self] snapshot, newPoints in
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    self?.apply(snapshot, newPoints)
                }
            }
        }
        arView.session.delegate = processor
        arView.session.delegateQueue = arQueue
        applyMeshVisibility()

        engine.onUpdate = { [weak self] status, point, features in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.vlio = status
                self.features = features
                if let point {
                    self.vlioTrajectory.append(point)
                    if self.vlioTrajectory.count > 6000 {
                        self.vlioTrajectory = stride(from: 0, to: self.vlioTrajectory.count, by: 2).map { self.vlioTrajectory[$0] }
                    }
                }
            }
        }
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.thermal = ProcessInfo.processInfo.thermalState }
        }
    }

    // MARK: Session control

    func start() {
        guard !running else { return }
        running = true
        arView.session.run(makeConfiguration(), options: [])
        startIMU()
    }

    /// Applies a new capture frame rate (restarts the session configuration, keeps the map).
    func applyCaptureFPS(_ fps: Int) {
        guard fps != captureFPS else { return }
        captureFPS = fps
        if running { arView.session.run(makeConfiguration(), options: []) }
    }

    func pause() {
        guard running else { return }
        running = false
        arView.session.pause()
        motion.stopAccelerometerUpdates()
        motion.stopGyroUpdates()
        motion.stopMagnetometerUpdates()
    }

    /// Restart tracking from scratch: new world origin, mesh and trajectory cleared.
    func reset() {
        arView.session.run(makeConfiguration(), options: [.resetTracking, .removeExistingAnchors])
        let processor = self.processor
        arQueue.async { processor.resetStats() }
        engine.reset()
        trajectory.removeAll()
        vlioTrajectory.removeAll()
        telemetry.distance = 0
    }

    // MARK: Recording

    func startRecording(_ settings: RecordingSettings) throws {
        ProcessingCenter.shared.setScanning(true)
        var settings = settings
        let format = arView.session.configuration?.videoFormat
        settings.frameRate = format?.framesPerSecond ?? 60
        var info = Self.deviceInfo(lidar: telemetry.depthSize)
        info["conditions_at_start"] = DeviceConditions.snapshot
        if let format {
            info["video_format"] = "\(Int(format.imageResolution.width))x\(Int(format.imageResolution.height))@\(format.framesPerSecond)"
        }
        try recorder.start(settings: settings, deviceInfo: info)
        if let dir = recorder.directoryURL { engine.startRecording(to: dir) }
    }

    func stopRecording(completion: @escaping (RecordingSummary?) -> Void) {
        let anchors = arView.session.currentFrame?.anchors.compactMap { $0 as? ARMeshAnchor } ?? []
        engine.stopRecording()
        recorder.stop(meshAnchors: anchors) { summary in
            // Heavy, precise processing happens now, after the scan, at utility QoS.
            if let summary { ProcessingCenter.shared.enqueue(summary.url) }
            ProcessingCenter.shared.setScanning(false)
            completion(summary)
        }
    }

    // MARK: Private

    private func makeConfiguration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        // Explicit video format: 1920x1440 at the chosen rate (Apple: pick the format the app needs).
        let formats = ARWorldTrackingConfiguration.supportedVideoFormats.filter {
            Int($0.imageResolution.width) == 1920 && Int($0.imageResolution.height) == 1440
        }
        if let f = formats.first(where: { $0.framesPerSecond == captureFPS }) ?? formats.first {
            config.videoFormat = f
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)          // LiDAR depth, 256x192
        }
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh                 // LiDAR mesh
        }
        return config
    }

    private func applyMeshVisibility() {
        if showMesh {
            arView.debugOptions.insert(.showSceneUnderstanding)
        } else {
            arView.debugOptions.remove(.showSceneUnderstanding)
        }
    }

    private func apply(_ snapshot: Telemetry, _ newPoints: [SIMD2<Float>]) {
        telemetry = snapshot
        if !newPoints.isEmpty {
            trajectory.append(contentsOf: newPoints)
            if trajectory.count > 6000 {              // keep the map cheap to draw
                trajectory = stride(from: 0, to: trajectory.count, by: 2).map { trajectory[$0] }
            }
        }
    }

    private func startIMU() {
        let processor = self.processor
        if motion.isAccelerometerAvailable {
            motion.accelerometerUpdateInterval = 1.0 / 100.0
            motion.startAccelerometerUpdates(to: imuQueue) { data, _ in
                if let data { processor.ingest(accel: data) }
            }
        }
        if motion.isGyroAvailable {
            motion.gyroUpdateInterval = 1.0 / 100.0
            motion.startGyroUpdates(to: imuQueue) { data, _ in
                if let data { processor.ingest(gyro: data) }
            }
        }
        if motion.isMagnetometerAvailable {
            motion.magnetometerUpdateInterval = 1.0 / 50.0
            motion.startMagnetometerUpdates(to: imuQueue) { data, _ in
                if let data { processor.ingest(mag: data) }
            }
        }
    }

    private static func deviceInfo(lidar: String) -> [String: Any] {
        var sys = utsname()
        uname(&sys)
        let model = withUnsafeBytes(of: &sys.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        return [
            "model": model,
            "system": "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
            "lidar": ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth),
            "depth_resolution": lidar,
        ]
    }
}

/// Runs on the ARKit delegate queue / IMU queue. Feeds the recorder and produces UI snapshots.
final class FrameProcessor: NSObject, ARSessionDelegate, @unchecked Sendable {
    var onTelemetry: ((Telemetry, [SIMD2<Float>]) -> Void)?
    private let recorder: Recorder
    private let engine: SlamEngine

    // ARKit queue state
    private var lastTimestamp: TimeInterval = 0
    private var cameraHz: Double = 0
    private var frameCounter = 0
    private var lastTrajectoryPoint: SIMD3<Float>?
    private var distance: Float = 0
    private var pendingPoints: [SIMD2<Float>] = []
    private var statusOverride: String?

    // IMU queue state (rate read from the ARKit queue under the lock)
    private let imuLock = NSLock()
    private var gyroCount = 0
    private var imuWindowStart: TimeInterval = 0
    private var imuHz: Double = 0

    init(recorder: Recorder, engine: SlamEngine) {
        self.recorder = recorder
        self.engine = engine
    }

    func resetStats() {
        lastTrajectoryPoint = nil
        distance = 0
        pendingPoints.removeAll()
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        recorder.ingest(frame: frame)
        engine.ingest(frame: frame)

        if lastTimestamp > 0 {
            let dt = frame.timestamp - lastTimestamp
            if dt > 0 { cameraHz = cameraHz == 0 ? 1 / dt : cameraHz * 0.95 + 0.05 / dt }
        }
        lastTimestamp = frame.timestamp

        let T = frame.camera.transform
        let p = SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z)
        if case .normal = frame.camera.trackingState {
            if let last = lastTrajectoryPoint {
                let d = simd_distance(last, p)
                if d > 0.03 {
                    distance += d
                    lastTrajectoryPoint = p
                    pendingPoints.append(SIMD2(p.x, p.z))
                }
            } else {
                lastTrajectoryPoint = p
                pendingPoints.append(SIMD2(p.x, p.z))
            }
        }

        frameCounter += 1
        guard frameCounter % 6 == 0 else { return }     // ~10 Hz UI updates

        var t = Telemetry()
        (t.quality, t.trackingText) = Self.describe(frame.camera.trackingState)
        if let statusOverride { t.trackingText = statusOverride; t.quality = .unavailable }
        t.cameraHz = cameraHz
        t.imuHz = imuLock.withLock { imuHz }
        if let depth = frame.sceneDepth?.depthMap {
            t.depthSize = "\(CVPixelBufferGetWidth(depth))×\(CVPixelBufferGetHeight(depth))"
        }
        t.position = p
        let forward = SIMD2<Float>(-T.columns.2.x, -T.columns.2.z)
        let len = simd_length(forward)
        t.heading = len > 1e-3 ? forward / len : SIMD2(0, -1)
        t.meshAnchors = frame.anchors.reduce(0) { $0 + ($1 is ARMeshAnchor ? 1 : 0) }
        t.distance = distance

        let points = pendingPoints
        pendingPoints.removeAll(keepingCapacity: true)
        onTelemetry?(t, points)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        statusOverride = "Error: \(error.localizedDescription)"
    }

    func sessionWasInterrupted(_ session: ARSession) {
        statusOverride = "Interrupted"
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        statusOverride = nil
    }

    // MARK: IMU (called on the IMU queue)

    func ingest(accel d: CMAccelerometerData) {
        // iOS reports g with gravity as -1 g; convert to specific force in m/s^2 (Android / ROS convention).
        let g = -9.80665
        recorder.ingestIMU(.accel, t: d.timestamp, d.acceleration.x * g, d.acceleration.y * g, d.acceleration.z * g)
        engine.ingest(accel: d)
    }

    func ingest(gyro d: CMGyroData) {
        recorder.ingestIMU(.gyro, t: d.timestamp, d.rotationRate.x, d.rotationRate.y, d.rotationRate.z)
        engine.ingest(gyro: d)
        imuLock.withLock {
            gyroCount += 1
            if imuWindowStart == 0 { imuWindowStart = d.timestamp }
            let dt = d.timestamp - imuWindowStart
            if dt >= 1 {
                imuHz = Double(gyroCount) / dt
                gyroCount = 0
                imuWindowStart = d.timestamp
            }
        }
    }

    func ingest(mag d: CMMagnetometerData) {
        recorder.ingestIMU(.mag, t: d.timestamp, d.magneticField.x, d.magneticField.y, d.magneticField.z)
    }

    private static func describe(_ s: ARCamera.TrackingState) -> (TrackingQuality, String) {
        switch s {
        case .normal: return (.good, "Tracking")
        case .notAvailable: return (.unavailable, "No tracking")
        case .limited(let reason):
            switch reason {
            case .initializing: return (.limited, "Initializing")
            case .excessiveMotion: return (.limited, "Moving too fast")
            case .insufficientFeatures: return (.limited, "Low detail")
            case .relocalizing: return (.limited, "Relocalizing")
            @unknown default: return (.limited, "Limited")
            }
        }
    }
}
