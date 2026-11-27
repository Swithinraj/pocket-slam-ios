import RealityKit
import SwiftUI

struct ContentView: View {
    @StateObject private var sensors = SensorSession()
    @Environment(\.scenePhase) private var scenePhase

    @AppStorage("imageRate") private var imageRate = 10
    @AppStorage("fullResRGB") private var fullResRGB = false
    @AppStorage("saveMesh") private var saveMesh = true
    @AppStorage("showMap") private var showMap = true
    @AppStorage("showMesh") private var showMesh = true
    @AppStorage("showFeatures") private var showFeatures = true
    @AppStorage("showVlioPath") private var showVlioPath = true
    @AppStorage("captureFPS") private var captureFPS = 60
    @AppStorage("liveVlio") private var liveVlio = true
    @ObservedObject private var processing = ProcessingCenter.shared

    @State private var isRecording = false
    @State private var isSaving = false
    @State private var showDetails = false
    @State private var showSettings = false
    @State private var showRecordings = false
    @State private var toast: String?
    @State private var errorMessage: String?

    var body: some View {
        ZStack {
            ARViewContainer(arView: sensors.arView)
                .ignoresSafeArea()

            if liveVlio && showFeatures {
                FeatureOverlay(points: sensors.features, arView: sensors.arView)
            }

            if isRecording {
                RoundedRectangle(cornerRadius: 44)
                    .strokeBorder(Color.red.opacity(0.8), lineWidth: 4)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            }

            VStack(spacing: 12) {
                topBar
                if showDetails && sensors.isTracking {
                    TelemetryCard(telemetry: sensors.telemetry, vlio: liveVlio ? sensors.vlio : nil)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
                Spacer()
                if showMap && sensors.isTracking {
                    HStack {
                        Spacer()
                        TrajectoryMapView(points: sensors.trajectory,
                                          current: SIMD2(sensors.telemetry.position.x, sensors.telemetry.position.z),
                                          heading: sensors.telemetry.heading,
                                          secondary: liveVlio && showVlioPath ? sensors.vlioTrajectory : [])
                            .frame(width: 150, height: 150)
                    }
                }
                if let job = processing.job, !isRecording {
                    ProcessingBanner(job: job, queued: processing.pending.count)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                if isRecording {
                    RecordingBadge(recorder: sensors.recorder)
                        .transition(.scale.combined(with: .opacity))
                }
                bottomBar
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)

            if isSaving {
                SavingOverlay()
            }

            if let toast {
                VStack {
                    Text(toast)
                        .font(.callout.weight(.medium))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.top, 60)
                    Spacer()
                }
                .transition(.move(edge: .top).combined(with: .opacity))
                .allowsHitTesting(false)
            }
        }
        .preferredColorScheme(.dark)
        .statusBarHidden(true)
        .persistentSystemOverlays(.hidden)
        .animation(.spring(response: 0.35, dampingFraction: 0.85), value: showDetails)
        .animation(.spring(response: 0.35, dampingFraction: 0.85), value: isRecording)
        .animation(.easeInOut(duration: 0.25), value: toast)
        .animation(.easeInOut(duration: 0.25), value: processing.job == nil)
        .onAppear {
            sensors.showMesh = showMesh
            sensors.captureFPS = captureFPS
            sensors.liveVlio = liveVlio
            sensors.start()
        }
        .onChange(of: showMesh) { _, newValue in sensors.showMesh = newValue }
        .onChange(of: captureFPS) { _, fps in sensors.applyCaptureFPS(fps) }
        .onChange(of: liveVlio) { _, on in sensors.liveVlio = on }
        .onChange(of: processing.message) { _, message in
            if let message { flash(message) }
        }
        .onChange(of: sensors.thermal) { _, state in
            // Apple: at .critical stop or sharply reduce camera use. Save the scan before iOS throttles it.
            if state == .critical && isRecording {
                toggleRecording()
                flash("Device too hot: scan saved")
            }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active:
                sensors.start()
            case .background:
                if isRecording { toggleRecording() }
                sensors.pause()
            default:
                break
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(imageRate: $imageRate, fullResRGB: $fullResRGB, saveMesh: $saveMesh, showMap: $showMap,
                         showFeatures: $showFeatures, showVlioPath: $showVlioPath, captureFPS: $captureFPS,
                         liveVlio: $liveVlio)
        }
        .sheet(isPresented: $showRecordings) {
            RecordingsView()
        }
        .alert("Could not start recording", isPresented: Binding(get: { errorMessage != nil },
                                                                  set: { if !$0 { errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage ?? "")
        }
    }

    // MARK: Bars

    private var topBar: some View {
        HStack(spacing: 10) {
            Button {
                showDetails.toggle()
            } label: {
                StatusPill(telemetry: sensors.telemetry, vlio: liveVlio ? sensors.vlio : nil, thermal: sensors.thermal,
                           tracking: sensors.isTracking, expanded: showDetails)
            }
            .buttonStyle(.plain)
            .disabled(!sensors.isTracking)

            Spacer()

            CircleIconButton(systemName: "gearshape") { showSettings = true }
                .disabled(isRecording)
                .opacity(isRecording ? 0.4 : 1)
        }
    }

    private var bottomBar: some View {
        HStack {
            CircleIconButton(systemName: showMesh ? "cube.transparent.fill" : "cube.transparent",
                             size: 52, highlighted: showMesh) {
                showMesh.toggle()
            }
            Spacer()
            RecordButton(isRecording: isRecording, isBusy: isSaving) {
                toggleRecording()
            }
            Spacer()
            CircleIconButton(systemName: "folder", size: 52) { showRecordings = true }
                .disabled(isRecording)
                .opacity(isRecording ? 0.4 : 1)
                .accessibilityLabel("Recordings")
        }
        .padding(.horizontal, 8)
    }

    // MARK: Actions

    private func toggleRecording() {
        guard !isSaving else { return }
        let haptic = UIImpactFeedbackGenerator(style: .medium)
        if isRecording {
            haptic.impactOccurred()
            isRecording = false
            isSaving = true
            sensors.stopRecording { summary in
                isSaving = false
                UIApplication.shared.isIdleTimerDisabled = false
                if let summary {
                    flash("Saved · \(summary.durationText) · \(summary.sizeText) · processing…")
                } else {
                    flash("Recording failed")
                }
            }
        } else {
            do {
                try sensors.startRecording(RecordingSettings(imageRate: imageRate,
                                                             fullResolutionRGB: fullResRGB,
                                                             saveMesh: saveMesh))
                haptic.impactOccurred()
                isRecording = true
                showDetails = false
                UIApplication.shared.isIdleTimerDisabled = true
            } catch {
                ProcessingCenter.shared.setScanning(false)
                errorMessage = error.localizedDescription
            }
        }
    }

    private func flash(_ message: String) {
        toast = message
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
            if toast == message { toast = nil }
        }
    }
}

// MARK: - Components

/// Tracked VLIO features drawn over the camera image (sensor image coords -> view via ARKit's display transform).
struct FeatureOverlay: View {
    let points: [CGPoint]
    let arView: ARView

    var body: some View {
        Canvas { ctx, size in
            guard let frame = arView.session.currentFrame else { return }
            let transform = frame.displayTransform(for: .portrait, viewportSize: size)
            for p in points {
                let q = p.applying(transform)
                let s = CGPoint(x: q.x * size.width, y: q.y * size.height)
                ctx.fill(Path(ellipseIn: CGRect(x: s.x - 2.5, y: s.y - 2.5, width: 5, height: 5)),
                         with: .color(.green.opacity(0.9)))
            }
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }
}

struct ARViewContainer: UIViewRepresentable {
    let arView: ARView
    func makeUIView(context: Context) -> ARView { arView }
    func updateUIView(_ uiView: ARView, context: Context) {}
}

private extension TrackingQuality {
    var color: Color {
        switch self {
        case .good: return .green
        case .limited: return .yellow
        case .unavailable: return .red
        }
    }
}

/// Post-scan processing progress (runs after the scan so scanning itself stays light).
struct ProcessingBanner: View {
    let job: ProcessingCenter.Job
    let queued: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: "gearshape.2.fill").foregroundStyle(.orange)
                Text("Processing scan").font(.footnote.weight(.semibold))
                Spacer()
                Text("\(Int(job.fraction * 100))%").font(.footnote.monospacedDigit().weight(.semibold))
            }
            ProgressView(value: job.fraction).tint(.orange)
            Text(queued > 0 ? "\(job.step) · \(queued) more queued" : job.step)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

struct StatusPill: View {
    let telemetry: Telemetry
    let vlio: VlioStatus?
    let thermal: ProcessInfo.ThermalState
    let tracking: Bool
    let expanded: Bool

    private var dotColor: Color { tracking ? telemetry.quality.color : .white.opacity(0.6) }

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(dotColor)
                .frame(width: 9, height: 9)
                .shadow(color: dotColor.opacity(0.8), radius: 4)
            Text(tracking ? telemetry.trackingText : "Ready to scan")
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)
            if tracking && telemetry.depthSize != "—" {
                chip("LiDAR", .white.opacity(0.15))
            }
            if tracking, let vlio {
                chip(vlio.initialized ? String(format: "VLIO %.0f Hz", vlio.hz) : "VLIO …",
                     vlio.initialized ? Color.orange.opacity(0.35) : Color.white.opacity(0.1))
            }
            switch thermal {
            case .fair: chip("Warm", Color.yellow.opacity(0.35))
            case .serious: chip("Hot", Color.orange.opacity(0.5))
            case .critical: chip("Too hot", Color.red.opacity(0.6))
            default: EmptyView()
            }
            if tracking {
                Image(systemName: "chevron.down")
                    .font(.caption.weight(.bold))
                    .rotationEffect(.degrees(expanded ? 180 : 0))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: Capsule())
        .contentShape(Capsule())
    }

    private func chip(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.bold).monospacedDigit())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color, in: Capsule())
    }
}

struct TelemetryCard: View {
    let telemetry: Telemetry
    let vlio: VlioStatus?

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
            header("ARKit + sensors")
            row("camera.fill", "Camera", String(format: "%.0f Hz", telemetry.cameraHz))
            row("gyroscope", "IMU", String(format: "%.0f Hz", telemetry.imuHz))
            row("sensor.fill", "LiDAR depth", telemetry.depthSize)
            row("location.fill", "Position",
                String(format: "%.2f  %.2f  %.2f m", telemetry.position.x, telemetry.position.y, telemetry.position.z))
            row("figure.walk", "Distance", String(format: "%.2f m", telemetry.distance))
            row("cube.transparent", "Mesh anchors", "\(telemetry.meshAnchors)")
            if let vlio {
                header("VLIO live preview")
                row("circle.dotted", "Features / landmarks", vlio.initialized ? "\(vlio.tracked) / \(vlio.landmarks)" : "starting…")
                row("square.stack.3d.up", "Keyframes / restarts", "\(vlio.keyframes) / \(vlio.reinits)")
                row("timer", "Rate / solve", String(format: "%.0f Hz / %.0f ms", vlio.hz, vlio.solveMs))
                row("arrow.left.and.right", "Offset to ARKit", String(format: "%.1f cm", vlio.drift * 100))
            }
        }
        .font(.footnote)
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func header(_ title: String) -> some View {
        GridRow {
            Text(title.uppercased())
                .font(.caption2.weight(.bold))
                .foregroundStyle(.tertiary)
                .gridCellColumns(2)
                .padding(.top, 2)
        }
    }

    private func row(_ icon: String, _ label: String, _ value: String) -> some View {
        GridRow {
            Label(label, systemImage: icon)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.footnote.monospacedDigit().weight(.medium))
        }
    }
}

struct CircleIconButton: View {
    let systemName: String
    var size: CGFloat = 44
    var highlighted = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: size * 0.4, weight: .semibold))
                .frame(width: size, height: size)
                .foregroundStyle(highlighted ? Color.black : Color.white)
                .background {
                    if highlighted {
                        Circle().fill(Color.white)
                    } else {
                        Circle().fill(.ultraThinMaterial)
                    }
                }
        }
        .buttonStyle(.plain)
    }
}

struct RecordButton: View {
    let isRecording: Bool
    let isBusy: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .strokeBorder(Color.white, lineWidth: 5)
                    .frame(width: 82, height: 82)
                if isBusy {
                    ProgressView().tint(.white)
                } else {
                    RoundedRectangle(cornerRadius: isRecording ? 8 : 33, style: .continuous)
                        .fill(Color.red)
                        .frame(width: isRecording ? 32 : 66, height: isRecording ? 32 : 66)
                }
            }
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isRecording)
        }
        .buttonStyle(.plain)
        .disabled(isBusy)
        .accessibilityLabel(isRecording ? "Stop recording" : "Start recording")
    }
}

struct RecordingBadge: View {
    let recorder: Recorder
    @State private var blink = false

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { context in
            let stats = recorder.currentStats
            let elapsed = context.date.timeIntervalSince(stats.startedAt)
            HStack(spacing: 10) {
                Circle()
                    .fill(Color.red)
                    .frame(width: 10, height: 10)
                    .opacity(blink ? 0.3 : 1)
                Text(RecordingSummary.formatDuration(elapsed))
                    .font(.headline.monospacedDigit())
                Divider().frame(height: 16)
                Text("\(stats.images) RGB-D")
                Text(ByteCountFormatter.string(fromByteCount: stats.bytes, countStyle: .file))
                if stats.droppedImages > 0 {
                    Text("\(stats.droppedImages) dropped").foregroundStyle(.yellow)
                }
                if stats.missedFrames > 0 {
                    Text("\(stats.missedFrames) missed").foregroundStyle(.orange)
                }
            }
            .font(.footnote.monospacedDigit())
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.6).repeatForever()) { blink = true }
        }
    }
}

struct SavingOverlay: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()
            VStack(spacing: 14) {
                ProgressView().controlSize(.large).tint(.white)
                Text("Saving recording…").font(.headline)
                Text("Writing images and exporting the mesh").font(.footnote).foregroundStyle(.secondary)
            }
            .padding(28)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        }
    }
}
