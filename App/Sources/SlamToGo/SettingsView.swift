import SwiftUI

struct SettingsView: View {
    @Binding var imageRate: Int
    @Binding var fullResRGB: Bool
    @Binding var saveMesh: Bool
    @Binding var showMap: Bool
    @Binding var showFeatures: Bool
    @Binding var showVlioPath: Bool
    @Binding var captureFPS: Int
    @Binding var liveVlio: Bool
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("RGB-D rate", selection: $imageRate) {
                        Text("Off").tag(0)
                        Text("5 Hz").tag(5)
                        Text("10 Hz").tag(10)
                        Text("15 Hz").tag(15)
                        Text("30 Hz").tag(30)
                    }
                    .pickerStyle(.segmented)
                    Picker("RGB-D size", selection: $fullResRGB) {
                        Text("960×720").tag(false)
                        Text("1920×1440").tag(true)
                    }
                    Toggle("Export LiDAR mesh on stop", isOn: $saveMesh)
                } header: {
                    Text("Recording")
                } footer: {
                    Text("Poses (60 Hz) and IMU (accelerometer and gyroscope at 100 Hz, magnetometer at 50 Hz) are always recorded. Each RGB-D frame saves the image and a depth map of the same size, pixel-aligned (LiDAR upsampled with RGB-guided edge-preserving filtering), plus the raw 256×192 LiDAR depth and confidence. 1920×1440 at 30 Hz is roughly 1.5 GB per minute.")
                }

                Section {
                    Picker("Camera rate", selection: $captureFPS) {
                        Text("60 fps").tag(60)
                        Text("30 fps").tag(30)
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("Capture")
                } footer: {
                    Text("Scanning only records raw sensor data. Depth alignment and the full-precision visual-LiDAR-inertial optimisation run automatically after each scan (pausing while you scan).")
                }

                Section {
                    Toggle("Trajectory map", isOn: $showMap)
                    Toggle("Live VLIO preview", isOn: $liveVlio)
                    if liveVlio {
                        Toggle("VLIO path on the map", isOn: $showVlioPath)
                        Toggle("Tracked features", isOn: $showFeatures)
                    }
                } header: {
                    Text("Display")
                } footer: {
                    Text("The live preview runs our VLIO core during the scan for monitoring (orange path, green features). It is optional: the precise result is always computed after the scan.")
                }

                Section {
                    LabeledContent("Pose", value: "frames.csv, trajectory_tum.txt")
                    LabeledContent("IMU", value: "accel / gyro / mag .csv")
                    LabeledContent("RGB-D", value: "rgb/*.jpg + depth/*.png (same size, mm)")
                    LabeledContent("LiDAR", value: "lidar/*.png 256×192 (mm)")
                    LabeledContent("Mesh", value: "mesh.obj")
                } header: {
                    Text("Dataset format")
                } footer: {
                    Text("Recordings are in the Files app under On My iPhone › SLAM to Go, and in the Apple Devices app on Windows under Files. All conventions are described in each recording's meta.json.")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .preferredColorScheme(.dark)
    }
}
