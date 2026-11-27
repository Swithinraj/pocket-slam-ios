import SwiftUI
import UIKit

struct RecordingsView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var processing = ProcessingCenter.shared
    @State private var recordings: [RecordingSummary] = []
    @State private var preparing: String?
    @State private var shareItem: ShareItem?
    @State private var pendingDelete: [RecordingSummary] = []
    @State private var selecting = false
    @State private var selection = Set<String>()
    @State private var deleting = false

    private var totalBytes: Int64 { recordings.reduce(0) { $0 + $1.bytes } }

    var body: some View {
        NavigationStack {
            Group {
                if recordings.isEmpty {
                    ContentUnavailableView("No recordings yet",
                                           systemImage: "record.circle",
                                           description: Text("Tap the red button to record a session."))
                } else {
                    List(selection: $selection) {
                        Section {
                            ForEach(recordings) { recording in
                                row(recording)
                                    .tag(recording.id)
                                    .swipeActions {
                                        Button(role: .destructive) {
                                            pendingDelete = [recording]
                                        } label: {
                                            Label("Delete", systemImage: "trash")
                                        }
                                    }
                            }
                        } header: {
                            Text("\(recordings.count) recordings · \(ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file))")
                        } footer: {
                            Text("Swipe left on a recording, use its ⋯ menu, or tap Select to delete several at once.")
                        }
                    }
                    .environment(\.editMode, .constant(selecting ? .active : .inactive))
                }
            }
            .navigationTitle("Recordings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    if !recordings.isEmpty {
                        Button(selecting ? "Cancel" : "Select") {
                            selecting.toggle()
                            selection.removeAll()
                        }
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
                if selecting {
                    ToolbarItemGroup(placement: .bottomBar) {
                        Button(selection.count == recordings.count ? "Deselect All" : "Select All") {
                            selection = selection.count == recordings.count ? [] : Set(recordings.map(\.id))
                        }
                        Spacer()
                        Button(role: .destructive) {
                            pendingDelete = recordings.filter { selection.contains($0.id) }
                        } label: {
                            Text(selection.isEmpty ? "Delete" : "Delete (\(selection.count))")
                        }
                        .disabled(selection.isEmpty)
                    }
                }
            }
            .onAppear(perform: reload)
            .onChange(of: processing.job == nil) { _, _ in reload() }
            .sheet(item: $shareItem) { item in
                ActivityView(items: [item.url])
            }
            .confirmationDialog(deleteTitle,
                                isPresented: Binding(get: { !pendingDelete.isEmpty },
                                                     set: { if !$0 { pendingDelete = [] } }),
                                titleVisibility: .visible) {
                Button("Delete \(ByteCountFormatter.string(fromByteCount: pendingDelete.reduce(0) { $0 + $1.bytes }, countStyle: .file))",
                       role: .destructive) {
                    delete(pendingDelete)
                }
            } message: {
                Text("This permanently removes the recorded images, depth, IMU data and results from the phone.")
            }
            .overlay {
                if deleting {
                    ProgressView("Deleting…")
                        .padding(24)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    private var deleteTitle: String {
        pendingDelete.count == 1 ? "Delete this recording?" : "Delete \(pendingDelete.count) recordings?"
    }

    private func row(_ recording: RecordingSummary) -> some View {
        HStack(spacing: 12) {
            Image(systemName: recording.hasMesh ? "cube.transparent.fill" : "point.3.connected.trianglepath.dotted")
                .font(.title3)
                .foregroundStyle(.cyan)
                .frame(width: 32)
            VStack(alignment: .leading, spacing: 3) {
                Text(recording.title)
                    .font(.body.weight(.medium))
                Text("\(recording.durationText) · \(recording.rgbdFrames) RGB-D · \(recording.sizeText)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                processingStatus(recording)
            }
            Spacer()
            if preparing == recording.id {
                ProgressView()
            } else if !selecting {
                Menu {
                    Button {
                        share(recording)
                    } label: {
                        Label("Share (zip)", systemImage: "square.and.arrow.up")
                    }
                    .disabled(preparing != nil)
                    if !recording.processed && processing.fraction(for: recording.url) == nil && !processing.isQueued(recording.url) {
                        Button {
                            processing.enqueue(recording.url)
                        } label: {
                            Label("Process", systemImage: "gearshape.2")
                        }
                    }
                    Divider()
                    Button(role: .destructive) {
                        pendingDelete = [recording]
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.title3)
                        .frame(width: 36, height: 36)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Actions for \(recording.title)")
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func processingStatus(_ recording: RecordingSummary) -> some View {
        if let f = processing.fraction(for: recording.url) {
            HStack(spacing: 6) {
                ProgressView(value: f).tint(.orange).frame(width: 80)
                Text("Processing \(Int(f * 100))%").font(.caption2).foregroundStyle(.orange)
            }
        } else if processing.isQueued(recording.url) {
            Label("Queued for processing", systemImage: "clock").font(.caption2).foregroundStyle(.secondary)
        } else if recording.processed {
            Label("Processed", systemImage: "checkmark.seal.fill").font(.caption2).foregroundStyle(.green)
        } else {
            Label("Not processed", systemImage: "circle.dashed").font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func reload() {
        recordings = RecordingSummary.all()
        selection.formIntersection(recordings.map(\.id))
    }

    /// Stops any processing of these recordings, then removes their folders off the main thread
    /// (a scan can be thousands of files).
    private func delete(_ items: [RecordingSummary]) {
        let urls = items.map(\.url)
        pendingDelete = []
        deleting = true
        let group = DispatchGroup()
        for url in urls {
            group.enter()
            processing.discard(url) { group.leave() }
        }
        group.notify(queue: .global(qos: .userInitiated)) {
            let fm = FileManager.default
            for url in urls {
                try? fm.removeItem(at: url)
                try? fm.removeItem(at: fm.temporaryDirectory.appendingPathComponent(url.lastPathComponent + ".zip"))
            }
            DispatchQueue.main.async {
                deleting = false
                selecting = false
                selection.removeAll()
                reload()
            }
        }
    }

    private func share(_ recording: RecordingSummary) {
        preparing = recording.id
        let url = recording.url
        DispatchQueue.global(qos: .userInitiated).async {
            let zipped = try? RecordingSummary.zip(url)
            DispatchQueue.main.async {
                preparing = nil
                if let zipped { shareItem = ShareItem(url: zipped) }
            }
        }
    }
}

struct ShareItem: Identifiable {
    let url: URL
    var id: String { url.path }
}

struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
