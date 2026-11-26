import SwiftUI
import UIKit

struct RecordingsView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var processing = ProcessingCenter.shared
    @State private var recordings: [RecordingSummary] = []
    @State private var preparing: String?
    @State private var shareItem: ShareItem?
    @State private var pendingDelete: RecordingSummary?

    var body: some View {
        NavigationStack {
            Group {
                if recordings.isEmpty {
                    ContentUnavailableView("No recordings yet",
                                           systemImage: "record.circle",
                                           description: Text("Tap the red button to record a session."))
                } else {
                    List {
                        Section {
                            ForEach(recordings) { recording in
                                row(recording)
                                    .swipeActions {
                                        Button(role: .destructive) {
                                            pendingDelete = recording
                                        } label: {
                                            Label("Delete", systemImage: "trash")
                                        }
                                    }
                            }
                        } footer: {
                            Text("Total \(ByteCountFormatter.string(fromByteCount: recordings.reduce(0) { $0 + $1.bytes }, countStyle: .file))")
                        }
                    }
                }
            }
            .navigationTitle("Recordings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear(perform: reload)
            .onChange(of: processing.job == nil) { _, _ in reload() }
            .sheet(item: $shareItem) { item in
                ActivityView(items: [item.url])
            }
            .confirmationDialog("Delete this recording?",
                                isPresented: Binding(get: { pendingDelete != nil },
                                                     set: { if !$0 { pendingDelete = nil } }),
                                titleVisibility: .visible,
                                presenting: pendingDelete) { recording in
                Button("Delete \(recording.sizeText)", role: .destructive) {
                    try? FileManager.default.removeItem(at: recording.url)
                    reload()
                }
            }
        }
        .preferredColorScheme(.dark)
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
            } else {
                Button {
                    share(recording)
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .buttonStyle(.borderless)
                .disabled(preparing != nil)
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
            Button {
                processing.enqueue(recording.url)
            } label: {
                Label("Process", systemImage: "gearshape.2").font(.caption2.weight(.semibold))
            }
            .buttonStyle(.borderless)
        }
    }

    private func reload() {
        recordings = RecordingSummary.all()
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
