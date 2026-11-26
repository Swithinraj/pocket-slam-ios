import Foundation

/// A finished recording on disk, read back from its meta.json.
struct RecordingSummary: Identifiable, Hashable {
    let url: URL
    let duration: Double
    let frames: Int
    let rgbdFrames: Int
    let bytes: Int64
    let hasMesh: Bool
    let processed: Bool

    var id: String { url.lastPathComponent }
    var name: String { url.lastPathComponent }

    var date: Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return formatter.date(from: name)
    }

    var title: String {
        guard let date else { return name }
        return date.formatted(date: .abbreviated, time: .standard)
    }

    var durationText: String { Self.formatDuration(duration) }
    var sizeText: String { ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file) }

    static func formatDuration(_ seconds: Double) -> String {
        let s = max(0, Int(seconds.rounded()))
        return String(format: "%d:%02d", s / 60, s % 60)
    }

    static func load(from url: URL) -> RecordingSummary {
        var duration = 0.0, frames = 0, rgbd = 0
        var hasMesh = FileManager.default.fileExists(atPath: url.appendingPathComponent("mesh.obj").path)
        if let data = try? Data(contentsOf: url.appendingPathComponent("meta.json")),
           let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            duration = meta["duration_s"] as? Double ?? 0
            frames = meta["frames"] as? Int ?? 0
            rgbd = meta["rgbd_frames"] as? Int ?? 0
            hasMesh = hasMesh || meta["mesh"] != nil
        }
        return RecordingSummary(url: url, duration: duration, frames: frames, rgbdFrames: rgbd,
                                bytes: directorySize(url), hasMesh: hasMesh,
                                processed: RecordingProcessor.status(of: url) == "done")
    }

    static func all() -> [RecordingSummary] {
        let root = Recorder.recordingsRoot
        let dirs = (try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])) ?? []
        return dirs
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
            .map(load(from:))
    }

    static func directorySize(_ url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    /// Zips the recording folder (system zip via NSFileCoordinator) into the temp directory for sharing.
    static func zip(_ url: URL) throws -> URL {
        var coordinatorError: NSError?
        var result: Result<URL, Error> = .failure(CocoaError(.fileReadUnknown))
        NSFileCoordinator().coordinate(readingItemAt: url, options: .forUploading, error: &coordinatorError) { zipURL in
            let destination = FileManager.default.temporaryDirectory.appendingPathComponent(url.lastPathComponent + ".zip")
            try? FileManager.default.removeItem(at: destination)
            do {
                try FileManager.default.copyItem(at: zipURL, to: destination)
                result = .success(destination)
            } catch {
                result = .failure(error)
            }
        }
        if let coordinatorError { throw coordinatorError }
        return try result.get()
    }
}
