import ARKit
import Foundation

/// Exports ARKit LiDAR mesh anchors as a single Wavefront OBJ in world coordinates (metres, +y up).
enum MeshExporter {
    static func writeOBJ(anchors: [ARMeshAnchor], to url: URL) -> (vertices: Int, faces: Int)? {
        guard FileManager.default.createFile(atPath: url.path, contents: nil),
              let handle = try? FileHandle(forWritingTo: url) else { return nil }
        defer { try? handle.close() }

        var buffer = "# SLAM to Go - ARKit LiDAR mesh, world frame, metres, +y up\n"
        buffer.reserveCapacity(1 << 21)
        func flushIfNeeded(force: Bool = false) {
            if force || buffer.utf8.count > (1 << 20) {
                try? handle.write(contentsOf: Data(buffer.utf8))
                buffer.removeAll(keepingCapacity: true)
            }
        }

        var vertexOffset = 0
        var faceTotal = 0
        for anchor in anchors {
            let geometry = anchor.geometry
            let transform = anchor.transform

            let vertices = geometry.vertices
            let vertexBase = vertices.buffer.contents().advanced(by: vertices.offset)
            for i in 0..<vertices.count {
                let v = vertexBase.advanced(by: i * vertices.stride).assumingMemoryBound(to: Float.self)
                let w = transform * SIMD4<Float>(v[0], v[1], v[2], 1)
                buffer += "v \(w.x) \(w.y) \(w.z)\n"
                flushIfNeeded()
            }

            let faces = geometry.faces
            let indexBase = faces.buffer.contents()
            let perFace = faces.indexCountPerPrimitive
            let bytesPerIndex = faces.bytesPerIndex
            for f in 0..<faces.count {
                buffer += "f"
                for k in 0..<perFace {
                    let byteOffset = (f * perFace + k) * bytesPerIndex
                    let index = bytesPerIndex == 4
                        ? Int(indexBase.load(fromByteOffset: byteOffset, as: UInt32.self))
                        : Int(indexBase.load(fromByteOffset: byteOffset, as: UInt16.self))
                    buffer += " \(index + vertexOffset + 1)"
                }
                buffer += "\n"
                flushIfNeeded()
            }
            vertexOffset += vertices.count
            faceTotal += faces.count
        }
        flushIfNeeded(force: true)
        return (vertexOffset, faceTotal)
    }
}
