import SwiftUI
import simd

/// Top-down (x, z) view of the camera path. Forward (-z at session start) points up.
struct TrajectoryMapView: View {
    let points: [SIMD2<Float>]
    let current: SIMD2<Float>
    let heading: SIMD2<Float>
    var secondary: [SIMD2<Float>] = []      // VLIO path (same world frame)

    var body: some View {
        Canvas { ctx, size in
            let all = points + secondary + [current]
            var lo = all[0], hi = all[0]
            for p in all {
                lo = simd_min(lo, p)
                hi = simd_max(hi, p)
            }
            let span = max(hi.x - lo.x, hi.y - lo.y, 2.0) * 1.2          // at least 2 m visible
            let center = (lo + hi) / 2
            let side = Float(min(size.width, size.height))
            let scale = side / span

            func toScreen(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(x: size.width / 2 + CGFloat((p.x - center.x) * scale),
                        y: size.height / 2 + CGFloat((p.y - center.y) * scale))
            }

            // 1 m grid
            var grid = Path()
            let half = span / 2
            var gx = (center.x - half).rounded(.down)
            while gx <= center.x + half {
                let x = toScreen(SIMD2(gx, 0)).x
                grid.move(to: CGPoint(x: x, y: 0))
                grid.addLine(to: CGPoint(x: x, y: size.height))
                gx += 1
            }
            var gz = (center.y - half).rounded(.down)
            while gz <= center.y + half {
                let y = toScreen(SIMD2(0, gz)).y
                grid.move(to: CGPoint(x: 0, y: y))
                grid.addLine(to: CGPoint(x: size.width, y: y))
                gz += 1
            }
            ctx.stroke(grid, with: .color(.white.opacity(0.08)), lineWidth: 1)

            // Path
            if points.count > 1 {
                var path = Path()
                path.move(to: toScreen(points[0]))
                for p in points.dropFirst() { path.addLine(to: toScreen(p)) }
                path.addLine(to: toScreen(current))
                ctx.stroke(path, with: .color(.cyan), style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round))
            }

            // VLIO path
            if secondary.count > 1 {
                var path = Path()
                path.move(to: toScreen(secondary[0]))
                for p in secondary.dropFirst() { path.addLine(to: toScreen(p)) }
                ctx.stroke(path, with: .color(.orange), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
            }

            // Start marker
            if let first = points.first {
                let s = toScreen(first)
                ctx.fill(Path(ellipseIn: CGRect(x: s.x - 4, y: s.y - 4, width: 8, height: 8)), with: .color(.green))
            }

            // Current position + heading wedge
            let c = toScreen(current)
            let h = CGPoint(x: CGFloat(heading.x), y: CGFloat(heading.y))
            let n = CGPoint(x: -h.y, y: h.x)
            var wedge = Path()
            wedge.move(to: CGPoint(x: c.x + h.x * 14, y: c.y + h.y * 14))
            wedge.addLine(to: CGPoint(x: c.x + n.x * 6, y: c.y + n.y * 6))
            wedge.addLine(to: CGPoint(x: c.x - n.x * 6, y: c.y - n.y * 6))
            wedge.closeSubpath()
            ctx.fill(wedge, with: .color(.white.opacity(0.85)))
            ctx.fill(Path(ellipseIn: CGRect(x: c.x - 5, y: c.y - 5, width: 10, height: 10)), with: .color(.white))

            // Scale bar (1 m)
            let barLength = CGFloat(scale)
            if barLength < size.width - 20 {
                var bar = Path()
                bar.move(to: CGPoint(x: 10, y: size.height - 12))
                bar.addLine(to: CGPoint(x: 10 + barLength, y: size.height - 12))
                ctx.stroke(bar, with: .color(.white.opacity(0.7)), lineWidth: 2)
                ctx.draw(Text("1 m").font(.caption2).foregroundColor(.white.opacity(0.8)),
                         at: CGPoint(x: 10 + barLength / 2, y: size.height - 22))
            }
        }
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(alignment: .topLeading) {
            HStack(spacing: 8) {
                legend(.cyan, "ARKit")
                if !secondary.isEmpty { legend(.orange, "VLIO") }
            }
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .padding(8)
        }
    }

    private func legend(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 3) {
            Capsule().fill(color).frame(width: 10, height: 3)
            Text(label)
        }
    }
}
