"""Interactive 3D viewer for a processed recording (run run_ba.py first).

    python view_reconstruction.py <recording_dir> [--screenshot out.png]

Keys:  P  point cloud / mesh          B  BA poses / ARKit poses reconstruction
       T  show / hide trajectories    L  show / hide ARKit LiDAR mesh (mesh.obj)
       mouse: left-drag rotate, right-drag / ctrl+drag pan, wheel zoom.   Q / Esc: quit
Trajectories: orange = our VLIO SLAM, blue = bundle adjusted, grey = ARKit VIO, green sphere = start.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def load_tum(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    return data[:, 1:4]


def line_set(points: np.ndarray, color, step: int = 3) -> o3d.geometry.LineSet:
    pts = points[::step]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                              o3d.utility.Vector2iVector(np.stack([np.arange(len(pts) - 1), np.arange(1, len(pts))], 1)))
    ls.paint_uniform_color(color)
    return ls


def vlio_in_arkit_world(root: Path) -> np.ndarray | None:
    """Our VLIO trajectory (own z-up world, arbitrary yaw) rigidly aligned onto the ARKit trajectory.
    Prefers the full-precision on-device result, then the live preview, then the PC replay."""
    for name in ("vlio_frames_tum.txt", "vlio_live_frames_tum.txt", "_vlio/vlio_frames_tum.txt"):
        path = root / name
        if path.exists() and path.stat().st_size > 200:
            break
    else:
        return None
    v = np.loadtxt(path, comments="#", ndmin=2)
    a = np.loadtxt(root / "trajectory_tum.txt", comments="#", ndmin=2)
    if len(v) < 3:
        return None
    A = np.stack([np.interp(v[:, 0], a[:, 0], a[:, k]) for k in (1, 2, 3)], 1)
    V = v[:, 1:4]
    mv, ma = V.mean(0), A.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (V - mv))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    print(f"VLIO trajectory: {path.relative_to(root)} ({len(v)} poses)")
    return (V - mv) @ R.T + ma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--screenshot", help="render one image offscreen and exit")
    args = ap.parse_args()
    root = Path(args.recording)
    slam = root / "_slam"

    geoms: dict[str, o3d.geometry.Geometry] = {}
    for tag in ("ba", "arkit"):
        p = slam / f"tsdf_{tag}_points.ply"
        m = slam / f"tsdf_{tag}.ply"
        if p.exists():
            geoms[f"points_{tag}"] = o3d.io.read_point_cloud(str(p))
        if m.exists():
            mesh = o3d.io.read_triangle_mesh(str(m))
            mesh.compute_vertex_normals()
            geoms[f"mesh_{tag}"] = mesh
    if (root / "mesh.obj").exists():
        lm = o3d.io.read_triangle_mesh(str(root / "mesh.obj"))
        lm.compute_vertex_normals()
        lm.paint_uniform_color([0.85, 0.55, 0.2])
        geoms["lidar_mesh"] = lm
    if not any(k.startswith(("points", "mesh")) for k in geoms):
        raise SystemExit(f"No reconstruction in {slam}. Run: python run_ba.py {root}")

    arkit = np.loadtxt(root / "trajectory_tum.txt", comments="#")[:, 1:4]
    traj = [line_set(arkit, [0.55, 0.55, 0.55])]
    if (slam / "trajectory_ba_tum.txt").exists():
        traj.append(line_set(load_tum(slam / "trajectory_ba_tum.txt"), [0.1, 0.45, 1.0]))
    vlio = vlio_in_arkit_world(root)
    if vlio is not None:
        traj.append(line_set(vlio, [1.0, 0.55, 0.0]))
    start = o3d.geometry.TriangleMesh.create_sphere(0.04)
    start.translate(arkit[0])
    start.paint_uniform_color([0.1, 0.8, 0.2])
    traj.append(start)

    state = {"kind": "points", "poses": "ba" if "points_ba" in geoms else "arkit", "traj": True, "lidar": False}

    def scene():
        key = f"{state['kind']}_{state['poses']}"
        out = [geoms[key]] if key in geoms else []
        if state["traj"]:
            out += traj
        if state["lidar"] and "lidar_mesh" in geoms:
            out.append(geoms["lidar_mesh"])
        return out

    def describe():
        print(f"showing: {state['kind']} | poses: {state['poses'].upper()} | trajectories: {'on' if state['traj'] else 'off'}"
              f" | ARKit mesh: {'on' if state['lidar'] else 'off'}", flush=True)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f"SLAM to Go - {root.name}", width=1400, height=900, visible=args.screenshot is None)
    for g in scene():
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.08, 0.08, 0.1])
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_up([0.0, 1.0, 0.0])                     # ARKit world: +y up
    ctr.set_front([0.35, 0.8, 0.5])                 # look down at the room at an angle
    ctr.set_zoom(0.6)

    def redraw(v):
        cam = v.get_view_control().convert_to_pinhole_camera_parameters()
        v.clear_geometries()
        for g in scene():
            v.add_geometry(g, reset_bounding_box=False)
        v.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
        describe()
        return False

    def toggle(field, values):
        def cb(v):
            state[field] = values[(values.index(state[field]) + 1) % len(values)]
            return redraw(v)
        return cb

    vis.register_key_callback(ord("P"), toggle("kind", ["points", "mesh"]))
    vis.register_key_callback(ord("B"), toggle("poses", ["ba", "arkit"]))
    vis.register_key_callback(ord("T"), toggle("traj", [True, False]))
    vis.register_key_callback(ord("L"), toggle("lidar", [False, True]))

    if args.screenshot:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.screenshot, do_render=True)
        vis.destroy_window()
        print(f"saved {args.screenshot}")
        return
    print(__doc__)
    describe()
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
