"""Offline keyframe bundle adjustment + TSDF fusion for a slamtogo recording.

    python run_ba.py <recording_dir> [--out DIR] [--no-tsdf]

Pipeline
  1. keyframes from RGB-D frames (motion spaced)            4. landmarks: LiDAR back-projection / triangulation
  2. SIFT + LiDAR depth per keypoint (edge/conf filtered)   5. GTSAM BA: reprojection + LiDAR depth (virtual stereo)
  3. matching: sequential window + loop candidates,              + ARKit odometry + gravity attitude, Huber, 2 rounds
     mutual ratio test, MAGSAC fundamental verification    6. full-rate trajectory, TSDF meshes, metrics, plots
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from slamtogo import backend as be
from slamtogo import frontend as fe
from slamtogo.dataset import R_ARKIT_TO_CV, Recording, load


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- metrics
def depth_consistency(rec: Recording, kfs, poses, pairs, step: int = 2) -> np.ndarray:
    """|z_a->b - lidar_b| for LiDAR points of keyframe a reprojected into keyframe b (occlusions excluded)."""
    errs = []
    for a, b in pairs:
        A, B = kfs[a], kfs[b]
        la, ca, lb = rec.lidar_m(A.frame), rec.confidence(A.frame), rec.lidar_m(B.frame)
        Ka, Kb = rec.K_depth(A.frame, la.shape), rec.K_depth(B.frame, lb.shape)
        v, u = np.nonzero((la > 0.1) & (ca == 2))
        v, u = v[::step], u[::step]
        d = la[v, u]
        pc = (np.linalg.inv(Ka) @ np.stack([u, v, np.ones_like(u)]).astype(float)) * d
        pw = poses[a][:3, :3] @ pc + poses[a][:3, 3:4]
        pb = poses[b][:3, :3].T @ (pw - poses[b][:3, 3:4])
        z = pb[2]
        uv = (Kb @ pb)[:2] / np.where(z > 1e-6, z, 1)
        ub, vb = np.round(uv[0]).astype(int), np.round(uv[1]).astype(int)
        inside = (z > 0.1) & (ub >= 0) & (vb >= 0) & (ub < lb.shape[1]) & (vb < lb.shape[0])
        db = lb[vb[inside], ub[inside]]
        diff = z[inside] - db
        keep = (db > 0.1) & (np.abs(diff) < 0.15)
        errs.append(np.abs(diff[keep]))
    return np.concatenate(errs) if errs else np.zeros(0)


# ---------------------------------------------------------------- trajectory
def corrected_trajectory(rec: Recording, kfs, poses) -> list[np.ndarray]:
    """Propagate keyframe corrections C_k = T_ba,k * T_arkit,k^-1 to all frames (slerp/lerp in time)."""
    t_kf = rec.timestamps[[k.frame for k in kfs]]
    C = np.array([P @ np.linalg.inv(k.T0) for k, P in zip(kfs, poses)])
    out = []
    if len(kfs) == 1:
        return [C[0] @ rec.T_world_cam_arkit(i) for i in range(len(rec.frames))]
    slerp = Slerp(t_kf, Rotation.from_matrix(C[:, :3, :3]))
    for i, t in enumerate(rec.timestamps):
        tc = np.clip(t, t_kf[0], t_kf[-1])
        Ct = np.eye(4)
        Ct[:3, :3] = slerp([tc]).as_matrix()[0]
        Ct[:3, 3] = [np.interp(tc, t_kf, C[:, r, 3]) for r in range(3)]
        out.append(Ct @ rec.T_world_cam_arkit(i))
    return out


def write_tum(path: Path, ts, poses_arkit) -> None:
    with path.open("w") as fh:
        fh.write("# timestamp tx ty tz qx qy qz qw  (ARKit world, camera-to-world, bundle adjusted)\n")
        for t, T in zip(ts, poses_arkit):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            fh.write(f"{t:.6f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}\n")


# ---------------------------------------------------------------- fusion
def tsdf_fuse(rec: Recording, frames, poses_cv, path: Path, voxel: float = 0.01) -> int:
    """Sparse TSDF (Open3D VoxelBlockGrid; the legacy ScalableTSDFVolume integrates nothing in this build)."""
    import open3d as o3d
    import open3d.core as o3c
    dev = o3c.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=16, block_count=100000, device=dev)
    for i, T in zip(frames, poses_cv):
        rgb = np.ascontiguousarray(rec.rgb(i)[:, :, ::-1])
        if rec.has_raw_lidar:
            depth, color, K = rec.depth_aligned_m(i), rgb, rec.K_rgb(i)
        else:
            depth = rec.lidar_m(i)
            color = np.ascontiguousarray(cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA))
            K = rec.K_depth(i, depth.shape)
        conf = cv2.resize(rec.confidence(i), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth = np.where(conf >= 1, depth, 0)
        d_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray((depth * 1000).astype(np.uint16)))).to(dev)
        c_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(color, dtype=np.uint8))).to(dev)
        intr = o3c.Tensor(K, dtype=o3c.float64)
        extr = o3c.Tensor(np.linalg.inv(T), dtype=o3c.float64)
        blocks = vbg.compute_unique_block_coordinates(d_img, intr, extr, 1000.0, 4.0)
        vbg.integrate(blocks, d_img, c_img, intr, intr, extr, 1000.0, 4.0)
    mesh = vbg.extract_triangle_mesh().to_legacy()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh)
    pcd = vbg.extract_point_cloud().to_legacy()                  # surface points with colour
    o3d.io.write_point_cloud(str(path.with_name(path.stem + "_points.ply")), pcd)
    return len(mesh.triangles)


def write_ply_points(path: Path, pts: np.ndarray) -> None:
    with path.open("w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(fh, pts, fmt="%.5f")


# ---------------------------------------------------------------- report
def plot(out: Path, rec, traj_arkit, traj_ba, kfs, poses, r_ark, r_ba, dc_ark, dc_ba) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    pa = np.array([T[:3, 3] for T in traj_arkit])
    pb = np.array([T[:3, 3] for T in traj_ba])
    ax[0].plot(pa[:, 0], pa[:, 2], "-", color="0.6", label="ARKit VIO")
    ax[0].plot(pb[:, 0], pb[:, 2], "-", color="C0", label="Bundle adjusted")
    ax[0].plot([P[0, 3] for P in poses], [P[2, 3] for P in poses], "o", ms=3, color="C3", label="keyframes")
    ax[0].set_aspect("equal"); ax[0].invert_yaxis(); ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("z [m]")
    ax[0].set_title("Top-down trajectory"); ax[0].legend()
    bins = np.linspace(0, 4, 60)
    ax[1].hist(r_ark, bins, alpha=0.6, color="0.5", label=f"ARKit poses  RMSE {np.sqrt(np.mean(r_ark**2)):.2f}px")
    ax[1].hist(r_ba, bins, alpha=0.6, color="C0", label=f"BA poses     RMSE {np.sqrt(np.mean(r_ba**2)):.2f}px")
    ax[1].set_xlabel("reprojection error [px]"); ax[1].set_title("Reprojection"); ax[1].legend()
    bins = np.linspace(0, 0.06, 60)
    ax[2].hist(dc_ark, bins, alpha=0.6, color="0.5", label=f"ARKit median {1000*np.median(dc_ark):.1f} mm")
    ax[2].hist(dc_ba, bins, alpha=0.6, color="C0", label=f"BA median {1000*np.median(dc_ba):.1f} mm")
    ax[2].set_xlabel("|depth disagreement| between views [m]"); ax[2].set_title("Multi-view LiDAR consistency"); ax[2].legend()
    fig.tight_layout()
    fig.savefig(out / "ba_report.png", dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--out")
    ap.add_argument("--min-trans", type=float, default=0.05)
    ap.add_argument("--min-rot", type=float, default=5.0)
    ap.add_argument("--features", type=int, default=2000)
    ap.add_argument("--no-tsdf", action="store_true")
    args = ap.parse_args()

    rec = load(args.recording)
    out = Path(args.out) if args.out else rec.root / "_slam"
    out.mkdir(parents=True, exist_ok=True)
    log(f"{rec.root.name}: {len(rec.frames)} frames, {len(rec.image_frames)} RGB-D, "
        f"{'raw LiDAR + aligned depth' if rec.has_raw_lidar else 'v1 (LiDAR only)'}")

    kf_frames = fe.select_keyframes(rec, args.min_trans, args.min_rot)
    if len(kf_frames) < 3:
        kf_frames = [int(i) for i in rec.image_frames]          # tiny clip: use every RGB-D frame
    kfs = fe.extract(rec, kf_frames, args.features)
    n_depth = sum(np.isfinite(k.depth).sum() for k in kfs)
    n_kp = sum(len(k.kps) for k in kfs)
    log(f"{len(kfs)} keyframes, {n_kp} SIFT features, {100 * n_depth / max(n_kp, 1):.0f}% with valid LiDAR depth")

    pairs = fe.candidate_pairs(kfs)
    matches, loops = {}, 0
    for a, b, is_loop in pairs:
        m = fe.match_pair(kfs[a], kfs[b])
        if m is not None:
            matches[(a, b)] = m
            loops += is_loop
    log(f"{len(pairs)} candidate pairs -> {len(matches)} verified ({loops} loop closures)")

    tracks = fe.build_tracks(matches)
    landmarks = be.init_landmarks(kfs, tracks)
    log(f"{len(tracks)} tracks -> {len(landmarks)} landmarks, "
        f"{sum(len(l.obs) for l in landmarks)} observations")

    cfg = be.BAConfig()
    # Round 1 + outlier rejection, round 2
    r1 = be.bundle_adjust(kfs, be.reject_outliers(kfs, [k.T0 for k in kfs], landmarks, max_px=8.0), cfg)
    inliers = be.reject_outliers(kfs, r1.poses, r1.landmarks, max_px=3.0)
    final = be.bundle_adjust(kfs, inliers, cfg, init_poses=r1.poses)
    log(f"BA: {final.stats['factors']} factors ({final.stats['depth_observations']} depth, "
        f"{final.stats['mono_observations']} mono), error {final.stats['graph_error_initial']:.1f} -> "
        f"{final.stats['graph_error_final']:.1f}")

    # Fair baseline: same inlier landmarks, poses fixed to ARKit, structure optimised
    base = be.bundle_adjust(kfs, inliers, be.BAConfig(fix_poses=True))
    r_ark, d_ark = be.residuals(kfs, base.poses, base.landmarks)
    r_ba, d_ba = be.residuals(kfs, final.poses, final.landmarks)

    verified = list(matches.keys())
    dc_ark = depth_consistency(rec, kfs, [k.T0 for k in kfs], verified)
    dc_ba = depth_consistency(rec, kfs, final.poses, verified)

    corr_t = [np.linalg.norm(P[:3, 3] - k.T0[:3, 3]) for k, P in zip(kfs, final.poses)]
    corr_r = [fe.rotation_angle_deg(k.T0[:3, :3].T @ P[:3, :3]) for k, P in zip(kfs, final.poses)]

    # Outputs
    traj_ba = corrected_trajectory(rec, kfs, final.poses)
    traj_ark = [rec.T_world_cam_arkit(i) for i in range(len(rec.frames))]
    write_tum(out / "trajectory_ba_tum.txt", rec.timestamps, traj_ba)
    write_ply_points(out / "landmarks_ba.ply", np.array([l.p for l in final.landmarks]))
    plot(out, rec, traj_ark, traj_ba, kfs, final.poses, r_ark, r_ba, dc_ark, dc_ba)

    summary = {
        "keyframes": len(kfs), "verified_pairs": len(matches), "loop_closures": loops,
        "landmarks": len(final.landmarks), **final.stats,
        "reprojection_rmse_px": {"arkit": float(np.sqrt(np.mean(r_ark ** 2))), "ba": float(np.sqrt(np.mean(r_ba ** 2)))},
        "feature_depth_rmse_mm": {"arkit": float(1000 * np.sqrt(np.mean(d_ark ** 2))), "ba": float(1000 * np.sqrt(np.mean(d_ba ** 2)))},
        "multiview_lidar_consistency_mm": {
            "arkit_median": float(1000 * np.median(dc_ark)), "ba_median": float(1000 * np.median(dc_ba)),
            "arkit_mean": float(1000 * np.mean(dc_ark)), "ba_mean": float(1000 * np.mean(dc_ba))},
        "keyframe_correction": {"trans_mean_mm": float(1000 * np.mean(corr_t)), "trans_max_mm": float(1000 * np.max(corr_t)),
                                "rot_mean_deg": float(np.mean(corr_r)), "rot_max_deg": float(np.max(corr_r))},
    }

    if not args.no_tsdf:
        rgbd = [int(i) for i in rec.image_frames]
        to_cv = lambda T: np.block([[T[:3, :3] @ R_ARKIT_TO_CV, T[:3, 3:4]], [np.zeros((1, 3)), np.ones((1, 1))]])
        n1 = tsdf_fuse(rec, rgbd, [to_cv(traj_ark[i]) for i in rgbd], out / "tsdf_arkit.ply")
        n2 = tsdf_fuse(rec, rgbd, [to_cv(traj_ba[i]) for i in rgbd], out / "tsdf_ba.ply")
        summary["tsdf_triangles"] = {"arkit": n1, "ba": n2}
        log(f"TSDF meshes: {n1} / {n2} triangles")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    s = summary
    print("\n================ Bundle adjustment vs ARKit VIO ================")
    print(f"keyframes {s['keyframes']}, verified pairs {s['verified_pairs']} ({s['loop_closures']} loops), landmarks {s['landmarks']}")
    print(f"reprojection RMSE        ARKit {s['reprojection_rmse_px']['arkit']:.3f} px  ->  BA {s['reprojection_rmse_px']['ba']:.3f} px")
    print(f"feature depth RMSE       ARKit {s['feature_depth_rmse_mm']['arkit']:.1f} mm  ->  BA {s['feature_depth_rmse_mm']['ba']:.1f} mm")
    mv = s["multiview_lidar_consistency_mm"]
    print(f"multi-view LiDAR median  ARKit {mv['arkit_median']:.2f} mm  ->  BA {mv['ba_median']:.2f} mm   (mean {mv['arkit_mean']:.2f} -> {mv['ba_mean']:.2f})")
    kc = s["keyframe_correction"]
    print(f"pose correction          mean {kc['trans_mean_mm']:.1f} mm / {kc['rot_mean_deg']:.3f} deg, max {kc['trans_max_mm']:.1f} mm / {kc['rot_max_deg']:.3f} deg")
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
