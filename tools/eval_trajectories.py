"""Compare pose sources on one recording with the same metrics and the same frame pairs.

    python eval_trajectories.py <recording_dir>

Sources: ARKit VIO (trajectory_tum.txt), offline BA (_slam/trajectory_ba_tum.txt),
on-device VLIO core real-time output (_vlio/vlio_frames_tum.txt) and its smoothed keyframes, and the
final loop-closed trajectory (slam_frames_tum.txt on the phone, _vlio/slam_frames_tum.txt from the PC replay).
Metrics: ATE after SE(3) alignment to the BA trajectory, Sim(3) scale, and dense multi-view LiDAR
consistency (LiDAR of frame A reprojected into frame B), which needs no ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from slamtogo.dataset import R_ARKIT_TO_CV, load


def read_tum(path: Path) -> dict[float, np.ndarray]:
    out = {}
    for row in np.loadtxt(path, comments="#", ndmin=2):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        T[:3, 3] = row[1:4]
        out[round(float(row[0]), 4)] = T
    return out


def interpolate(track: dict[float, np.ndarray], query: dict[int, float], max_gap: float = 0.15) -> dict[int, np.ndarray]:
    """Pose at each query time (slerp / lerp between the two nearest samples no further than max_gap)."""
    from scipy.spatial.transform import Slerp
    ts = np.array(sorted(track))
    Ts = np.array([track[t] for t in ts])
    out = {}
    for i, t in query.items():
        k = np.searchsorted(ts, t)
        if k == 0 or k >= len(ts) or ts[k] - ts[k - 1] > max_gap:
            continue
        a, b = ts[k - 1], ts[k]
        w = (t - a) / (b - a)
        T = np.eye(4)
        T[:3, :3] = Slerp([0, 1], Rotation.from_matrix(Ts[k - 1:k + 1, :3, :3]))([w]).as_matrix()[0]
        T[:3, 3] = (1 - w) * Ts[k - 1][:3, 3] + w * Ts[k][:3, 3]
        out[i] = T
    return out


def arkit_to_cv(T: np.ndarray) -> np.ndarray:
    T = T.copy()
    T[:3, :3] = T[:3, :3] @ R_ARKIT_TO_CV
    return T


def umeyama(src: np.ndarray, dst: np.ndarray, scale: bool):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean() if scale else 1.0
    return s, R, md - s * R @ ms


def local_scale(P: dict[int, np.ndarray], ref: dict[int, np.ndarray], ts, window: float = 2.0) -> float:
    """Median |displacement| / |reference displacement| over ~window seconds: metric scale, unaffected by drift."""
    keys = sorted(k for k in P if k in ref)
    t = np.array([ts[k] for k in keys])
    r = []
    for a, ka in enumerate(keys):
        b = np.searchsorted(t, t[a] + window)
        if b >= len(keys):
            break
        kb = keys[b]
        d_ref = np.linalg.norm(ref[kb][:3, 3] - ref[ka][:3, 3])
        if d_ref > 0.10:
            r.append(np.linalg.norm(P[kb][:3, 3] - P[ka][:3, 3]) / d_ref)
    return float(np.median(r)) if r else float("nan")


def consistency(rec, poses: dict[int, np.ndarray], pairs, step: int = 3) -> np.ndarray:
    errs = []
    for a, b in pairs:
        if a not in poses or b not in poses:
            continue
        la, ca, lb = rec.lidar_m(a), rec.confidence(a), rec.lidar_m(b)
        Ka, Kb = rec.K_depth(a, la.shape), rec.K_depth(b, lb.shape)
        v, u = np.nonzero((la > 0.1) & (ca == 2))
        v, u = v[::step], u[::step]
        pc = (np.linalg.inv(Ka) @ np.stack([u, v, np.ones_like(u)]).astype(float)) * la[v, u]
        Ta, Tb = poses[a], poses[b]
        pb = Tb[:3, :3].T @ (Ta[:3, :3] @ pc + Ta[:3, 3:4] - Tb[:3, 3:4])
        z = pb[2]
        uv = (Kb @ pb)[:2] / np.where(z > 1e-6, z, 1)
        ub, vb = np.round(uv[0]).astype(int), np.round(uv[1]).astype(int)
        ok = (z > 0.1) & (ub >= 0) & (vb >= 0) & (ub < lb.shape[1]) & (vb < lb.shape[0])
        db = lb[vb[ok], ub[ok]]
        d = z[ok] - db
        keep = (db > 0.1) & (np.abs(d) < 0.15)
        errs.append(np.abs(d[keep]))
    return np.concatenate(errs) if errs else np.zeros(1)


def load_sources(rec) -> dict[str, dict[int, np.ndarray]]:
    """Camera-to-world poses (OpenCV camera) at every RGB-D frame for every available source."""
    ts = rec.timestamps
    key = lambda t: round(float(t), 4)
    rgbd = [int(i) for i in rec.image_frames]
    sources: dict[str, dict[int, np.ndarray]] = {}
    ark = read_tum(rec.root / "trajectory_tum.txt")
    sources["ARKit VIO"] = {i: arkit_to_cv(ark[key(ts[i])]) for i in rgbd if key(ts[i]) in ark}
    if (rec.root / "_slam/trajectory_ba_tum.txt").exists():
        ba = read_tum(rec.root / "_slam/trajectory_ba_tum.txt")
        sources["Offline BA"] = {i: arkit_to_cv(ba[key(ts[i])]) for i in rgbd if key(ts[i]) in ba}
    q = {i: float(ts[i]) for i in rgbd}
    if (rec.root / "vlio_live_frames_tum.txt").exists():      # live preview during the scan (phone)
        sources["VLIO live"] = interpolate(read_tum(rec.root / "vlio_live_frames_tum.txt"), q)
    if (rec.root / "vlio_frames_tum.txt").exists():           # computed on the phone after the scan
        sources["VLIO on-device"] = interpolate(read_tum(rec.root / "vlio_frames_tum.txt"), q)
        sources["VLIO device KF"] = interpolate(read_tum(rec.root / "vlio_keyframes_tum.txt"), q, max_gap=0.5)
    if (rec.root / "_vlio/vlio_frames_tum.txt").exists():      # PC replay of the same core
        vl = read_tum(rec.root / "_vlio/vlio_frames_tum.txt")
        sources["VLIO PC replay"] = {i: vl[key(ts[i])] for i in rgbd if key(ts[i]) in vl}
        vk = read_tum(rec.root / "_vlio/vlio_keyframes_tum.txt")
        sources["VLIO PC KF"] = {i: vk[key(ts[i])] for i in rgbd if key(ts[i]) in vk}
    # Final trajectories after loop closure (on the phone after processing v2+, or the PC replay).
    if (rec.root / "slam_frames_tum.txt").exists():
        sources["SLAM on-device"] = interpolate(read_tum(rec.root / "slam_frames_tum.txt"), q)
    if (rec.root / "_vlio/slam_frames_tum.txt").exists():
        sources["SLAM PC replay"] = interpolate(read_tum(rec.root / "_vlio/slam_frames_tum.txt"), q)
    return sources


def frame_pairs(rec, reference: dict[int, np.ndarray]):
    """Sequential pairs (3 and 10 RGB-D frames apart) and revisit pairs (> 15 s apart, < 0.6 m, < 30 deg)."""
    ts = rec.timestamps
    rgbd = [i for i in (int(i) for i in rec.image_frames) if i in reference]
    C = np.array([reference[i][:3, 3] for i in rgbd])
    F = np.array([reference[i][:3, 2] for i in rgbd])
    seq = [(rgbd[k], rgbd[k + d]) for d in (3, 10) for k in range(0, len(rgbd) - d, 4)]
    rev = []
    for a in range(0, len(rgbd), 3):
        for b in range(a + 1, len(rgbd)):
            if ts[rgbd[b]] - ts[rgbd[a]] > 15 and np.linalg.norm(C[a] - C[b]) < 0.6 and F[a] @ F[b] > np.cos(np.radians(30)):
                rev.append((rgbd[a], rgbd[b]))
    return seq, rev[:: max(1, len(rev) // 150)]


def evaluate(rec, sources, refname: str, seq, rev) -> dict[str, dict]:
    """Per source: ATE (SE3-aligned), Sim3 scale, local scale, multi-view LiDAR consistency (arrays in metres)."""
    out = {}
    ref = sources[refname]
    for name, P in sources.items():
        common = [i for i in sorted(P) if i in ref]
        A = np.array([P[i][:3, 3] for i in common])
        B = np.array([ref[i][:3, 3] for i in common])
        _, R, t = umeyama(A, B, scale=False)
        s, _, _ = umeyama(A, B, scale=True)
        out[name] = {
            "frames": len(P),
            "ate_m": float(np.sqrt(((A @ R.T + t - B) ** 2).sum(1).mean())),
            "sim3_scale": float(s),
            "local_scale": local_scale(P, ref, rec.timestamps),
            "lidar_seq": consistency(rec, P, seq),
            "lidar_revisit": consistency(rec, P, rev),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    args = ap.parse_args()
    rec = load(args.recording)
    sources = load_sources(rec)
    seq, rev = frame_pairs(rec, sources["ARKit VIO"])
    print(f"{len(rec.image_frames)} RGB-D frames, {len(seq)} sequential pairs, {len(rev)} revisit pairs (>15 s apart)")
    print()
    refname = "Offline BA" if "Offline BA" in sources else "ARKit VIO"
    print(f"{'source':16s} {'frames':>6s} {'ATE vs ' + refname:>18s} {'Sim3 scale':>10s} {'local scale':>11s} "
          f"{'LiDAR seq med/mean':>20s} {'LiDAR revisit med/mean':>24s}")
    for name, r in evaluate(rec, sources, refname, seq, rev).items():
        cs, cr = r["lidar_seq"], r["lidar_revisit"]
        print(f"{name:16s} {r['frames']:6d} {100 * r['ate_m']:15.1f} cm {r['sim3_scale']:10.4f} {r['local_scale']:11.4f} "
              f"{1000 * np.median(cs):9.2f} /{1000 * np.mean(cs):6.2f} mm {1000 * np.median(cr):11.2f} /{1000 * np.mean(cr):6.2f} mm")


if __name__ == "__main__":
    main()
