"""Sanity-check a slamtogo recording: timing, rates, depth encoding/alignment, IMU axes vs ARKit.

Usage: python inspect_recording.py <recording_dir> [--out <dir>]
Writes overlay images to <out> (default: <recording_dir>/_inspect).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from slamtogo.dataset import R_CAM_DEV, load

OK, WARN, BAD = "OK  ", "WARN", "FAIL"


def report(status: str, msg: str) -> None:
    print(f"[{status}] {msg}")


def rate_stats(t: np.ndarray) -> tuple[float, float, float]:
    dt = np.diff(t)
    return 1.0 / np.median(dt), dt.max() * 1000, (dt <= 0).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--out")
    args = ap.parse_args()
    rec = load(args.recording)
    out = Path(args.out) if args.out else rec.root / "_inspect"
    out.mkdir(parents=True, exist_ok=True)
    m = rec.meta
    print(f"Recording {rec.root.name}: {m['device']['model']} {m['device']['system']}, "
          f"{m['duration_s']:.1f} s, format {m['format']}\n")

    # ---------- timing ----------
    print("== Streams ==")
    t = rec.timestamps
    for name, ts, expect in [("frames", t, 60), ("accel", rec.accel[:, 0], 100),
                             ("gyro", rec.gyro[:, 0], 100), ("mag", rec.mag[:, 0], 50)]:
        if len(ts) < 2:
            report(BAD, f"{name}: no data")
            continue
        hz, max_gap, nonmono = rate_stats(ts)
        st = OK if abs(hz - expect) / expect < 0.1 and nonmono == 0 and max_gap < 5000 / expect else WARN
        report(st, f"{name:6s} n={len(ts):5d}  {hz:6.1f} Hz (expect {expect})  max gap {max_gap:6.1f} ms  non-monotonic {nonmono}")
    span = (max(t[0], rec.gyro[0, 0], rec.accel[0, 0]), min(t[-1], rec.gyro[-1, 0], rec.accel[-1, 0]))
    report(OK if span[1] - span[0] > 0.9 * (t[-1] - t[0]) else WARN,
           f"camera/IMU overlap {span[1] - span[0]:.2f} s of {t[-1] - t[0]:.2f} s (same clock)")
    trk = rec.frames["tracking"]
    report(OK if (trk == 2).mean() > 0.9 else WARN,
           f"tracking normal {100 * (trk == 2).mean():.1f}%  limited {100 * (trk == 1).mean():.1f}%  none {100 * (trk == 0).mean():.1f}%")

    # ---------- trajectory ----------
    print("\n== Trajectory (ARKit VIO) ==")
    P = np.stack([rec.frames[k] for k in ("tx", "ty", "tz")], axis=1)
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    speed = steps / np.diff(t)
    report(OK, f"path length {steps.sum():.2f} m, bbox {np.ptp(P, axis=0).round(2)} m, "
               f"mean speed {speed.mean():.2f} m/s, max {speed.max():.2f} m/s")
    jumps = np.flatnonzero(speed > 1.5)           # a real jump implies an impossible hand-held speed
    report(OK if len(jumps) == 0 else WARN, f"pose jumps (implied speed > 1.5 m/s): {len(jumps)}")
    gaps = np.diff(t) > 1.5 / 60
    missed = int(np.round(np.diff(t)[gaps] * 60).sum() - gaps.sum())
    report(OK if missed < 0.01 * len(t) else WARN,
           f"ARKit frames missed by the app: {missed} ({100 * missed / (len(t) + missed):.1f}%), "
           f"{gaps.sum()} gaps, longest {1000 * np.diff(t).max():.0f} ms")

    # ---------- IMU axes vs ARKit ----------
    print("\n== IMU frame check against ARKit orientation ==")
    R_wc = Rotation.from_quat(np.stack([rec.frames[k] for k in ("qx", "qy", "qz", "qw")], axis=1))
    acc_t = rec.accel[:, 0]
    inside = (acc_t >= t[0]) & (acc_t <= t[-1])
    idx = np.clip(np.searchsorted(t, acc_t[inside]), 0, len(t) - 1)
    a_dev = rec.accel[inside, 1:4]
    a_world = R_wc[idx].apply((R_CAM_DEV @ a_dev.T).T)
    g_mean = a_world.mean(axis=0)
    ang = np.degrees(np.arccos(np.clip(g_mean[1] / np.linalg.norm(g_mean), -1, 1)))
    report(OK if ang < 5 and abs(np.linalg.norm(g_mean) - 9.81) < 0.5 else BAD,
           f"mean specific force in world {g_mean.round(3)} |{np.linalg.norm(g_mean):.2f}| m/s^2, "
           f"{ang:.1f} deg from +y (expect ~0 deg, 9.81)")

    # gyro vs ARKit angular velocity (camera frame), with time-offset search
    dR = R_wc[:-1].inv() * R_wc[1:]
    w_arkit = dR.as_rotvec() / np.diff(t)[:, None]          # body (camera) rates
    t_mid = 0.5 * (t[:-1] + t[1:])
    w_gyro_cam = (R_CAM_DEV @ rec.gyro[:, 1:4].T).T
    best = (None, -1.0)
    for off_ms in range(-50, 51, 1):
        tt = t_mid + off_ms / 1000.0
        g = np.stack([np.interp(tt, rec.gyro[:, 0], w_gyro_cam[:, k]) for k in range(3)], axis=1)
        num = (g * w_arkit).sum()
        den = np.linalg.norm(g) * np.linalg.norm(w_arkit) + 1e-12
        if num / den > best[1]:
            best = (off_ms, num / den, g)
    off, corr, g = best
    rms = np.sqrt(((g - w_arkit) ** 2).sum(axis=1).mean())
    report(OK if corr > 0.9 else BAD,
           f"gyro vs ARKit rotation rate: correlation {corr:.3f}, RMS diff {np.degrees(rms):.2f} deg/s, "
           f"best IMU time offset {off:+d} ms (gyro t + offset = camera t)")

    # ---------- RGB-D ----------
    print("\n== RGB-D ==")
    ids = rec.image_frames
    report(OK if len(ids) else BAD, f"{len(ids)} RGB-D frames ({len(ids) / (t[-1] - t[0]):.1f} Hz)")
    if len(ids):
        rgb = rec.rgb(ids[0])
        name = f"{int(rec.frames[ids[0]]['frame']):06d}.png"
        d_raw = cv2.imread(str(rec.root / ("lidar" if rec.has_raw_lidar else "depth") / name), cv2.IMREAD_UNCHANGED)
        conf = rec.confidence(ids[0])
        report(OK, f"rgb {rgb.shape[1]}x{rgb.shape[0]}  lidar {d_raw.shape[1]}x{d_raw.shape[0]} {d_raw.dtype}  "
                   f"confidence {conf.shape[1]}x{conf.shape[0]} {conf.dtype}")
        same_aspect = abs(rgb.shape[1] / rgb.shape[0] - d_raw.shape[1] / d_raw.shape[0]) < 1e-3
        report(OK if same_aspect else BAD, f"RGB/LiDAR aspect ratio match: {same_aspect}  "
                   f"(scale RGB/LiDAR = {rgb.shape[1] / d_raw.shape[1]:.3f})")
        if rec.has_raw_lidar:
            d_al = cv2.imread(str(rec.root / "depth" / name), cv2.IMREAD_UNCHANGED)
            same = d_al is not None and d_al.shape[:2] == rgb.shape[:2]
            report(OK if same else BAD, f"aligned depth/ is the same size as rgb/: {same} ({d_al.shape[1]}x{d_al.shape[0]})")
            lid_up = cv2.resize(rec.lidar_m(ids[0]), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            al = rec.depth_aligned_m(ids[0])
            rel = np.median(np.abs(al - lid_up)[lid_up > 0] / lid_up[lid_up > 0])
            report(OK if rel < 0.03 else BAD, f"aligned depth vs raw LiDAR (no flip / scale error): median |diff| {100 * rel:.2f}%")
        else:
            report(WARN, "v1 recording: depth/ is raw 256x192 LiDAR (no RGB-aligned same-size depth)")
        meds, valid, highs = [], [], []
        for i in ids:
            d = rec.lidar_m(i)
            v = d > 0
            valid.append(v.mean())
            meds.append(np.median(d[v]) if v.any() else 0)
            highs.append((rec.confidence(i) == 2).mean())
        meds = np.array(meds)
        report(OK if 0.1 < np.median(meds) < 10 else BAD,
               f"depth median per frame {meds.min():.2f}..{meds.max():.2f} m (sane range means PNG byte order is right)")
        report(OK, f"valid depth {100 * np.mean(valid):.1f}%   high-confidence {100 * np.mean(highs):.1f}%")
        vals = np.unique(rec.confidence(ids[0]))
        report(OK if set(vals.tolist()) <= {0, 1, 2} else BAD, f"confidence values {vals.tolist()}")

        # overlay: depth (upsampled) on RGB + edge comparison for a few frames
        for i in ids[[0, len(ids) // 2, -1]]:
            rgb = rec.rgb(i)
            d_up = rec.depth_aligned_m(i, rgb.shape[:2])
            dn = np.clip(d_up / max(np.percentile(d_up[d_up > 0], 98), 1e-3), 0, 1)
            color = cv2.applyColorMap((255 * (1 - dn)).astype(np.uint8), cv2.COLORMAP_TURBO)
            blend = cv2.addWeighted(rgb, 0.5, color, 0.5, 0)
            edges_rgb = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), 60, 160)
            dz = cv2.Sobel(np.log(np.maximum(d_up, 0.05)), cv2.CV_32F, 1, 0) ** 2 + \
                cv2.Sobel(np.log(np.maximum(d_up, 0.05)), cv2.CV_32F, 0, 1) ** 2
            edges_d = (dz > 0.02).astype(np.uint8) * 255
            edge_vis = np.zeros_like(rgb)
            edge_vis[..., 1] = edges_rgb
            edge_vis[..., 2] = edges_d
            panel = np.hstack([blend, edge_vis])
            cv2.imwrite(str(out / f"overlay_{int(rec.frames[i]['frame']):06d}.jpg"), panel)
        report(OK, f"overlays written to {out} (left: depth over RGB, right: green RGB edges / red depth edges)")

    # ---------- mesh ----------
    mesh = rec.root / "mesh.obj"
    if mesh.exists():
        nv = nf = 0
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        with mesh.open() as fh:
            for line in fh:
                if line.startswith("v "):
                    p = np.array(line.split()[1:4], float)
                    lo, hi = np.minimum(lo, p), np.maximum(hi, p)
                    nv += 1
                elif line.startswith("f "):
                    nf += 1
        report(OK if nv == m.get("mesh", {}).get("vertices") else WARN,
               f"mesh.obj {nv} vertices, {nf} faces, extent {(hi - lo).round(2)} m")


if __name__ == "__main__":
    main()
