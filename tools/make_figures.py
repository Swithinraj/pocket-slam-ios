"""Regenerate every figure and number in docs/ from recordings.

    python tools/make_figures.py --recordings <folder with recordings> --out docs \
        [--latest 2026-11-26_14-08-02] [--scale-set 2026-11-26_00-15-22 2026-11-26_02-09-36 2026-11-26_14-08-02] \
        [--with-triangulation]

Each recording must have been processed with tools/run_ba.py (offline BA reference in _slam/) and contain
our VLIO trajectory (vlio_*_tum.txt from the app, or _vlio/ from core/run_recording). The ablation figure
needs tools/run_ablation.sh to have been run on --latest.
Outputs: docs/figures/*.png and docs/results/metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from eval_trajectories import (arkit_to_cv, evaluate, frame_pairs, load_sources, local_scale, read_tum, umeyama)
from slamtogo.dataset import load

# Validated categorical palette (slots 1-3, all-pairs CVD-safe); colour follows the entity in every figure.
C_BA, C_VLIO, C_ARKIT = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
NAMES = {"Offline BA": "Offline BA (loop-closed reference)", "ARKit VIO": "ARKit VIO", "VLIO": "Our VLIO"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 1.6, "font.family": "sans-serif",
})


def vlio_source(sources: dict) -> tuple[str, dict]:
    """Our VLIO keyframes: computed on the phone if available, else the PC replay."""
    for key in ("VLIO device KF", "VLIO PC KF"):
        if key in sources and len(sources[key]) > 10:
            return key, sources[key]
    raise SystemExit("no VLIO trajectory in this recording")


def aligned_positions(P: dict, ref: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = [i for i in sorted(P) if i in ref]
    A = np.array([P[i][:3, 3] for i in keys])
    B = np.array([ref[i][:3, 3] for i in keys])
    _, R, t = umeyama(A, B, scale=False)
    return np.array(keys), A @ R.T + t, B


def heading_drift(P: dict, ref: dict, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation about gravity of the estimate relative to the reference since the first common frame [deg]."""
    keys = [i for i in sorted(P) if i in ref]
    k0 = keys[0]
    R0, Rb0 = P[k0][:3, :3], ref[k0][:3, :3]
    up_cam0 = Rb0.T @ np.array([0.0, 1.0, 0.0])            # ARKit / BA world: +y up
    t, yaw = [], []
    for k in keys:
        E = (R0.T @ P[k][:3, :3]) @ (Rb0.T @ ref[k][:3, :3]).T
        yaw.append(np.degrees(Rotation.from_matrix(E).as_rotvec() @ up_cam0))
        t.append(ts[k] - ts[keys[0]])
    return np.array(t), np.array(yaw)


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.plot(x[-1], y[-1], "o", ms=4, color=color)


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------- figures
def fig_trajectories(rec, sources, vkey, out: Path):
    ba = sources["Offline BA"]
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for name, P, color, lw in (("ARKit VIO", sources["ARKit VIO"], C_ARKIT, 1.2), ("Offline BA", ba, C_BA, 1.8),
                               ("VLIO", sources[vkey], C_VLIO, 1.2)):
        _, A, _ = aligned_positions(P, ba)
        ax.plot(A[:, 0], A[:, 2], color=color, lw=lw, label=NAMES[name])
    B = np.array([ba[i][:3, 3] for i in sorted(ba)])
    ax.plot(B[0, 0], B[0, 2], "o", ms=8, color=INK, mec="white", mew=2, label="Start")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"Top-down trajectories, {rec.root.name} ({rec.meta['duration_s']:.0f} s)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out / "fig1_trajectories.png")


def fig_error_over_time(rec, sources, vkey, out: Path):
    ba = sources["Offline BA"]
    ts = rec.timestamps
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for name, P, color in (("ARKit VIO", sources["ARKit VIO"], C_ARKIT), ("VLIO", sources[vkey], C_VLIO)):
        keys, A, B = aligned_positions(P, ba)
        t = ts[keys] - ts[keys[0]]
        e = 100 * np.linalg.norm(A - B, axis=1)
        ax.plot(t, e, color=color, label=NAMES[name])
        end_label(ax, t, e, f"{NAMES[name]}  {e[-1]:.0f} cm", color)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position difference to BA [cm]")
    ax.set_title("Position difference to the loop-closed reference (rigidly aligned)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    save(fig, out / "fig2_error_over_time.png")


def fig_heading(rec, sources, vkey, out: Path):
    ba = sources["Offline BA"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.axhline(0, color=AXIS, lw=1)
    for name, P, color in (("ARKit VIO", sources["ARKit VIO"], C_ARKIT), ("VLIO", sources[vkey], C_VLIO)):
        t, yaw = heading_drift(P, ba, rec.timestamps)
        ax.plot(t, yaw, color=color, label=NAMES[name])
        end_label(ax, t, yaw, f"{NAMES[name]}  {yaw[-1]:+.1f}°", color)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("heading error [deg]")
    ax.set_title("Heading (yaw) drift relative to the loop-closed reference")
    ax.legend(loc="upper left")
    save(fig, out / "fig3_heading_drift.png")


def fig_scale(rows: list[dict], out: Path):
    """rows: {recording, source, sim3, local}"""
    recs = list(dict.fromkeys(r["recording"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True, sharex=True)
    for ax, metric, title in ((axes[0], "sim3", "Whole-trajectory Sim(3) scale"),
                              (axes[1], "local", "Local scale (2 s motions)")):
        ax.axvline(1.0, color=INK2, lw=1)
        for j, (src, color, off) in enumerate((("ARKit VIO", C_ARKIT, 0.12), ("VLIO", C_VLIO, -0.12))):
            ys, xs = [], []
            for i, rname in enumerate(recs):
                v = [r[metric] for r in rows if r["recording"] == rname and r["source"] == src]
                if v:
                    ys.append(i + off)
                    xs.append(v[0])
            ax.plot(xs, ys, "o", ms=8, color=color, mec="white", mew=2, label=NAMES[src] if metric == "sim3" else None)
            for x, y in zip(xs, ys):
                ax.annotate(f"{x:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8,
                            color=INK2)
        ax.set_title(title)
        ax.set_xlabel("estimate / reference (1.0 = exact)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(recs)))
    axes[0].set_yticklabels(recs)
    axes[0].set_ylim(-0.6, len(recs) - 0.4)
    axes[0].legend(loc="lower left")
    save(fig, out / "fig4_scale_local_vs_global.png")


def fig_lidar(res: dict, vkey: str, out: Path):
    order = (("Offline BA", C_BA), ("ARKit VIO", C_ARKIT), (vkey, C_VLIO))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.9), sharey=True)
    for ax, field, title in ((axes[0], "lidar_seq", "Neighbouring frames"), (axes[1], "lidar_revisit", "Revisits (> 15 s apart)")):
        vals = [1000 * float(np.median(res[n][field])) for n, _ in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, height=0.55, color=[c for _, c in order])
        for yy, v in zip(y, vals):
            ax.annotate(f"{v:.1f} mm", (v, yy), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_title(title)
        ax.set_xlabel("median disagreement [mm], lower is better")
        ax.set_xlim(0, max(vals) * 1.3)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order))[::-1])
    axes[0].set_yticklabels([NAMES.get(n, "Our VLIO") for n, _ in order])
    fig.suptitle("Multi-view LiDAR consistency (no ground truth needed)", fontsize=11, fontweight="bold", color=INK)
    save(fig, out / "fig5_lidar_consistency.png")


ABLATION = [("base", "Full VLIO (default)"), ("nodepth", "No LiDAR"), ("hostonly", "LiDAR on first observation only"),
            ("targetonly", "LiDAR on later observations only"), ("alldepth", "LiDAR on all features (no flatness test)"),
            ("loosedepth", "LiDAR noise x3"), ("tightpx", "Pixel noise x2"), ("age10", "Tracks capped at 10 frames")]


def ablation_rows(rec, ba: dict) -> list[dict]:
    ts = rec.timestamps
    lut = {round(float(t), 4): i for i, t in enumerate(ts)}
    rows = []
    for key, label in ABLATION:
        f = rec.root / "_var" / key / "vlio_keyframes_tum.txt"
        if not f.exists():
            continue
        P = {lut[t]: T for t, T in read_tum(f).items() if t in lut}
        keys, A, B = aligned_positions(P, ba)
        s, _, _ = umeyama(np.array([P[k][:3, 3] for k in keys]), B, scale=True)
        rows.append({"variant": key, "label": label, "ate_m": float(np.sqrt(((A - B) ** 2).sum(1).mean())),
                     "sim3_scale": float(s), "local_scale": local_scale(P, ba, ts)})
    return rows


def fig_ablation(rows: list[dict], rec_name: str, out: Path):
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, field, title, fmt, ref in ((axes[0], "local_scale", "Local scale (2 s)", "{:.3f}", 1.0),
                                       (axes[1], "sim3_scale", "Whole-trajectory Sim(3) scale", "{:.3f}", 1.0),
                                       (axes[2], "ate_m", "Keyframe ATE [cm]", "{:.0f}", None)):
        v = np.array([r[field] for r in rows]) * (100 if field == "ate_m" else 1)
        ax.plot(v, y, "o", ms=8, color=C_VLIO, mec="white", mew=2)
        for vv, yy in zip(v, y):
            ax.annotate(fmt.format(vv), (vv, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK2)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=1)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["label"] for r in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.suptitle(f"VLIO ablation, {rec_name} (PC replay of the core)", fontsize=11, fontweight="bold", color=INK)
    save(fig, out / "fig6_ablation.png")


def lidar_vs_triangulation(rec, span) -> dict:
    """Median relative difference (triangulated depth - LiDAR) / LiDAR, with ARKit poses vs BA poses."""
    from slamtogo import backend as be
    from slamtogo import frontend as fe
    from slamtogo.dataset import R_ARKIT_TO_CV
    ba = read_tum(rec.root / "_slam/trajectory_ba_tum.txt")
    ts = rec.timestamps
    frames = [int(i) for i in rec.image_frames if span[0] <= ts[i] - ts[0] < span[1]][::3]
    kfs = fe.extract(rec, frames, 1500)
    out = {}
    for src in ("ARKit VIO", "Offline BA"):
        for k in kfs:
            if src == "Offline BA":
                T = ba[round(float(ts[k.frame]), 4)].copy()
                T[:3, :3] = T[:3, :3] @ R_ARKIT_TO_CV
                k.T0 = T
            else:
                k.T0 = rec.T_world_cam_cv(k.frame)
        rows = []
        for a in range(len(kfs)):
            for b in range(a + 2, min(a + 6, len(kfs))):
                A, Bk = kfs[a], kfs[b]
                if np.linalg.norm(A.T0[:3, 3] - Bk.T0[:3, 3]) < 0.08:
                    continue
                m = fe.match_pair(A, Bk)
                if m is None:
                    continue
                Pa = A.K @ np.linalg.inv(A.T0)[:3]
                Pb = Bk.K @ np.linalg.inv(Bk.T0)[:3]
                for i, j in m:
                    d = A.depth[i]
                    if not np.isfinite(d):
                        continue
                    X = be.cv_triangulate(Pa, Pb, A.kps[i], Bk.kps[j])
                    if X is None:
                        continue
                    _, z = be.project(A.T0, A.K, X)
                    ra, rb = X - A.T0[:3, 3], X - Bk.T0[:3, 3]
                    par = np.degrees(np.arccos(np.clip(ra @ rb / np.linalg.norm(ra) / np.linalg.norm(rb), -1, 1)))
                    if z > 0 and par > 4:
                        rows.append((z - d) / d)
        r = np.array(rows)
        out[src] = {"n": len(r), "median": float(np.median(r)), "q25": float(np.percentile(r, 25)),
                    "q75": float(np.percentile(r, 75))}
    return out


def fig_triangulation(tri: dict, out: Path):
    recs = list(tri)
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.axvline(0, color=INK2, lw=1)
    for src, color, off in (("Offline BA", C_BA, 0.12), ("ARKit VIO", C_ARKIT, -0.12)):
        for i, r in enumerate(recs):
            s = tri[r][src]
            y = i + off
            ax.plot([100 * s["q25"], 100 * s["q75"]], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(100 * s["median"], y, "o", ms=8, color=color, mec="white", mew=2,
                    label=f"{NAMES[src]} poses" if i == 0 else None)
            ax.annotate(f"{100 * s['median']:+.2f} %", (100 * s["median"], y), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(recs)))
    ax.set_yticklabels(recs)
    ax.set_ylim(-0.6, len(recs) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("(triangulated - LiDAR) / LiDAR depth [%]   dot = median, bar = interquartile range")
    ax.set_title("Is the LiDAR biased? Feature triangulation vs LiDAR on flat surfaces")
    ax.set_ylim(-0.6, len(recs) + 0.3)
    ax.legend(loc="upper left", ncols=2)
    save(fig, out / "fig7_lidar_vs_triangulation.png")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--out", default="docs")
    ap.add_argument("--latest", default="2026-11-26_14-08-02")
    ap.add_argument("--scale-set", nargs="*", default=["2026-11-26_00-15-22", "2026-11-26_02-09-36", "2026-11-26_14-08-02"])
    ap.add_argument("--with-triangulation", action="store_true")
    args = ap.parse_args()
    root = Path(args.recordings)
    figs = Path(args.out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "results").mkdir(parents=True, exist_ok=True)
    metrics: dict = {"recordings": {}}

    scale_rows = []
    for name in dict.fromkeys(args.scale_set + [args.latest]):
        rec = load(root / name)
        sources = load_sources(rec)
        vkey, _ = vlio_source(sources)
        seq, rev = frame_pairs(rec, sources["ARKit VIO"])
        res = evaluate(rec, sources, "Offline BA", seq, rev)
        metrics["recordings"][name] = {
            "duration_s": rec.meta["duration_s"], "rgbd_frames": len(rec.image_frames),
            "arkit_fps": rec.meta.get("settings", {}).get("arkit_fps"), "vlio_source": vkey,
            "sources": {k: {"ate_m": v["ate_m"], "sim3_scale": v["sim3_scale"], "local_scale": v["local_scale"],
                            "lidar_seq_median_mm": 1000 * float(np.median(v["lidar_seq"])),
                            "lidar_revisit_median_mm": 1000 * float(np.median(v["lidar_revisit"]))}
                        for k, v in res.items()},
        }
        if name in args.scale_set:
            for src, key in (("ARKit VIO", "ARKit VIO"), ("VLIO", vkey)):
                scale_rows.append({"recording": name, "source": src, "sim3": res[key]["sim3_scale"],
                                   "local": res[key]["local_scale"]})
        if name == args.latest:
            fig_trajectories(rec, sources, vkey, figs)
            fig_error_over_time(rec, sources, vkey, figs)
            fig_heading(rec, sources, vkey, figs)
            fig_lidar(res, vkey, figs)
            rows = ablation_rows(rec, sources["Offline BA"])
            metrics["ablation"] = {"recording": name, "rows": rows}
            fig_ablation(rows, name, figs)
    fig_scale(scale_rows, figs)

    if args.with_triangulation:
        tri = {}
        for name, span in ((args.latest, (20, 140)), (args.scale_set[0], (0, 80))):
            tri[name] = lidar_vs_triangulation(load(root / name), span)
        metrics["lidar_vs_triangulation"] = tri
        fig_triangulation(tri, figs)

    (Path(args.out) / "results" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", Path(args.out) / "results" / "metrics.json")


if __name__ == "__main__":
    main()
