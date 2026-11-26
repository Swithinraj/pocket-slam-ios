"""Keyframes, features with LiDAR depth, pairwise matching (sequential + loop) and feature tracks."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .dataset import Recording


@dataclass
class Keyframe:
    frame: int                 # index into rec.frames
    T0: np.ndarray             # initial camera-to-world (OpenCV camera), from ARKit
    K: np.ndarray              # 3x3 intrinsics of the RGB image
    kps: np.ndarray            # (N, 2) pixel coordinates
    desc: np.ndarray           # (N, 128) SIFT descriptors
    depth: np.ndarray          # (N,) LiDAR depth at the keypoint [m], nan if unusable
    depth_sigma: np.ndarray    # (N,) 1-sigma depth noise [m]
    ang_rate: float = 0.0      # camera angular speed at capture [rad/s] (rolling-shutter / blur indicator)


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def angular_rate(rec: Recording, i: int) -> float:
    """Angular speed around frame i from neighbouring ARKit poses [rad/s]."""
    a, b = max(i - 1, 0), min(i + 1, len(rec.frames) - 1)
    if a == b:
        return 0.0
    R = rec.T_world_cam_arkit(a)[:3, :3].T @ rec.T_world_cam_arkit(b)[:3, :3]
    return np.radians(rotation_angle_deg(R)) / (rec.timestamps[b] - rec.timestamps[a])


def select_keyframes(rec: Recording, min_trans: float = 0.05, min_rot_deg: float = 5.0,
                     max_rate_deg: float = 40.0) -> list[int]:
    """RGB-D frames with normal tracking, spaced by motion. Once the motion threshold is reached, frames
    rotating faster than `max_rate_deg`/s are skipped (rolling shutter + blur) unless nothing calmer follows."""
    out: list[int] = []
    last = None
    pending = None
    for i in rec.image_frames:
        if rec.frames["tracking"][i] != 2:
            continue
        T = rec.T_world_cam_cv(i)
        if last is not None:
            d = np.linalg.inv(last) @ T
            if np.linalg.norm(d[:3, 3]) < min_trans and rotation_angle_deg(d[:3, :3]) < min_rot_deg:
                continue
            if np.degrees(angular_rate(rec, int(i))) > max_rate_deg:
                pending = pending if pending is not None else int(i)
                if np.linalg.norm(d[:3, 3]) < 3 * min_trans and rotation_angle_deg(d[:3, :3]) < 3 * min_rot_deg:
                    continue
                i = pending                      # waited too long: take the first candidate
        out.append(int(i))
        last = rec.T_world_cam_cv(int(i))
        pending = None
    return out


def lidar_depth_at(kps: np.ndarray, lidar: np.ndarray, conf: np.ndarray, rgb_shape) -> tuple[np.ndarray, np.ndarray]:
    """Sample raw LiDAR at keypoints; reject low confidence and depth edges."""
    h, w = rgb_shape[:2]
    lh, lw = lidar.shape
    u = np.clip(np.round((kps[:, 0] + 0.5) * lw / w - 0.5).astype(int), 2, lw - 3)
    v = np.clip(np.round((kps[:, 1] + 0.5) * lh / h - 0.5).astype(int), 2, lh - 3)
    d = lidar[v, u].astype(np.float64)
    c = conf[v, u]
    # Only trust LiDAR where it is locally flat: the 256x192 map is smoothed across object boundaries,
    # and corner-like features sit on exactly those boundaries (foreground corners get background depth).
    r = range(-2, 3)
    patch = np.stack([lidar[v + dv, u + du] for dv in r for du in r], axis=1)
    spread = patch.max(axis=1) - patch.min(axis=1)
    ok = (d > 0.1) & (d < 5.0) & (c >= 1) & (spread < 0.01 + 0.01 * d)
    sigma = np.where(c == 2, 0.01 + 0.01 * d, 0.03 + 0.03 * d)
    d = np.where(ok, d, np.nan)
    return d, sigma


def extract(rec: Recording, kf_frames: list[int], n_features: int = 2000) -> list[Keyframe]:
    sift = cv2.SIFT_create(nfeatures=n_features)
    kfs = []
    for i in kf_frames:
        rgb = rec.rgb(i)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kp], dtype=np.float64).reshape(-1, 2)
        if desc is None:
            desc = np.zeros((0, 128), np.float32)
        d, s = lidar_depth_at(pts, rec.lidar_m(i), rec.confidence(i), rgb.shape)
        kfs.append(Keyframe(frame=i, T0=rec.T_world_cam_cv(i), K=rec.K_rgb(i), kps=pts, desc=desc,
                            depth=d, depth_sigma=s, ang_rate=angular_rate(rec, i)))
    return kfs


def candidate_pairs(kfs: list[Keyframe], window: int = 4, loop_dist: float = 1.0,
                    loop_angle_deg: float = 35.0, max_loops: int = 5) -> list[tuple[int, int, bool]]:
    """Sequential neighbours plus non-adjacent keyframes that look at the same place (loop candidates)."""
    pairs = []
    n = len(kfs)
    centers = np.array([k.T0[:3, 3] for k in kfs])
    fwd = np.array([k.T0[:3, 2] for k in kfs])       # OpenCV camera z axis = viewing direction
    for a in range(n):
        for b in range(a + 1, min(n, a + 1 + window)):
            pairs.append((a, b, False))
        far = np.arange(a + 1 + window, n)
        if len(far):
            dist = np.linalg.norm(centers[far] - centers[a], axis=1)
            ang = np.degrees(np.arccos(np.clip(fwd[far] @ fwd[a], -1, 1)))
            ok = far[(dist < loop_dist) & (ang < loop_angle_deg)]
            ok = ok[np.argsort(np.linalg.norm(centers[ok] - centers[a], axis=1))][:max_loops]
            pairs += [(a, int(b), True) for b in ok]
    return pairs


def match_pair(ka: Keyframe, kb: Keyframe, ratio: float = 0.8, min_inliers: int = 25):
    """Ratio-test + mutual SIFT matches, verified with a MAGSAC fundamental matrix."""
    if len(ka.desc) < 8 or len(kb.desc) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ab = bf.knnMatch(ka.desc, kb.desc, k=2)
    ba = bf.match(kb.desc, ka.desc)
    back = {m.queryIdx: m.trainIdx for m in ba}
    m = [(x.queryIdx, x.trainIdx) for x, y in (p for p in ab if len(p) == 2)
         if x.distance < ratio * y.distance and back.get(x.trainIdx) == x.queryIdx]
    if len(m) < min_inliers:
        return None
    m = np.array(m)
    try:
        F, mask = cv2.findFundamentalMat(ka.kps[m[:, 0]], kb.kps[m[:, 1]], cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
    except cv2.error:            # degenerate configuration: MAGSAC asserts instead of returning None
        return None
    if F is None or mask is None:
        return None
    m = m[mask.ravel().astype(bool)]
    return m if len(m) >= min_inliers else None


class UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


KEY = 1_000_000


def build_tracks(matches: dict[tuple[int, int], np.ndarray]) -> list[list[tuple[int, int]]]:
    """Merge pairwise matches into multi-view tracks [(kf, kp), ...]; drop inconsistent ones."""
    uf = UnionFind()
    for (a, b), m in matches.items():
        for i, j in m:
            uf.union(a * KEY + int(i), b * KEY + int(j))
    groups: dict[int, list[int]] = {}
    for node in uf.parent:
        groups.setdefault(uf.find(node), []).append(node)
    tracks = []
    for nodes in groups.values():
        obs = sorted((n // KEY, n % KEY) for n in nodes)
        kfs = [o[0] for o in obs]
        if len(obs) >= 2 and len(set(kfs)) == len(kfs):      # one observation per keyframe
            tracks.append(obs)
    return tracks
