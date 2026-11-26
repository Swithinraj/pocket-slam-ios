"""GTSAM bundle adjustment: reprojection + LiDAR depth (virtual stereo) + ARKit odometry + gravity."""
from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from .frontend import Keyframe, rotation_angle_deg

VIRTUAL_BASELINE = 0.1      # m; depth d is encoded as a right-image u' = u - fx * b / d (ORB-SLAM RGB-D trick)


@dataclass
class Landmark:
    obs: list[tuple[int, int]]          # (kf, kp)
    p: np.ndarray                       # world point


@dataclass
class BAConfig:
    pixel_sigma: float = 1.0
    huber_k: float = 1.345
    rolling_shutter_s: float = 0.012    # assumed sensor readout time; inflates pixel noise when rotating
    odom_trans_sigma: float = 0.002     # m, plus 1% of step (ARKit VIO is very good over short intervals)
    odom_rot_sigma_deg: float = 0.1     # plus 1% of step angle
    gravity_sigma_deg: float = 0.5
    use_odometry: bool = True
    use_gravity: bool = True
    fix_poses: bool = False             # structure-only (evaluate ARKit poses fairly)
    max_iterations: int = 100


@dataclass
class BAResult:
    poses: list[np.ndarray]
    landmarks: list[Landmark]
    stats: dict = field(default_factory=dict)


def project(T_wc: np.ndarray, K: np.ndarray, p_w: np.ndarray) -> tuple[np.ndarray, float]:
    p_c = T_wc[:3, :3].T @ (p_w - T_wc[:3, 3])
    uv = (K @ p_c)[:2] / p_c[2]
    return uv, p_c[2]


def init_landmarks(kfs: list[Keyframe], tracks, max_px: float = 8.0) -> list[Landmark]:
    """Back-project LiDAR depth with ARKit poses (median over views); triangulate if no depth."""
    out = []
    for obs in tracks:
        pts = []
        for k, i in obs:
            d = kfs[k].depth[i]
            if np.isfinite(d):
                ray = np.linalg.inv(kfs[k].K) @ np.array([*kfs[k].kps[i], 1.0])
                pts.append(kfs[k].T0[:3, :3] @ (ray * d) + kfs[k].T0[:3, 3])
        if pts:
            p = np.median(np.array(pts), axis=0)
        else:
            p = triangulate(kfs, obs)
            if p is None:
                continue
        good = []
        for k, i in obs:
            uv, z = project(kfs[k].T0, kfs[k].K, p)
            if z > 0.05 and np.linalg.norm(uv - kfs[k].kps[i]) < max_px:
                good.append((k, i))
        if len(good) >= 2:
            out.append(Landmark(good, p))
    return out


def triangulate(kfs: list[Keyframe], obs) -> np.ndarray | None:
    (ka, ia), (kb, ib) = obs[0], obs[-1]
    A, B = kfs[ka], kfs[kb]
    if np.linalg.norm(A.T0[:3, 3] - B.T0[:3, 3]) < 0.05:
        return None
    Pa = A.K @ np.linalg.inv(A.T0)[:3]
    Pb = B.K @ np.linalg.inv(B.T0)[:3]
    X4 = cv_triangulate(Pa, Pb, A.kps[ia], B.kps[ib])
    return X4


def cv_triangulate(Pa, Pb, xa, xb):
    A = np.stack([xa[0] * Pa[2] - Pa[0], xa[1] * Pa[2] - Pa[1], xb[0] * Pb[2] - Pb[0], xb[1] * Pb[2] - Pb[1]])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-9:
        return None
    return X[:3] / X[3]


def _robust(base, k):
    return gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k), base)


def bundle_adjust(kfs: list[Keyframe], landmarks: list[Landmark], cfg: BAConfig,
                  init_poses: list[np.ndarray] | None = None) -> BAResult:
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    poses0 = init_poses or [k.T0 for k in kfs]

    for k, T in enumerate(poses0):
        values.insert(X(k), gtsam.Pose3(T))

    # Gauge: first pose fixed (or all poses fixed for structure-only evaluation).
    tight = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
    for k in (range(len(kfs)) if cfg.fix_poses else [0]):
        graph.add(gtsam.PriorFactorPose3(X(k), gtsam.Pose3(poses0[k]), tight))

    if not cfg.fix_poses and cfg.use_odometry:
        for k in range(len(kfs) - 1):
            rel = np.linalg.inv(kfs[k].T0) @ kfs[k + 1].T0           # ARKit VIO relative motion
            dist = np.linalg.norm(rel[:3, 3])
            ang = rotation_angle_deg(rel[:3, :3])
            rs = np.radians(cfg.odom_rot_sigma_deg + 0.01 * ang)
            ts = cfg.odom_trans_sigma + 0.01 * dist
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([rs, rs, rs, ts, ts, ts]))
            graph.add(gtsam.BetweenFactorPose3(X(k), X(k + 1), gtsam.Pose3(rel), noise))

    if not cfg.fix_poses and cfg.use_gravity and hasattr(gtsam, "Pose3AttitudeFactor"):
        # ARKit world is gravity aligned (+y up): keep each camera's measured "up" direction.
        g_noise = gtsam.noiseModel.Isotropic.Sigma(2, np.radians(cfg.gravity_sigma_deg))
        n_up = gtsam.Unit3(np.array([0.0, 1.0, 0.0]))
        for k, kf in enumerate(kfs):
            b_up = kf.T0[:3, :3].T @ np.array([0.0, 1.0, 0.0])
            graph.add(gtsam.Pose3AttitudeFactor(X(k), n_up, g_noise, gtsam.Unit3(b_up)))

    calib_mono = {}
    calib_stereo = {}
    n_depth = n_mono = 0
    for j, lm in enumerate(landmarks):
        values.insert(L(j), lm.p)
        for k, i in lm.obs:
            kf = kfs[k]
            fx, fy, cx, cy = kf.K[0, 0], kf.K[1, 1], kf.K[0, 2], kf.K[1, 2]
            # Rolling shutter: rows are exposed up to +-readout/2 apart -> image motion of w * t/2 * f pixels.
            px = np.hypot(cfg.pixel_sigma, kf.ang_rate * cfg.rolling_shutter_s / 2 * fx)
            u, v = kf.kps[i]
            d = kf.depth[i]
            if np.isfinite(d):
                if k not in calib_stereo:
                    calib_stereo[k] = gtsam.Cal3_S2Stereo(fx, fy, 0.0, cx, cy, VIRTUAL_BASELINE)
                s_ur = np.hypot(px, fx * VIRTUAL_BASELINE * kf.depth_sigma[i] / d ** 2)
                noise = _robust(gtsam.noiseModel.Diagonal.Sigmas(np.array([px, s_ur, px])), cfg.huber_k)
                meas = gtsam.StereoPoint2(u, u - fx * VIRTUAL_BASELINE / d, v)
                graph.add(gtsam.GenericStereoFactor3D(meas, noise, X(k), L(j), calib_stereo[k]))
                n_depth += 1
            else:
                if k not in calib_mono:
                    calib_mono[k] = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
                noise = _robust(gtsam.noiseModel.Isotropic.Sigma(2, px), cfg.huber_k)
                graph.add(gtsam.GenericProjectionFactorCal3_S2(np.array([u, v]), noise, X(k), L(j), calib_mono[k]))
                n_mono += 1

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(cfg.max_iterations)
    params.setRelativeErrorTol(1e-7)
    err0 = graph.error(values)
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
    err1 = graph.error(result)

    poses = [result.atPose3(X(k)).matrix() for k in range(len(kfs))]
    lms = [Landmark(lm.obs, result.atPoint3(L(j))) for j, lm in enumerate(landmarks)]
    return BAResult(poses, lms, {"graph_error_initial": err0, "graph_error_final": err1,
                                 "depth_observations": n_depth, "mono_observations": n_mono,
                                 "factors": graph.size()})


def residuals(kfs: list[Keyframe], poses: list[np.ndarray], landmarks: list[Landmark]):
    """Per-observation reprojection error [px] and depth error [m] (where LiDAR depth exists)."""
    reproj, depth = [], []
    for lm in landmarks:
        for k, i in lm.obs:
            uv, z = project(poses[k], kfs[k].K, lm.p)
            reproj.append(np.linalg.norm(uv - kfs[k].kps[i]))
            d = kfs[k].depth[i]
            if np.isfinite(d):
                depth.append(z - d)
    return np.array(reproj), np.array(depth)


def reject_outliers(kfs, poses, landmarks: list[Landmark], max_px: float = 3.0, max_depth_sigma: float = 3.0):
    out = []
    for lm in landmarks:
        good = []
        for k, i in lm.obs:
            uv, z = project(poses[k], kfs[k].K, lm.p)
            if z <= 0.05 or np.linalg.norm(uv - kfs[k].kps[i]) > max_px:
                continue
            d = kfs[k].depth[i]
            if np.isfinite(d) and abs(z - d) > max_depth_sigma * kfs[k].depth_sigma[i]:
                continue
            good.append((k, i))
        if len(good) >= 2:
            out.append(Landmark(good, lm.p))
    return out
