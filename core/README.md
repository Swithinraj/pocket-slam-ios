# slamcore: shared VLIO core (C++17, Eigen only)

Tightly-coupled **visual-LiDAR-inertial odometry** with a sliding-window factor graph. One code base
for iOS (SwiftPM, C API), Android (NDK/JNI, C API) and Linux tools.

```
include/slamcore/   types, so3, preintegration, tracker, estimator, system (C++ API)
src/                implementation + slamcore_c.cpp (C API)
src/public/         slamcore_c.h (the only header Swift / JNI see)
tools/              run_recording: replays a slamtogo recording through the core (causal, like the app)
tests/              synthetic ground-truth tests
third_party/        eigen 3.4.0 (MPL2), stb_image (public domain, tools only)
```

## Pipeline

| Stage | What |
|---|---|
| Front end | Shi-Tomasi corners (grid-spread, 150 max) + pyramidal inverse-compositional KLT; gyro-predicted initial flow; forward-backward check (1 px) |
| LiDAR | depth sampled at features only where the 5x5 LiDAR neighbourhood is flat (the 256x192 map is smoothed across edges); sigma 1 cm + 1 % (high conf) |
| Keyframes | rotation-compensated parallax > 12 px, < 60 % tracks surviving, or > 0.5 s |
| State | per keyframe: R, p, v, b_a, b_g (15 dof); landmarks as inverse depth in the host keyframe |
| Factors | IMU preintegration (Forster on-manifold, bias Jacobians, re-integration on bias change) + bias random walk; reprojection (px); LiDAR depth on every observation that has one; marginalisation prior |
| Solver | Levenberg-Marquardt, landmarks eliminated by Schur complement, Huber, central-difference Jacobians on the manifold |
| Window | 10 keyframes; the oldest is marginalised (Schur complement → eigen-decomposed linear prior) |
| Init | gravity from the accelerometer; metric scale comes from LiDAR, so no visual-inertial scale initialisation is needed |
| Loop closure (post-scan, `slam_enable_loop_closure` + `slam_finish`) | ORB-style keyframe features with LiDAR depth; pose-based candidates; guided + appearance matching, 3-point LiDAR RANSAC, reprojection refinement; 4-DoF pose graph (sparse Cholesky, annealed Cauchy); every frame corrected |

## Build and test (Linux / WSL)

```bash
# from the repo root
cmake -S core -B ~/build/slamcore -DCMAKE_BUILD_TYPE=Release && cmake --build ~/build/slamcore -j
~/build/slamcore/test_core                                   # synthetic ground truth
mkdir -p <rec>/_vlio && ~/build/slamcore/run_recording <rec> <rec>/_vlio    # replay a recording
python tools/eval_trajectories.py <rec>                      # compare with ARKit / offline BA
```

## Results

- Synthetic (12 s, 4.8 m, noisy IMU with biases, 0.7 px features, 1 cm LiDAR): **9.4 mm RMSE**,
  0.32 deg max rotation error, gyro bias recovered (`test_core`).
- Real recordings: see [../docs/FINDINGS.md](../docs/FINDINGS.md). Summary: metric local scale within 1-2 %
  (better than ARKit), local accuracy close to ARKit, global error dominated by heading drift (no loop closure yet).
- Speed on one PC core: front end ~8-9 ms per frame, solve ~21-28 ms mean per keyframe.

## Runner options (for experiments)

`run_recording <rec> <out> [--track-width 480] [--no-depth] [--no-host-depth] [--no-target-depth]
[--flatness F] [--depth-sigma-scale S] [--pixel-sigma P] [--max-age N] [--pbc x y z] [--no-loop]`

Writes `vlio_*` (odometry) and, unless `--no-loop`, `slam_frames_tum.txt` / `slam_keyframes_tum.txt` /
`slam_loops.txt` (after loop closure).

## Next

1. Appearance-only place recognition (bag of words) for very large drift and across map segments.
2. Per-frame motion-only refinement (non-keyframe poses).
3. Analytic Jacobians (5-10x faster solve).
