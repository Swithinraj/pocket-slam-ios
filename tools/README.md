# tools: offline analysis (Python, WSL)

```bash
bash tools/setup_env.sh                  # creates ~/venvs/slamtogo with the pinned requirements
source ~/venvs/slamtogo/bin/activate
```

| Command | What it does |
|---|---|
| `python tools/inspect_recording.py <rec>` | stream rates and gaps, camera/IMU clock overlap, IMU axes vs ARKit (gravity, gyro correlation, time offset), depth encoding, RGB/depth size and alignment, mesh; overlays in `<rec>/_inspect/` |
| `python tools/run_ba.py <rec>` | offline keyframe bundle adjustment (reference) + TSDF meshes and point clouds → `<rec>/_slam/` |
| `python tools/eval_trajectories.py <rec>` | ARKit vs offline BA vs our VLIO: ATE, Sim(3) scale, **local scale**, multi-view LiDAR consistency |
| `python tools/view_reconstruction.py <rec>` | 3D viewer (WSLg). Keys: **P** points/mesh, **B** BA/ARKit poses, **T** trajectories, **L** ARKit mesh |
| `bash tools/run_ablation.sh <rec>` | replays the C++ core with ablation settings → `<rec>/_var/` |
| `python tools/make_figures.py --recordings <dir> --out docs [--with-triangulation]` | regenerates every figure and number in `docs/` |

`run_ba.py` pipeline: keyframes every 5 cm / 5° (avoiding fast rotation) → SIFT + LiDAR depth at
locally flat keypoints → sequential + loop-candidate matching (mutual ratio test, MAGSAC) → GTSAM bundle
adjustment (reprojection, LiDAR depth as virtual stereo, ARKit odometry, gravity prior, Huber, two rounds) →
full-rate trajectory, 1 cm TSDF (Open3D `VoxelBlockGrid`), `ba_report.png`, `summary.json`.
