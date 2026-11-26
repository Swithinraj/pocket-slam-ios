# Recording format (`slamtogo-v2`)

One folder per scan, `Documents/recordings/<yyyy-MM-dd_HH-mm-ss>/` on the phone.

| File | Content |
|---|---|
| `meta.json` | device, iOS, video format, settings, thermal state and Low Power Mode at start/end, counts, all conventions |
| `frames.csv` | every ARKit frame: `frame,timestamp,tracking,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,width,height,exposure_s,image` |
| `trajectory_tum.txt` | ARKit camera poses in TUM format (`t tx ty tz qx qy qz qw`) |
| `accel.csv` / `gyro.csv` / `mag.csv` | raw IMU: m/s² (specific force, +9.81 "up" at rest), rad/s, µT; 100 / 100 / 50 Hz |
| `rgb/NNNNNN.jpg` | camera image at the RGB-D rate (960x720 default, or 1920x1440) |
| `lidar/NNNNNN.png` | raw ARKit LiDAR depth 256x192, uint16 millimetres (0 = invalid) |
| `confidence/NNNNNN.png` | LiDAR confidence 0 low / 1 medium / 2 high, 256x192 |
| `mesh.obj` | ARKit LiDAR mesh, world frame |
| `depth/NNNNNN.png` | *written by post-scan processing*: depth pixel-aligned with `rgb/`, same size, uint16 mm |
| `vlio_frames_tum.txt` | *post-scan*: our VLIO camera pose for every RGB-D frame |
| `vlio_keyframes_tum.txt` | *post-scan*: final (sliding-window smoothed) keyframe poses |
| `vlio_live_*_tum.txt` | live preview output (only if enabled) |
| `processing.json` | processing status, duration, keyframes, restarts, mean solve time, conditions |

## Conventions

* **Clock**: seconds of iOS system uptime, shared by ARKit and CoreMotion (camera-IMU offset measured 0 ms).
* **ARKit world**: gravity aligned, +y up, origin at session start. ARKit camera: +x right, +y up, -z forward,
  sensor in landscape. `frame` indices name the image files.
* **VLIO world**: gravity aligned, +z up, yaw fixed at the first keyframe. VLIO poses use the OpenCV camera
  (+x right, +y down, +z forward). Compare with ARKit after a rigid alignment.
* **Intrinsics** in `frames.csv` are for the full 1920x1440 image; multiply by `settings.rgb_scale` for
  `rgb/`. LiDAR has the same field of view (scale by 256 / width).
* **IMU axes** (device): x right, y up, z out of the screen (portrait). Camera = device rotated:
  `R_bc = [[0,-1,0],[-1,0,0],[0,0,-1]]` (OpenCV camera to device).
