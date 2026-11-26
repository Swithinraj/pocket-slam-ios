"""Loader for slamtogo-v1 recordings (see iphone/README.md for the format)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Device (portrait: x right, y up, z out of screen) -> ARKit camera (landscape-right: x right, y up, z back).
# camera x = -device y, camera y = device x, camera z = device z
R_CAM_DEV = np.array([[0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])

# ARKit camera (x right, y up, z back) -> OpenCV camera (x right, y down, z forward)
R_ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0])


@dataclass
class Recording:
    root: Path
    meta: dict
    frames: np.ndarray                 # structured array from frames.csv
    accel: np.ndarray                  # (N, 4) t, ax, ay, az   [m/s^2, device axes]
    gyro: np.ndarray                   # (N, 4) t, gx, gy, gz   [rad/s, device axes]
    mag: np.ndarray                    # (N, 4) t, mx, my, mz   [uT]
    image_frames: np.ndarray = field(init=False)   # indices into `frames` that have RGB-D

    def __post_init__(self):
        self.image_frames = np.flatnonzero(self.frames["image"] == 1)

    # ---------- poses ----------
    @property
    def timestamps(self) -> np.ndarray:
        return self.frames["timestamp"]

    def T_world_cam_arkit(self, i: int) -> np.ndarray:
        """4x4 camera-to-world, ARKit camera convention."""
        f = self.frames[i]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([f["qx"], f["qy"], f["qz"], f["qw"]]).as_matrix()
        T[:3, 3] = [f["tx"], f["ty"], f["tz"]]
        return T

    def T_world_cam_cv(self, i: int) -> np.ndarray:
        """4x4 camera-to-world with an OpenCV camera (x right, y down, z forward)."""
        T = self.T_world_cam_arkit(i)
        T[:3, :3] = T[:3, :3] @ R_ARKIT_TO_CV
        return T

    # ---------- images ----------
    def rgb_scale(self) -> float:
        return float(self.meta.get("settings", {}).get("rgb_scale", 1.0))

    def K_rgb(self, i: int) -> np.ndarray:
        f, s = self.frames[i], self.rgb_scale()
        return np.array([[f["fx"] * s, 0, f["cx"] * s],
                         [0, f["fy"] * s, f["cy"] * s],
                         [0, 0, 1.0]])

    def K_depth(self, i: int, depth_shape: tuple[int, int]) -> np.ndarray:
        f = self.frames[i]
        sx = depth_shape[1] / f["width"]
        sy = depth_shape[0] / f["height"]
        return np.array([[f["fx"] * sx, 0, f["cx"] * sx],
                         [0, f["fy"] * sy, f["cy"] * sy],
                         [0, 0, 1.0]])

    def _name(self, i: int) -> str:
        return f"{int(self.frames[i]['frame']):06d}"

    def rgb(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.root / "rgb" / f"{self._name(i)}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self._name(i))
        return img

    def depth_m(self, i: int, folder: str = "depth") -> np.ndarray:
        d = cv2.imread(str(self.root / folder / f"{self._name(i)}.png"), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(self._name(i))
        return d.astype(np.float32) / 1000.0

    @property
    def has_raw_lidar(self) -> bool:
        """v2 recordings keep raw LiDAR in lidar/ and an RGB-aligned, same-size depth in depth/."""
        return (self.root / "lidar").is_dir()

    def lidar_m(self, i: int) -> np.ndarray:
        """Raw measured LiDAR depth (256x192) in metres."""
        return self.depth_m(i, "lidar" if self.has_raw_lidar else "depth")

    def depth_aligned_m(self, i: int, shape: tuple[int, int] | None = None) -> np.ndarray:
        """Depth with the same size as the RGB image (v1: nearest-neighbour upsampled LiDAR)."""
        if self.has_raw_lidar:
            return self.depth_m(i, "depth")
        h, w = shape if shape is not None else self.rgb(i).shape[:2]
        return cv2.resize(self.lidar_m(i), (w, h), interpolation=cv2.INTER_NEAREST)

    def confidence(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.root / "confidence" / f"{self._name(i)}.png"), cv2.IMREAD_UNCHANGED)


def _read_imu(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4))
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[np.argsort(data[:, 0], kind="stable")]


def load(root: str | Path) -> Recording:
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    frames = np.genfromtxt(root / "frames.csv", delimiter=",", names=True)
    return Recording(root=root, meta=meta, frames=np.atleast_1d(frames),
                     accel=_read_imu(root / "accel.csv"),
                     gyro=_read_imu(root / "gyro.csv"),
                     mag=_read_imu(root / "mag.csv"))
