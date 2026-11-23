"""Unified Robot Configuration and Device Selection for AckBot."""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class RobotConfig:
    """
    Canonical physical, kinematic, and sensor specification for AckBot.

    Observation Contract (426-D float32, values in [0, 1] / [-1, 1]):
        - [0 : 64]    Front camera depth rays (64 samples), normalized to [0, 1]
                      between [cam_range_min, cam_range_max], ordered right-to-left
                      (-cam_fov_deg/2 to +cam_fov_deg/2 in REP 103 yaw, matching
                      ActorCritic `cam_angles` in [-1, +1]).
        - [64 : 424]  Planar 2D LiDAR rays (360 samples), normalized to [0, 1]
                      between [lidar_range_min, lidar_range_max], ordered
                      counter-clockwise starting from +X (0 to 2*pi, 1 deg step).
        - [424 : 426] Proprioceptive state [v_norm, steer_norm] in [-1, 1],
                      corresponding to physical [v, steer] via normalize_action().
    """

    # Chassis Geometry & Mass
    chassis_length: float = 0.30
    chassis_width: float = 0.20
    chassis_mass: float = 5.0
    total_mass: float = 6.6
    ground_clearance: float = 0.06

    # Wheels & Axles
    wheel_diameter: float = 0.12
    wheel_radius: float = 0.06
    wheel_width: float = 0.06
    wheel_mass: float = 0.40
    wheelbase: float = 0.30
    track_width: float = 0.30
    overall_width: float = 0.36
    chassis_gap: float = 0.02

    # Kinematics & Limits
    min_turning_radius: float = 0.485
    max_steering_angle: float = 0.553953  # rad (~31.74 deg)
    v_max: float = 0.80  # m/s
    v_min: float = -0.40  # m/s
    max_steering_rate: float = 1.0471975512  # rad/s (60 deg/s)
    dt: float = 0.05  # 20 Hz simulation / control loop

    # Camera RGB-D Parameters
    cam_pos_x: float = 0.15
    cam_pos_y: float = 0.00
    cam_pos_z: float = 0.25
    cam_fov_deg: float = 85.0
    cam_fov_rad: float = math.radians(85.0)
    cam_range_min: float = 0.15
    cam_range_max: float = 5.00
    cam_samples: int = 64

    # LiDAR 2D Parameters
    lidar_pos_x: float = -0.15
    lidar_pos_y: float = 0.00
    lidar_pos_z: float = 0.30
    lidar_fov_deg: float = 360.0
    lidar_range_min: float = 0.10
    lidar_range_max: float = 3.00
    lidar_samples: int = 360

    # Observation & Action Dimensions
    obs_dim: int = 64 + 360 + 2  # 426
    act_dim: int = 2

    # Following Goal Target
    target_follow_distance: float = 0.80

    def denormalize_cam(self, cam_norm: np.ndarray) -> np.ndarray:
        """Denormalize [0, 1] camera depth back to meters."""
        return cam_norm * (self.cam_range_max - self.cam_range_min) + self.cam_range_min

    def normalize_cam(self, cam_meters: np.ndarray) -> np.ndarray:
        """Normalize camera depth meters to [0, 1]."""
        clipped = np.clip(cam_meters, self.cam_range_min, self.cam_range_max)
        return (clipped - self.cam_range_min) / (self.cam_range_max - self.cam_range_min)

    def denormalize_lidar(self, lidar_norm: np.ndarray) -> np.ndarray:
        """Denormalize [0, 1] LiDAR range back to meters."""
        return lidar_norm * (self.lidar_range_max - self.lidar_range_min) + self.lidar_range_min

    def normalize_lidar(self, lidar_meters: np.ndarray) -> np.ndarray:
        """Normalize LiDAR range meters to [0, 1]."""
        clipped = np.clip(lidar_meters, self.lidar_range_min, self.lidar_range_max)
        return (clipped - self.lidar_range_min) / (self.lidar_range_max - self.lidar_range_min)

    def normalize_state(self, v: float, steer: float) -> np.ndarray:
        """Map physical proprioception [v, steer] to normalized [-1, 1] observation slice."""
        return self.normalize_action(v, steer)

    def normalize_action(self, v: float, steer: float) -> np.ndarray:
        """Map physical [v, steer] to normalized [-1, 1] space."""
        v_scale = (self.v_max - self.v_min) / 2.0
        v_bias = (self.v_max + self.v_min) / 2.0
        norm_v = (v - v_bias) / v_scale
        norm_steer = steer / self.max_steering_angle
        return np.array(
            [np.clip(norm_v, -1.0, 1.0), np.clip(norm_steer, -1.0, 1.0)], dtype=np.float32
        )

    def unscale_action(self, norm_v: float, norm_steer: float) -> tuple[float, float]:
        """Map normalized action [-1, 1] back to physical [v, steer]."""
        v_scale = (self.v_max - self.v_min) / 2.0
        v_bias = (self.v_max + self.v_min) / 2.0
        v = float(np.clip(norm_v * v_scale + v_bias, self.v_min, self.v_max))
        steer = float(
            np.clip(
                norm_steer * self.max_steering_angle,
                -self.max_steering_angle,
                self.max_steering_angle,
            )
        )
        return v, steer


def _resolve_default_config_path() -> Path | None:
    candidates = [
        Path(__file__).resolve().parent.parent / "configs" / "robot_params.yaml",
        Path.cwd() / "configs" / "robot_params.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


@lru_cache(maxsize=8)
def _load_robot_config_cached(resolved_path: str | None) -> RobotConfig:
    if resolved_path and Path(resolved_path).exists():
        with open(resolved_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        chassis = data.get("chassis", {})
        wheels = data.get("wheels", {})
        kinematics = data.get("kinematics", {})
        camera = data.get("camera_rgbd", {})
        lidar = data.get("lidar_2d", {})

        cam_samples = int(camera.get("samples", 64))
        lidar_samples = int(lidar.get("samples", 360))
        cam_fov_deg = float(camera.get("fov_h_deg", 85.0))

        return RobotConfig(
            chassis_length=float(chassis.get("length_x", 0.30)),
            chassis_width=float(chassis.get("width_y", 0.20)),
            chassis_mass=float(chassis.get("mass", 5.0)),
            total_mass=float(chassis.get("total_mass", 6.6)),
            ground_clearance=float(chassis.get("ground_clearance", 0.06)),
            wheel_diameter=float(wheels.get("diameter", 0.12)),
            wheel_radius=float(wheels.get("radius", 0.06)),
            wheel_width=float(wheels.get("width", 0.06)),
            wheel_mass=float(wheels.get("mass", 0.40)),
            wheelbase=float(wheels.get("wheelbase", 0.30)),
            track_width=float(wheels.get("track_width_front", 0.30)),
            overall_width=float(wheels.get("overall_width", 0.36)),
            chassis_gap=float(wheels.get("chassis_gap", 0.02)),
            min_turning_radius=float(kinematics.get("min_turning_radius", 0.485)),
            max_steering_angle=float(kinematics.get("max_steering_angle", 0.553953)),
            v_max=float(kinematics.get("max_linear_speed", 0.80)),
            v_min=float(kinematics.get("min_linear_speed", -0.40)),
            max_steering_rate=float(kinematics.get("max_steering_rate", 1.0471975512)),
            dt=0.05,
            cam_pos_x=float(camera.get("pos_x", 0.15)),
            cam_pos_y=float(camera.get("pos_y", 0.00)),
            cam_pos_z=float(camera.get("pos_z", 0.25)),
            cam_fov_deg=cam_fov_deg,
            cam_fov_rad=math.radians(cam_fov_deg),
            cam_range_min=float(camera.get("depth_min", 0.15)),
            cam_range_max=float(camera.get("depth_max", 5.00)),
            cam_samples=cam_samples,
            lidar_pos_x=float(lidar.get("pos_x", -0.15)),
            lidar_pos_y=float(lidar.get("pos_y", 0.00)),
            lidar_pos_z=float(lidar.get("pos_z", 0.30)),
            lidar_fov_deg=float(lidar.get("fov_deg", 360.0)),
            lidar_range_min=float(lidar.get("range_min", 0.10)),
            lidar_range_max=float(lidar.get("range_max", 3.00)),
            lidar_samples=lidar_samples,
            obs_dim=cam_samples + lidar_samples + 2,
        )
    return RobotConfig()


def get_robot_config(config_path: str | Path | None = None) -> RobotConfig:
    """Load and cache RobotConfig keyed by resolved YAML path so custom paths never pollute defaults."""
    if config_path is None:
        resolved = _resolve_default_config_path()
    else:
        resolved = Path(config_path).resolve()
    return _load_robot_config_cached(str(resolved) if resolved is not None else None)


def configure_cpu_threads(
    num_threads: int | str | None = "auto",
    num_envs: int = 1,
    vec_env: str = "sync",
) -> int:
    """
    Configure PyTorch and BLAS/OpenMP CPU threads to avoid core oversubscription.

    When running `AsyncVectorEnv` on modest core counts (e.g., 6-core Linux VM), child
    worker processes inherit `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` while the main
    process uses a bounded thread count for PyTorch BPTT updates.
    """
    cpu_count = os.cpu_count() or 1
    if num_threads is None or num_threads == "auto":
        if vec_env == "async" and num_envs > 1:
            # Leave headroom for OS/IPC and avoid contention if workers share cores.
            threads = max(1, min(4, cpu_count // 2 if cpu_count >= 4 else cpu_count))
        else:
            threads = max(1, min(4, cpu_count))
    else:
        threads = max(1, int(num_threads))

    if vec_env == "async" and num_envs > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

    torch.set_num_threads(threads)
    return threads


def select_device(requested: str | None = "auto") -> torch.device:
    """Select the best available compute device (CUDA, MPS on Apple Silicon, or CPU)."""
    req = (requested or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn("CUDA requested but unavailable; falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")
    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS requested but unavailable; falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")
    return torch.device(req)
