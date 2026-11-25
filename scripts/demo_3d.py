#!/usr/bin/env python3
"""
Interactive 3D Demo Show & Playable Human-Target Game for AckBot (MuJoCo + GLFW).

Features:
  - Live 3D OpenGL rendering directly via MuJoCo (`mjr_render`) — works with `uv run python`
    on macOS, Linux, and Windows (no `mjpython` wrapper required).
  - Choice of two modes (at startup or toggled live anytime via `M`):
      1. Auto Demo Show: Autonomous bipedal walker + trained AckBot across Stages 1..7.
      2. Human Game Control: Control the human target with WASD / Arrow keys (+ Shift sprint)
         with camera-relative steering and smooth wall sliding while AckBot tracks you!
  - Live 3D sensor overlays (360-deg LiDAR hits, 85-deg camera FOV fan, 0.80m goal ring, trails),
    multiple camera presets (`C`), mouse orbit/pan/zoom, and Robot Front-Cam Picture-in-Picture (`P`).
  - Optional headless 3D MP4 demo video export (`--record outputs/video/demo_3d.mp4`).

Examples:
    uv run python scripts/demo_3d.py
    uv run python scripts/demo_3d.py --mode human --stage 7
    uv run python scripts/demo_3d.py --mode auto --tour
    uv run python scripts/demo_3d.py --record outputs/video/demo_3d.mp4 --stages 1 3 5 7
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

import ackermann_follow.envs  # noqa: F401
from ackermann_follow.config import select_device
from ackermann_follow.engine.evaluator import _infer_action, _init_agent_state
from ackermann_follow.envs.mujoco_follow_env import MuJoCoAckermannFollowEnv, _wrap_angle
from scripts.eval_policy import load_policy

STAGE_NAMES = {
    1: "Free Obstacle Arena",
    2: "Straight Corridor",
    3: "L-Turn Outside Wall",
    4: "L-Turn Inside Corridor",
    5: "Dead-End (Wall-Hug)",
    6: "Dead-End (Middle Walk)",
    7: "L-Turn -> Dead-End",
}

CAMERA_NAMES = (
    "Chase AckBot",
    "Follow Human (Game)",
    "Cinema Orbit",
    "Arena 3D Overview",
)


def find_default_checkpoint(experiments_root: str | Path = "outputs/experiments") -> Path | None:
    """Locate the newest trained best_model.pt (or latest_model.pt) checkpoint on disk."""
    root = Path(experiments_root)
    if not root.exists():
        return None
    for filename in ("best_model.pt", "latest_model.pt"):
        candidates = sorted(
            root.glob(f"*/checkpoints/{filename}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive 3D MuJoCo Demo Show & Playable Human-Target Game for AckBot"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained policy checkpoint (.pt); auto-detects latest best_model.pt if omitted",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="choose",
        choices=["choose", "auto", "human"],
        help="Control mode: interactive prompt ('choose'), autonomous showcase ('auto'), or playable human ('human')",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Initial curriculum stage (1..7)",
    )
    parser.add_argument(
        "--tour",
        action="store_true",
        default=None,
        help="Automatically cycle through Stages 1..7 in Auto mode",
    )
    parser.add_argument(
        "--no-tour",
        action="store_false",
        dest="tour",
        help="Stay on the selected stage across resets",
    )
    parser.add_argument("--width", type=int, default=1280, help="Window / video width")
    parser.add_argument("--height", type=int, default=800, help="Window / video height")
    parser.add_argument("--seed", type=int, default=101, help="Base RNG seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Policy inference device",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Export a headless 3D MP4 demo show instead of opening an interactive window",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="outputs/video",
        help="Output folder for live recordings toggled during the interactive demo (X / F9)",
    )
    parser.add_argument(
        "--start-recording",
        action="store_true",
        help="Automatically begin recording MP4 as soon as the interactive 3D window opens",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7],
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Stages to include when exporting with --record",
    )
    parser.add_argument(
        "--max-record-steps",
        type=int,
        default=500,
        help="Maximum steps per stage clip when recording with --record",
    )
    return parser.parse_args()


def prompt_startup_choice(default_stage: int | None) -> tuple[str, int, bool]:
    """Prompt the user in terminal to pick Auto Demo Show vs Human Game Control."""
    print("=" * 68)
    print("         ACKBOT 3D DEMO SHOW & PLAYABLE TARGET SIMULATOR")
    print("=" * 68)
    print("Choose simulation mode:")
    print("  [1] Auto Demo Show     - Watch AckBot follow an AI walker (Stages 1..7)")
    print("  [2] Human Game Control - Control the human with WASD / Arrows!")
    print("      (Tip: Press 'M' anytime in the 3D window to switch modes live)")
    print("-" * 68)

    raw_mode = input("Select mode [1/2, default=1]: ").strip().lower()
    mode = "human" if raw_mode in ("2", "h", "human", "game") else "auto"

    if default_stage is not None:
        return mode, default_stage, False

    print("\nSelect starting stage:")
    if mode == "auto":
        print("  [0] All-Stages Showcase Tour (1 -> 7)")
    for stg in range(1, 8):
        print(f"  [{stg}] Stage {stg}: {STAGE_NAMES[stg]}")

    default_choice = "0" if mode == "auto" else "1"
    raw_stg = input(f"Select stage [default={default_choice}]: ").strip()
    if raw_stg == "" and mode == "auto":
        return mode, 1, True
    if raw_stg == "0" and mode == "auto":
        return mode, 1, True
    try:
        stg_num = int(raw_stg) if raw_stg else 1
        stg_num = int(np.clip(stg_num, 1, 7))
    except ValueError:
        stg_num = 1
    return mode, stg_num, False


def _active_wall_boxes(env: MuJoCoAckermannFollowEnv) -> list[tuple[float, float, float, float]]:
    """Return (cx, cy, hx, hy) for all active static group-0 box walls in the scene."""
    boxes: list[tuple[float, float, float, float]] = []
    m = env.model
    d = env.data
    human_set = set(int(g) for g in env._human_gids)
    for gid in range(m.ngeom):
        if m.geom_group[gid] != 0 or gid in human_set:
            continue
        if int(m.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        gx = float(d.geom_xpos[gid, 0])
        gy = float(d.geom_xpos[gid, 1])
        hx = float(m.geom_size[gid, 0])
        hy = float(m.geom_size[gid, 1])
        boxes.append((gx, gy, hx, hy))
    return boxes


def resolve_human_wall_collisions(
    x: float,
    y: float,
    prev_x: float,
    prev_y: float,
    boxes: list[tuple[float, float, float, float]],
    radius: float = 0.22,
    arena_limit: float = 7.75,
) -> tuple[float, float]:
    """
    Resolve 2D circle-vs-AABB collisions for the player-controlled human with smooth wall sliding.
    Tests X and Y axes sequentially so walking diagonally into a corridor wall slides cleanly.
    """
    nx = float(np.clip(x, -arena_limit, arena_limit))
    ny = float(np.clip(y, -arena_limit, arena_limit))

    # Resolve X motion first against boxes
    for cx, cy, hx, hy in boxes:
        if abs(prev_y - cy) < (hy + radius) and abs(nx - cx) < (hx + radius):
            if prev_x <= cx:
                nx = min(nx, cx - (hx + radius))
            else:
                nx = max(nx, cx + (hx + radius))

    # Resolve Y motion next against boxes
    for cx, cy, hx, hy in boxes:
        if abs(nx - cx) < (hx + radius) and abs(ny - cy) < (hy + radius):
            if prev_y <= cy:
                ny = min(ny, cy - (hy + radius))
            else:
                ny = max(ny, cy + (hy + radius))

    # Final push-out pass in case of corner overlap
    for cx, cy, hx, hy in boxes:
        dx = nx - cx
        dy = ny - cy
        px = (hx + radius) - abs(dx)
        py = (hy + radius) - abs(dy)
        if px > 0.0 and py > 0.0:
            if px < py:
                nx += px if dx >= 0.0 else -px
            else:
                ny += py if dy >= 0.0 else -py

    return float(np.clip(nx, -arena_limit, arena_limit)), float(np.clip(ny, -arena_limit, arena_limit))


class HumanGameController:
    """Translates camera-relative keyboard input into smooth bipedal human movement."""

    def __init__(
        self,
        walk_speed: float = 0.42,
        sprint_speed: float = 1.05,
        slow_speed: float = 0.24,
        accel_rate: float = 1.20,
        decel_rate: float = 1.80,
    ) -> None:
        self.walk_speed = float(walk_speed)
        self.sprint_speed = float(sprint_speed)
        self.slow_speed = float(slow_speed)
        self.accel_rate = float(accel_rate)
        self.decel_rate = float(decel_rate)
        self.cur_speed = 0.0

    def step_from_input(
        self,
        env: MuJoCoAckermannFollowEnv,
        forward_input: float,
        right_input: float,
        cam_forward_xy: tuple[float, float],
        sprint: bool = False,
        slow: bool = False,
        boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> None:
        """Advance the human target one env step using camera-relative WASD/Arrow axes."""
        fx, fy = cam_forward_xy
        fnorm = math.hypot(fx, fy)
        if fnorm < 1e-6:
            fx, fy = 1.0, 0.0
        else:
            fx /= fnorm
            fy /= fnorm
        # Right vector perpendicular to forward in +Z-up world
        rx, ry = fy, -fx

        move_x = forward_input * fx + right_input * rx
        move_y = forward_input * fy + right_input * ry
        mag = math.hypot(move_x, move_y)

        target_speed = 0.0
        if mag > 1e-4:
            target_speed = self.sprint_speed if sprint else (self.slow_speed if slow else self.walk_speed)
            desired_yaw = math.atan2(move_y, move_x)
            yaw_err = _wrap_angle(desired_yaw - env.target.yaw)
            env.target.omega = float(np.clip(7.0 * yaw_err, -6.5, 6.5))
        else:
            env.target.omega = 0.0

        # Smooth acceleration / deceleration so holding Shift progressively speeds up
        dt = env.cfg.dt
        if target_speed > self.cur_speed:
            self.cur_speed = min(target_speed, self.cur_speed + self.accel_rate * dt)
        elif target_speed < self.cur_speed:
            self.cur_speed = max(target_speed, self.cur_speed - self.decel_rate * dt)
        if self.cur_speed < 0.015 and target_speed == 0.0:
            self.cur_speed = 0.0

        prev_x, prev_y = env.target.x, env.target.y
        env.target.speed = self.cur_speed
        env.target.step(env.cfg.dt)

        if boxes is None:
            boxes = _active_wall_boxes(env)
        env.target.x, env.target.y = resolve_human_wall_collisions(
            env.target.x, env.target.y, prev_x, prev_y, boxes
        )
        env._sync_human_mocap()


def _add_connector(
    scn: mujoco.MjvScene,
    geom_type: int,
    width: float,
    p0: np.ndarray,
    p1: np.ndarray,
    rgba: tuple[float, float, float, float],
) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        geom_type,
        float(width),
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scn.ngeom += 1


def _add_sphere(
    scn: mujoco.MjvScene,
    pos: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def append_3d_overlays(
    scn: mujoco.MjvScene,
    env: MuJoCoAckermannFollowEnv,
    obs: np.ndarray,
    info: dict[str, Any],
    robot_trail: list[tuple[float, float]],
    human_trail: list[tuple[float, float]],
) -> None:
    """Inject 3D LiDAR hits, 85-deg camera FOV rays, 0.80m goal ring, and trails into `scn`."""
    cfg = env.cfg
    rx, ry, ryaw, _, _ = env._get_robot_pose_and_vel()
    hx, hy = env.target.x, env.target.y

    cam_end = cfg.cam_samples
    mask_end = cam_end + cfg.cam_mask_samples
    lidar_end = mask_end + cfg.lidar_samples

    cam_dists = cfg.denormalize_cam(obs[:cam_end])
    cam_mask = obs[cam_end:mask_end]
    lidar_dists = cfg.denormalize_lidar(obs[mask_end:lidar_end])

    # 1. Target follow ring (0.80 m ideal distance around human)
    in_fov = bool(info.get("target_in_fov", False))
    coll = bool(info.get("collision", False))
    ring_col = (0.95, 0.25, 0.20, 0.90) if coll else ((0.18, 0.88, 0.45, 0.80) if in_fov else (0.96, 0.78, 0.16, 0.80))
    n_seg = 24
    r_goal = cfg.target_follow_distance
    for k in range(n_seg):
        a0 = 2.0 * math.pi * k / n_seg
        a1 = 2.0 * math.pi * (k + 1) / n_seg
        p0 = np.array([hx + r_goal * math.cos(a0), hy + r_goal * math.sin(a0), 0.015])
        p1 = np.array([hx + r_goal * math.cos(a1), hy + r_goal * math.sin(a1), 0.015])
        _add_connector(scn, int(mujoco.mjtGeom.mjGEOM_CAPSULE), 0.012, p0, p1, ring_col)

    # 2. Front RGB-D camera rays (every 3rd ray + all target-locked rays)
    cam_origin = np.array(env.data.site_xpos[env._cam_site_id], dtype=np.float64)
    cam_angles = np.linspace(-0.5 * cfg.cam_fov_rad, 0.5 * cfg.cam_fov_rad, cfg.cam_samples)
    for idx in range(cfg.cam_samples):
        is_target = cam_mask[idx] > 0.5
        if not is_target and (idx % 4 != 0) and idx not in (0, cfg.cam_samples - 1):
            continue
        ang = ryaw + float(cam_angles[idx])
        dist_i = float(cam_dists[idx])
        end_pt = np.array(
            [cam_origin[0] + dist_i * math.cos(ang), cam_origin[1] + dist_i * math.sin(ang), cam_origin[2]]
        )
        col = (0.15, 0.95, 0.42, 0.75) if is_target else (0.20, 0.78, 0.95, 0.22)
        w = 0.008 if is_target else 0.004
        _add_connector(scn, int(mujoco.mjtGeom.mjGEOM_LINE), w, cam_origin, end_pt, col)

    # 3. 360-deg LiDAR obstacle hit points (sampled every 4 deg for crisp visuals)
    lidar_origin = np.array(env.data.site_xpos[env._lidar_site_id], dtype=np.float64)
    for idx in range(0, cfg.lidar_samples, 4):
        dist_i = float(lidar_dists[idx])
        if dist_i >= cfg.lidar_range_max - 0.05:
            continue
        ang = ryaw + idx * (2.0 * math.pi / cfg.lidar_samples)
        hit_pt = np.array(
            [lidar_origin[0] + dist_i * math.cos(ang), lidar_origin[1] + dist_i * math.sin(ang), lidar_origin[2]]
        )
        _add_sphere(scn, hit_pt, 0.022, (1.0, 0.48, 0.12, 0.85))

    # 4. Ground trails for robot (blue) & human (coral red)
    if len(robot_trail) > 1:
        pts = robot_trail[-160::3]
        for a, b in zip(pts[:-1], pts[1:]):
            _add_connector(
                scn,
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                0.012,
                np.array([a[0], a[1], 0.012]),
                np.array([b[0], b[1], 0.012]),
                (0.20, 0.58, 0.95, 0.65),
            )
    if len(human_trail) > 1:
        pts = human_trail[-160::3]
        for a, b in zip(pts[:-1], pts[1:]):
            _add_connector(
                scn,
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                0.010,
                np.array([a[0], a[1], 0.010]),
                np.array([b[0], b[1], 0.010]),
                (0.92, 0.30, 0.25, 0.55),
            )


def update_camera_preset(
    cam: mujoco.MjvCamera,
    preset_idx: int,
    env: MuJoCoAckermannFollowEnv,
    yaw_offset_deg: float,
    elev_offset_deg: float,
    dist_scale: float,
    pan_offset: np.ndarray,
    sim_time: float,
) -> None:
    """Position the main 3D MjvCamera according to the active camera preset + user mouse offsets."""
    rx, ry, ryaw, _, _ = env._get_robot_pose_and_vel()
    hx, hy = env.target.x, env.target.y

    if preset_idx == 0:  # Chase AckBot
        cam.lookat[:] = [0.65 * rx + 0.35 * hx + pan_offset[0], 0.65 * ry + 0.35 * hy + pan_offset[1], 0.45 + pan_offset[2]]
        cam.azimuth = math.degrees(ryaw) + yaw_offset_deg
        cam.elevation = float(np.clip(-42.0 + elev_offset_deg, -85.0, -5.0))
        cam.distance = float(np.clip(5.2 * dist_scale, 1.2, 22.0))
    elif preset_idx == 1:  # Follow Human (3rd-person game cam)
        cam.lookat[:] = [hx + pan_offset[0], hy + pan_offset[1], 0.85 + pan_offset[2]]
        cam.azimuth = math.degrees(env.target.yaw) + yaw_offset_deg
        cam.elevation = float(np.clip(-44.0 + elev_offset_deg, -85.0, -5.0))
        cam.distance = float(np.clip(5.6 * dist_scale, 1.5, 22.0))
    elif preset_idx == 2:  # Cinema Orbit
        cam.lookat[:] = [0.5 * (rx + hx) + pan_offset[0], 0.5 * (ry + hy) + pan_offset[1], 0.55 + pan_offset[2]]
        cam.azimuth = 35.0 + sim_time * 10.0 + yaw_offset_deg
        cam.elevation = float(np.clip(-46.0 + elev_offset_deg, -85.0, -5.0))
        cam.distance = float(np.clip(6.8 * dist_scale, 1.8, 24.0))
    else:  # Arena 3D Overview
        cam.lookat[:] = [0.5 * (rx + hx) + pan_offset[0], 0.5 * (ry + hy) + pan_offset[1], 0.25 + pan_offset[2]]
        cam.azimuth = 0.0 + yaw_offset_deg
        cam.elevation = float(np.clip(-68.0 + elev_offset_deg, -88.0, -15.0))
        cam.distance = float(np.clip(12.5 * dist_scale, 2.5, 28.0))


def configure_robot_pip_camera(pip_cam: mujoco.MjvCamera, env: MuJoCoAckermannFollowEnv) -> None:
    """Align `pip_cam` with AckBot's front RGB-D camera looking straight ahead."""
    rx, ry, ryaw, _, _ = env._get_robot_pose_and_vel()
    cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)
    # Offset slightly ahead of cam_housing (x=0.14..0.155) so housing never occludes near plane
    cx = rx + 0.175 * cos_y
    cy = ry + 0.175 * sin_y
    cz = 0.26
    look_dist = 2.0
    pip_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    pip_cam.lookat[:] = [cx + look_dist * cos_y, cy + look_dist * sin_y, cz]
    pip_cam.distance = look_dist
    pip_cam.azimuth = math.degrees(ryaw)
    pip_cam.elevation = -3.0


def _step_env_with_human_choice(
    env: MuJoCoAckermannFollowEnv,
    action: np.ndarray,
    human_mode: bool,
    human_ctrl: HumanGameController,
    fwd_in: float,
    right_in: float,
    cam_fwd_xy: tuple[float, float],
    sprint: bool,
    slow: bool,
    boxes: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """Execute one 20 Hz step using either AI target or player-controlled human target."""
    if not human_mode:
        return env.step(action)

    act = np.asarray(action, dtype=np.float32).reshape(2)
    env._prev_steer_cmd = env._steer_cmd
    env._apply_ackermann_control(float(act[0]), float(act[1]))
    human_ctrl.step_from_input(
        env,
        forward_input=fwd_in,
        right_input=right_in,
        cam_forward_xy=cam_fwd_xy,
        sprint=sprint,
        slow=slow,
        boxes=boxes,
    )

    for _ in range(env.n_substeps):
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)
    env._step_count += 1

    obs, min_lidar = env._get_obs()
    env._last_min_lidar = min_lidar
    dist, bearing, geom_in_fov = env._compute_target_metrics()
    in_fov = geom_in_fov and (env._last_cam_target_visible or dist <= 1.15)
    collision = env._check_collision()
    terminated = collision
    truncated = (not terminated) and (env._step_count >= env.max_steps)
    reward = -25.0 if collision else math.exp(-((abs(dist - env.cfg.target_follow_distance) / 0.65) ** 2))
    info = {
        "stage": env.stage,
        "unlocked_stage": env.unlocked_stage,
        "target_dist": dist,
        "target_bearing": bearing,
        "target_in_fov": in_fov,
        "min_lidar": min_lidar,
        "collision": collision,
    }
    return obs, float(reward), terminated, truncated, info


def _open_ffmpeg_writer(width: int, height: int, fps: int, out_path: Path) -> subprocess.Popen[bytes]:
    """Launch an ffmpeg rawvideo -> H.264 MP4 encoder process."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found in PATH.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    return proc


class LiveVideoRecorder:
    """Streams live OpenGL framebuffer captures to an MP4 file during the interactive demo."""

    def __init__(self, fps: int = 20) -> None:
        self.fps = int(fps)
        self.proc: subprocess.Popen[bytes] | None = None
        self.out_path: Path | None = None
        self.width = 0
        self.height = 0
        self.frame_count = 0

    @property
    def is_recording(self) -> bool:
        return self.proc is not None

    def start(self, width: int, height: int, output_path: Path) -> Path:
        if self.is_recording:
            self.stop()
        # libx264 yuv420p requires even width and height
        rec_w = max(2, int(width) - (int(width) % 2))
        rec_h = max(2, int(height) - (int(height) % 2))
        self.proc = _open_ffmpeg_writer(rec_w, rec_h, self.fps, output_path)
        self.out_path = output_path
        self.width = rec_w
        self.height = rec_h
        self.frame_count = 0
        print(f"[REC] Started live recording -> {output_path} ({rec_w}x{rec_h} @ {self.fps} fps)")
        return output_path

    def write_frame(self, rgb_frame: np.ndarray, flip_vertical: bool = True) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        frame = np.flipud(rgb_frame) if flip_vertical else rgb_frame
        fh, fw = frame.shape[0], frame.shape[1]
        if fw != self.width or fh != self.height:
            from PIL import Image

            frame = np.asarray(
                Image.fromarray(frame).resize((self.width, self.height), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        else:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
        self.proc.stdin.write(frame.tobytes())
        self.frame_count += 1

    def stop(self) -> tuple[Path | None, int]:
        if self.proc is None:
            return None, 0
        proc = self.proc
        out_path = self.out_path
        frames = self.frame_count
        self.proc = None
        self.out_path = None
        self.frame_count = 0

        if proc.stdin is not None:
            proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        retcode = proc.wait()
        if retcode != 0:
            raise RuntimeError(f"ffmpeg failed (exit {retcode}): {stderr}")
        if out_path is not None and out_path.exists():
            size_kb = out_path.stat().st_size / 1024.0
            print(f"[REC] Saved {frames} frames ({frames / self.fps:.1f}s) to {out_path} ({size_kb:.1f} KB)")
        return out_path, frames


def run_interactive_3d(
    args: argparse.Namespace,
    model,
    device: torch.device,
    mode: str,
    start_stage: int,
    tour_enabled: bool,
) -> None:
    """Run the interactive live GLFW + MuJoCo 3D window."""
    import glfw

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW for 3D demo window.")

    env = MuJoCoAckermannFollowEnv(sensor_noise=False)
    window = None
    ctx = None

    try:
        glfw.window_hint(glfw.SAMPLES, 4)
        window = glfw.create_window(
            args.width,
            args.height,
            "AckBot 3D Demo Show & Human Control Game",
            None,
            None,
        )
        if not window:
            raise RuntimeError("Failed to create GLFW OpenGL window.")
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        opt = mujoco.MjvOption()
        cam = mujoco.MjvCamera()
        pip_cam = mujoco.MjvCamera()
        scn = mujoco.MjvScene(env.model, maxgeom=4000)
        pip_scn = mujoco.MjvScene(env.model, maxgeom=1500)
        ctx = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        state: dict[str, Any] = {
            "human_mode": mode == "human",
            "stage": int(start_stage),
            "tour": bool(tour_enabled),
            "cam_preset": 1 if mode == "human" else 0,
            "show_sensors": True,
            "show_pip": True,
            "show_help": True,
            "paused": False,
            "reset_req": False,
            "rec_toggle_req": bool(args.start_recording),
            "quit_req": False,
            "yaw_off": 0.0,
            "elev_off": 0.0,
            "dist_scale": 1.0,
            "pan_off": np.zeros(3, dtype=np.float64),
            "mouse_left": False,
            "mouse_right": False,
            "shift_held": False,
            "ctrl_held": False,
            "last_mx": 0.0,
            "last_my": 0.0,
        }

        def on_key(win, key, scancode, action, mods):
            del win, scancode
            state["shift_held"] = bool(mods & glfw.MOD_SHIFT)
            state["ctrl_held"] = bool(mods & glfw.MOD_CONTROL)
            if key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT):
                state["shift_held"] = action != glfw.RELEASE
                return
            if key in (glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL):
                state["ctrl_held"] = action != glfw.RELEASE
                return
            if action != glfw.PRESS:
                return
            if key == glfw.KEY_ESCAPE:
                state["quit_req"] = True
            elif key == glfw.KEY_M:
                state["human_mode"] = not state["human_mode"]
                if state["human_mode"] and state["cam_preset"] == 0:
                    state["cam_preset"] = 1
                elif not state["human_mode"] and state["cam_preset"] == 1:
                    state["cam_preset"] = 0
                state["yaw_off"] = 0.0
                state["elev_off"] = 0.0
                state["pan_off"][:] = 0.0
            elif key == glfw.KEY_C:
                state["cam_preset"] = (state["cam_preset"] + 1) % len(CAMERA_NAMES)
                state["yaw_off"] = 0.0
                state["elev_off"] = 0.0
                state["dist_scale"] = 1.0
                state["pan_off"][:] = 0.0
            elif key == glfw.KEY_V:
                state["show_sensors"] = not state["show_sensors"]
            elif key == glfw.KEY_P:
                state["show_pip"] = not state["show_pip"]
            elif key == glfw.KEY_H:
                state["show_help"] = not state["show_help"]
            elif key == glfw.KEY_SPACE:
                state["paused"] = not state["paused"]
            elif key == glfw.KEY_R:
                state["reset_req"] = True
            elif key == glfw.KEY_N:
                state["stage"] = (state["stage"] % 7) + 1
                state["reset_req"] = True
            elif key == glfw.KEY_T:
                state["tour"] = not state["tour"]
            elif key in (glfw.KEY_X, glfw.KEY_F9):
                state["rec_toggle_req"] = True
            elif glfw.KEY_1 <= key <= glfw.KEY_7:
                state["stage"] = (key - glfw.KEY_1) + 1
                state["tour"] = False
                state["reset_req"] = True

        def on_mouse_btn(win, button, action, mods):
            del mods
            mx, my = glfw.get_cursor_pos(win)
            state["last_mx"], state["last_my"] = mx, my
            if button == glfw.MOUSE_BUTTON_LEFT:
                state["mouse_left"] = action == glfw.PRESS
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                state["mouse_right"] = action == glfw.PRESS

        def on_cursor(win, mx, my):
            dx = mx - state["last_mx"]
            dy = my - state["last_my"]
            state["last_mx"], state["last_my"] = mx, my
            if state["mouse_left"]:
                state["yaw_off"] -= dx * 0.30
                state["elev_off"] -= dy * 0.25
            elif state["mouse_right"]:
                az_rad = math.radians(cam.azimuth)
                right_x, right_y = math.sin(az_rad), -math.cos(az_rad)
                fwd_x, fwd_y = math.cos(az_rad), math.sin(az_rad)
                scale = 0.005 * max(1.0, cam.distance)
                state["pan_off"][0] -= (dx * right_x - dy * fwd_x) * scale
                state["pan_off"][1] -= (dx * right_y - dy * fwd_y) * scale

        def on_scroll(win, xoff, yoff):
            del win, xoff
            state["dist_scale"] = float(np.clip(state["dist_scale"] * (0.90 ** yoff), 0.35, 3.5))

        glfw.set_key_callback(window, on_key)
        glfw.set_mouse_button_callback(window, on_mouse_btn)
        glfw.set_cursor_pos_callback(window, on_cursor)
        glfw.set_scroll_callback(window, on_scroll)

        human_ctrl = HumanGameController()
        episode_idx = 0

        def reset_episode(stg: int) -> tuple[np.ndarray, dict[str, Any], torch.Tensor, list, list, list, float, np.ndarray]:
            nonlocal episode_idx
            episode_idx += 1
            opts = {"stage": stg, "force_front_spawn": True} if state["human_mode"] or state["tour"] else {"stage": stg}
            obs_0, info_0 = env.reset(seed=args.seed + stg * 100 + episode_idx, options=opts)
            hx_0 = _init_agent_state(model, device)
            boxes_0 = _active_wall_boxes(env)
            human_ctrl.cur_speed = 0.0
            return obs_0, info_0, hx_0, boxes_0, [], [], 0.0, np.zeros(2, dtype=np.float32)

        obs, info, hx, boxes, robot_trail, human_trail, ep_ret, last_action = reset_episode(state["stage"])
        dt = env.cfg.dt
        last_step_t = time.perf_counter()
        last_rec_t = last_step_t
        sim_time = 0.0
        cam_fwd_xy = (1.0, 0.0)
        collision_cooldown = 0.0
        recorder = LiveVideoRecorder(fps=20)
        rec_notice = ""
        rec_notice_until = 0.0
        pixel_buf: np.ndarray | None = None

        while not glfw.window_should_close(window) and not state["quit_req"]:
            glfw.poll_events()

            if state["reset_req"]:
                state["reset_req"] = False
                collision_cooldown = 0.0
                obs, info, hx, boxes, robot_trail, human_trail, ep_ret, last_action = reset_episode(state["stage"])

            now = time.perf_counter()
            capture_rec_frame = False
            if state["rec_toggle_req"]:
                state["rec_toggle_req"] = False
                if recorder.is_recording:
                    saved_path, saved_frames = recorder.stop()
                    if saved_path is not None:
                        rec_notice = f"Saved {saved_frames / 20.0:.1f}s -> {saved_path}"
                        rec_notice_until = now + 4.0
                else:
                    fb_w_cur, fb_h_cur = glfw.get_framebuffer_size(window)
                    if fb_w_cur > 0 and fb_h_cur > 0:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_dir = Path(args.record_dir)
                        rec_path = out_dir / f"live_demo_{ts}.mp4"
                        try:
                            recorder.start(fb_w_cur, fb_h_cur, rec_path)
                            last_rec_t = now - dt
                        except RuntimeError as exc:
                            rec_notice = f"Record error: {exc}"
                            rec_notice_until = now + 4.0

            if recorder.is_recording and (now - last_rec_t) >= dt:
                last_rec_t = now
                capture_rec_frame = True
            if not state["paused"] and (now - last_step_t) >= dt:
                last_step_t = now
                if collision_cooldown > 0.0:
                    collision_cooldown -= dt
                    if collision_cooldown <= 0.0:
                        if state["tour"] and not state["human_mode"]:
                            state["stage"] = (state["stage"] % 7) + 1
                        obs, info, hx, boxes, robot_trail, human_trail, ep_ret, last_action = reset_episode(state["stage"])
                else:
                    sim_time += dt
                    with torch.no_grad():
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        act_t, hx = _infer_action(model, obs_t, hx)
                        last_action = act_t.squeeze(0).cpu().numpy()

                    fwd_in = 0.0
                    right_in = 0.0
                    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS or glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
                        fwd_in += 1.0
                    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS or glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
                        fwd_in -= 1.0
                    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
                        right_in += 1.0
                    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS or glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
                        right_in -= 1.0

                    sprint = (
                        state["shift_held"]
                        or glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                        or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
                    )
                    slow = (
                        state["ctrl_held"]
                        or glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
                        or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
                    )

                    obs, reward, terminated, truncated, info = _step_env_with_human_choice(
                        env=env,
                        action=last_action,
                        human_mode=state["human_mode"],
                        human_ctrl=human_ctrl,
                        fwd_in=fwd_in,
                        right_in=right_in,
                        cam_fwd_xy=cam_fwd_xy,
                        sprint=sprint,
                        slow=slow,
                        boxes=boxes,
                    )
                    ep_ret += reward

                    rx, ry, _, _, _ = env._get_robot_pose_and_vel()
                    robot_trail.append((rx, ry))
                    human_trail.append((env.target.x, env.target.y))

                    if terminated:
                        collision_cooldown = 1.2
                    elif truncated:
                        if state["human_mode"]:
                            # Keep playing seamlessly in human game mode (no forced cutoff!)
                            env._step_count = 0
                        else:
                            if state["tour"]:
                                state["stage"] = (state["stage"] % 7) + 1
                            obs, info, hx, boxes, robot_trail, human_trail, ep_ret, last_action = reset_episode(
                                state["stage"]
                            )

            # Update camera & render main 3D viewport
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            if fb_w <= 0 or fb_h <= 0:
                time.sleep(0.01)
                continue

            update_camera_preset(
                cam=cam,
                preset_idx=state["cam_preset"],
                env=env,
                yaw_offset_deg=state["yaw_off"],
                elev_offset_deg=state["elev_off"],
                dist_scale=state["dist_scale"],
                pan_offset=state["pan_off"],
                sim_time=sim_time,
            )
            mujoco.mjv_updateScene(
                env.model,
                env.data,
                opt,
                None,
                cam,
                int(mujoco.mjtCatBit.mjCAT_ALL.value),
                scn,
            )
            cam_fwd_xy = (float(scn.camera[0].forward[0]), float(scn.camera[0].forward[1]))

            if state["show_sensors"]:
                append_3d_overlays(scn, env, obs, info, robot_trail, human_trail)

            viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
            mujoco.mjr_render(viewport, scn, ctx)

            # HUD overlays
            _, _, _, v_fwd, steer_c = env._get_robot_pose_and_vel()
            mode_label = "HUMAN GAME CONTROL (WASD/Arrows)" if state["human_mode"] else "AUTO DEMO SHOW"
            if state["tour"] and not state["human_mode"]:
                mode_label += " [TOUR]"
            status_str = (
                "COLLISION! Resetting..."
                if info.get("collision", False)
                else ("FOV LOCKED" if info.get("target_in_fov", False) else "SEARCHING (GRU)")
            )
            if state["paused"]:
                status_str = "PAUSED"

            rec_line = ""
            if recorder.is_recording and recorder.out_path is not None:
                rec_sec = recorder.frame_count / float(recorder.fps)
                rec_line = f"\n[REC {int(rec_sec // 60):02d}:{int(rec_sec % 60):02d}] -> {recorder.out_path.name}"
            elif now < rec_notice_until and rec_notice:
                rec_line = f"\n[{rec_notice}]"

            sprint_tag = (
                " [SPRINT]"
                if state["human_mode"]
                and (
                    state["shift_held"]
                    or glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                    or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
                )
                else ""
            )
            top_left = (
                f"MODE: {mode_label}\n"
                f"Stage {env.stage}/7: {STAGE_NAMES[env.stage]} | Step {env._step_count}/{env.max_steps}\n"
                f"Status: {status_str} | Dist: {info.get('target_dist', 0.0):.2f} m (goal 0.80 m)\n"
                f"AckBot: v={v_fwd:+.2f} m/s | steer={math.degrees(steer_c):+.1f} deg | Return={ep_ret:+.1f}\n"
                f"Human : v={env.target.speed:.2f} m/s{sprint_tag} | Camera: {CAMERA_NAMES[state['cam_preset']]}"
                f"{rec_line}"
            )
            mujoco.mjr_overlay(
                int(mujoco.mjtFont.mjFONT_SHADOW.value),
                int(mujoco.mjtGridPos.mjGRID_TOPLEFT.value),
                viewport,
                top_left,
                "",
                ctx,
            )

            if state["show_help"]:
                help_txt = (
                    "CONTROLS:\n"
                    "  M : Toggle Human Game / Auto Show | X / F9 : Start/Stop Record MP4\n"
                    "  WASD / Arrows : Move Human (Shift=Sprint)\n"
                    "  1..7 / N : Switch Stage (T=Auto Tour)\n"
                    "  C : Camera Preset | Mouse Drag/Wheel: Orbit/Zoom\n"
                    "  V : 3D Sensors | P : Robot Cam PiP | R : Reset | Space : Pause | H : Hide Help"
                )
                mujoco.mjr_overlay(
                    int(mujoco.mjtFont.mjFONT_SHADOW.value),
                    int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value),
                    viewport,
                    help_txt,
                    "",
                    ctx,
                )

            # Robot Front-Camera Picture-in-Picture (bottom-right)
            if state["show_pip"] and fb_w >= 480 and fb_h >= 360:
                pip_w = max(220, int(fb_w * 0.27))
                pip_h = int(pip_w * 0.72)
                margin = 16
                pip_rect = mujoco.MjrRect(fb_w - pip_w - margin, margin, pip_w, pip_h)
                configure_robot_pip_camera(pip_cam, env)
                mujoco.mjv_updateScene(
                    env.model,
                    env.data,
                    opt,
                    None,
                    pip_cam,
                    int(mujoco.mjtCatBit.mjCAT_ALL.value),
                    pip_scn,
                )
                mujoco.mjr_render(pip_rect, pip_scn, ctx)
                mujoco.mjr_overlay(
                    int(mujoco.mjtFont.mjFONT_SHADOW.value),
                    int(mujoco.mjtGridPos.mjGRID_TOPLEFT.value),
                    pip_rect,
                    "ROBOT FRONT CAM (85 deg)",
                    "",
                    ctx,
                )

            if capture_rec_frame and recorder.is_recording:
                if pixel_buf is None or pixel_buf.shape[:2] != (fb_h, fb_w):
                    pixel_buf = np.empty((fb_h, fb_w, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(pixel_buf, None, viewport, ctx)
                recorder.write_frame(pixel_buf, flip_vertical=True)

            glfw.swap_buffers(window)
            time.sleep(0.002)
    finally:
        recorder.stop()
        if ctx is not None:
            ctx.free()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()
        env.close()


def record_3d_demo_video(
    args: argparse.Namespace,
    model,
    device: torch.device,
) -> None:
    """Record a 3D multi-stage showcase MP4 using MuJoCo Renderer + 3D sensor overlays + Robot Cam PiP."""
    from PIL import Image, ImageDraw
    from scripts.record_video import _load_font

    out_path = Path(args.record)
    env = MuJoCoAckermannFollowEnv(sensor_noise=False)
    env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), int(args.width))
    env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), int(args.height))
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    pip_w = max(240, int(args.width * 0.26))
    pip_h = int(pip_w * 0.75)
    pip_renderer = mujoco.Renderer(env.model, height=pip_h, width=pip_w)
    cam = mujoco.MjvCamera()
    pip_cam = mujoco.MjvCamera()
    font_lg = _load_font(20)
    font_sm = _load_font(15)

    proc = _open_ffmpeg_writer(args.width, args.height, 20, out_path)

    total_frames = 0
    try:
        for stg in args.stages:
            obs, info = env.reset(
                seed=args.seed + stg * 100, options={"stage": stg, "force_front_spawn": True}
            )
            hx = _init_agent_state(model, device)
            ep_ret = 0.0
            robot_trail: list[tuple[float, float]] = []
            human_trail: list[tuple[float, float]] = []
            step_idx = 0
            done = False

            while not done and step_idx < args.max_record_steps:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    act_t, hx = _infer_action(model, obs_t, hx)
                    action = act_t.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                step_idx += 1
                done = terminated or truncated

                rx, ry, _, v_fwd, steer_c = env._get_robot_pose_and_vel()
                robot_trail.append((rx, ry))
                human_trail.append((env.target.x, env.target.y))

                # 1. Main 3D Chase View + 3D Overlays
                update_camera_preset(
                    cam=cam,
                    preset_idx=0,
                    env=env,
                    yaw_offset_deg=-22.0,
                    elev_offset_deg=-4.0,
                    dist_scale=1.0,
                    pan_offset=np.zeros(3),
                    sim_time=step_idx * env.cfg.dt,
                )
                renderer.update_scene(env.data, camera=cam)
                append_3d_overlays(renderer.scene, env, obs, info, robot_trail, human_trail)
                main_rgb = renderer.render()

                # 2. Robot Front-Cam PiP
                configure_robot_pip_camera(pip_cam, env)
                pip_renderer.update_scene(env.data, camera=pip_cam)
                pip_rgb = pip_renderer.render()

                pil_img = Image.fromarray(main_rgb)
                margin = 16
                pip_x = args.width - pip_w - margin
                pip_y = args.height - pip_h - margin
                pil_img.paste(Image.fromarray(pip_rgb), (pip_x, pip_y))

                draw = ImageDraw.Draw(pil_img, "RGBA")
                draw.rectangle(
                    [pip_x - 2, pip_y - 24, pip_x + pip_w + 2, pip_y + pip_h + 2],
                    outline=(70, 180, 255, 230),
                    width=2,
                )
                draw.rectangle([pip_x, pip_y - 22, pip_x + pip_w, pip_y], fill=(16, 22, 32, 210))
                draw.text((pip_x + 8, pip_y - 19), "ROBOT FRONT CAM (85°)", fill=(160, 220, 255), font=font_sm)

                # Top-left HUD
                draw.rectangle([14, 14, 560, 108], fill=(16, 22, 32, 200), outline=(70, 88, 110, 255))
                in_fov = bool(info.get("target_in_fov", False))
                coll = bool(info.get("collision", False))
                st_col = (231, 76, 60) if coll else ((46, 204, 113) if in_fov else (241, 196, 15))
                st_txt = "COLLISION" if coll else ("FOV LOCKED" if in_fov else "SEARCHING (GRU)")
                draw.text(
                    (26, 22),
                    f"STAGE {stg}/7: {STAGE_NAMES[stg]} | Step {step_idx}",
                    fill=(242, 246, 250),
                    font=font_lg,
                )
                draw.text(
                    (26, 50),
                    f"Dist: {info.get('target_dist', 0.0):.2f} m | v={v_fwd:+.2f} m/s | steer={math.degrees(steer_c):+.1f} deg",
                    fill=(200, 212, 226),
                    font=font_sm,
                )
                draw.text(
                    (26, 74),
                    f"Status: {st_txt} | Return: {ep_ret:+.1f}",
                    fill=st_col,
                    font=font_sm,
                )

                proc.stdin.write(np.asarray(pil_img, dtype=np.uint8).tobytes())
                total_frames += 1

            print(f"Recorded 3D Stage {stg}: {step_idx} frames, return={ep_ret:+.1f}")

        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        retcode = proc.wait()
        if retcode != 0:
            raise RuntimeError(f"ffmpeg failed (exit {retcode}): {stderr}")
    finally:
        renderer.close()
        pip_renderer.close()
        env.close()

    size_kb = out_path.stat().st_size / 1024.0
    print(f"Saved {total_frames} 3D demo frames to {out_path} ({size_kb:.1f} KB)")


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint is not None else find_default_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            "No trained checkpoint found. Pass --checkpoint <path/to/best_model.pt>."
        )

    device = select_device(args.device)
    model, ckpt = load_policy(ckpt_path, device=device)
    print(
        f"Loaded policy: {ckpt_path} "
        f"(step={ckpt.get('step', '?')}, unlocked_stage={ckpt.get('unlocked_stage', '?')}, device={device})"
    )

    if args.record is not None:
        record_3d_demo_video(args, model, device)
        return

    if args.mode == "choose" and sys.stdin.isatty():
        mode, start_stage, tour_default = prompt_startup_choice(args.stage)
    else:
        mode = "human" if args.mode == "human" else "auto"
        start_stage = args.stage if args.stage is not None else 1
        tour_default = (mode == "auto") and (args.stage is None)

    tour_enabled = args.tour if args.tour is not None else tour_default
    run_interactive_3d(
        args=args,
        model=model,
        device=device,
        mode=mode,
        start_stage=start_stage,
        tour_enabled=tour_enabled,
    )


if __name__ == "__main__":
    main()
