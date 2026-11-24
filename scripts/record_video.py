#!/usr/bin/env python3
"""
Record MP4 Evaluation Videos for AckBot Recurrent PPO Checkpoints.

Uses MuJoCo 3D `rgb_array` rendering when OpenGL is available (e.g., MacBook Pro M4 Pro)
and automatically falls back to a crisp 2D top-down MuJoCo physics + sensor-ray renderer
on headless Linux VMs without OpenGL.

Example:
    uv run python scripts/record_video.py \
        --checkpoint outputs/experiments/.../checkpoints/best_model.pt \
        --stages 1 2 \
        --output outputs/video/ackbot_stage1_stage2.mp4
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import ackerman_follow.envs  # noqa: F401
from ackerman_follow.config import select_device
from scripts.eval_policy import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record MP4 evaluation video of AckBot policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2],
        choices=[1, 2, 3, 4, 5],
        help="Stages to record sequentially into the output video (1..5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/video/ackbot_eval.mp4",
        help="Output MP4 file path",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "mujoco", "topdown"],
        help="Rendering backend (auto falls back to topdown when OpenGL is unavailable)",
    )
    parser.add_argument("--width", type=int, default=960, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS (20 Hz matches env dt)")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
    )
    return parser.parse_args()


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class TopDownRenderer:
    """Crisp 2D top-down renderer reading live MuJoCo geom poses and raycasts."""

    def __init__(self, width: int = 960, height: int = 720) -> None:
        self.width = int(width)
        self.height = int(height)
        self.font_lg = _load_font(18)
        self.font_sm = _load_font(14)

    def render_frame(
        self,
        env_unwrapped,
        obs: np.ndarray,
        info: dict,
        step_idx: int,
        ep_ret: float,
        action: np.ndarray,
        robot_trail: list[tuple[float, float]],
        human_trail: list[tuple[float, float]],
    ) -> np.ndarray:
        rx, ry, ryaw, v_fwd, steer_c = env_unwrapped._get_robot_pose_and_vel()
        hx, hy = env_unwrapped.target.x, env_unwrapped.target.y

        # Smooth camera follow centered mid-way between robot and target (fixed 10m span)
        cx = 0.55 * rx + 0.45 * hx
        cy = 0.55 * ry + 0.45 * hy
        scale = min(self.width, self.height) / 10.0  # ~10 m visible span

        def to_px(wx: float, wy: float) -> tuple[float, float]:
            px = self.width * 0.5 + (wx - cx) * scale
            py = self.height * 0.5 - (wy - cy) * scale
            return px, py

        img = Image.new("RGB", (self.width, self.height), (242, 245, 249))
        draw = ImageDraw.Draw(img, "RGBA")

        # 1. World 1-meter grid
        x_min, x_max = int(math.floor(cx - 7)), int(math.ceil(cx + 7))
        y_min, y_max = int(math.floor(cy - 7)), int(math.ceil(cy + 7))
        for gx in range(x_min, x_max + 1):
            p0 = to_px(gx, y_min)
            p1 = to_px(gx, y_max)
            draw.line([p0, p1], fill=(224, 229, 236, 255), width=1)
        for gy in range(y_min, y_max + 1):
            p0 = to_px(x_min, gy)
            p1 = to_px(x_max, gy)
            draw.line([p0, p1], fill=(224, 229, 236, 255), width=1)

        # 2. Active group-0 arena walls & stage obstacles (skip human geoms)
        m = env_unwrapped.model
        d = env_unwrapped.data
        human_gids = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "human_leg_l_geom"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "human_leg_r_geom"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "human_torso_geom"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "human_head_geom"),
        }
        for gid in range(m.ngeom):
            if m.geom_group[gid] != 0 or gid in human_gids:
                continue
            gtype = int(m.geom_type[gid])
            gx, gy = float(d.geom_xpos[gid, 0]), float(d.geom_xpos[gid, 1])
            rgba = m.geom_rgba[gid]
            fill_col = (
                int(rgba[0] * 255),
                int(rgba[1] * 255),
                int(rgba[2] * 255),
                230,
            )
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                sx, sy = float(m.geom_size[gid, 0]), float(m.geom_size[gid, 1])
                p0 = to_px(gx - sx, gy + sy)
                p1 = to_px(gx + sx, gy - sy)
                draw.rectangle([p0[0], p0[1], p1[0], p1[1]], fill=fill_col, outline=(40, 46, 56))
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                rad = float(m.geom_size[gid, 0]) * scale
                px, py = to_px(gx, gy)
                draw.ellipse(
                    [px - rad, py - rad, px + rad, py + rad], fill=fill_col, outline=(40, 46, 56)
                )

        # 3. LiDAR & Camera rays + RGB target mask from current observation
        cfg = env_unwrapped.cfg
        cam_end = cfg.cam_samples
        mask_end = cam_end + cfg.cam_mask_samples
        lidar_end = mask_end + cfg.lidar_samples

        cam_dists = cfg.denormalize_cam(obs[:cam_end])
        cam_mask = obs[cam_end:mask_end]
        lidar_dists = cfg.denormalize_lidar(obs[mask_end:lidar_end])

        lx_w = rx - 0.15 * math.cos(ryaw)
        ly_w = ry - 0.15 * math.sin(ryaw)
        lp = to_px(lx_w, ly_w)
        for i in range(0, cfg.lidar_samples, 3):
            ang = ryaw + i * (2.0 * math.pi / cfg.lidar_samples)
            dist_i = float(lidar_dists[i])
            ex, ey = to_px(lx_w + dist_i * math.cos(ang), ly_w + dist_i * math.sin(ang))
            col = (240, 120, 40, 110) if dist_i < cfg.lidar_range_max - 0.05 else (140, 180, 230, 45)
            draw.line([lp, (ex, ey)], fill=col, width=1)

        cx_w = rx + 0.15 * math.cos(ryaw)
        cy_w = ry + 0.15 * math.sin(ryaw)
        cp = to_px(cx_w, cy_w)
        cam_poly = [cp]
        cam_angles = np.linspace(-0.5 * cfg.cam_fov_rad, 0.5 * cfg.cam_fov_rad, cfg.cam_samples)
        target_ray_endpoints = []
        for ang_local, dist_i, is_target in zip(cam_angles, cam_dists, cam_mask):
            ang = ryaw + float(ang_local)
            ep = to_px(cx_w + float(dist_i) * math.cos(ang), cy_w + float(dist_i) * math.sin(ang))
            cam_poly.append(ep)
            if is_target > 0.5:
                target_ray_endpoints.append(ep)
        draw.polygon(cam_poly, fill=(46, 204, 113, 40), outline=(39, 174, 96, 130))
        for ep in target_ray_endpoints:
            draw.line([cp, ep], fill=(39, 220, 100, 180), width=2)

        # 4. Trails
        if len(human_trail) > 1:
            draw.line([to_px(x, y) for x, y in human_trail[-150:]], fill=(231, 76, 60, 150), width=2)
        if len(robot_trail) > 1:
            draw.line([to_px(x, y) for x, y in robot_trail[-150:]], fill=(41, 128, 185, 180), width=3)

        # 5. Human target (goal ring + torso + alternating legs)
        goal_r = cfg.target_follow_distance * scale
        hp = to_px(hx, hy)
        draw.ellipse(
            [hp[0] - goal_r, hp[1] - goal_r, hp[0] + goal_r, hp[1] + goal_r],
            outline=(46, 204, 113, 110),
            width=2,
        )
        left_leg, right_leg = env_unwrapped.target.get_feet_positions()
        leg_r = max(4.0, env_unwrapped.target.leg_radius * scale)
        for leg in (left_leg, right_leg):
            lp_px = to_px(leg.x, leg.y)
            draw.ellipse(
                [lp_px[0] - leg_r, lp_px[1] - leg_r, lp_px[0] + leg_r, lp_px[1] + leg_r],
                fill=(45, 52, 64, 230),
                outline=(255, 255, 255, 200),
            )
        torso_r = max(6.0, env_unwrapped.target.torso_radius * scale)
        shirt_col = tuple(int(c * 255) for c in env_unwrapped.target.color_rgb) + (240,)
        draw.ellipse(
            [hp[0] - torso_r, hp[1] - torso_r, hp[0] + torso_r, hp[1] + torso_r],
            fill=shirt_col,
            outline=(20, 20, 20, 255),
            width=2,
        )

        # 6. AckBot chassis + 4 wheels
        cos_r, sin_r = math.cos(ryaw), math.sin(ryaw)

        def rot_pt(lx: float, ly: float) -> tuple[float, float]:
            return to_px(rx + lx * cos_r - ly * sin_r, ry + lx * sin_r + ly * cos_r)

        chassis_poly = [
            rot_pt(0.15, 0.10),
            rot_pt(0.15, -0.10),
            rot_pt(-0.15, -0.10),
            rot_pt(-0.15, 0.10),
        ]
        draw.polygon(chassis_poly, fill=(32, 42, 58, 245), outline=(255, 255, 255, 240))

        fl_steer = float(d.qpos[env_unwrapped._fl_steer_qpos])
        fr_steer = float(d.qpos[env_unwrapped._fr_steer_qpos])
        wheels = [
            (0.15, 0.15, fl_steer),
            (0.15, -0.15, fr_steer),
            (-0.15, 0.15, 0.0),
            (-0.15, -0.15, 0.0),
        ]
        for wx_l, wy_l, w_ang in wheels:
            c_w, s_w = math.cos(w_ang), math.sin(w_ang)
            pts = []
            for dx, dy in ((0.06, 0.03), (0.06, -0.03), (-0.06, -0.03), (-0.06, 0.03)):
                pts.append(rot_pt(wx_l + dx * c_w - dy * s_w, wy_l + dx * s_w + dy * c_w))
            draw.polygon(pts, fill=(20, 20, 24, 255), outline=(180, 190, 205, 255))

        # Heading arrow on chassis
        draw.line([rot_pt(0.0, 0.0), rot_pt(0.22, 0.0)], fill=(52, 152, 219, 255), width=3)

        # 7. HUD Overlay
        draw.rectangle([12, 12, 540, 108], fill=(18, 24, 34, 205), outline=(70, 85, 105, 255))
        dist = float(info.get("target_dist", 0.0))
        in_fov = bool(info.get("target_in_fov", False))
        coll = bool(info.get("collision", False))
        stg = int(info.get("stage", env_unwrapped.stage))

        status_col = (231, 76, 60) if coll else ((46, 204, 113) if in_fov else (241, 196, 15))
        status_txt = "COLLISION!" if coll else ("FOV LOCKED" if in_fov else "SEARCHING (GRU)")

        draw.text(
            (24, 20),
            f"STAGE {stg} | Step {step_idx:3d}/{env_unwrapped.max_steps} | Return: {ep_ret:+7.1f}",
            fill=(240, 244, 248),
            font=self.font_lg,
        )
        draw.text(
            (24, 48),
            f"Dist: {dist:4.2f} m (goal 0.80 m) | v={v_fwd:+.2f} m/s | steer={math.degrees(steer_c):+5.1f} deg",
            fill=(200, 210, 222),
            font=self.font_sm,
        )
        draw.text(
            (24, 74),
            f"Status: {status_txt} | Cmd: v={float(action[0]):+.2f}, steer={math.degrees(float(action[1])):+5.1f} deg",
            fill=status_col,
            font=self.font_sm,
        )

        return np.asarray(img, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg binary not found in PATH.")

    device = select_device(args.device)
    model, ckpt = load_policy(args.checkpoint, device=device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend = args.backend
    if backend == "auto":
        try:
            test_env = gym.make("AckermanFollow-v0", render_mode="rgb_array")
            test_env.reset(seed=args.seed)
            _ = test_env.render()
            test_env.close()
            backend = "mujoco"
        except Exception:
            backend = "topdown"

    render_mode = "rgb_array" if backend == "mujoco" else None
    env = gym.make("AckermanFollow-v0", render_mode=render_mode)
    topdown = TopDownRenderer(width=args.width, height=args.height) if backend == "topdown" else None

    frame_w, frame_h = (640, 480) if backend == "mujoco" else (args.width, args.height)
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
        f"{frame_w}x{frame_h}",
        "-r",
        str(args.fps),
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

    total_frames = 0
    try:
        for stg in args.stages:
            obs, info = env.reset(seed=args.seed + stg * 100, options={"stage": stg})
            hx = model.init_hidden(1, device=device)
            ep_ret = 0.0
            step_idx = 0
            done = False
            robot_trail: list[tuple[float, float]] = []
            human_trail: list[tuple[float, float]] = []

            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    act_t, hx = model.get_deterministic_action(obs_t, hx=hx, return_hidden=True)
                    action = act_t.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                step_idx += 1
                done = terminated or truncated

                rx, ry, _, _, _ = env.unwrapped._get_robot_pose_and_vel()
                robot_trail.append((rx, ry))
                human_trail.append((env.unwrapped.target.x, env.unwrapped.target.y))

                if backend == "mujoco":
                    frame = env.render()
                else:
                    frame = topdown.render_frame(
                        env.unwrapped, obs, info, step_idx, ep_ret, action, robot_trail, human_trail
                    )

                proc.stdin.write(frame.tobytes())
                total_frames += 1

            print(
                f"Recorded Stage {stg}: steps={step_idx}, return={ep_ret:+.2f}, "
                f"collision={info.get('collision', False)}"
            )

        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        retcode = proc.wait()
        if retcode != 0:
            raise RuntimeError(f"ffmpeg failed (exit {retcode}): {stderr}")
    finally:
        env.close()

    size_kb = out_path.stat().st_size / 1024.0
    print(f"Saved {total_frames} frames ({backend}) to {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
