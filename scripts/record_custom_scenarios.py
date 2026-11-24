#!/usr/bin/env python3
"""
Record custom scenario showcase videos for AckBot:
  1. Open start -> L-turn -> dead end -> 180-deg return to initial position
  2. Far car + human right at L-wall corner about to hide behind the wall
  3a. Dead-end U-turn: human hugs right wall, curves left across dead end, passes car
  3b. Dead-end U-turn: human hugs left wall, curves right across dead end, passes car
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import ImageDraw

import ackerman_follow.envs  # noqa: F401
from ackerman_follow.config import select_device
from ackerman_follow.envs.mujoco_follow_env import MuJoCoAckermanFollowEnv, _yaw_to_quat
from scripts.eval_policy import load_policy
from scripts.record_video import TopDownRenderer, _load_font


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record custom AckBot scenario videos")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/experiments/exp_20261124_143534_vm_rgbd_mask_stage5_finetune/checkpoints/best_model.pt",
        help="Path to checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/video/ackbot_custom_scenarios.mp4",
        help="Output MP4 file path",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu", "cuda"])
    return parser.parse_args()


def _set_custom_pose(
    env: MuJoCoAckermanFollowEnv,
    robot_pose: tuple[float, float, float],
    human_pose: tuple[float, float, float, float],
    waypoints: list[tuple[float, float]],
    max_steps: int,
    adaptive_pacing: bool = False,
) -> np.ndarray:
    """Configure deterministic custom robot/human poses and waypoints on a reset env."""
    import mujoco

    rx, ry, ryaw = robot_pose
    hx, hy, hyaw, hspeed = human_pose

    env.max_steps = int(max_steps)
    env._adaptive_pacing = bool(adaptive_pacing)
    env._allow_pauses = False
    env._allow_reversals = False
    env._wp_dir = 1
    env._wp_loop = True
    env._speed_mult = 1.0
    env._behavior_timer = int(max_steps) + 100
    env.data.qpos[env._robot_free_qpos : env._robot_free_qpos + 3] = [rx, ry, 0.0]
    env.data.qpos[env._robot_free_qpos + 3 : env._robot_free_qpos + 7] = _yaw_to_quat(ryaw)

    env._waypoints = list(waypoints)
    env._wp_idx = 0
    env._base_target_speed = float(hspeed)
    env.target.reset(x=hx, y=hy, yaw=hyaw, speed=float(hspeed), rng=env.np_random)
    env.model.geom_rgba[env._shirt_geom_id, :3] = env.target.color_rgb
    env._sync_human_mocap()

    mujoco.mj_forward(env.model, env.data)
    obs, min_lidar = env._get_obs()
    env._last_min_lidar = min_lidar
    dist, bearing, _ = env._compute_target_metrics()
    env._prev_dist_err = abs(dist - env.cfg.target_follow_distance)
    env._prev_abs_bearing = abs(bearing)
    return obs


def _run_and_write_clip(
    env: MuJoCoAckermanFollowEnv,
    model,
    device: torch.device,
    renderer: TopDownRenderer,
    title: str,
    obs: np.ndarray,
    proc_stdin,
    font_title,
) -> tuple[int, float, bool]:
    hx = model.init_hidden(1, device=device)
    ep_ret = 0.0
    step_idx = 0
    done = False
    robot_trail: list[tuple[float, float]] = []
    human_trail: list[tuple[float, float]] = []
    info: dict = {}

    while not done:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            act_t, hx = model.get_deterministic_action(obs_t, hx=hx, return_hidden=True)
            action = act_t.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += float(reward)
        step_idx += 1
        done = terminated or truncated

        rx, ry, _, _, _ = env._get_robot_pose_and_vel()
        robot_trail.append((rx, ry))
        human_trail.append((env.target.x, env.target.y))

        frame = renderer.render_frame(
            env, obs, info, step_idx, ep_ret, action, robot_trail, human_trail
        )
        # Overlay custom scenario title banner in top-right
        from PIL import Image

        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img, "RGBA")
        draw.rectangle([552, 12, renderer.width - 12, 54], fill=(18, 24, 34, 210), outline=(70, 85, 105, 255))
        draw.text((564, 24), title, fill=(255, 220, 110), font=font_title)
        proc_stdin.write(np.asarray(pil_img, dtype=np.uint8).tobytes())

    return step_idx, ep_ret, bool(info.get("collision", False))


def main() -> None:
    args = parse_args()
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found in PATH")

    device = select_device(args.device)
    model, _ = load_policy(args.checkpoint, device=device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = TopDownRenderer(width=args.width, height=args.height)
    font_title = _load_font(15)

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
        f"{args.width}x{args.height}",
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

    env_std = MuJoCoAckermanFollowEnv()
    try:
        # 1. Open -> L-Turn -> Dead End -> 180 back to initial position (native Stage 7 geometry)
        env_std.reset(seed=101, options={"stage": 7, "force_front_spawn": True})
        obs1 = _set_custom_pose(
            env_std,
            robot_pose=(0.0, 0.0, 0.0),
            human_pose=(1.7, 0.0, 0.0, 0.36),
            waypoints=[
                (4.45, 0.00),
                (4.45, -3.55),
                (4.90, -3.65),
                (4.90, -0.35),
                (4.25, 0.35),
                (1.50, 0.35),
                (0.0, 0.0),
            ],
            max_steps=950,
            adaptive_pacing=True,
        )
        s1, r1, c1 = _run_and_write_clip(
            env_std, model, device, renderer,
            "1: Open -> L-Turn -> Dead-End -> 180 Back",
            obs1, proc.stdin, font_title,
        )
        print(f"Cond 1: steps={s1}, return={r1:+.2f}, collision={c1}")

        # 2. Far car + human right at L-turn corner about to hide beside the wall (native Stage 3)
        env_std.reset(seed=202, options={"stage": 3, "force_front_spawn": True})
        obs2 = _set_custom_pose(
            env_std,
            robot_pose=(0.0, 1.15, 0.0),
            human_pose=(4.15, 1.05, -0.10, 0.42),
            waypoints=[
                (5.25, 1.05),
                (5.25, -2.85),
                (1.90, -2.85),
                (1.90, 1.15),
            ],
            max_steps=650,
            adaptive_pacing=False,
        )
        s2, r2, c2 = _run_and_write_clip(
            env_std, model, device, renderer,
            "2: Far Car -> Target Hides Around L-Wall",
            obs2, proc.stdin, font_title,
        )
        print(f"Cond 2: steps={s2}, return={r2:+.2f}, collision={c2}")

        # 3a. Dead-end U-turn: walk slightly to RIGHT wall, turn LEFT inside dead end
        env_std.reset(seed=303, options={"stage": 5, "force_front_spawn": True})
        obs3a = _set_custom_pose(
            env_std,
            robot_pose=(0.0, 0.0, 0.0),
            human_pose=(1.7, -0.25, -0.15, 0.36),
            waypoints=[
                (3.85, -0.75),
                (4.20, 0.00),
                (3.85, 0.72),
                (1.60, 0.72),
                (-2.20, 0.45),
                (-4.50, -1.20),
            ],
            max_steps=700,
            adaptive_pacing=True,
        )
        s3a, r3a, c3a = _run_and_write_clip(
            env_std, model, device, renderer,
            "3A: Dead-End Right Wall -> Turn Left",
            obs3a, proc.stdin, font_title,
        )
        print(f"Cond 3A: steps={s3a}, return={r3a:+.2f}, collision={c3a}")

        # 3b. Dead-end U-turn: walk slightly to LEFT wall, turn RIGHT inside dead end
        env_std.reset(seed=404, options={"stage": 5, "force_front_spawn": True})
        obs3b = _set_custom_pose(
            env_std,
            robot_pose=(0.0, 0.0, 0.0),
            human_pose=(1.7, 0.25, 0.15, 0.36),
            waypoints=[
                (3.85, 0.75),
                (4.20, 0.00),
                (3.85, -0.72),
                (1.60, -0.72),
                (-2.20, -0.45),
                (-4.50, 1.20),
            ],
            max_steps=700,
            adaptive_pacing=True,
        )
        s3b, r3b, c3b = _run_and_write_clip(
            env_std, model, device, renderer,
            "3B: Dead-End Left Wall -> Turn Right",
            obs3b, proc.stdin, font_title,
        )
        print(f"Cond 3B: steps={s3b}, return={r3b:+.2f}, collision={c3b}")

        # 4. Dead-end U-turn: human walks straight down the middle (native Stage 6)
        env_std.reset(seed=505, options={"stage": 6, "force_front_spawn": True})
        obs4 = _set_custom_pose(
            env_std,
            robot_pose=(0.0, 0.0, 0.0),
            human_pose=(1.7, 0.0, 0.0, 0.36),
            waypoints=[
                (3.85, 0.00),
                (4.20, 0.00),
                (1.60, 0.10),
                (-2.20, 0.00),
                (-4.50, -1.20),
            ],
            max_steps=750,
            adaptive_pacing=True,
        )
        s4, r4, c4 = _run_and_write_clip(
            env_std, model, device, renderer,
            "4: Dead-End Middle Walk -> 180 Return",
            obs4, proc.stdin, font_title,
        )
        print(f"Cond 4: steps={s4}, return={r4:+.2f}, collision={c4}")
    finally:
        env_std.close()

    proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
    retcode = proc.wait()
    if retcode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {retcode}): {stderr}")

    size_kb = out_path.stat().st_size / 1024.0
    print(f"Saved custom scenarios video to {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
