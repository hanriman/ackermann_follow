#!/usr/bin/env python3
"""
Deterministic Evaluation & Interactive MuJoCo Viewer for AckBot Recurrent PPO Checkpoints.

Headless benchmark across stages:
    uv run python scripts/eval_policy.py --checkpoint outputs/experiments/.../checkpoints/best_model.pt

Interactive 3D viewer (use `mjpython` on macOS Apple Silicon):
    uv run mjpython scripts/eval_policy.py --checkpoint outputs/experiments/.../checkpoints/best_model.pt --stage 2 --render
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import ackerman_follow.envs  # noqa: F401
from ackerman_follow.config import get_robot_config, select_device
from ackerman_follow.ppo.actor_critic import ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained AckBot Recurrent PPO policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pt) containing model_state_dict",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Evaluate a single stage (1..4); omit to evaluate all 4 stages",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of deterministic episodes per stage",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open MuJoCo interactive viewer (use `uv run mjpython` on macOS)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        default=True,
        help="Pace interactive viewer at real-time 20 Hz (enabled by default with --render)",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_false",
        dest="realtime",
        help="Run viewer as fast as possible without 20 Hz sleep",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Compute device",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Base evaluation seed")
    return parser.parse_args()


def load_policy(checkpoint_path: str | Path, device: torch.device) -> tuple[ActorCritic, dict]:
    """Load ActorCritic weights from a training checkpoint."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    robot_cfg = get_robot_config()
    model = ActorCritic(
        obs_dim=robot_cfg.obs_dim,
        act_dim=robot_cfg.act_dim,
        robot_cfg=robot_cfg,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


def run_evaluation(
    model: ActorCritic,
    stages: tuple[int, ...],
    n_episodes: int,
    device: torch.device,
    seed: int = 1000,
    render: bool = False,
    realtime: bool = True,
) -> dict[int, dict[str, float]]:
    """Run deterministic policy rollouts across the requested curriculum stages."""
    render_mode = "human" if render else None
    env = gym.make("AckermanFollow-v0", render_mode=render_mode)
    dt = env.unwrapped.cfg.dt
    results: dict[int, dict[str, float]] = {}

    try:
        for stg in stages:
            rewards, lengths, mean_dists, fov_rates, collisions = [], [], [], [], []

            for ep_idx in range(n_episodes):
                ep_seed = seed + stg * 100 + ep_idx
                obs, _ = env.reset(seed=ep_seed, options={"stage": stg})
                hx = model.init_hidden(1, device=device)

                ep_ret = 0.0
                ep_len = 0
                ep_dists = []
                fov_steps = 0
                done = False
                collided = False

                while not done:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        act_t, hx = model.get_deterministic_action(obs_t, hx=hx, return_hidden=True)
                        action = act_t.squeeze(0).cpu().numpy()

                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_ret += float(reward)
                    ep_len += 1
                    ep_dists.append(float(info.get("target_dist", 0.0)))
                    if info.get("target_in_fov", False):
                        fov_steps += 1
                    if terminated:
                        collided = True
                    done = terminated or truncated

                    if render:
                        viewer = getattr(env.unwrapped, "_viewer", None)
                        if viewer is not None and not viewer.is_running():
                            done = True
                        if realtime:
                            elapsed = time.perf_counter() - t0
                            if elapsed < dt:
                                time.sleep(dt - elapsed)

                rewards.append(ep_ret)
                lengths.append(ep_len)
                mean_dists.append(float(np.mean(ep_dists)) if ep_dists else 0.0)
                fov_rates.append(float(fov_steps / max(1, ep_len)))
                collisions.append(1.0 if collided else 0.0)

            summary = {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_length": float(np.mean(lengths)),
                "mean_dist": float(np.mean(mean_dists)),
                "fov_rate": float(np.mean(fov_rates)),
                "collision_rate": float(np.mean(collisions)),
            }
            results[stg] = summary
            print(
                f"Stage {stg} | Return: {summary['mean_reward']:+7.2f} ± {summary['std_reward']:5.2f} | "
                f"Len: {summary['mean_length']:5.1f} | Dist: {summary['mean_dist']:4.2f} m | "
                f"FOV: {summary['fov_rate']*100:5.1f}% | Coll: {summary['collision_rate']*100:5.1f}%"
            )
    finally:
        env.close()

    return results


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, ckpt = load_policy(args.checkpoint, device=device)

    step = ckpt.get("step", "?")
    unlocked = ckpt.get("unlocked_stage", "?")
    print(f"Loaded checkpoint: {args.checkpoint} (step={step}, unlocked_stage={unlocked}, device={device})")

    stages = (args.stage,) if args.stage is not None else (1, 2, 3, 4)
    results = run_evaluation(
        model=model,
        stages=stages,
        n_episodes=args.episodes,
        device=device,
        seed=args.seed,
        render=args.render,
        realtime=args.realtime,
    )

    if len(results) > 1:
        comp_rew = float(np.mean([r["mean_reward"] for r in results.values()]))
        comp_fov = float(np.mean([r["fov_rate"] for r in results.values()]))
        comp_col = float(np.mean([r["collision_rate"] for r in results.values()]))
        print("-" * 72)
        print(
            f"Composite (S1..S4) | Return: {comp_rew:+7.2f} | "
            f"FOV: {comp_fov*100:5.1f}% | Coll: {comp_col*100:5.1f}%"
        )


if __name__ == "__main__":
    main()
