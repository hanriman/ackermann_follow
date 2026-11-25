#!/usr/bin/env python3
"""
Deterministic Evaluation & Interactive MuJoCo Viewer for ANY AckBot Checkpoint
(`gru_ppo`, `lewm`, `lewm_rl_modelfree`, `lewm_tdmpc2`).

Headless benchmark across stages:
    uv run python scripts/eval_policy.py --checkpoint outputs/experiments/.../checkpoints/best_model.pt

Interactive 3D viewer (use `mjpython` on macOS Apple Silicon):
    uv run mjpython scripts/eval_policy.py --checkpoint outputs/experiments/.../checkpoints/best_model.pt --stage 2 --render
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import ackermann_follow.envs  # noqa: F401
from ackermann_follow.algos.base import BaseAgent
from ackermann_follow.algos.registry import load_agent
from ackermann_follow.config import select_device
from ackermann_follow.engine.evaluator import evaluate_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate any trained AckBot policy/world-model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Evaluate a single stage (1..7); omit to evaluate stages 1..4",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of deterministic episodes per stage")
    parser.add_argument("--render", action="store_true", help="Open MuJoCo interactive viewer")
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--no-realtime", action="store_false", dest="realtime")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1000, help="Base evaluation seed")
    return parser.parse_args()


def load_policy(checkpoint_path: str | Path, device: torch.device) -> tuple[BaseAgent, dict]:
    """Load any registered algorithm checkpoint (including legacy ActorCritic)."""
    return load_agent(checkpoint_path, device=device)


def run_evaluation(
    model,
    stages: tuple[int, ...],
    n_episodes: int,
    device: torch.device,
    seed: int = 1000,
    render: bool = False,
    realtime: bool = True,
) -> dict[int, dict[str, float]]:
    """Run deterministic policy rollouts across the requested curriculum stages."""
    render_mode = "human" if render else None
    env = gym.make("AckermannFollow-v0", render_mode=render_mode)
    try:
        metrics = evaluate_agent(
            agent=model,
            env=env,
            n_episodes_per_stage=n_episodes,
            stages=stages,
            device=device,
            seed=seed,
            render=render,
            realtime=realtime,
        )
        results: dict[int, dict[str, float]] = metrics["stage_breakdown"]
        for stg in stages:
            s = results[stg]
            print(
                f"Stage {stg} | Return: {s['mean_reward']:+7.2f} ± {s['std_reward']:5.2f} | "
                f"Len: {s['mean_length']:5.1f} | Dist: {s['mean_dist']:4.2f} m | "
                f"FOV: {s['fov_rate'] * 100:5.1f}% | Coll: {s['collision_rate'] * 100:5.1f}%"
            )
        return results
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, ckpt = load_policy(args.checkpoint, device=device)

    step = ckpt.get("step", "?")
    unlocked = ckpt.get("unlocked_stage", "?")
    algo = ckpt.get("algo", getattr(model, "algo_name", "gru_ppo"))
    print(
        f"Loaded checkpoint: {args.checkpoint} "
        f"(algo={algo}, step={step}, unlocked_stage={unlocked}, device={device})"
    )

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
            f"Composite (S1..S{max(stages)}) | Return: {comp_rew:+7.2f} | "
            f"FOV: {comp_fov * 100:5.1f}% | Coll: {comp_col * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()
