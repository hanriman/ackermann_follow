#!/usr/bin/env python3
"""
Backward-compatible Recurrent PPO Training Entrypoint.
Delegates to the unified modular training engine (`ackerman_follow.engine.run_training`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ackerman_follow.engine.evaluator import evaluate_agent
from ackerman_follow.engine.runner import run_training
from ackerman_follow.engine.vec_env import _EnvFactory, _VectorSafeInfoWrapper  # noqa: F401

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "ppo_follow.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="AckBot Recurrent PPO Training Pipeline")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to config yaml")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel environments")
    parser.add_argument("--exp-name", type=str, default="ppo_follow", help="Experiment name tag")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--vec-env", type=str, default=None, choices=["sync", "async"])
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum", action="store_true", default=True)
    parser.add_argument("--no-curriculum", action="store_false", dest="curriculum")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--sensor-noise", action="store_true", default=True)
    parser.add_argument("--no-sensor-noise", action="store_false", dest="sensor_noise")
    parser.add_argument("--warm-start", type=str, default=None)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_policy(model, env, n_episodes_per_stage=3, stages=(1, 2, 3, 4), device="cpu"):
    return evaluate_agent(model, env, n_episodes_per_stage=n_episodes_per_stage, stages=stages, device=device)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_training(
        algo_name="gru_ppo",
        cfg=cfg,
        exp_name=args.exp_name,
        total_steps_override=args.total_steps,
        num_envs_override=args.num_envs,
        device_override=args.device,
        vec_env_override=args.vec_env,
        torch_threads_override=args.torch_threads,
        seed=args.seed,
        curriculum_flag=args.curriculum,
        start_stage=args.start_stage,
        sensor_noise=args.sensor_noise,
        warm_start=args.warm_start,
    )


if __name__ == "__main__":
    main()
