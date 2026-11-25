#!/usr/bin/env python3
"""
Unified Multi-Algorithm Training Entrypoint for AckBot.

Supports:
  - gru_ppo           : Recurrent Spatial PPO (on-policy BPTT)
  - lewm              : LeWorldModel (JEPA + SIGReg + MPPI planning)
  - lewm_rl_modelfree : LeWM + Model-Free RL (PPO or SAC)
  - lewm_tdmpc2       : LeWM + TD-MPC2 (JEPA dynamics + Two-Hot Q/Reward + MPPI)

Examples:
  uv run python scripts/train.py --algo gru_ppo --exp-name ppo_run
  uv run python scripts/train.py --algo lewm --exp-name lewm_wm
  uv run python scripts/train.py --algo lewm_rl_modelfree --exp-name lewm_ppo
  uv run python scripts/train.py --algo lewm_tdmpc2 --exp-name tdmpc2_run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from ackermann_follow.algos.registry import ALGO_REGISTRY
from ackermann_follow.engine.runner import run_training

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CURRICULUM_CFG = CONFIGS_DIR / "curriculum.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified AckBot Multi-Algorithm Trainer")
    parser.add_argument(
        "--algo",
        type=str,
        default="gru_ppo",
        choices=sorted(ALGO_REGISTRY.keys()),
        help="Algorithm to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to algorithm YAML config (defaults to configs/algos/<algo>.yaml)",
    )
    parser.add_argument(
        "--curriculum-config",
        type=str,
        default=str(DEFAULT_CURRICULUM_CFG),
        help="Path to shared curriculum YAML config",
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment tag")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total environment steps")
    parser.add_argument("--num-envs", type=int, default=None, help="Override parallel env count")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "mps", "cpu", "cuda"],
        help="Compute device override",
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        default=None,
        choices=["sync", "async"],
        help="Vectorized env mode override",
    )
    parser.add_argument("--torch-threads", type=int, default=None, help="PyTorch CPU threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--curriculum", action="store_true", default=True, help="Enable curriculum")
    parser.add_argument("--no-curriculum", action="store_false", dest="curriculum", help="Disable curriculum")
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Initial curriculum stage (1..7)",
    )
    parser.add_argument("--sensor-noise", action="store_true", default=True, help="Enable sensory noise")
    parser.add_argument("--no-sensor-noise", action="store_false", dest="sensor_noise", help="Disable sensory noise")
    parser.add_argument("--warm-start", type=str, default=None, help="Full agent checkpoint to resume from")
    parser.add_argument("--warm-start-wm", type=str, default=None, help="Pretrained LeWM checkpoint for `wm.*` weights")
    parser.add_argument("--dataset", type=str, default=None, help="Offline `.npz` replay dataset for `lewm` pretraining")
    return parser.parse_args()


def load_combined_config(algo: str, algo_config_path: str | None, curriculum_path: str | None) -> dict[str, Any]:
    cfg_path = Path(algo_config_path) if algo_config_path else CONFIGS_DIR / "algos" / f"{algo}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if "curriculum" not in cfg and curriculum_path and Path(curriculum_path).exists():
        with open(curriculum_path, "r", encoding="utf-8") as f:
            cfg["curriculum"] = yaml.safe_load(f) or {}
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_combined_config(args.algo, args.config, args.curriculum_config)
    exp_name = args.exp_name or args.algo

    run_training(
        algo_name=args.algo,
        cfg=cfg,
        exp_name=exp_name,
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
        warm_start_wm=args.warm_start_wm,
        dataset_path=args.dataset,
    )


if __name__ == "__main__":
    main()
