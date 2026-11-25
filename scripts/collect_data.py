#!/usr/bin/env python3
"""
Collect multi-stage trajectory datasets into a SequenceReplayBuffer (`.npz`)
for offline LeWorldModel (JEPA) pretraining.

Examples:
  uv run python scripts/collect_data.py --steps 50000 --output outputs/datasets/mixed_7stage.npz
  uv run python scripts/collect_data.py --checkpoint <best_model.pt> --noise-std 0.25 --steps 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ackerman_follow.algos.registry import load_agent
from ackerman_follow.buffers.sequence_replay import SequenceReplayBuffer
from ackerman_follow.config import select_device
from ackerman_follow.engine.vec_env import build_vec_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect AckBot multi-stage dataset for LeWM pretraining")
    parser.add_argument("--steps", type=int, default=40960, help="Total environment transitions to collect")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--vec-env", type=str, default="sync", choices=["sync", "async"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional policy checkpoint (random if omitted)")
    parser.add_argument("--noise-std", type=float, default=0.25, help="Action exploration noise when using checkpoint")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7], choices=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--output", type=str, default="outputs/datasets/ackbot_7stage.npz", help="Output `.npz` path")
    parser.add_argument("--seed", type=int, default=77, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)
    agent = load_agent(args.checkpoint, device=device)[0] if args.checkpoint else None

    env = build_vec_env(
        num_envs=args.num_envs,
        vec_env_mode=args.vec_env,
        auto_curriculum=True,
        start_stage=max(args.stages),
        replay_previous_ratio=0.70,
        sensor_noise=False,
    )
    try:
        obs, infos = env.reset(seed=args.seed)
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        n_steps_total = (args.steps + args.num_envs - 1) // args.num_envs
        buf = SequenceReplayBuffer(
            capacity=n_steps_total * args.num_envs,
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_envs=args.num_envs,
            device="cpu",
        )
        state = agent.init_state(args.num_envs, device=device) if agent is not None else None
        active_stages = np.asarray(infos.get("stage", np.ones(args.num_envs)), dtype=np.int32)

        for _ in range(n_steps_total):
            if agent is None:
                actions = np.stack([env.single_action_space.sample() for _ in range(args.num_envs)]).astype(np.float32)
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    act_t, state = agent.act(obs_t, state=state, deterministic=True)
                    actions = act_t.cpu().numpy()
                if args.noise_std > 0.0:
                    actions[:, 0] = np.clip(actions[:, 0] + rng.normal(0.0, args.noise_std, size=args.num_envs), -0.4, 0.8)
                    actions[:, 1] = np.clip(actions[:, 1] + rng.normal(0.0, args.noise_std, size=args.num_envs), -0.553953, 0.553953)

            next_obs, rewards, terminateds, truncateds, step_infos = env.step(actions)
            dones_ep = np.logical_or(terminateds, truncateds)
            buf.add(obs, actions, rewards, next_obs, terminateds, dones_ep, stage=active_stages)
            obs = next_obs

            if np.any(dones_ep):
                obs, reset_infos = env.reset(options={"reset_mask": dones_ep})
                if agent is not None:
                    state = agent.reset_state_mask(state, dones_ep)
                if "stage" in reset_infos:
                    for i in np.flatnonzero(dones_ep):
                        active_stages[i] = int(reset_infos["stage"][i])

        saved = buf.save(Path(args.output))
        print(f"Saved {len(buf):,} transitions to {saved}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
