"""Deterministic multi-stage evaluation for any BaseAgent or legacy ActorCritic."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch


def _infer_action(agent: Any, obs_t: torch.Tensor, state: Any):
    """Support both `BaseAgent.act(...)` and legacy `ActorCritic.get_deterministic_action(...)`."""
    if hasattr(agent, "act"):
        return agent.act(obs_t, state=state, deterministic=True)
    return agent.get_deterministic_action(obs_t, hx=state, return_hidden=True)


def _init_agent_state(agent: Any, device: torch.device | str):
    if hasattr(agent, "init_state"):
        return agent.init_state(1, device=device)
    return agent.init_hidden(1, device=device)


def evaluate_agent(
    agent: Any,
    env: gym.Env,
    n_episodes_per_stage: int = 3,
    stages: tuple[int, ...] = (1, 2, 3, 4),
    device: str | torch.device = "cpu",
    seed: int = 1000,
    render: bool = False,
    realtime: bool = True,
) -> dict[str, Any]:
    """Run deterministic evaluation episodes across the requested curriculum stages."""
    all_rewards, all_lengths, all_dists = [], [], []
    total_collisions = 0
    total_episodes = 0
    total_fov_steps = 0
    total_steps = 0
    stage_breakdown: dict[int, dict[str, float]] = {}
    dt = env.unwrapped.cfg.dt

    was_training = agent.training
    agent.eval()
    try:
        for stg in stages:
            stg_rewards, stg_lengths, stg_dists = [], [], []
            stg_collisions = 0
            stg_fov_steps = 0
            stg_steps = 0

            for ep_idx in range(n_episodes_per_stage):
                obs, _ = env.reset(seed=seed + stg * 100 + ep_idx, options={"stage": stg})
                state = _init_agent_state(agent, device)
                done = False
                ep_ret = 0.0
                ep_len = 0
                ep_dists = []

                while not done:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        action_t, state = _infer_action(agent, obs_t, state)
                        action = action_t.squeeze(0).cpu().numpy()

                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_ret += float(reward)
                    ep_len += 1
                    ep_dists.append(float(info.get("target_dist", 0.0)))
                    if info.get("target_in_fov", False):
                        stg_fov_steps += 1
                    stg_steps += 1

                    done = terminated or truncated
                    if terminated:
                        stg_collisions += 1

                    if render:
                        viewer = getattr(env.unwrapped, "_viewer", None)
                        if viewer is not None and not viewer.is_running():
                            done = True
                        if realtime:
                            elapsed = time.perf_counter() - t0
                            if elapsed < dt:
                                time.sleep(dt - elapsed)

                stg_rewards.append(ep_ret)
                stg_lengths.append(ep_len)
                stg_dists.append(float(np.mean(ep_dists)) if ep_dists else 0.0)

            mean_stg_rew = float(np.mean(stg_rewards))
            mean_stg_dst = float(np.mean(stg_dists))
            col_rate = float(stg_collisions / max(1, n_episodes_per_stage))
            fov_rate = float(stg_fov_steps / max(1, stg_steps))

            stage_breakdown[stg] = {
                "mean_reward": mean_stg_rew,
                "std_reward": float(np.std(stg_rewards)),
                "mean_length": float(np.mean(stg_lengths)),
                "mean_dist": mean_stg_dst,
                "collision_rate": col_rate,
                "fov_rate": fov_rate,
            }
            all_rewards.extend(stg_rewards)
            all_lengths.extend(stg_lengths)
            all_dists.extend(stg_dists)
            total_collisions += stg_collisions
            total_episodes += n_episodes_per_stage
            total_fov_steps += stg_fov_steps
            total_steps += stg_steps
    finally:
        if was_training:
            agent.train()

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_length": float(np.mean(all_lengths)),
        "mean_target_dist": float(np.mean(all_dists)),
        "collision_rate": float(total_collisions / max(1, total_episodes)),
        "target_in_fov_rate": float(total_fov_steps / max(1, total_steps)),
        "stage_breakdown": stage_breakdown,
    }
