"""Spawn-safe vectorized Gymnasium environment builder for AckBot."""

from __future__ import annotations

import gymnasium as gym
import torch

import ackermann_follow.envs  # noqa: F401


class _VectorSafeInfoWrapper(gym.Wrapper):
    """Drop variable-schema nested info dicts that break Gymnasium VectorEnv info batching."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info.pop("association_result", None)
        info.pop("tracker_status", None)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.pop("association_result", None)
        info.pop("tracker_status", None)
        return obs, reward, terminated, truncated, info


class _EnvFactory:
    """Top-level picklable env factory for spawn-safe AsyncVectorEnv on macOS & Linux."""

    def __init__(
        self,
        auto_curriculum: bool,
        start_stage: int,
        min_episodes_per_stage: int,
        stage_thresholds: dict[int, float],
        replay_previous_ratio: float,
        worker_threads: int | None = None,
        sensor_noise: bool = True,
    ) -> None:
        self.auto_curriculum = auto_curriculum
        self.start_stage = start_stage
        self.min_episodes_per_stage = min_episodes_per_stage
        self.stage_thresholds = stage_thresholds
        self.replay_previous_ratio = replay_previous_ratio
        self.worker_threads = worker_threads
        self.sensor_noise = sensor_noise

    def __call__(self):
        if self.worker_threads is not None:
            torch.set_num_threads(self.worker_threads)
        base_env = gym.make(
            "AckermannFollow-v0",
            auto_curriculum=self.auto_curriculum,
            stage=self.start_stage,
            min_episodes_per_stage=self.min_episodes_per_stage,
            stage_thresholds=self.stage_thresholds,
            replay_previous_ratio=self.replay_previous_ratio,
            sensor_noise=self.sensor_noise,
        )
        return _VectorSafeInfoWrapper(base_env)


def build_vec_env(
    num_envs: int,
    vec_env_mode: str = "sync",
    auto_curriculum: bool = True,
    start_stage: int = 1,
    min_episodes_per_stage: int = 35,
    stage_thresholds: dict[int, float] | None = None,
    replay_previous_ratio: float = 0.45,
    sensor_noise: bool = True,
) -> gym.vector.VectorEnv:
    """Create a SyncVectorEnv or spawn-safe AsyncVectorEnv for training."""
    thresholds = stage_thresholds or {
        1: 540.0,
        2: 430.0,
        3: 680.0,
        4: 620.0,
        5: 600.0,
        6: 580.0,
    }
    mode = vec_env_mode.lower()
    worker_threads = 1 if (mode == "async" and num_envs > 1) else None
    env_fns = [
        _EnvFactory(
            auto_curriculum=auto_curriculum,
            start_stage=start_stage,
            min_episodes_per_stage=min_episodes_per_stage,
            stage_thresholds=thresholds,
            replay_previous_ratio=replay_previous_ratio,
            worker_threads=worker_threads,
            sensor_noise=sensor_noise,
        )
        for _ in range(num_envs)
    ]
    if mode == "async" and num_envs > 1:
        return gym.vector.AsyncVectorEnv(
            env_fns,
            autoreset_mode=gym.vector.AutoresetMode.DISABLED,
            context="spawn",
        )
    return gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.DISABLED)
