"""Off-policy / world-model sequence replay buffer respecting episode boundaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SequenceReplayBuffer:
    """
    Circular multi-environment sequence replay buffer stored in host RAM (float32 NumPy)
    and transferred to target device only on minibatch sampling.

    Stores `(obs_t, next_obs_t, action_t, reward_t, terminated_t, episode_done_t)` per env
    so contiguous horizon-`H` sequences `[H, B, ...]` can be sampled cleanly for:
      - LeWM (JEPA multi-step latent prediction)
      - LeWM + RL (SAC / auxiliary JEPA)
      - LeWM + TD-MPC2 (horizon-H latent dynamics + reward + TD targets)
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity_per_env = max(16, int(capacity) // max(1, int(num_envs)))
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        shape = (self.capacity_per_env, self.num_envs)
        self.obs = np.zeros((*shape, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((*shape, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((*shape, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.terminateds = np.zeros(shape, dtype=np.float32)
        self.episode_dones = np.zeros(shape, dtype=np.float32)
        self.stages = np.ones(shape, dtype=np.int32)

        self.ptr = 0
        self.size_per_env = 0

    def __len__(self) -> int:
        return self.size_per_env * self.num_envs

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray | float,
        next_obs: np.ndarray,
        terminated: np.ndarray | bool,
        episode_done: np.ndarray | bool,
        stage: np.ndarray | int = 1,
    ) -> None:
        """Append one vectorized step `[num_envs, ...]` into the circular buffer."""
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(self.num_envs, self.obs_dim)
        next_obs_arr = np.asarray(next_obs, dtype=np.float32).reshape(self.num_envs, self.obs_dim)
        act_arr = np.asarray(action, dtype=np.float32).reshape(self.num_envs, self.act_dim)
        rew_arr = np.broadcast_to(np.asarray(reward, dtype=np.float32), (self.num_envs,))
        term_arr = np.broadcast_to(np.asarray(terminated, dtype=np.float32), (self.num_envs,))
        ep_done_arr = np.broadcast_to(np.asarray(episode_done, dtype=np.float32), (self.num_envs,))
        stage_arr = np.broadcast_to(np.asarray(stage, dtype=np.int32), (self.num_envs,))

        self.obs[self.ptr] = obs_arr
        self.next_obs[self.ptr] = next_obs_arr
        self.actions[self.ptr] = act_arr
        self.rewards[self.ptr] = rew_arr
        self.terminateds[self.ptr] = term_arr
        self.episode_dones[self.ptr] = ep_done_arr
        self.stages[self.ptr] = stage_arr

        self.ptr = (self.ptr + 1) % self.capacity_per_env
        self.size_per_env = min(self.capacity_per_env, self.size_per_env + 1)

    def can_sample(self, horizon: int) -> bool:
        return self.size_per_env > horizon + 1

    def sample_sequences(
        self,
        batch_size: int,
        horizon: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Sample `batch_size` contiguous horizon-`horizon` chunks without crossing the write cursor.

        Returns dict of device tensors:
          - `obs`:          [H, B, obs_dim]
          - `next_obs`:     [H, B, obs_dim]
          - `actions`:      [H, B, act_dim]
          - `rewards`:      [H, B]
          - `terminateds`:  [H, B]
          - `episode_dones`:[H, B]
          - `valid_mask`:   [H, B] (1.0 up to and including the step where episode_done=1, 0.0 after)
        """
        if not self.can_sample(horizon):
            raise ValueError(
                f"Not enough steps in replay ({self.size_per_env} per env) for horizon={horizon}."
            )
        gen = rng if rng is not None else np.random.default_rng()

        if self.size_per_env < self.capacity_per_env:
            # Buffer has not wrapped yet: valid starts are [0, size_per_env - horizon]
            valid_starts = np.arange(0, self.size_per_env - horizon + 1, dtype=np.int64)
        else:
            # Wrapped circularly: oldest element is at self.ptr; exclude windows crossing self.ptr - 1 -> self.ptr
            ordered = (self.ptr + np.arange(self.capacity_per_env - horizon + 1)) % self.capacity_per_env
            valid_starts = ordered

        start_t = gen.choice(valid_starts, size=batch_size, replace=True)
        env_idx = gen.integers(0, self.num_envs, size=batch_size)
        offsets = np.arange(horizon, dtype=np.int64)[:, None]  # [H, 1]
        t_idx = (start_t[None, :] + offsets) % self.capacity_per_env  # [H, B]
        e_idx = np.broadcast_to(env_idx[None, :], (horizon, batch_size))

        obs_seq = self.obs[t_idx, e_idx]
        next_obs_seq = self.next_obs[t_idx, e_idx]
        act_seq = self.actions[t_idx, e_idx]
        rew_seq = self.rewards[t_idx, e_idx]
        term_seq = self.terminateds[t_idx, e_idx]
        ep_done_seq = self.episode_dones[t_idx, e_idx]

        # Build mask so steps strictly after an episode_done inside the chunk are zero-weighted
        valid_mask = np.ones((horizon, batch_size), dtype=np.float32)
        for t in range(1, horizon):
            valid_mask[t] = valid_mask[t - 1] * (1.0 - ep_done_seq[t - 1])

        return {
            "obs": torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(next_obs_seq, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(act_seq, dtype=torch.float32, device=self.device),
            "rewards": torch.as_tensor(rew_seq, dtype=torch.float32, device=self.device),
            "terminateds": torch.as_tensor(term_seq, dtype=torch.float32, device=self.device),
            "episode_dones": torch.as_tensor(ep_done_seq, dtype=torch.float32, device=self.device),
            "valid_mask": torch.as_tensor(valid_mask, dtype=torch.float32, device=self.device),
        }

    def save(self, path: str | Path) -> Path:
        """Persist populated replay buffer slices to disk (`.npz`)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.size_per_env < self.capacity_per_env:
            order = np.arange(self.size_per_env)
        else:
            order = (self.ptr + np.arange(self.capacity_per_env)) % self.capacity_per_env
        np.savez_compressed(
            out,
            obs=self.obs[order],
            next_obs=self.next_obs[order],
            actions=self.actions[order],
            rewards=self.rewards[order],
            terminateds=self.terminateds[order],
            episode_dones=self.episode_dones[order],
            stages=self.stages[order],
        )
        return out

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "SequenceReplayBuffer":
        """Load a saved `.npz` replay dataset into a SequenceReplayBuffer."""
        data = np.load(Path(path))
        obs = data["obs"]
        t_len, num_envs, obs_dim = obs.shape
        act_dim = data["actions"].shape[-1]
        buf = cls(
            capacity=max(16, t_len) * num_envs,
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_envs=num_envs,
            device=device,
        )
        buf.obs[:t_len] = obs
        buf.next_obs[:t_len] = data["next_obs"]
        buf.actions[:t_len] = data["actions"]
        buf.rewards[:t_len] = data["rewards"]
        buf.terminateds[:t_len] = data["terminateds"]
        buf.episode_dones[:t_len] = data["episode_dones"]
        buf.stages[:t_len] = data["stages"]
        buf.size_per_env = t_len
        buf.ptr = t_len % buf.capacity_per_env
        return buf
