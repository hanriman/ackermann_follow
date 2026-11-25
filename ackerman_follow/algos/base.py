"""Base agent interface shared across all RL and World-Model algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ackerman_follow.config import RobotConfig
from ackerman_follow.models.heads import ActionScalerMixin


class BaseAgent(ActionScalerMixin):
    """
    Unified contract for all AckBot algorithms (GRU PPO, LeWM, LeWM+RL, LeWM+TD-MPC2).

    Every algorithm implements:
      - `init_state(batch_size, device)`
      - `reset_state_mask(state, done_mask)`
      - `act(obs, state=None, deterministic=True) -> (phys_action, next_state)`
    so `engine/evaluator.py`, `scripts/eval_policy.py`, `scripts/demo_3d.py`, and
    `scripts/record_video.py` work identically regardless of the underlying algorithm.
    """

    algo_name: str = "base"
    training_mode: str = "on_policy"  # "on_policy" | "off_policy" | "offline_wm"

    def __init__(self, robot_cfg: RobotConfig | None = None) -> None:
        super().__init__(robot_cfg=robot_cfg)

    def init_state(self, batch_size: int, device: torch.device) -> Any:
        """Initialize recurrent/belief state for a batch of environments."""
        del batch_size, device
        return None

    def reset_state_mask(self, state: Any, done_mask: np.ndarray | torch.Tensor) -> Any:
        """Zero out recurrent/belief state entries where `done_mask` is True."""
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            mask = torch.as_tensor(done_mask, dtype=torch.bool, device=state.device)
            if state.dim() == 3:
                state[:, mask, :] = 0.0
            elif state.dim() == 2:
                state[mask, :] = 0.0
            return state
        if isinstance(state, tuple):
            return tuple(self.reset_state_mask(s, done_mask) for s in state)
        if isinstance(state, dict):
            return {k: self.reset_state_mask(v, done_mask) for k, v in state.items()}
        return state

    def act(
        self,
        obs: torch.Tensor,
        state: Any = None,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, Any]:
        """Compute physical `[v, steer]` actions for `[B, obs_dim]` observations."""
        raise NotImplementedError

    def export_metadata(self) -> dict[str, Any]:
        """Return algorithm hyperparameters stored inside checkpoints for reload."""
        return {}
