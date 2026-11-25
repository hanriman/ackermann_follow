"""Protocol defining the shared latent world-model interface."""

from __future__ import annotations

from typing import Protocol

import torch


class WorldModelProtocol(Protocol):
    latent_dim: int

    def encode(self, obs: torch.Tensor) -> torch.Tensor: ...

    def init_belief(self, batch_size: int, device: torch.device) -> torch.Tensor | None: ...

    def step_belief(
        self,
        obs: torch.Tensor,
        prev_action_norm: torch.Tensor | None,
        belief: torch.Tensor | None,
        episode_done: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def predict_next(
        self,
        latent: torch.Tensor,
        norm_action: torch.Tensor,
    ) -> torch.Tensor: ...
