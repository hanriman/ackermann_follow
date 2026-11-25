"""Backward-compatible shim exposing ActorCritic backed by modular GRUPPOAgent."""

from __future__ import annotations

import torch

from ackerman_follow.algos.gru_ppo import GRUPPOAgent, remap_legacy_actor_critic_state_dict
from ackerman_follow.models.encoders import layer_init


class ActorCritic(GRUPPOAgent):
    """Compatibility alias for `GRUPPOAgent` supporting legacy `ActorCritic` checkpoints."""

    def get_deterministic_action(self, x: torch.Tensor, hx=None, return_hidden: bool = False):
        action, next_hx = self.act(x, state=hx, deterministic=True)
        if return_hidden:
            return action, next_hx
        return action

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        remapped = remap_legacy_actor_critic_state_dict(dict(state_dict))
        return super().load_state_dict(remapped, strict=strict, assign=assign)


__all__ = ["ActorCritic", "layer_init"]
