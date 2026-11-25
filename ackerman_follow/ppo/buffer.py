"""Backward-compatible shim re-exporting RolloutBuffer from ackerman_follow.buffers."""

from ackerman_follow.buffers.rollout import RolloutBuffer

__all__ = ["RolloutBuffer"]
