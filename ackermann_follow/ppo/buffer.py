"""Backward-compatible shim re-exporting RolloutBuffer from ackermann_follow.buffers."""

from ackermann_follow.buffers.rollout import RolloutBuffer

__all__ = ["RolloutBuffer"]
