"""Gymnasium environment registration for MuJoCo Ackermann target following."""

from gymnasium.envs.registration import register

from ackerman_follow.envs.human_target import BipedalHumanTarget
from ackerman_follow.envs.mujoco_follow_env import MuJoCoAckermanFollowEnv

register(
    id="AckermanFollow-v0",
    entry_point="ackerman_follow.envs.mujoco_follow_env:MuJoCoAckermanFollowEnv",
)

__all__ = ["BipedalHumanTarget", "MuJoCoAckermanFollowEnv"]
