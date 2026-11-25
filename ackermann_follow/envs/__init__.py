"""Gymnasium environment registration for MuJoCo Ackermann target following."""

from gymnasium.envs.registration import register

from ackermann_follow.envs.human_target import BipedalHumanTarget
from ackermann_follow.envs.mujoco_follow_env import MuJoCoAckermannFollowEnv

register(
    id="AckermannFollow-v0",
    entry_point="ackermann_follow.envs.mujoco_follow_env:MuJoCoAckermannFollowEnv",
)

__all__ = ["BipedalHumanTarget", "MuJoCoAckermannFollowEnv"]
