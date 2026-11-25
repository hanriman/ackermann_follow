"""Modular RL and World-Model algorithms implementing the shared BaseAgent contract."""

from ackermann_follow.algos.base import BaseAgent
from ackermann_follow.algos.gru_ppo import GRUPPOAgent
from ackermann_follow.algos.lewm import LeWMAgent
from ackermann_follow.algos.lewm_rl_modelfree import LeWMModelFreeAgent
from ackermann_follow.algos.lewm_tdmpc2 import LeWMTDMPC2Agent
from ackermann_follow.algos.registry import ALGO_REGISTRY, build_agent, load_agent

__all__ = [
    "ALGO_REGISTRY",
    "BaseAgent",
    "GRUPPOAgent",
    "LeWMAgent",
    "LeWMModelFreeAgent",
    "LeWMTDMPC2Agent",
    "build_agent",
    "load_agent",
]
