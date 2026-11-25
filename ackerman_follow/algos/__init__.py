"""Modular RL and World-Model algorithms implementing the shared BaseAgent contract."""

from ackerman_follow.algos.base import BaseAgent
from ackerman_follow.algos.gru_ppo import GRUPPOAgent
from ackerman_follow.algos.lewm import LeWMAgent
from ackerman_follow.algos.lewm_rl_modelfree import LeWMModelFreeAgent
from ackerman_follow.algos.lewm_tdmpc2 import LeWMTDMPC2Agent
from ackerman_follow.algos.registry import ALGO_REGISTRY, build_agent, load_agent

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
