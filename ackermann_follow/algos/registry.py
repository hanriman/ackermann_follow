"""Algorithm registry & unified checkpoint loader across GRU PPO, LeWM, LeWM+RL, and LeWM+TD-MPC2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from ackermann_follow.algos.base import BaseAgent
from ackermann_follow.algos.gru_ppo import (
    GRUPPOAgent,
    build_gru_ppo_from_cfg,
    remap_legacy_actor_critic_state_dict,
)
from ackermann_follow.algos.lewm import LeWMAgent, build_lewm_from_cfg
from ackermann_follow.algos.lewm_rl_modelfree import (
    LeWMModelFreeAgent,
    build_lewm_rl_modelfree_from_cfg,
)
from ackermann_follow.algos.lewm_tdmpc2 import LeWMTDMPC2Agent, build_lewm_tdmpc2_from_cfg
from ackermann_follow.config import RobotConfig, get_robot_config

BuilderFn = Callable[[int, int, dict[str, Any], RobotConfig | None], BaseAgent]

ALGO_REGISTRY: dict[str, BuilderFn] = {
    "gru_ppo": build_gru_ppo_from_cfg,
    "lewm": build_lewm_from_cfg,
    "lewm_rl_modelfree": build_lewm_rl_modelfree_from_cfg,
    "lewm_tdmpc2": build_lewm_tdmpc2_from_cfg,
}

AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "gru_ppo": GRUPPOAgent,
    "lewm": LeWMAgent,
    "lewm_rl_modelfree": LeWMModelFreeAgent,
    "lewm_tdmpc2": LeWMTDMPC2Agent,
}


def build_agent(
    algo_name: str,
    obs_dim: int | None = None,
    act_dim: int | None = None,
    algo_cfg: dict[str, Any] | None = None,
    robot_cfg: RobotConfig | None = None,
) -> BaseAgent:
    """Instantiate any registered RL or world-model agent by name."""
    key = algo_name.lower()
    if key not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm {algo_name!r}. Available: {sorted(ALGO_REGISTRY)}")
    cfg = robot_cfg if robot_cfg is not None else get_robot_config()
    return ALGO_REGISTRY[key](
        int(obs_dim if obs_dim is not None else cfg.obs_dim),
        int(act_dim if act_dim is not None else cfg.act_dim),
        algo_cfg or {},
        cfg,
    )


def load_agent(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[BaseAgent, dict[str, Any]]:
    """
    Load any trained agent checkpoint (`gru_ppo`, `lewm`, `lewm_rl_modelfree`, `lewm_tdmpc2`),
    including legacy `ActorCritic` checkpoints without an explicit `algo` key.
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    dev = torch.device(device)
    ckpt = torch.load(ckpt_file, map_location=dev, weights_only=False)
    robot_cfg = get_robot_config()

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = ckpt
    else:
        state_dict = ckpt
        meta = {}

    algo_name = str(meta.get("algo", "gru_ppo")).lower()
    algo_meta = meta.get("algo_meta") or meta.get("algo_config") or {}
    if not algo_meta and isinstance(meta.get("config"), dict):
        algo_meta = meta["config"].get("algorithm", {})

    if algo_name == "gru_ppo":
        state_dict = remap_legacy_actor_critic_state_dict(state_dict)
        agent: BaseAgent = build_gru_ppo_from_cfg(
            robot_cfg.obs_dim, robot_cfg.act_dim, algo_meta, robot_cfg
        )
    elif algo_name in AGENT_CLASSES and meta.get("algo_meta"):
        agent = AGENT_CLASSES[algo_name](robot_cfg=robot_cfg, **meta["algo_meta"])
    else:
        agent = build_agent(
            algo_name=algo_name,
            obs_dim=robot_cfg.obs_dim,
            act_dim=robot_cfg.act_dim,
            algo_cfg=algo_meta,
            robot_cfg=robot_cfg,
        )

    agent = agent.to(dev)
    agent.load_state_dict(state_dict, strict=True)
    agent.eval()
    return agent, meta
