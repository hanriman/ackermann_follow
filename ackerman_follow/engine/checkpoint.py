"""Self-describing checkpoint saving across all AckBot algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ackerman_follow.algos.base import BaseAgent


def save_checkpoint(
    path: str | Path,
    agent: BaseAgent,
    step: int,
    unlocked_stage: int,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save agent weights together with algorithm identifier and constructor metadata."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "algo": agent.algo_name,
        "algo_meta": agent.export_metadata(),
        "step": int(step),
        "unlocked_stage": int(unlocked_stage),
        "model_state_dict": agent.state_dict(),
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, out)
    return out
