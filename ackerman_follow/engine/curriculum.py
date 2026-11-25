"""Synchronized 7-stage curriculum graduation tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


class CurriculumTracker:
    """Tracks frontier-stage episode returns and triggers synchronized stage graduation."""

    def __init__(
        self,
        enabled: bool = True,
        start_stage: int = 1,
        min_episodes_per_stage: int = 35,
        stage_thresholds: dict[int, float] | None = None,
        max_stage: int = 7,
    ) -> None:
        self.enabled = bool(enabled)
        self.unlocked_stage = int(np.clip(start_stage, 1, max_stage))
        self.min_episodes_per_stage = int(min_episodes_per_stage)
        self.stage_thresholds = stage_thresholds or {
            1: 540.0,
            2: 430.0,
            3: 680.0,
            4: 620.0,
            5: 600.0,
            6: 580.0,
        }
        self.max_stage = int(max_stage)
        self.frontier_ep_rewards: list[float] = []

    def record_episode(
        self,
        worker_stage: int,
        ep_ret: float,
        vec_env: Any,
        on_graduate: Callable[[int, int], None] | None = None,
    ) -> bool:
        """Record an episode return; graduates & notifies `vec_env` if frontier threshold is met."""
        if not self.enabled or worker_stage != self.unlocked_stage:
            return False
        self.frontier_ep_rewards.append(float(ep_ret))
        if len(self.frontier_ep_rewards) < self.min_episodes_per_stage:
            return False

        mean_frontier = float(np.mean(self.frontier_ep_rewards[-self.min_episodes_per_stage :]))
        thresh = self.stage_thresholds.get(self.unlocked_stage)
        if thresh is not None and mean_frontier >= thresh and self.unlocked_stage < self.max_stage:
            grad_stage = self.unlocked_stage
            self.unlocked_stage += 1
            self.frontier_ep_rewards.clear()
            if vec_env is not None:
                vec_env.call("set_unlocked_stage", self.unlocked_stage)
            if on_graduate is not None:
                on_graduate(grad_stage, self.unlocked_stage)
            return True
        return False


def parse_curriculum_config(curr_cfg: dict[str, Any]) -> tuple[int, dict[int, float], float]:
    """Extract `(min_episodes_per_stage, stage_thresholds, replay_previous_ratio)` from YAML dict."""
    min_eps = int(curr_cfg.get("min_episodes_per_stage", 35))
    raw_thresholds = curr_cfg.get("stage_thresholds", {1: 540.0, 2: 430.0, 3: 680.0, 4: 620.0, 5: 600.0, 6: 580.0})
    thresholds = {
        int(k): float(v["min_return"] if isinstance(v, dict) else v)
        for k, v in raw_thresholds.items()
    }
    replay_ratio = float(curr_cfg.get("replay_previous_ratio", 0.45))
    return min_eps, thresholds, replay_ratio


def save_milestone_path(ckpt_dir: Path, grad_stage: int) -> Path:
    return ckpt_dir / f"graduated_stage_{grad_stage}.pt"
