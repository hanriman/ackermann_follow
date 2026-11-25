"""Shared training, evaluation, curriculum, logging, and checkpoint engine for all algorithms."""

from ackerman_follow.engine.checkpoint import save_checkpoint
from ackerman_follow.engine.curriculum import CurriculumTracker
from ackerman_follow.engine.evaluator import evaluate_agent
from ackerman_follow.engine.logging import DualLogger, MetricsLogger
from ackerman_follow.engine.runner import run_training
from ackerman_follow.engine.vec_env import _EnvFactory, _VectorSafeInfoWrapper, build_vec_env

__all__ = [
    "CurriculumTracker",
    "DualLogger",
    "MetricsLogger",
    "_EnvFactory",
    "_VectorSafeInfoWrapper",
    "build_vec_env",
    "evaluate_agent",
    "run_training",
    "save_checkpoint",
]
