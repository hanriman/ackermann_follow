"""Latent world-model architectures (LeWM / JEPA)."""

from ackermann_follow.models.world_models.base import WorldModelProtocol
from ackermann_follow.models.world_models.lewm_jepa import LeWMJEPA, sigreg_epps_pulley_loss

__all__ = ["LeWMJEPA", "WorldModelProtocol", "sigreg_epps_pulley_loss"]
