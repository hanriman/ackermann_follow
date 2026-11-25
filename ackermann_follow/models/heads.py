"""Reusable actor, critic, Q-ensemble, reward, and termination heads for RL & World Models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from ackermann_follow.config import RobotConfig, get_robot_config
from ackermann_follow.models.encoders import layer_init


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm: sign(x) * log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def two_hot_encode(
    x: torch.Tensor,
    num_bins: int = 101,
    vmin: float = -10.0,
    vmax: float = 10.0,
) -> torch.Tensor:
    """Encode scalar values (typically symlog-scaled) into two-hot categorical weights."""
    x_sym = torch.clamp(symlog(x), vmin, vmax)
    bin_width = (vmax - vmin) / float(num_bins - 1)
    pos = (x_sym - vmin) / bin_width
    low = torch.floor(pos).long().clamp(0, num_bins - 2)
    high = low + 1
    weight_high = (pos - low.float()).clamp(0.0, 1.0)
    weight_low = 1.0 - weight_high

    target = torch.zeros((*x.shape, num_bins), dtype=torch.float32, device=x.device)
    target.scatter_(-1, low.unsqueeze(-1), weight_low.unsqueeze(-1))
    target.scatter_add_(-1, high.unsqueeze(-1), weight_high.unsqueeze(-1))
    return target


def two_hot_decode(
    logits: torch.Tensor,
    num_bins: int = 101,
    vmin: float = -10.0,
    vmax: float = 10.0,
) -> torch.Tensor:
    """Decode categorical logits over symlog bins back into scalar expectations."""
    probs = F.softmax(logits, dim=-1)
    bins = torch.linspace(vmin, vmax, num_bins, dtype=logits.dtype, device=logits.device)
    val_sym = (probs * bins).sum(dim=-1)
    return symexp(val_sym)


class ActionScalerMixin(nn.Module):
    """Registers RobotConfig action bounds as device-safe buffers for [-1, 1] <-> physical [v, steer]."""

    def __init__(self, robot_cfg: RobotConfig | None = None) -> None:
        super().__init__()
        cfg = robot_cfg if robot_cfg is not None else get_robot_config()
        self.robot_cfg = cfg
        v_scale = (cfg.v_max - cfg.v_min) / 2.0
        v_bias = (cfg.v_max + cfg.v_min) / 2.0
        self.register_buffer("v_min", torch.tensor(cfg.v_min, dtype=torch.float32))
        self.register_buffer("v_max", torch.tensor(cfg.v_max, dtype=torch.float32))
        self.register_buffer("steer_max", torch.tensor(cfg.max_steering_angle, dtype=torch.float32))
        self.register_buffer("v_scale", torch.tensor(v_scale, dtype=torch.float32))
        self.register_buffer("v_bias", torch.tensor(v_bias, dtype=torch.float32))
        self.register_buffer("steer_scale", torch.tensor(cfg.max_steering_angle, dtype=torch.float32))

    def scale_action(self, norm_action: torch.Tensor) -> torch.Tensor:
        """Map normalized action [-1, 1] to physical [v, steer]."""
        v = norm_action[..., 0] * self.v_scale + self.v_bias
        steer = norm_action[..., 1] * self.steer_scale
        return torch.stack([v, steer], dim=-1)

    def unscale_action(self, phys_action: torch.Tensor) -> torch.Tensor:
        """Map physical [v, steer] to normalized [-1, 1]."""
        norm_v = (phys_action[..., 0] - self.v_bias) / self.v_scale
        norm_steer = phys_action[..., 1] / self.steer_scale
        return torch.stack([norm_v, norm_steer], dim=-1)


class TanhGaussianActor(ActionScalerMixin):
    """Squashed Gaussian policy head outputting physical [v, steer] or normalized actions."""

    def __init__(
        self,
        in_dim: int,
        act_dim: int = 2,
        hidden_dims: tuple[int, ...] = (128, 64),
        state_dependent_std: bool = False,
        robot_cfg: RobotConfig | None = None,
    ) -> None:
        super().__init__(robot_cfg=robot_cfg)
        self.act_dim = int(act_dim)
        self.state_dependent_std = bool(state_dependent_std)

        layers: list[nn.Module] = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(cur, h)))
            layers.append(nn.Tanh())
            cur = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = layer_init(nn.Linear(cur, self.act_dim), std=0.01)
        if self.state_dependent_std:
            self.logstd_head = layer_init(nn.Linear(cur, self.act_dim), std=0.01)
        else:
            self.logstd = nn.Parameter(torch.zeros(1, self.act_dim) - 0.5)

    @staticmethod
    def tanh_log_det_jacobian(u: torch.Tensor) -> torch.Tensor:
        """Numerically stable log|det(d tanh(u) / du)|."""
        return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))

    def _dist(self, features: torch.Tensor) -> Normal:
        h = self.trunk(features)
        raw_mean = 3.0 * torch.tanh(self.mean_head(h))
        if self.state_dependent_std:
            raw_logstd = self.logstd_head(h)
        else:
            raw_logstd = self.logstd.expand_as(raw_mean)
        action_logstd = torch.clamp(raw_logstd, -2.5, -0.35)
        return Normal(raw_mean, torch.exp(action_logstd))

    def sample(
        self,
        features: torch.Tensor,
        raw_action: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        return_physical: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(action, log_prob, entropy, raw_u)`."""
        dist = self._dist(features)
        if action is None and raw_action is None:
            u = dist.rsample()
            norm_act = torch.tanh(u)
            out_act = self.scale_action(norm_act) if return_physical else norm_act
        elif raw_action is not None:
            u = raw_action
            norm_act = torch.tanh(u)
            out_act = self.scale_action(norm_act) if return_physical else norm_act
        else:
            norm_act = self.unscale_action(action) if return_physical else action
            norm_act = torch.clamp(norm_act, -0.999999, 0.999999)
            u = torch.atanh(norm_act)
            out_act = action

        log_prob = (dist.log_prob(u) - self.tanh_log_det_jacobian(u)).sum(-1)
        u_ent = dist.rsample()
        entropy = -(dist.log_prob(u_ent) - self.tanh_log_det_jacobian(u_ent)).sum(-1)
        return out_act, log_prob, entropy, u

    def deterministic(self, features: torch.Tensor, return_physical: bool = True) -> torch.Tensor:
        """Greedy mean action `tanh(raw_mean)`."""
        dist = self._dist(features)
        norm_mean = torch.tanh(dist.loc)
        return self.scale_action(norm_mean) if return_physical else norm_mean


class ValueCritic(nn.Module):
    """Scalar state value network V(s)."""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(cur, h)))
            layers.append(nn.Tanh())
            cur = h
        layers.append(layer_init(nn.Linear(cur, 1), std=1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _QBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            layer_init(nn.Linear(hidden_dim, out_dim), std=0.01 if out_dim > 1 else 1.0),
        )

    def forward(self, z_a: torch.Tensor) -> torch.Tensor:
        return self.net(z_a)


class TwinQNet(nn.Module):
    """Twin Q-function ensemble supporting scalar (SAC) or Two-Hot distributional (TD-MPC2) outputs."""

    def __init__(
        self,
        latent_dim: int,
        act_dim: int = 2,
        hidden_dim: int = 256,
        num_bins: int = 1,
        vmin: float = -10.0,
        vmax: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        in_dim = latent_dim + act_dim
        self.q1 = _QBranch(in_dim, hidden_dim, self.num_bins)
        self.q2 = _QBranch(in_dim, hidden_dim, self.num_bins)

    def forward_logits(self, z: torch.Tensor, norm_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        za = torch.cat([z, norm_action], dim=-1)
        return self.q1(za), self.q2(za)

    def forward(self, z: torch.Tensor, norm_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        l1, l2 = self.forward_logits(z, norm_action)
        if self.num_bins == 1:
            return l1.squeeze(-1), l2.squeeze(-1)
        return (
            two_hot_decode(l1, self.num_bins, self.vmin, self.vmax),
            two_hot_decode(l2, self.num_bins, self.vmin, self.vmax),
        )

    def min_q(self, z: torch.Tensor, norm_action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(z, norm_action)
        return torch.minimum(q1, q2)


class RewardHead(nn.Module):
    """Predicts step rewards R(z, a) as scalar or Two-Hot symlog distribution."""

    def __init__(
        self,
        latent_dim: int,
        act_dim: int = 2,
        hidden_dim: int = 256,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.net = nn.Sequential(
            layer_init(nn.Linear(latent_dim + act_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            layer_init(nn.Linear(hidden_dim, self.num_bins), std=0.01 if self.num_bins > 1 else 1.0),
        )

    def forward_logits(self, z: torch.Tensor, norm_action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, norm_action], dim=-1))

    def forward(self, z: torch.Tensor, norm_action: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(z, norm_action)
        if self.num_bins == 1:
            return logits.squeeze(-1)
        return two_hot_decode(logits, self.num_bins, self.vmin, self.vmax)


class TerminationHead(nn.Module):
    """Predicts collision/termination probability logit from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(latent_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            layer_init(nn.Linear(hidden_dim, 1), std=0.01),
        )

    def forward_logit(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logit(z))
