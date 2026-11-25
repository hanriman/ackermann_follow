"""
LeWorldModel (LeWM) Joint-Embedding Predictive Architecture (JEPA) for AckBot.

Key components:
  - Online SpatialSensorEncoder + latent projector E_theta(o_t) -> z_t
  - Stop-gradient EMA Target Encoder E_{theta_bar}(o_{t+k}) -> z_tgt
  - Action-conditioned latent dynamics predictor P_phi(z_t, a_t) -> z_hat_{t+1}
  - Optional recurrent GRU belief integration for occluded target tracking (POMDP)
  - MPS/CPU-safe real-valued Epps-Pulley SIGReg collapse-prevention regularizer
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from ackerman_follow.config import RobotConfig, get_robot_config
from ackerman_follow.models.encoders import SpatialSensorEncoder, layer_init


def sigreg_epps_pulley_loss(
    z: torch.Tensor,
    num_slices: int = 64,
    num_t: int = 17,
    t_max: float = 4.0,
) -> torch.Tensor:
    """
    Sketched Isotropic Gaussian Regularization (SIGReg) via 1D sliced Epps-Pulley test.

    Uses purely real cos/sin characteristic function terms so it runs identically on CPU,
    CUDA, and Apple Silicon MPS without complex tensor kernels.
    """
    z_flat = z.reshape(-1, z.shape[-1])
    n, d = z_flat.shape
    if n < 2:
        return z_flat.new_zeros(())

    # Sample random unit projection directions u ~ S^{d-1}
    proj_dirs = torch.randn((d, num_slices), dtype=z_flat.dtype, device=z_flat.device)
    proj_dirs = F.normalize(proj_dirs, dim=0)
    slices = z_flat @ proj_dirs  # [N, M]

    # Integration grid t in [-t_max, t_max] (symmetric -> imaginary part sin integrates nicely)
    t_grid = torch.linspace(0.0, t_max, num_t, dtype=z_flat.dtype, device=z_flat.device)
    # Target N(0, 1) characteristic function exp(-0.5 * t^2) and Gaussian weight
    phi_target = torch.exp(-0.5 * (t_grid ** 2))  # [K]
    weights = phi_target  # [K]

    # angles: [N, M, K]
    angles = slices.unsqueeze(-1) * t_grid.view(1, 1, num_t)
    mean_cos = torch.cos(angles).mean(dim=0)  # [M, K]
    mean_sin = torch.sin(angles).mean(dim=0)  # [M, K]

    err_sq = (mean_cos - phi_target.unsqueeze(0)) ** 2 + mean_sin ** 2
    # Trapezoidal weighting (count t > 0 twice due to symmetry of (-t, +t))
    sym_mult = torch.ones_like(t_grid)
    sym_mult[1:] = 2.0
    loss = (err_sq * (weights * sym_mult).unsqueeze(0)).mean() * float(n)
    # Normalize by batch size scale so loss magnitude stays O(1) across batch sizes
    return loss / max(1.0, float(n) * 0.25)


class LeWMJEPA(nn.Module):
    """Action-conditioned JEPA latent world model with EMA target encoder and SIGReg."""

    def __init__(
        self,
        obs_dim: int | None = None,
        act_dim: int = 2,
        latent_dim: int = 128,
        predictor_hidden: int = 256,
        ema_tau: float = 0.99,
        sigreg_coef: float = 0.25,
        pred_decay: float = 0.90,
        use_recurrent_belief: bool = True,
        robot_cfg: RobotConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = robot_cfg if robot_cfg is not None else get_robot_config()
        self.robot_cfg = cfg
        self.obs_dim = int(obs_dim if obs_dim is not None else cfg.obs_dim)
        self.act_dim = int(act_dim)
        self.latent_dim = int(latent_dim)
        self.ema_tau = float(ema_tau)
        self.sigreg_coef = float(sigreg_coef)
        self.pred_decay = float(pred_decay)
        self.use_recurrent_belief = bool(use_recurrent_belief)

        # Online encoder: SpatialSensorEncoder (490 -> 128) + linear latent projection
        self.online_backbone = SpatialSensorEncoder(
            obs_dim=self.obs_dim,
            latent_dim=128,
            robot_cfg=cfg,
        )
        self.online_proj = nn.Sequential(
            layer_init(nn.Linear(128, self.latent_dim)),
            nn.LayerNorm(self.latent_dim),
        )

        # Stop-gradient EMA target encoder
        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_proj = copy.deepcopy(self.online_proj)
        for p in self.target_backbone.parameters():
            p.requires_grad_(False)
        for p in self.target_proj.parameters():
            p.requires_grad_(False)

        # Optional recurrent GRU belief integrator for POMDP target memory
        if self.use_recurrent_belief:
            self.belief_gru = nn.GRU(self.latent_dim + self.act_dim, self.latent_dim)
            for name, param in self.belief_gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0.0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
            self.belief_fuse = nn.Sequential(
                layer_init(nn.Linear(self.latent_dim * 2, self.latent_dim)),
                nn.LayerNorm(self.latent_dim),
            )
        else:
            self.belief_gru = None
            self.belief_fuse = None

        # Action-conditioned latent residual predictor: z_{t+1} = LayerNorm(z_t + f(z_t, a_t))
        self.predictor_net = nn.Sequential(
            layer_init(nn.Linear(self.latent_dim + self.act_dim, predictor_hidden)),
            nn.LayerNorm(predictor_hidden),
            nn.Mish(),
            layer_init(nn.Linear(predictor_hidden, predictor_hidden)),
            nn.LayerNorm(predictor_hidden),
            nn.Mish(),
            layer_init(nn.Linear(predictor_hidden, self.latent_dim), std=0.5),
        )
        self.predictor_norm = nn.LayerNorm(self.latent_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation `[..., obs_dim]` with online JEPA encoder."""
        h = self.online_backbone(obs)
        return self.online_proj(h)

    @torch.no_grad()
    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation `[..., obs_dim]` with stop-gradient EMA target encoder."""
        h = self.target_backbone(obs)
        return self.target_proj(h)

    def init_belief(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if not self.use_recurrent_belief:
            return None
        return torch.zeros((1, batch_size, self.latent_dim), dtype=torch.float32, device=device)

    def step_belief(
        self,
        obs: torch.Tensor,
        prev_action_norm: torch.Tensor | None = None,
        belief: torch.Tensor | None = None,
        episode_done: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute current latent representation `z_t` and updated recurrent belief `next_belief`.
        Supports `[B, obs_dim]` step inputs.
        """
        z_obs = self.encode(obs)
        if not self.use_recurrent_belief or self.belief_gru is None or self.belief_fuse is None:
            return z_obs, None

        bsz = z_obs.shape[0]
        if belief is None:
            belief = self.init_belief(bsz, z_obs.device)
        if prev_action_norm is None:
            prev_action_norm = torch.zeros((bsz, self.act_dim), dtype=z_obs.dtype, device=z_obs.device)

        gru_in = torch.cat([z_obs, prev_action_norm], dim=-1).unsqueeze(0)
        gru_out, next_belief = self.belief_gru(gru_in, belief)
        if episode_done is not None:
            mask = (1.0 - episode_done).view(1, bsz, 1)
            next_belief = next_belief * mask
        z_state = self.belief_fuse(torch.cat([z_obs, gru_out.squeeze(0)], dim=-1))
        return z_state, next_belief

    def encode_sequence(
        self,
        obs_seq: torch.Tensor,
        actions_norm_seq: torch.Tensor,
        episode_dones: torch.Tensor | None = None,
        init_belief: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode `[H, B, obs_dim]` sequence into `(z_obs_seq, z_state_seq)` respecting resets.
        """
        horizon, bsz, _ = obs_seq.shape
        z_obs_seq = self.encode(obs_seq)
        if not self.use_recurrent_belief or self.belief_gru is None or self.belief_fuse is None:
            return z_obs_seq, z_obs_seq

        cur_belief = init_belief if init_belief is not None else self.init_belief(bsz, obs_seq.device)
        states = []
        zero_act = torch.zeros((bsz, self.act_dim), dtype=obs_seq.dtype, device=obs_seq.device)
        for t in range(horizon):
            prev_a = zero_act if t == 0 else actions_norm_seq[t - 1]
            gru_in = torch.cat([z_obs_seq[t], prev_a], dim=-1).unsqueeze(0)
            gru_out, cur_belief = self.belief_gru(gru_in, cur_belief)
            z_t = self.belief_fuse(torch.cat([z_obs_seq[t], gru_out.squeeze(0)], dim=-1))
            states.append(z_t)
            if episode_dones is not None:
                cur_belief = cur_belief * (1.0 - episode_dones[t]).view(1, bsz, 1)
        return z_obs_seq, torch.stack(states, dim=0)

    def predict_next(self, latent: torch.Tensor, norm_action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state `z_{t+1}` given current latent `z_t` and normalized `a_t in [-1, 1]`."""
        delta = self.predictor_net(torch.cat([latent, norm_action], dim=-1))
        return self.predictor_norm(latent + delta)

    def rollout_predictions(
        self, z0: torch.Tensor, norm_actions_seq: torch.Tensor
    ) -> torch.Tensor:
        """Imagine open-loop latent trajectory `[H, B, D]` starting from `z0 [B, D]`."""
        preds = []
        z = z0
        for t in range(norm_actions_seq.shape[0]):
            z = self.predict_next(z, norm_actions_seq[t])
            preds.append(z)
        return torch.stack(preds, dim=0)

    @torch.no_grad()
    def update_ema(self, tau: float | None = None) -> None:
        """Polyak average update for the EMA target encoder."""
        m = self.ema_tau if tau is None else float(tau)
        for p_on, p_tg in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            p_tg.data.mul_(m).add_(p_on.data, alpha=1.0 - m)
        for p_on, p_tg in zip(self.online_proj.parameters(), self.target_proj.parameters()):
            p_tg.data.mul_(m).add_(p_on.data, alpha=1.0 - m)

    def compute_jepa_loss(
        self,
        obs_seq: torch.Tensor,
        next_obs_seq: torch.Tensor,
        actions_norm_seq: torch.Tensor,
        valid_mask: torch.Tensor,
        episode_dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute multi-step JEPA latent prediction loss + SIGReg anti-collapse loss.

        Args:
            obs_seq:         [H, B, obs_dim]
            next_obs_seq:    [H, B, obs_dim]
            actions_norm_seq:[H, B, act_dim] in [-1, 1]
            valid_mask:      [H, B] (1.0 for valid transitions before any reset)
        """
        horizon = obs_seq.shape[0]
        z_obs_seq, z_state_seq = self.encode_sequence(
            obs_seq, actions_norm_seq, episode_dones=episode_dones
        )

        # Open-loop latent predictions starting from z_state_seq[0]
        pred_latents = self.rollout_predictions(z_state_seq[0], actions_norm_seq)  # [H, B, D]

        with torch.no_grad():
            tgt_obs_latents = self.encode_target(next_obs_seq)  # [H, B, D]
            if self.use_recurrent_belief and self.belief_gru is not None and self.belief_fuse is not None:
                # Target state aligns with target observation representation projected cleanly
                tgt_latents = tgt_obs_latents
            else:
                tgt_latents = tgt_obs_latents

        # Temporal discount weights across rollout horizon
        step_weights = torch.tensor(
            [math.pow(self.pred_decay, t) for t in range(horizon)],
            dtype=obs_seq.dtype,
            device=obs_seq.device,
        ).unsqueeze(-1)  # [H, 1]
        weights = valid_mask * step_weights

        # Normalized cosine + MSE prediction error in latent space
        pred_norm = F.normalize(pred_latents, dim=-1)
        tgt_norm = F.normalize(tgt_latents, dim=-1)
        step_err = (pred_norm - tgt_norm).pow(2).mean(dim=-1) + 0.1 * F.mse_loss(
            pred_latents, tgt_latents, reduction="none"
        ).mean(dim=-1)
        pred_loss = (step_err * weights).sum() / weights.sum().clamp_min(1.0)

        # SIGReg applied to online observation latents to enforce isotropic Gaussian distribution
        z_valid = z_obs_seq[valid_mask > 0.5]
        sigreg_loss = (
            sigreg_epps_pulley_loss(z_valid)
            if z_valid.shape[0] >= 4
            else obs_seq.new_zeros(())
        )

        total_loss = pred_loss + self.sigreg_coef * sigreg_loss
        info = {
            "jepa_pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "z_state_seq": z_state_seq,
            "pred_latents": pred_latents,
            "z_std": z_obs_seq.std(dim=(0, 1)).mean().detach(),
        }
        return total_loss, info
