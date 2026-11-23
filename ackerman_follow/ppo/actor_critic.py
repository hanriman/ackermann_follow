"""
Recurrent Spatial Actor-Critic neural network architecture for AckBot target following with PPO.

Features:
- Pure 426-D sensor input contract (64 camera depth + 360 LiDAR + 2 state)
- Proximity inversion (1.0 - range_norm) so sparse targets stand out over empty background
- Angular CoordConv 1D encoders (with circular padding for 360-deg LiDAR)
- Differentiable polar soft-argmin & angular sector summary features
- GRU(128, 128) temporal memory with current-frame residual skip for velocity & FOV-exit memory
- Decoupled Actor and Critic trunks with non-saturating scaled-tanh action mean
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

from ackerman_follow.config import RobotConfig, get_robot_config


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class _CircularConv1d(nn.Module):
    """1D convolution with explicit circular padding (MPS- and CPU-safe)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.pad = padding
        self.conv = layer_init(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad), mode="circular")
        return self.conv(x)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int | None = None,
        act_dim: int | None = None,
        cam_dim: int | None = None,
        cam_mask_dim: int | None = None,
        lidar_dim: int | None = None,
        gru_hidden_dim: int = 128,
        robot_cfg: RobotConfig | None = None,
    ):
        super().__init__()
        cfg = robot_cfg if robot_cfg is not None else get_robot_config()
        self.robot_cfg = cfg
        self.cam_dim = int(cam_dim if cam_dim is not None else cfg.cam_samples)
        self.cam_mask_dim = int(
            cam_mask_dim if cam_mask_dim is not None else cfg.cam_mask_samples
        )
        self.lidar_dim = int(lidar_dim if lidar_dim is not None else cfg.lidar_samples)
        self.obs_dim = int(
            obs_dim
            if obs_dim is not None
            else (self.cam_dim + self.cam_mask_dim + self.lidar_dim + 2)
        )
        self.act_dim = int(act_dim if act_dim is not None else cfg.act_dim)
        self.state_dim = self.obs_dim - self.cam_dim - self.cam_mask_dim - self.lidar_dim  # 2 (v, steer)
        self.gru_hidden_dim = int(gru_hidden_dim)

        # Action physical bounds & scaling constants registered as buffers so .to(device) moves all
        v_scale = (cfg.v_max - cfg.v_min) / 2.0
        v_bias = (cfg.v_max + cfg.v_min) / 2.0
        self.register_buffer("v_min", torch.tensor(cfg.v_min, dtype=torch.float32))
        self.register_buffer("v_max", torch.tensor(cfg.v_max, dtype=torch.float32))
        self.register_buffer("steer_max", torch.tensor(cfg.max_steering_angle, dtype=torch.float32))
        self.register_buffer("v_scale", torch.tensor(v_scale, dtype=torch.float32))
        self.register_buffer("v_bias", torch.tensor(v_bias, dtype=torch.float32))
        self.register_buffer("steer_scale", torch.tensor(cfg.max_steering_angle, dtype=torch.float32))

        # Registered angular coordinate buffers for CoordConv and polar soft-argmin
        self.register_buffer(
            "cam_angles", torch.linspace(-1.0, 1.0, self.cam_dim, dtype=torch.float32)
        )
        lidar_angles = torch.linspace(0.0, 2.0 * math.pi, self.lidar_dim + 1, dtype=torch.float32)[
            :-1
        ]
        self.register_buffer("lidar_cos", torch.cos(lidar_angles))
        self.register_buffer("lidar_sin", torch.sin(lidar_angles))

        # 1. Front RGB-D Camera Branch (64 depth + 64 target mask + angle -> 3-ch CoordConv + 6 polar)
        self.cam_conv = nn.Sequential(
            layer_init(nn.Conv1d(3, 16, kernel_size=5, stride=2, padding=2)),
            nn.SiLU(),
            layer_init(nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)),
            nn.SiLU(),
        )
        self.cam_proj = nn.Sequential(
            layer_init(nn.Linear(32 * 8 + 6, 64)),
            nn.LayerNorm(64),
            nn.Tanh(),
        )

        # 2. 360-deg LiDAR Branch (360 rays -> circular 1D CoordConv + Max/AvgPool(8) + 11 polar features)
        self.lidar_conv = nn.Sequential(
            _CircularConv1d(3, 16, kernel_size=7, stride=3, padding=3),
            nn.SiLU(),
            _CircularConv1d(16, 32, kernel_size=5, stride=3, padding=2),
            nn.SiLU(),
        )
        self.lidar_proj = nn.Sequential(
            layer_init(nn.Linear(32 * 16 + 11, 64)),
            nn.LayerNorm(64),
            nn.Tanh(),
        )

        # 3. Vehicle Proprioceptive State Branch (2 features: v, steer)
        self.state_net = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 16)),
            nn.LayerNorm(16),
            nn.Tanh(),
        )

        # 4. Fused Multi-Modal Spatial Representation (64 + 64 + 16 = 144 -> 128)
        self.fused_net = nn.Sequential(
            layer_init(nn.Linear(144, 128)),
            nn.Tanh(),
        )

        # 5. Temporal Memory (GRU over fused spatial features)
        self.gru = nn.GRU(128, self.gru_hidden_dim)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # 6. Decoupled Critic & Actor Heads (input = 128 current spatial + 128 GRU temporal = 256)
        rep_dim = 128 + self.gru_hidden_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(rep_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(rep_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.act_dim), std=0.01),
        )

        # Learnable log standard deviation for continuous Gaussian policy
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim) - 0.5)

    def init_hidden(self, batch_size=1, device=None):
        """Return zero-initialized GRU hidden state (1, batch_size, gru_hidden_dim)."""
        if device is None:
            device = self.v_min.device
        return torch.zeros((1, batch_size, self.gru_hidden_dim), dtype=torch.float32, device=device)

    def _extract_spatial_features(self, x):
        """Extract CoordConv + target-locked polar soft-argmin features for [N, obs_dim]."""
        cam_end = self.cam_dim
        mask_end = cam_end + self.cam_mask_dim
        lidar_end = mask_end + self.lidar_dim

        cam_feats = x[:, :cam_end]
        cam_mask = x[:, cam_end:mask_end]
        lidar_feats = x[:, mask_end:lidar_end]
        state_feats = x[:, lidar_end:]

        # Invert normalized ranges -> proximity (0.0 = empty max-range, 1.0 = close hit)
        cam_prox = 1.0 - cam_feats
        lidar_prox = 1.0 - lidar_feats

        # Target-locked polar soft-argmin + symmetric 3-sector obstacle proximity
        cam_target_vis = cam_mask.max(dim=-1, keepdim=True).values
        cam_w = torch.softmax(cam_mask * 15.0 + cam_prox * 3.0, dim=-1)
        cam_soft_bearing = (cam_w * self.cam_angles).sum(dim=-1, keepdim=True) * cam_target_vis
        cam_target_prox = (cam_prox * cam_mask).max(dim=-1, keepdim=True).values

        edge = self.cam_dim // 3
        cam_right_prox = cam_prox[:, :edge].max(dim=-1, keepdim=True).values
        cam_center_prox = cam_prox[:, edge : self.cam_dim - edge].max(dim=-1, keepdim=True).values
        cam_left_prox = cam_prox[:, self.cam_dim - edge :].max(dim=-1, keepdim=True).values
        cam_polar = torch.cat(
            [
                cam_soft_bearing,
                cam_target_prox,
                cam_target_vis,
                cam_right_prox,
                cam_center_prox,
                cam_left_prox,
            ],
            dim=-1,
        )

        cam_coord = self.cam_angles.unsqueeze(0).expand_as(cam_prox)
        cam_in = torch.stack([cam_prox, cam_mask, cam_coord], dim=1)  # [N, 3, 64]
        cam_conv_raw = self.cam_conv(cam_in)
        cam_conv_out = torch.cat(
            [
                F.adaptive_max_pool1d(cam_conv_raw, 4).flatten(start_dim=1),
                F.adaptive_avg_pool1d(cam_conv_raw, 4).flatten(start_dim=1),
            ],
            dim=-1,
        )
        h_cam = self.cam_proj(torch.cat([cam_conv_out, cam_polar], dim=-1))

        # Differentiable circular polar soft-argmin & 8-sector summaries for 360-deg LiDAR
        lidar_w = torch.softmax(lidar_prox * 12.0, dim=-1)
        lidar_soft_cos = (lidar_w * self.lidar_cos).sum(dim=-1, keepdim=True)
        lidar_soft_sin = (lidar_w * self.lidar_sin).sum(dim=-1, keepdim=True)
        lidar_max_prox = lidar_prox.max(dim=-1, keepdim=True).values
        sector_size = self.lidar_dim // 8
        lidar_sectors = lidar_prox[:, : sector_size * 8].view(-1, 8, sector_size).max(dim=-1).values
        lidar_polar = torch.cat(
            [lidar_soft_cos, lidar_soft_sin, lidar_max_prox, lidar_sectors], dim=-1
        )

        lidar_cos_exp = self.lidar_cos.unsqueeze(0).expand_as(lidar_prox)
        lidar_sin_exp = self.lidar_sin.unsqueeze(0).expand_as(lidar_prox)
        lidar_in = torch.stack([lidar_prox, lidar_cos_exp, lidar_sin_exp], dim=1)  # [N, 3, 360]
        lidar_conv_raw = self.lidar_conv(lidar_in)
        lidar_conv_out = torch.cat(
            [
                F.adaptive_max_pool1d(lidar_conv_raw, 8).flatten(start_dim=1),
                F.adaptive_avg_pool1d(lidar_conv_raw, 8).flatten(start_dim=1),
            ],
            dim=-1,
        )
        h_lidar = self.lidar_proj(torch.cat([lidar_conv_out, lidar_polar], dim=-1))

        h_state = self.state_net(state_feats)

        fused = torch.cat([h_cam, h_lidar, h_state], dim=-1)
        return self.fused_net(fused)

    def _extract_features(self, x, hx=None, episode_dones=None):
        """
        Encode observations into combined spatial + GRU temporal representation.

        Supports:
        - Step input `x`: [B, obs_dim] with optional `hx`: [1, B, H] -> returns ([B, 256], next_hx [1, B, H])
        - Sequence input `x`: [T, B, obs_dim] with `hx`: [1, B, H] and optional `episode_dones`: [T, B]
          -> returns ([T, B, 256], next_hx [1, B, H])
        """
        is_seq = x.dim() == 3
        if not is_seq:
            spatial = self._extract_spatial_features(x)  # [B, 128]
            bsz = spatial.shape[0]
            if hx is None:
                hx = self.init_hidden(bsz, device=spatial.device)
            gru_out, next_hx = self.gru(spatial.unsqueeze(0), hx)
            gru_out = gru_out.squeeze(0)
            if episode_dones is not None:
                mask = (1.0 - episode_dones).view(1, bsz, 1)
                next_hx = next_hx * mask
            return torch.cat([spatial, gru_out], dim=-1), next_hx

        seq_len, bsz, _ = x.shape
        spatial_flat = self._extract_spatial_features(x.reshape(seq_len * bsz, self.obs_dim))
        spatial_seq = spatial_flat.view(seq_len, bsz, 128)
        if hx is None:
            hx = self.init_hidden(bsz, device=x.device)

        if episode_dones is None or not torch.any(episode_dones[:-1] > 0.5):
            gru_seq, next_hx = self.gru(spatial_seq, hx)
            if episode_dones is not None:
                next_hx = next_hx * (1.0 - episode_dones[-1]).view(1, bsz, 1)
        else:
            outputs = []
            cur_hx = hx
            for t in range(seq_len):
                out_t, cur_hx = self.gru(spatial_seq[t : t + 1], cur_hx)
                outputs.append(out_t)
                cur_hx = cur_hx * (1.0 - episode_dones[t]).view(1, bsz, 1)
            gru_seq = torch.cat(outputs, dim=0)
            next_hx = cur_hx

        return torch.cat([spatial_seq, gru_seq], dim=-1), next_hx

    def get_value(self, x, hx=None, episode_dones=None):
        features, _ = self._extract_features(x, hx=hx, episode_dones=episode_dones)
        return self.critic(features)

    def scale_action(self, norm_action):
        """Map normalized action [-1, 1] to physical vehicle commands."""
        v = norm_action[..., 0] * self.v_scale + self.v_bias
        steer = norm_action[..., 1] * self.steer_scale
        return torch.stack([v, steer], dim=-1)

    def unscale_action(self, phys_action):
        """Map physical commands back to normalized [-1, 1] range."""
        norm_v = (phys_action[..., 0] - self.v_bias) / self.v_scale
        norm_steer = phys_action[..., 1] / self.steer_scale
        return torch.stack([norm_v, norm_steer], dim=-1)

    def _compute_action_mean(self, features):
        # Bound pre-tanh Gaussian mean to [-3.0, 3.0] so tanh reaches +/-0.995 without inf atanh
        return 3.0 * torch.tanh(self.actor_mean(features))

    @staticmethod
    def _tanh_log_det_jacobian(u: torch.Tensor) -> torch.Tensor:
        """Numerically stable log|det(d tanh(u) / du)| = sum log(1 - tanh^2(u))."""
        return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))

    def get_action_and_value(
        self,
        x,
        action=None,
        hx=None,
        episode_dones=None,
        return_hidden=False,
        raw_action=None,
        return_raw_action=False,
    ):
        features, next_hx = self._extract_features(x, hx=hx, episode_dones=episode_dones)
        raw_mean = self._compute_action_mean(features)

        # Clamp logstd so exploration std stays in [0.08, 0.70] and never inflates uncontrollably
        action_logstd = torch.clamp(self.actor_logstd.expand_as(raw_mean), -2.5, -0.35)
        action_std = torch.exp(action_logstd)

        dist = Normal(raw_mean, action_std)

        if action is None and raw_action is None:
            u = dist.sample()
            norm_action = torch.tanh(u)
            action = self.scale_action(norm_action)
        elif raw_action is not None:
            u = raw_action.unsqueeze(0) if raw_action.dim() == 1 else raw_action
            if action is None:
                action = self.scale_action(torch.tanh(u))
        else:
            if action.dim() == 1:
                action = action.unsqueeze(0)
            norm_action = torch.clamp(self.unscale_action(action), -0.999999, 0.999999)
            u = torch.atanh(norm_action)

        # Exact squashed-Gaussian log-prob using numerically stable tanh Jacobian
        log_prob = (dist.log_prob(u) - self._tanh_log_det_jacobian(u)).sum(-1)
        # True squashed policy entropy via 1-sample reparameterized Monte Carlo
        u_ent = dist.rsample()
        entropy = -(dist.log_prob(u_ent) - self._tanh_log_det_jacobian(u_ent)).sum(-1)
        value = self.critic(features)

        if return_hidden and return_raw_action:
            return action, log_prob, entropy, value, next_hx, u
        if return_hidden:
            return action, log_prob, entropy, value, next_hx
        if return_raw_action:
            return action, log_prob, entropy, value, u
        return action, log_prob, entropy, value

    def get_deterministic_action(self, x, hx=None, return_hidden=False):
        """Greedy inference for evaluation (no exploration noise)."""
        with torch.no_grad():
            features, next_hx = self._extract_features(x, hx=hx)
            norm_mean = torch.tanh(self._compute_action_mean(features))
            phys_action = self.scale_action(norm_mean)
            if return_hidden:
                return phys_action, next_hx
            return phys_action
