"""Shared 490-D spatial sensor encoder (CoordConv + Polar Soft-Argmin) for AckBot."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ackerman_follow.config import RobotConfig, get_robot_config


def layer_init(layer: nn.Module, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal weight initialization with constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class _CircularConv1d(nn.Module):
    """1D convolution with explicit circular padding (MPS- and CPU-safe)."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ) -> None:
        super().__init__()
        self.pad = padding
        self.conv = layer_init(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad), mode="circular")
        return self.conv(x)


class SpatialSensorEncoder(nn.Module):
    """
    Encodes the canonical 490-D AckBot observation into a compact spatial latent vector:
      - [0 : 64]    Front camera depth (normalized [0, 1])
      - [64 : 128]  Front camera RGB target mask ({0.0, 1.0})
      - [128 : 488] 360-deg planar LiDAR (normalized [0, 1])
      - [488 : 490] Proprioceptive state [v_norm, steer_norm] in [-1, 1]
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        cam_dim: int | None = None,
        cam_mask_dim: int | None = None,
        lidar_dim: int | None = None,
        latent_dim: int = 128,
        robot_cfg: RobotConfig | None = None,
    ) -> None:
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
        self.state_dim = self.obs_dim - self.cam_dim - self.cam_mask_dim - self.lidar_dim
        self.latent_dim = int(latent_dim)

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

        # 4. Fused Multi-Modal Spatial Representation (64 + 64 + 16 = 144 -> latent_dim)
        self.fused_net = nn.Sequential(
            layer_init(nn.Linear(144, self.latent_dim)),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `[..., obs_dim]` observations into `[..., latent_dim]` embeddings."""
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.obs_dim)

        cam_end = self.cam_dim
        mask_end = cam_end + self.cam_mask_dim
        lidar_end = mask_end + self.lidar_dim

        cam_feats = x_flat[:, :cam_end]
        cam_mask = x_flat[:, cam_end:mask_end]
        lidar_feats = x_flat[:, mask_end:lidar_end]
        state_feats = x_flat[:, lidar_end:]

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
        cam_in = torch.stack([cam_prox, cam_mask, cam_coord], dim=1)
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
        lidar_in = torch.stack([lidar_prox, lidar_cos_exp, lidar_sin_exp], dim=1)
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
        fused = self.fused_net(torch.cat([h_cam, h_lidar, h_state], dim=-1))
        return fused.view(*orig_shape, self.latent_dim)
