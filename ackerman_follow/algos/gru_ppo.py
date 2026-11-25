"""Recurrent Spatial PPO (GRU PPO) algorithm built on SpatialSensorEncoder + GRU."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from ackerman_follow.buffers.rollout import RolloutBuffer
from ackerman_follow.config import RobotConfig, get_robot_config
from ackerman_follow.algos.base import BaseAgent
from ackerman_follow.models.encoders import SpatialSensorEncoder
from ackerman_follow.models.heads import TanhGaussianActor, ValueCritic


def remap_legacy_actor_critic_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Map legacy `ActorCritic` state dict keys (`cam_conv.*`, `actor_mean.*`, `critic.*`, ...)
    to modular `GRUPPOAgent` keys (`encoder.*`, `actor.*`, `critic.net.*`).
    """
    if "encoder.cam_angles" in state_dict:
        return state_dict

    encoder_prefixes = (
        "cam_angles",
        "lidar_cos",
        "lidar_sin",
        "cam_conv.",
        "cam_proj.",
        "lidar_conv.",
        "lidar_proj.",
        "state_net.",
        "fused_net.",
    )
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(encoder_prefixes):
            remapped[f"encoder.{k}"] = v
        elif k.startswith("actor_mean."):
            # legacy Sequential indices: 0, 2 -> actor.trunk.0, 2; 4 -> actor.mean_head
            suffix = k[len("actor_mean.") :]
            if suffix.startswith("4."):
                remapped[f"actor.mean_head.{suffix[2:]}"] = v
            else:
                remapped[f"actor.trunk.{suffix}"] = v
        elif k == "actor_logstd":
            remapped["actor.logstd"] = v
        elif k.startswith("critic."):
            remapped[f"critic.net.{k[len('critic.'):]}"] = v
        else:
            remapped[k] = v

    for buf_name in ("v_min", "v_max", "steer_max", "v_scale", "v_bias", "steer_scale"):
        if buf_name in state_dict and f"actor.{buf_name}" not in remapped:
            remapped[f"actor.{buf_name}"] = state_dict[buf_name]
    return remapped


class GRUPPOAgent(BaseAgent):
    """Recurrent Spatial Actor-Critic trained with truncated-BPTT PPO."""

    algo_name = "gru_ppo"
    training_mode = "on_policy"

    def __init__(
        self,
        obs_dim: int | None = None,
        act_dim: int | None = None,
        cam_dim: int | None = None,
        cam_mask_dim: int | None = None,
        lidar_dim: int | None = None,
        gru_hidden_dim: int = 128,
        robot_cfg: RobotConfig | None = None,
    ) -> None:
        super().__init__(robot_cfg=robot_cfg)
        cfg = self.robot_cfg
        self.obs_dim = int(obs_dim if obs_dim is not None else cfg.obs_dim)
        self.act_dim = int(act_dim if act_dim is not None else cfg.act_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)

        self.encoder = SpatialSensorEncoder(
            obs_dim=self.obs_dim,
            cam_dim=cam_dim,
            cam_mask_dim=cam_mask_dim,
            lidar_dim=lidar_dim,
            latent_dim=128,
            robot_cfg=cfg,
        )
        self.gru = nn.GRU(128, self.gru_hidden_dim)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        rep_dim = 128 + self.gru_hidden_dim
        self.critic = ValueCritic(rep_dim, hidden_dims=(128, 64))
        self.actor = TanhGaussianActor(
            rep_dim,
            act_dim=self.act_dim,
            hidden_dims=(128, 64),
            state_dependent_std=False,
            robot_cfg=cfg,
        )

    def export_metadata(self) -> dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "gru_hidden_dim": self.gru_hidden_dim,
        }

    def init_state(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = self.v_min.device
        return torch.zeros((1, batch_size, self.gru_hidden_dim), dtype=torch.float32, device=device)

    def init_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        return self.init_state(batch_size=batch_size, device=device)

    def _extract_features(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
        episode_dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_seq = x.dim() == 3
        if not is_seq:
            spatial = self.encoder(x)
            bsz = spatial.shape[0]
            if hx is None:
                hx = self.init_state(bsz, device=spatial.device)
            gru_out, next_hx = self.gru(spatial.unsqueeze(0), hx)
            gru_out = gru_out.squeeze(0)
            if episode_dones is not None:
                next_hx = next_hx * (1.0 - episode_dones).view(1, bsz, 1)
            return torch.cat([spatial, gru_out], dim=-1), next_hx

        seq_len, bsz, _ = x.shape
        spatial_seq = self.encoder(x)
        if hx is None:
            hx = self.init_state(bsz, device=x.device)

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

    def get_value(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
        episode_dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features, _ = self._extract_features(x, hx=hx, episode_dones=episode_dones)
        return self.critic(features)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
        episode_dones: torch.Tensor | None = None,
        return_hidden: bool = False,
        raw_action: torch.Tensor | None = None,
        return_raw_action: bool = False,
    ):
        features, next_hx = self._extract_features(x, hx=hx, episode_dones=episode_dones)
        if action is not None and action.dim() == 1:
            action = action.unsqueeze(0)
        if raw_action is not None and raw_action.dim() == 1:
            raw_action = raw_action.unsqueeze(0)

        out_action, log_prob, entropy, u = self.actor.sample(
            features,
            raw_action=raw_action,
            action=action,
            return_physical=True,
        )
        value = self.critic(features)

        if return_hidden and return_raw_action:
            return out_action, log_prob, entropy, value, next_hx, u
        if return_hidden:
            return out_action, log_prob, entropy, value, next_hx
        if return_raw_action:
            return out_action, log_prob, entropy, value, u
        return out_action, log_prob, entropy, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            features, next_hx = self._extract_features(obs, hx=state)
            return self.actor.deterministic(features, return_physical=True), next_hx
        action, _, _, _, next_hx = self.get_action_and_value(obs, hx=state, return_hidden=True)
        return action, next_hx

    def update_on_policy(
        self,
        buffer: RolloutBuffer,
        optimizer: torch.optim.Optimizer,
        algo_cfg: dict[str, Any],
    ) -> dict[str, float]:
        """Run PPO epochs with sequence-chunk BPTT over a filled RolloutBuffer."""
        n_epochs = int(algo_cfg.get("n_epochs", 6))
        seq_len = int(algo_cfg.get("seq_len", 64))
        num_mini_batches = int(algo_cfg.get("num_mini_batches", 8))
        clip_range = float(algo_cfg.get("clip_range", 0.2))
        clip_range_vf = float(algo_cfg.get("clip_range_vf", 5.0))
        ent_coef = float(algo_cfg.get("ent_coef", 0.001))
        vf_coef = float(algo_cfg.get("vf_coef", 0.05))
        max_grad_norm = float(algo_cfg.get("max_grad_norm", 0.5))
        target_kl = float(algo_cfg.get("target_kl", 0.03))

        b_adv_all = buffer.advantages
        buffer.advantages = (b_adv_all - b_adv_all.mean()) / (b_adv_all.std() + 1e-8)

        pg_losses, val_losses, entropy_losses = [], [], []
        for _ in range(n_epochs):
            epoch_kls = []
            for (
                b_obs,
                b_actions,
                b_logprobs,
                b_adv,
                b_returns,
                b_vals,
                b_init_hx,
                b_ep_dones,
                b_raw_actions,
            ) in buffer.get_recurrent_batches(
                num_mini_batches=num_mini_batches,
                seq_len=seq_len,
                include_raw_actions=True,
            ):
                _, new_logprob, entropy, new_value = self.get_action_and_value(
                    b_obs,
                    action=b_actions,
                    raw_action=b_raw_actions,
                    hx=b_init_hx,
                    episode_dones=b_ep_dones,
                )

                new_logprob_f = new_logprob.reshape(-1)
                b_logprobs_f = b_logprobs.reshape(-1)
                b_adv_f = b_adv.reshape(-1)
                b_returns_f = b_returns.reshape(-1)
                b_vals_f = b_vals.reshape(-1)
                new_value_f = new_value.reshape(-1)
                entropy_f = entropy.reshape(-1)

                logratio = torch.clamp(new_logprob_f - b_logprobs_f, -20.0, 20.0)
                with torch.no_grad():
                    approx_kl = ((logratio.exp() - 1) - logratio).mean()
                    epoch_kls.append(approx_kl.item())

                ratio = logratio.exp()
                pg_loss1 = -b_adv_f * ratio
                pg_loss2 = -b_adv_f * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_value_f - b_returns_f) ** 2
                v_clipped = b_vals_f + torch.clamp(
                    new_value_f - b_vals_f, -clip_range_vf, clip_range_vf
                )
                v_loss_clipped = (v_clipped - b_returns_f) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy_f.mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss
                if not torch.isfinite(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                val_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())

            if len(epoch_kls) > 0 and float(np.mean(epoch_kls)) > target_kl:
                break

        return {
            "loss_pi": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "loss_v": float(np.mean(val_losses)) if val_losses else 0.0,
            "entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
        }


def build_gru_ppo_from_cfg(
    obs_dim: int,
    act_dim: int,
    algo_cfg: dict[str, Any],
    robot_cfg: RobotConfig | None = None,
) -> GRUPPOAgent:
    cfg = robot_cfg if robot_cfg is not None else get_robot_config()
    return GRUPPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        gru_hidden_dim=int(algo_cfg.get("gru_hidden_dim", 128)),
        robot_cfg=cfg,
    )
