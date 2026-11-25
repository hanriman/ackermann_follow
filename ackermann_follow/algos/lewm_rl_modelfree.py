"""
LeWM + Model-Free RL (`lewm_rl_modelfree`):
Combines a LeWorldModel (JEPA) latent state encoder/predictor with either:
  - `policy_type="ppo"` (on-policy recurrent actor-critic + auxiliary JEPA loss)
  - `policy_type="sac"` (off-policy twin-Q SAC + auto-entropy alpha + auxiliary JEPA loss)
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from ackermann_follow.algos.base import BaseAgent
from ackermann_follow.buffers.rollout import RolloutBuffer
from ackermann_follow.buffers.sequence_replay import SequenceReplayBuffer
from ackermann_follow.config import RobotConfig
from ackermann_follow.models.heads import TanhGaussianActor, TwinQNet, ValueCritic
from ackermann_follow.models.world_models.lewm_jepa import LeWMJEPA


class LeWMModelFreeAgent(BaseAgent):
    """Model-free RL (PPO or SAC) operating on LeWM JEPA latent representations."""

    algo_name = "lewm_rl_modelfree"

    def __init__(
        self,
        obs_dim: int | None = None,
        act_dim: int = 2,
        latent_dim: int = 128,
        gru_hidden_dim: int = 128,
        policy_type: str = "ppo",
        freeze_wm: bool = False,
        jepa_aux_coef: float = 0.25,
        ema_tau: float = 0.99,
        sigreg_coef: float = 0.25,
        use_recurrent_belief: bool = True,
        gamma: float = 0.99,
        sac_tau: float = 0.005,
        init_alpha: float = 0.2,
        robot_cfg: RobotConfig | None = None,
    ) -> None:
        super().__init__(robot_cfg=robot_cfg)
        cfg = self.robot_cfg
        self.obs_dim = int(obs_dim if obs_dim is not None else cfg.obs_dim)
        self.act_dim = int(act_dim)
        self.latent_dim = int(latent_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.policy_type = str(policy_type).lower()
        if self.policy_type not in ("ppo", "sac"):
            raise ValueError(f"Unsupported policy_type={policy_type!r}; choose 'ppo' or 'sac'.")
        self.training_mode = "on_policy" if self.policy_type == "ppo" else "off_policy"
        self.freeze_wm = bool(freeze_wm)
        self.jepa_aux_coef = float(jepa_aux_coef)
        self.gamma = float(gamma)
        self.sac_tau = float(sac_tau)

        self.wm = LeWMJEPA(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            latent_dim=self.latent_dim,
            ema_tau=ema_tau,
            sigreg_coef=sigreg_coef,
            use_recurrent_belief=use_recurrent_belief,
            robot_cfg=cfg,
        )
        if self.freeze_wm:
            for p in self.wm.parameters():
                p.requires_grad_(False)

        self.gru = nn.GRU(self.latent_dim, self.gru_hidden_dim)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        rep_dim = self.latent_dim + self.gru_hidden_dim
        self.actor = TanhGaussianActor(
            rep_dim,
            act_dim=self.act_dim,
            hidden_dims=(128, 64),
            state_dependent_std=(self.policy_type == "sac"),
            robot_cfg=cfg,
        )

        if self.policy_type == "ppo":
            self.critic = ValueCritic(rep_dim, hidden_dims=(128, 64))
            self.q_net = None
            self.target_q_net = None
            self.log_alpha = None
        else:
            self.critic = None
            self.q_net = TwinQNet(rep_dim, act_dim=self.act_dim, hidden_dim=256, num_bins=1)
            self.target_q_net = copy.deepcopy(self.q_net)
            for p in self.target_q_net.parameters():
                p.requires_grad_(False)
            self.log_alpha = nn.Parameter(torch.tensor(math_log := float(np.log(init_alpha)), dtype=torch.float32))
            del math_log
            self.target_entropy = -float(self.act_dim)

        self._init_kwargs = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "latent_dim": self.latent_dim,
            "gru_hidden_dim": self.gru_hidden_dim,
            "policy_type": self.policy_type,
            "freeze_wm": self.freeze_wm,
            "jepa_aux_coef": self.jepa_aux_coef,
            "ema_tau": float(ema_tau),
            "sigreg_coef": float(sigreg_coef),
            "use_recurrent_belief": bool(use_recurrent_belief),
            "gamma": self.gamma,
            "sac_tau": self.sac_tau,
            "init_alpha": float(init_alpha),
        }

    def export_metadata(self) -> dict[str, Any]:
        return dict(self._init_kwargs)

    def load_world_model_weights(self, wm_ckpt_path: str) -> None:
        """Optionally initialize `self.wm` from a pretrained `lewm` checkpoint."""
        ckpt = torch.load(wm_ckpt_path, map_location=self.v_min.device, weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        wm_sd = {k[len("wm."):]: v for k, v in sd.items() if k.startswith("wm.")}
        if wm_sd:
            self.wm.load_state_dict(wm_sd, strict=False)

    def init_state(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = self.v_min.device
        return torch.zeros((1, batch_size, self.gru_hidden_dim), dtype=torch.float32, device=device)

    def _encode_features(
        self,
        obs: torch.Tensor,
        hx: torch.Tensor | None = None,
        episode_dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_seq = obs.dim() == 3
        with torch.set_grad_enabled((not self.freeze_wm) and torch.is_grad_enabled()):
            z_obs = self.wm.encode(obs)
        if self.freeze_wm:
            z_obs = z_obs.detach()

        if not is_seq:
            bsz = z_obs.shape[0]
            if hx is None:
                hx = self.init_state(bsz, device=z_obs.device)
            gru_out, next_hx = self.gru(z_obs.unsqueeze(0), hx)
            if episode_dones is not None:
                next_hx = next_hx * (1.0 - episode_dones).view(1, bsz, 1)
            return torch.cat([z_obs, gru_out.squeeze(0)], dim=-1), next_hx, z_obs

        seq_len, bsz, _ = z_obs.shape
        if hx is None:
            hx = self.init_state(bsz, device=z_obs.device)
        if episode_dones is None or not torch.any(episode_dones[:-1] > 0.5):
            gru_seq, next_hx = self.gru(z_obs, hx)
            if episode_dones is not None:
                next_hx = next_hx * (1.0 - episode_dones[-1]).view(1, bsz, 1)
        else:
            outs = []
            cur_hx = hx
            for t in range(seq_len):
                out_t, cur_hx = self.gru(z_obs[t : t + 1], cur_hx)
                outs.append(out_t)
                cur_hx = cur_hx * (1.0 - episode_dones[t]).view(1, bsz, 1)
            gru_seq = torch.cat(outs, dim=0)
            next_hx = cur_hx
        return torch.cat([z_obs, gru_seq], dim=-1), next_hx, z_obs

    def get_value(
        self,
        obs: torch.Tensor,
        hx: torch.Tensor | None = None,
        episode_dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.critic is None:
            raise RuntimeError("get_value() is only available when policy_type='ppo'.")
        feats, _, _ = self._encode_features(obs, hx=hx, episode_dones=episode_dones)
        return self.critic(feats)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
        episode_dones: torch.Tensor | None = None,
        return_hidden: bool = False,
        raw_action: torch.Tensor | None = None,
        return_raw_action: bool = False,
    ):
        feats, next_hx, _ = self._encode_features(obs, hx=hx, episode_dones=episode_dones)
        out_action, log_prob, entropy, u = self.actor.sample(
            feats, raw_action=raw_action, action=action, return_physical=True
        )
        value = self.critic(feats) if self.critic is not None else torch.zeros_like(log_prob).unsqueeze(-1)
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
        feats, next_hx, _ = self._encode_features(obs, hx=state)
        if deterministic:
            return self.actor.deterministic(feats, return_physical=True), next_hx
        action, _, _, _ = self.actor.sample(feats, return_physical=True)
        return action, next_hx

    def update_on_policy(
        self,
        buffer: RolloutBuffer,
        optimizer: torch.optim.Optimizer,
        algo_cfg: dict[str, Any],
    ) -> dict[str, float]:
        """On-policy PPO update + optional auxiliary JEPA loss over sequence chunks."""
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

        pg_losses, val_losses, entropy_losses, jepa_losses = [], [], [], []
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

                # Auxiliary multi-step JEPA latent prediction + SIGReg over sequence chunk
                jepa_val = 0.0
                if (not self.freeze_wm) and self.jepa_aux_coef > 0.0 and b_obs.shape[0] > 2:
                    j_horizon = min(6, b_obs.shape[0] - 1)
                    act_norm = self.unscale_action(b_actions[:j_horizon]).clamp(-1.0, 1.0)
                    ep_sub = b_ep_dones[:j_horizon]
                    valid_mask = torch.ones_like(ep_sub)
                    for t in range(1, j_horizon):
                        valid_mask[t] = valid_mask[t - 1] * (1.0 - ep_sub[t - 1])
                    jepa_loss, _ = self.wm.compute_jepa_loss(
                        obs_seq=b_obs[:j_horizon],
                        next_obs_seq=b_obs[1 : j_horizon + 1],
                        actions_norm_seq=act_norm,
                        valid_mask=valid_mask,
                        episode_dones=ep_sub,
                    )
                    loss = loss + self.jepa_aux_coef * jepa_loss
                    jepa_val = float(jepa_loss.item())

                if not torch.isfinite(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()
                if not self.freeze_wm:
                    self.wm.update_ema()

                pg_losses.append(pg_loss.item())
                val_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                jepa_losses.append(jepa_val)

            if len(epoch_kls) > 0 and float(np.mean(epoch_kls)) > target_kl:
                break

        return {
            "loss_pi": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "loss_v": float(np.mean(val_losses)) if val_losses else 0.0,
            "entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "jepa_aux": float(np.mean(jepa_losses)) if jepa_losses else 0.0,
        }

    def update_from_replay(
        self,
        replay: SequenceReplayBuffer,
        optimizer: torch.optim.Optimizer,
        algo_cfg: dict[str, Any],
    ) -> dict[str, float]:
        """Off-policy SAC + auxiliary JEPA update when `policy_type='sac'`."""
        if self.q_net is None or self.target_q_net is None or self.log_alpha is None:
            raise RuntimeError("update_from_replay() requires policy_type='sac'.")

        batch_size = int(algo_cfg.get("batch_size", 64))
        horizon = int(algo_cfg.get("horizon", 8))
        grad_steps = int(algo_cfg.get("grad_steps_per_iter", 8))
        max_grad_norm = float(algo_cfg.get("max_grad_norm", 10.0))

        if not replay.can_sample(horizon):
            return {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0}

        pi_losses, q_losses, ent_vals = [], [], []
        for _ in range(grad_steps):
            batch = replay.sample_sequences(batch_size=batch_size, horizon=horizon)
            obs_seq = batch["obs"]
            next_obs_seq = batch["next_obs"]
            act_norm_seq = self.unscale_action(batch["actions"]).clamp(-1.0, 1.0)
            rewards = batch["rewards"]
            terminateds = batch["terminateds"]
            ep_dones = batch["episode_dones"]
            valid_mask = batch["valid_mask"]

            # Encode recurrent features along sequence
            feats, _, _ = self._encode_features(obs_seq, episode_dones=ep_dones)
            with torch.no_grad():
                next_feats, _, _ = self._encode_features(next_obs_seq, episode_dones=ep_dones)
                next_act_norm, next_logp, _, _ = self.actor.sample(next_feats, return_physical=False)
                alpha = self.log_alpha.exp().detach()
                target_q = self.target_q_net.min_q(next_feats, next_act_norm) - alpha * next_logp
                td_target = rewards + self.gamma * (1.0 - terminateds) * target_q

            q1, q2 = self.q_net.forward(feats, act_norm_seq)
            q_err = (q1 - td_target).pow(2) + (q2 - td_target).pow(2)
            q_loss = (q_err * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

            # Policy + temperature update on detached latent features
            feats_det = feats.detach()
            pi_act_norm, pi_logp, pi_ent, _ = self.actor.sample(feats_det, return_physical=False)
            q_pi = self.q_net.min_q(feats_det, pi_act_norm)
            alpha_det = self.log_alpha.exp().detach()
            pi_loss_t = alpha_det * pi_logp - q_pi
            pi_loss = (pi_loss_t * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

            alpha_loss_t = -self.log_alpha * (pi_logp.detach() + self.target_entropy)
            alpha_loss = (alpha_loss_t * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

            loss = q_loss + pi_loss + alpha_loss
            if (not self.freeze_wm) and self.jepa_aux_coef > 0.0:
                jepa_loss, _ = self.wm.compute_jepa_loss(
                    obs_seq=obs_seq,
                    next_obs_seq=next_obs_seq,
                    actions_norm_seq=act_norm_seq,
                    valid_mask=valid_mask,
                    episode_dones=ep_dones,
                )
                loss = loss + self.jepa_aux_coef * jepa_loss

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            optimizer.step()

            if not self.freeze_wm:
                self.wm.update_ema()
            with torch.no_grad():
                for p, p_tgt in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                    p_tgt.data.mul_(1.0 - self.sac_tau).add_(p.data, alpha=self.sac_tau)

            pi_losses.append(float(pi_loss.item()))
            q_losses.append(float(q_loss.item()))
            ent_vals.append(float(pi_ent.mean().item()))

        return {
            "loss_pi": float(np.mean(pi_losses)) if pi_losses else 0.0,
            "loss_v": float(np.mean(q_losses)) if q_losses else 0.0,
            "entropy": float(np.mean(ent_vals)) if ent_vals else 0.0,
        }


def build_lewm_rl_modelfree_from_cfg(
    obs_dim: int,
    act_dim: int,
    algo_cfg: dict[str, Any],
    robot_cfg: RobotConfig | None = None,
) -> LeWMModelFreeAgent:
    return LeWMModelFreeAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=int(algo_cfg.get("latent_dim", 128)),
        gru_hidden_dim=int(algo_cfg.get("gru_hidden_dim", 128)),
        policy_type=str(algo_cfg.get("policy_type", "ppo")),
        freeze_wm=bool(algo_cfg.get("freeze_wm", False)),
        jepa_aux_coef=float(algo_cfg.get("jepa_aux_coef", 0.25)),
        ema_tau=float(algo_cfg.get("ema_tau", 0.99)),
        sigreg_coef=float(algo_cfg.get("sigreg_coef", 0.25)),
        use_recurrent_belief=bool(algo_cfg.get("use_recurrent_belief", True)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        sac_tau=float(algo_cfg.get("sac_tau", 0.005)),
        init_alpha=float(algo_cfg.get("init_alpha", 0.2)),
        robot_cfg=robot_cfg,
    )
