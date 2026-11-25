"""
LeWM + TD-MPC2 (`lewm_tdmpc2`):
Combines LeWorldModel (JEPA + SIGReg latent dynamics) with TD-MPC2:
  - Distributional Two-Hot symlog RewardHead & TwinQNet (+ EMA Target Q)
  - TerminationHead for collision risk avoidance
  - Squashed Gaussian Policy Prior maximizing Q + entropy
  - Test-time MPPI trajectory optimization over imagined JEPA rollouts
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ackermann_follow.algos.base import BaseAgent
from ackermann_follow.buffers.sequence_replay import SequenceReplayBuffer
from ackermann_follow.config import RobotConfig
from ackermann_follow.models.heads import (
    RewardHead,
    TanhGaussianActor,
    TerminationHead,
    TwinQNet,
    two_hot_encode,
)
from ackermann_follow.models.world_models.lewm_jepa import LeWMJEPA
from ackermann_follow.planners.mppi import MPPIPlanner


class LeWMTDMPC2Agent(BaseAgent):
    """Joint JEPA World Model + TD-MPC2 value/policy learning and MPPI planning."""

    algo_name = "lewm_tdmpc2"
    training_mode = "off_policy"

    def __init__(
        self,
        obs_dim: int | None = None,
        act_dim: int = 2,
        latent_dim: int = 128,
        predictor_hidden: int = 256,
        ema_tau: float = 0.99,
        q_ema_tau: float = 0.01,
        sigreg_coef: float = 0.25,
        pred_decay: float = 0.90,
        use_recurrent_belief: bool = True,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        gamma: float = 0.99,
        jepa_coef: float = 2.0,
        reward_coef: float = 1.0,
        done_coef: float = 0.5,
        value_coef: float = 0.5,
        entropy_coef: float = 1e-4,
        horizon: int = 5,
        planner_samples: int = 256,
        planner_elites: int = 32,
        planner_iterations: int = 4,
        num_pi_trajs: int = 16,
        temperature: float = 0.5,
        robot_cfg: RobotConfig | None = None,
    ) -> None:
        super().__init__(robot_cfg=robot_cfg)
        cfg = self.robot_cfg
        self.obs_dim = int(obs_dim if obs_dim is not None else cfg.obs_dim)
        self.act_dim = int(act_dim)
        self.latent_dim = int(latent_dim)
        self.q_ema_tau = float(q_ema_tau)
        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.gamma = float(gamma)
        self.jepa_coef = float(jepa_coef)
        self.reward_coef = float(reward_coef)
        self.done_coef = float(done_coef)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.horizon = int(horizon)

        self.wm = LeWMJEPA(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            latent_dim=self.latent_dim,
            predictor_hidden=predictor_hidden,
            ema_tau=ema_tau,
            sigreg_coef=sigreg_coef,
            pred_decay=pred_decay,
            use_recurrent_belief=use_recurrent_belief,
            robot_cfg=cfg,
        )
        self.reward_head = RewardHead(
            latent_dim=self.latent_dim,
            act_dim=self.act_dim,
            hidden_dim=256,
            num_bins=self.num_bins,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        self.termination_head = TerminationHead(latent_dim=self.latent_dim, hidden_dim=128)
        self.q_net = TwinQNet(
            latent_dim=self.latent_dim,
            act_dim=self.act_dim,
            hidden_dim=256,
            num_bins=self.num_bins,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        self.target_q_net = copy.deepcopy(self.q_net)
        for p in self.target_q_net.parameters():
            p.requires_grad_(False)

        self.policy_prior = TanhGaussianActor(
            self.latent_dim,
            act_dim=self.act_dim,
            hidden_dims=(256, 128),
            state_dependent_std=True,
            robot_cfg=cfg,
        )
        self.planner = MPPIPlanner(
            act_dim=self.act_dim,
            horizon=self.horizon,
            num_samples=planner_samples,
            num_elites=planner_elites,
            num_iterations=planner_iterations,
            num_pi_trajs=num_pi_trajs,
            temperature=temperature,
            gamma=self.gamma,
        )

        self._init_kwargs = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "latent_dim": self.latent_dim,
            "predictor_hidden": int(predictor_hidden),
            "ema_tau": float(ema_tau),
            "q_ema_tau": self.q_ema_tau,
            "sigreg_coef": float(sigreg_coef),
            "pred_decay": float(pred_decay),
            "use_recurrent_belief": bool(use_recurrent_belief),
            "num_bins": self.num_bins,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "gamma": self.gamma,
            "jepa_coef": self.jepa_coef,
            "reward_coef": self.reward_coef,
            "done_coef": self.done_coef,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "horizon": self.horizon,
            "planner_samples": int(planner_samples),
            "planner_elites": int(planner_elites),
            "planner_iterations": int(planner_iterations),
            "num_pi_trajs": int(num_pi_trajs),
            "temperature": float(temperature),
        }

    def export_metadata(self) -> dict[str, Any]:
        return dict(self._init_kwargs)

    def init_state(self, batch_size: int = 1, device: torch.device | None = None) -> tuple[torch.Tensor | None, torch.Tensor]:
        if device is None:
            device = self.v_min.device
        belief = self.wm.init_belief(batch_size, device)
        prev_act = torch.zeros((batch_size, self.act_dim), dtype=torch.float32, device=device)
        self.planner.reset()
        return (belief, prev_act)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor] | None = None,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor]]:
        bsz = obs.shape[0]
        if state is None:
            belief, prev_act = self.init_state(bsz, obs.device)
        else:
            belief, prev_act = state

        z_t, next_belief = self.wm.step_belief(obs, prev_action_norm=prev_act, belief=belief)
        norm_act = self.planner.plan(
            z0=z_t,
            predict_next_fn=self.wm.predict_next,
            reward_fn=self.reward_head,
            value_fn=self.q_net.min_q,
            policy_fn=lambda z: self.policy_prior.deterministic(z, return_physical=False),
            termination_fn=self.termination_head,
            deterministic=deterministic,
        )
        phys_act = self.scale_action(norm_act)
        return phys_act, (next_belief, norm_act)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        obs_seq = batch["obs"]
        next_obs_seq = batch["next_obs"]
        act_norm_seq = self.unscale_action(batch["actions"]).clamp(-1.0, 1.0)
        rewards = batch["rewards"]
        terminateds = batch["terminateds"]
        ep_dones = batch["episode_dones"]
        valid_mask = batch["valid_mask"]
        horizon = obs_seq.shape[0]

        # 1. JEPA Latent Consistency + SIGReg Loss
        jepa_loss, jepa_info = self.wm.compute_jepa_loss(
            obs_seq=obs_seq,
            next_obs_seq=next_obs_seq,
            actions_norm_seq=act_norm_seq,
            valid_mask=valid_mask,
            episode_dones=ep_dones,
        )

        z_state_seq = jepa_info["z_state_seq"]
        pred_latents = jepa_info["pred_latents"]
        z_rollout = torch.cat([z_state_seq[:1], pred_latents[:-1]], dim=0)  # [H, B, D]

        step_weights = torch.tensor(
            [math.pow(self.wm.pred_decay, t) for t in range(horizon)],
            dtype=obs_seq.dtype,
            device=obs_seq.device,
        ).unsqueeze(-1)
        weights = valid_mask * step_weights

        # 2. Two-Hot Symlog Reward Loss along imagined JEPA rollout
        rew_logits = self.reward_head.forward_logits(z_rollout, act_norm_seq)
        rew_tgt = two_hot_encode(rewards, num_bins=self.num_bins, vmin=self.vmin, vmax=self.vmax)
        rew_ce = -(rew_tgt * F.log_softmax(rew_logits, dim=-1)).sum(dim=-1)
        rew_loss = (rew_ce * weights).sum() / weights.sum().clamp_min(1.0)

        # 3. Termination Loss
        done_logits = self.termination_head.forward_logit(pred_latents)
        done_bce = F.binary_cross_entropy_with_logits(done_logits, terminateds, reduction="none")
        done_loss = (done_bce * weights).sum() / weights.sum().clamp_min(1.0)

        # 4. Two-Hot TD Value Loss using EMA Target Q on real next_obs EMA latents
        with torch.no_grad():
            z_next_tgt = self.wm.encode_target(next_obs_seq)
            next_pi_act, _, _, _ = self.policy_prior.sample(z_next_tgt, return_physical=False)
            next_q_val = self.target_q_net.min_q(z_next_tgt, next_pi_act)
            td_scalar = rewards + self.gamma * (1.0 - terminateds) * next_q_val
            td_two_hot = two_hot_encode(td_scalar, num_bins=self.num_bins, vmin=self.vmin, vmax=self.vmax)

        q1_logits, q2_logits = self.q_net.forward_logits(z_rollout, act_norm_seq)
        q1_ce = -(td_two_hot * F.log_softmax(q1_logits, dim=-1)).sum(dim=-1)
        q2_ce = -(td_two_hot * F.log_softmax(q2_logits, dim=-1)).sum(dim=-1)
        td_loss = ((q1_ce + q2_ce) * weights).sum() / weights.sum().clamp_min(1.0)

        # 5. Policy Prior Loss (maximize Q + entropy on detached latents)
        z_pi_in = z_rollout.detach()
        pi_act, _, pi_ent, _ = self.policy_prior.sample(z_pi_in, return_physical=False)
        q_pi = self.q_net.min_q(z_pi_in, pi_act)
        pi_loss_t = -(q_pi + self.entropy_coef * pi_ent)
        pi_loss = (pi_loss_t * weights).sum() / weights.sum().clamp_min(1.0)

        total_loss = (
            self.jepa_coef * jepa_loss
            + self.reward_coef * rew_loss
            + self.done_coef * done_loss
            + self.value_coef * td_loss
            + pi_loss
        )
        metrics = {
            "loss_pi": float(pi_loss.item()),
            "loss_v": float(td_loss.item()),
            "entropy": float(pi_ent.mean().item()),
            "jepa_pred": float(jepa_info["jepa_pred_loss"].item()),
            "sigreg": float(jepa_info["sigreg_loss"].item()),
            "rew_loss": float(rew_loss.item()),
        }
        return total_loss, metrics

    def update_from_replay(
        self,
        replay: SequenceReplayBuffer,
        optimizer: torch.optim.Optimizer,
        algo_cfg: dict[str, Any],
    ) -> dict[str, float]:
        batch_size = int(algo_cfg.get("batch_size", 64))
        horizon = int(algo_cfg.get("horizon", self.horizon))
        grad_steps = int(algo_cfg.get("grad_steps_per_iter", 8))
        max_grad_norm = float(algo_cfg.get("max_grad_norm", 20.0))

        if not replay.can_sample(horizon):
            return {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0}

        agg: dict[str, list[float]] = {}
        for _ in range(grad_steps):
            batch = replay.sample_sequences(batch_size=batch_size, horizon=horizon)
            loss, metrics = self.compute_loss(batch)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            optimizer.step()

            self.wm.update_ema()
            with torch.no_grad():
                for p, p_tgt in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                    p_tgt.data.mul_(1.0 - self.q_ema_tau).add_(p.data, alpha=self.q_ema_tau)

            for k, v in metrics.items():
                agg.setdefault(k, []).append(v)

        return (
            {k: float(sum(vs) / len(vs)) for k, vs in agg.items()}
            if agg
            else {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0}
        )


def build_lewm_tdmpc2_from_cfg(
    obs_dim: int,
    act_dim: int,
    algo_cfg: dict[str, Any],
    robot_cfg: RobotConfig | None = None,
) -> LeWMTDMPC2Agent:
    return LeWMTDMPC2Agent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=int(algo_cfg.get("latent_dim", 128)),
        predictor_hidden=int(algo_cfg.get("predictor_hidden", 256)),
        ema_tau=float(algo_cfg.get("ema_tau", 0.99)),
        q_ema_tau=float(algo_cfg.get("q_ema_tau", 0.01)),
        sigreg_coef=float(algo_cfg.get("sigreg_coef", 0.25)),
        pred_decay=float(algo_cfg.get("pred_decay", 0.90)),
        use_recurrent_belief=bool(algo_cfg.get("use_recurrent_belief", True)),
        num_bins=int(algo_cfg.get("num_bins", 101)),
        vmin=float(algo_cfg.get("vmin", -10.0)),
        vmax=float(algo_cfg.get("vmax", 10.0)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        jepa_coef=float(algo_cfg.get("jepa_coef", 2.0)),
        reward_coef=float(algo_cfg.get("reward_coef", 1.0)),
        done_coef=float(algo_cfg.get("done_coef", 0.5)),
        value_coef=float(algo_cfg.get("value_coef", 0.5)),
        entropy_coef=float(algo_cfg.get("entropy_coef", 1e-4)),
        horizon=int(algo_cfg.get("horizon", 5)),
        planner_samples=int(algo_cfg.get("planner_samples", 256)),
        planner_elites=int(algo_cfg.get("planner_elites", 32)),
        planner_iterations=int(algo_cfg.get("planner_iterations", 4)),
        num_pi_trajs=int(algo_cfg.get("num_pi_trajs", 16)),
        temperature=float(algo_cfg.get("temperature", 0.5)),
        robot_cfg=robot_cfg,
    )
