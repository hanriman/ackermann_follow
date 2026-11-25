"""Model Predictive Path Integral (MPPI) planner in JEPA latent space (CPU/MPS/CUDA safe)."""

from __future__ import annotations

from typing import Callable

import torch


class MPPIPlanner:
    """
    Batched MPPI trajectory optimizer over a learned latent dynamics model.

    Supports both:
      - Pure LeWM planning (random Gaussian action sequences scored by RewardHead)
      - TD-MPC2 planning (mixture of PolicyPrior rollouts + Gaussian perturbations scored by R + gamma^H Q)
    """

    def __init__(
        self,
        act_dim: int = 2,
        horizon: int = 6,
        num_samples: int = 256,
        num_elites: int = 32,
        num_iterations: int = 4,
        num_pi_trajs: int = 16,
        temperature: float = 0.5,
        gamma: float = 0.99,
        min_std: float = 0.05,
        max_std: float = 1.2,
    ) -> None:
        self.act_dim = int(act_dim)
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.num_elites = min(int(num_elites), int(num_samples))
        self.num_iterations = int(num_iterations)
        self.num_pi_trajs = min(int(num_pi_trajs), int(num_samples))
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self._prev_mean: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev_mean = None

    @torch.no_grad()
    def plan(
        self,
        z0: torch.Tensor,
        predict_next_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        value_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        policy_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        termination_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Plan normalized actions in `[-1, 1]` for a batch `z0 [B, D]`.

        Returns:
            norm_action: `[B, act_dim]` in `[-1, 1]`.
        """
        bsz, latent_dim = z0.shape
        device = z0.device
        dtype = z0.dtype

        mean = torch.zeros((bsz, self.horizon, self.act_dim), dtype=dtype, device=device)
        if self._prev_mean is not None and self._prev_mean.shape == mean.shape:
            mean[:, :-1] = self._prev_mean[:, 1:]
        std = torch.full((bsz, self.horizon, self.act_dim), self.max_std, dtype=dtype, device=device)

        n_pi = self.num_pi_trajs if policy_fn is not None else 0
        n_gauss = self.num_samples - n_pi

        # Precompute policy-guided trajectories once from z0
        pi_actions = None
        if n_pi > 0 and policy_fn is not None:
            pi_acts_list = []
            z_pi = z0.unsqueeze(1).expand(bsz, n_pi, latent_dim).reshape(bsz * n_pi, latent_dim)
            for _ in range(self.horizon):
                a_pi = torch.clamp(policy_fn(z_pi), -1.0, 1.0)
                pi_acts_list.append(a_pi.view(bsz, n_pi, self.act_dim))
                z_pi = predict_next_fn(z_pi, a_pi)
            pi_actions = torch.stack(pi_acts_list, dim=2)  # [B, n_pi, H, A]

        best_actions = torch.zeros((bsz, self.act_dim), dtype=dtype, device=device)
        for _ in range(self.num_iterations):
            noise = torch.randn((bsz, n_gauss, self.horizon, self.act_dim), dtype=dtype, device=device)
            gauss_actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * noise, -1.0, 1.0)

            if pi_actions is not None:
                actions = torch.cat([pi_actions, gauss_actions], dim=1)  # [B, N, H, A]
            else:
                actions = gauss_actions

            # Evaluate all N candidate trajectories in parallel
            z_cur = z0.unsqueeze(1).expand(bsz, self.num_samples, latent_dim).reshape(
                bsz * self.num_samples, latent_dim
            )
            total_return = torch.zeros((bsz * self.num_samples,), dtype=dtype, device=device)
            alive = torch.ones((bsz * self.num_samples,), dtype=dtype, device=device)
            discount = 1.0

            for t in range(self.horizon):
                a_t = actions[:, :, t, :].reshape(bsz * self.num_samples, self.act_dim)
                r_t = reward_fn(z_cur, a_t)
                total_return = total_return + discount * alive * r_t
                z_cur = predict_next_fn(z_cur, a_t)
                if termination_fn is not None:
                    p_done = termination_fn(z_cur)
                    alive = alive * (1.0 - p_done)
                discount *= self.gamma

            if value_fn is not None:
                a_last = (
                    torch.clamp(policy_fn(z_cur), -1.0, 1.0)
                    if policy_fn is not None
                    else actions[:, :, -1, :].reshape(bsz * self.num_samples, self.act_dim)
                )
                total_return = total_return + discount * alive * value_fn(z_cur, a_last)

            scores = total_return.view(bsz, self.num_samples)  # [B, N]
            elite_vals, elite_idx = torch.topk(scores, k=self.num_elites, dim=1)
            elite_actions = torch.gather(
                actions,
                dim=1,
                index=elite_idx[:, :, None, None].expand(bsz, self.num_elites, self.horizon, self.act_dim),
            )

            max_val = elite_vals.max(dim=1, keepdim=True).values
            weights = torch.exp(self.temperature * (elite_vals - max_val))
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

            w_exp = weights[:, :, None, None]
            mean = (w_exp * elite_actions).sum(dim=1)
            var = (w_exp * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)
            std = torch.clamp(torch.sqrt(var + 1e-6), self.min_std, self.max_std)

        self._prev_mean = mean.detach()
        if deterministic:
            best_actions = mean[:, 0, :]
        else:
            best_actions = torch.clamp(mean[:, 0, :] + std[:, 0, :] * torch.randn_like(mean[:, 0, :]), -1.0, 1.0)
        return torch.clamp(best_actions, -1.0, 1.0)
