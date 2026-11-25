from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import ackerman_follow.envs  # noqa: F401
from ackerman_follow.algos.registry import build_agent, load_agent
from ackerman_follow.buffers.rollout import RolloutBuffer
from ackerman_follow.buffers.sequence_replay import SequenceReplayBuffer
from ackerman_follow.config import get_robot_config
from ackerman_follow.engine.checkpoint import save_checkpoint
from ackerman_follow.engine.curriculum import CurriculumTracker
from ackerman_follow.engine.evaluator import evaluate_agent
from ackerman_follow.models.encoders import SpatialSensorEncoder
from ackerman_follow.models.heads import two_hot_decode, two_hot_encode
from ackerman_follow.models.world_models.lewm_jepa import LeWMJEPA, sigreg_epps_pulley_loss
from ackerman_follow.planners.mppi import MPPIPlanner
from scripts.train import load_combined_config


def _test_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_spatial_encoder_and_two_hot_roundtrip():
    device = _test_device()
    enc = SpatialSensorEncoder(obs_dim=490, latent_dim=128).to(device)
    obs = torch.rand((3, 5, 490), dtype=torch.float32, device=device)
    z = enc(obs)
    assert z.shape == (3, 5, 128)
    assert torch.isfinite(z).all()

    vals = torch.tensor([-25.0, -1.2, 0.0, 2.5, 12.0], device=device)
    weights = two_hot_encode(vals, num_bins=101, vmin=-10.0, vmax=10.0)
    assert weights.shape == (5, 101)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5, device=device), atol=1e-5)
    bins = torch.linspace(-10.0, 10.0, 101, device=device)
    # Feed exact log-probabilities concentrating on the two-hot weights
    logits = torch.log(weights.clamp_min(1e-8))
    decoded = two_hot_decode(logits, num_bins=101, vmin=-10.0, vmax=10.0)
    assert torch.allclose(decoded, vals, atol=0.35)
    del bins


def test_sequence_replay_buffer_boundary_mask_and_io(tmp_path):
    buf = SequenceReplayBuffer(capacity=40, obs_dim=490, act_dim=2, num_envs=2, device="cpu")
    for t in range(12):
        obs = np.full((2, 490), float(t), dtype=np.float32)
        act = np.full((2, 2), 0.2, dtype=np.float32)
        rew = np.ones(2, dtype=np.float32)
        next_obs = np.full((2, 490), float(t + 1), dtype=np.float32)
        term = np.array([t == 4, False], dtype=np.float32)
        ep_done = np.array([t == 4, t == 7], dtype=np.float32)
        buf.add(obs, act, rew, next_obs, term, ep_done, stage=1)

    assert buf.can_sample(horizon=5)
    batch = buf.sample_sequences(batch_size=16, horizon=5, rng=np.random.default_rng(0))
    assert batch["obs"].shape == (5, 16, 490)
    assert batch["valid_mask"].shape == (5, 16)

    # Whenever episode_dones[t] == 1.0, subsequent steps t+1..H-1 must have valid_mask == 0.0
    dones_np = batch["episode_dones"].numpy()
    mask_np = batch["valid_mask"].numpy()
    for b in range(16):
        done_idx = np.flatnonzero(dones_np[:-1, b] > 0.5)
        if len(done_idx) > 0:
            first_done = int(done_idx[0])
            assert np.all(mask_np[first_done + 1 :, b] == 0.0)

    npz_path = buf.save(tmp_path / "replay.npz")
    reloaded = SequenceReplayBuffer.load(npz_path, device="cpu")
    assert len(reloaded) == len(buf)
    assert np.allclose(reloaded.obs[:12], buf.obs[:12])


def test_lewm_jepa_sigreg_and_mppi_planner():
    device = _test_device()
    z_iso = torch.randn((64, 128), device=device)
    z_col = torch.ones((64, 128), device=device)
    loss_iso = sigreg_epps_pulley_loss(z_iso)
    loss_col = sigreg_epps_pulley_loss(z_col)
    assert torch.isfinite(loss_iso) and torch.isfinite(loss_col)
    # Collapsed constant latents incur strictly larger SIGReg penalty than isotropic Gaussian latents
    assert loss_col > loss_iso

    wm = LeWMJEPA(obs_dim=490, act_dim=2, latent_dim=64).to(device)
    obs_seq = torch.rand((5, 8, 490), device=device)
    next_obs_seq = torch.rand((5, 8, 490), device=device)
    act_norm = torch.rand((5, 8, 2), device=device) * 2.0 - 1.0
    valid_mask = torch.ones((5, 8), device=device)

    loss, info = wm.compute_jepa_loss(obs_seq, next_obs_seq, act_norm, valid_mask)
    assert torch.isfinite(loss)
    loss.backward()
    wm.update_ema()
    assert torch.isfinite(info["sigreg_loss"])

    planner = MPPIPlanner(act_dim=2, horizon=4, num_samples=32, num_elites=8, num_iterations=2)
    z0 = torch.randn((2, 64), device=device)
    planned = planner.plan(
        z0=z0,
        predict_next_fn=wm.predict_next,
        reward_fn=lambda z, a: -z.pow(2).mean(dim=-1) - a.pow(2).mean(dim=-1),
    )
    assert planned.shape == (2, 2)
    assert (planned >= -1.0001).all() and (planned <= 1.0001).all()


def test_all_registered_algos_act_update_checkpoint_and_eval(tmp_path):
    device = _test_device()
    cfg = get_robot_config()

    # Build a tiny sequence replay buffer for off-policy / WM updates
    replay = SequenceReplayBuffer(capacity=64, obs_dim=490, act_dim=2, num_envs=2, device=device)
    for _ in range(16):
        replay.add(
            obs=np.random.rand(2, 490).astype(np.float32),
            action=np.array([[0.3, 0.1], [0.2, -0.1]], dtype=np.float32),
            reward=np.array([0.5, 0.4], dtype=np.float32),
            next_obs=np.random.rand(2, 490).astype(np.float32),
            terminated=np.zeros(2, dtype=np.float32),
            episode_done=np.zeros(2, dtype=np.float32),
        )

    algo_cases = [
        ("gru_ppo", {"gru_hidden_dim": 64, "seq_len": 4, "n_epochs": 1, "num_mini_batches": 2}),
        ("lewm", {"latent_dim": 64, "batch_size": 8, "horizon": 4, "grad_steps_per_iter": 2, "planner_samples": 32, "planner_elites": 8, "planner_iterations": 2}),
        ("lewm_rl_modelfree", {"policy_type": "ppo", "latent_dim": 64, "gru_hidden_dim": 64, "seq_len": 4, "n_epochs": 1, "num_mini_batches": 2}),
        ("lewm_rl_modelfree", {"policy_type": "sac", "latent_dim": 64, "gru_hidden_dim": 64, "batch_size": 8, "horizon": 4, "grad_steps_per_iter": 2}),
        ("lewm_tdmpc2", {"latent_dim": 64, "batch_size": 8, "horizon": 4, "grad_steps_per_iter": 2, "planner_samples": 32, "planner_elites": 8, "planner_iterations": 2, "num_pi_trajs": 8}),
    ]

    env = gym.make("AckermanFollow-v0")
    try:
        for idx, (algo_name, algo_cfg) in enumerate(algo_cases):
            agent = build_agent(algo_name, obs_dim=490, act_dim=2, algo_cfg=algo_cfg, robot_cfg=cfg).to(device)
            opt = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad], lr=3e-4)

            state = agent.init_state(2, device=device)
            dummy_obs = torch.rand((2, 490), dtype=torch.float32, device=device)
            phys_act, next_state = agent.act(dummy_obs, state=state, deterministic=True)
            assert phys_act.shape == (2, 2)
            assert (phys_act[:, 0] >= -0.401).all() and (phys_act[:, 0] <= 0.801).all()
            assert (phys_act[:, 1] >= -0.555).all() and (phys_act[:, 1] <= 0.555).all()
            _ = agent.reset_state_mask(next_state, np.array([True, False]))

            if agent.training_mode == "on_policy":
                rb = RolloutBuffer(buffer_size=8, obs_dim=490, act_dim=2, device=device, num_envs=2, gru_hidden_dim=64)
                hx = agent.init_state(2, device=device)
                for _ in range(8):
                    act_t, logp_t, _, val_t, hx, raw_t = agent.get_action_and_value(
                        dummy_obs, hx=hx, return_hidden=True, return_raw_action=True
                    )
                    rb.add(
                        dummy_obs.cpu().numpy(),
                        act_t.detach().cpu().numpy(),
                        logp_t.detach().cpu().numpy(),
                        np.ones(2, dtype=np.float32),
                        np.zeros(2, dtype=np.float32),
                        val_t.squeeze(-1).detach().cpu().numpy(),
                        hidden=hx.squeeze(0).detach().cpu().numpy(),
                        raw_action=raw_t.detach().cpu().numpy(),
                    )
                rb.compute_returns_and_advantages(np.zeros(2, dtype=np.float32))
                metrics = agent.update_on_policy(rb, opt, algo_cfg)
            else:
                metrics = agent.update_from_replay(replay, opt, algo_cfg)

            assert all(np.isfinite(v) for v in metrics.values())

            ckpt_path = save_checkpoint(
                tmp_path / f"{algo_name}_{idx}.pt",
                agent,
                step=100,
                unlocked_stage=1,
                config={"algorithm": algo_cfg},
            )
            reloaded, meta = load_agent(ckpt_path, device=device)
            assert meta["algo"] == algo_name
            eval_res = evaluate_agent(reloaded, env, n_episodes_per_stage=1, stages=(1,), device=device)
            assert np.isfinite(eval_res["mean_reward"])
    finally:
        env.close()


def test_curriculum_tracker_and_legacy_checkpoint_compat():
    tracker = CurriculumTracker(enabled=True, start_stage=1, min_episodes_per_stage=2, stage_thresholds={1: 100.0})
    graduated = []
    assert tracker.record_episode(1, 50.0, vec_env=None) is False
    assert tracker.record_episode(1, 160.0, vec_env=None, on_graduate=lambda g, u: graduated.append((g, u))) is True
    assert tracker.unlocked_stage == 2
    assert graduated == [(1, 2)]

    # Verify YAML configs load cleanly for all 4 algorithms
    for algo in ("gru_ppo", "lewm", "lewm_rl_modelfree", "lewm_tdmpc2"):
        combined = load_combined_config(algo, None, "configs/curriculum.yaml")
        assert combined["algo"] == algo
        assert "curriculum" in combined and "algorithm" in combined

    # Verify existing legacy ActorCritic checkpoint on disk loads via registry.load_agent
    legacy_ckpt = Path("outputs/experiments/exp_20261124_232557_vm_7stage_sensory_v4/checkpoints/best_model.pt")
    if legacy_ckpt.exists():
        agent, _ = load_agent(legacy_ckpt, device="cpu")
        state = agent.init_state(1, torch.device("cpu"))
        act, _ = agent.act(torch.zeros((1, 490)), state=state, deterministic=True)
        assert act.shape == (1, 2)
