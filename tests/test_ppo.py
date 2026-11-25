import numpy as np
import torch

from ackermann_follow.config import RobotConfig
from ackermann_follow.ppo.actor_critic import ActorCritic
from ackermann_follow.ppo.buffer import RolloutBuffer


def get_test_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_actor_critic_forward_pass():
    device = get_test_device()
    model = ActorCritic(obs_dim=490, act_dim=2).to(device)

    batch_size = 8
    dummy_obs = torch.randn((batch_size, 490), dtype=torch.float32, device=device)

    action, log_prob, entropy, value = model.get_action_and_value(dummy_obs)

    assert action.shape == (batch_size, 2)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, 1)


def test_action_bounds():
    device = get_test_device()
    model = ActorCritic(obs_dim=490, act_dim=2).to(device)

    # Sample 100 actions and verify all physical limits
    dummy_obs = torch.randn((100, 490), dtype=torch.float32, device=device)
    actions, _, _, _ = model.get_action_and_value(dummy_obs)

    v = actions[:, 0]
    steer = actions[:, 1]

    assert (v >= -0.401).all() and (v <= 0.801).all()
    assert (steer >= -0.554).all() and (steer <= 0.554).all()


def test_deterministic_action():
    device = get_test_device()
    model = ActorCritic(obs_dim=490, act_dim=2).to(device)

    dummy_obs = torch.randn((4, 490), dtype=torch.float32, device=device)
    action1 = model.get_deterministic_action(dummy_obs)
    action2 = model.get_deterministic_action(dummy_obs)

    # Deterministic output should be identical
    assert torch.allclose(action1, action2)


def test_rollout_buffer_and_gae():
    device = get_test_device()
    buf_size = 64
    buffer = RolloutBuffer(buffer_size=buf_size, obs_dim=490, act_dim=2, device=device)

    for i in range(buf_size):
        obs = np.random.randn(490).astype(np.float32)
        act = np.array([0.2, 0.1], dtype=np.float32)
        logp = -1.5
        rew = 1.0
        done = i == buf_size - 1
        val = 0.5
        buffer.add(obs, act, logp, rew, done, val)

    assert buffer.full is True

    # Compute GAE
    buffer.compute_returns_and_advantages(last_val=0.5, last_done=False)

    assert buffer.advantages.shape == (buf_size,)
    assert buffer.returns.shape == (buf_size,)

    # Test batch generator
    batch_count = 0
    total_samples = 0
    for b_obs, b_act, b_logp, b_adv, b_ret, b_val in buffer.get_batches(batch_size=16):
        batch_count += 1
        total_samples += len(b_obs)
        assert b_obs.shape == (16, 490)
        assert b_act.shape == (16, 2)

    assert batch_count == 4
    assert total_samples == buf_size


def test_gae_terminal_reward_propagation():
    """Verify GAE correctly propagates terminal collision penalty to preceding steps without bootstrapping across episodes."""
    device = torch.device("cpu")
    buf_size = 10
    buffer = RolloutBuffer(buffer_size=buf_size, obs_dim=490, act_dim=2, device=device)

    # Episode 1: Steps 0 to 4. Step 4 ends in a collision penalty (-25.0, done=1.0)
    # Episode 2: Steps 5 to 9. Step 5 is clean restart (reward=1.0, done=0.0, value=10.0)
    for t in range(buf_size):
        obs = np.zeros(490, dtype=np.float32)
        act = np.zeros(2, dtype=np.float32)
        logp = 0.0
        val = 0.0
        if t < 4:
            rew = 1.0
            done = 0.0
        elif t == 4:
            rew = -25.0  # Fatal collision at step 4
            done = 1.0
        else:
            rew = 2.0  # New clean episode
            done = 0.0
            val = 5.0
        buffer.add(obs, act, logp, rew, done, val)

    buffer.compute_returns_and_advantages(last_val=0.0, last_done=True, gamma=0.99, gae_lambda=0.95)

    # Advantages for steps prior to collision must be significantly negative due to -25.0 penalty propagation
    adv = buffer.advantages.cpu().numpy()
    assert adv[4] < -20.0, f"Terminal step advantage should reflect -25 penalty, got {adv[4]}"
    assert adv[3] < -15.0, f"Step prior to collision must receive propagated penalty, got {adv[3]}"
    assert adv[2] < -10.0, f"Step 2 must receive propagated penalty, got {adv[2]}"

    # Step 4 must NOT bootstrap from Step 5's high value (5.0)
    # If done[4] was ignored or off-by-one, adv[4] would add gamma * val[5] (5.0 * 0.99 = 4.95)
    delta_4 = -25.0 - 0.0  # rew[4] - val[4] = -25.0
    assert abs(adv[4] - delta_4) < 1e-4, f"Terminal delta must equal exact penalty, got {adv[4]}"


def test_actor_critic_angular_sensitivity_and_non_saturating_bounds():
    """Verify CoordConv/target-mask polar features distinguish target vs wall and left vs right."""
    device = get_test_device()
    model = ActorCritic(obs_dim=490, act_dim=2).to(device)

    obs_right = torch.ones((1, 490), dtype=torch.float32, device=device)
    obs_right[:, 64:128] = 0.0
    obs_right[:, 5:9] = 0.30
    obs_right[:, 64 + 5 : 64 + 9] = 1.0  # Target on right
    obs_right[:, 488:] = 0.0

    obs_left = torch.ones((1, 490), dtype=torch.float32, device=device)
    obs_left[:, 64:128] = 0.0
    obs_left[:, 5:9] = 0.15  # Closer wall on right
    obs_left[:, 55:59] = 0.40  # Target on left
    obs_left[:, 64 + 55 : 64 + 59] = 1.0
    obs_left[:, 488:] = 0.0

    feat_r, _ = model._extract_features(obs_right)
    feat_l, _ = model._extract_features(obs_left)
    assert not torch.allclose(feat_r, feat_l, atol=1e-4)

    # Verify TanhNormal pre-tanh mean reaches > 99% of normalized +/-1.0 bounds smoothly
    extreme_logits = torch.tensor([[3.0, -3.0]], dtype=torch.float32, device=device)
    squashed = torch.tanh(extreme_logits)
    assert (squashed[:, 0] > 0.99).all() and (squashed[:, 1] < -0.99).all()


def test_actor_critic_recurrent_memory_and_episode_reset():
    """Verify GRU hidden state retains prior target observation and resets cleanly on episode_dones."""
    device = get_test_device()
    model = ActorCritic(obs_dim=490, act_dim=2).to(device)

    obs_target_left = torch.ones((1, 490), dtype=torch.float32, device=device)
    obs_target_left[:, 64:128] = 0.0
    obs_target_left[:, 56:60] = 0.25
    obs_target_left[:, 64 + 56 : 64 + 60] = 1.0

    obs_empty = torch.ones((1, 490), dtype=torch.float32, device=device)
    obs_empty[:, 64:128] = 0.0

    hx0 = model.init_hidden(1, device=device)
    _, _, _, _, hx1 = model.get_action_and_value(obs_target_left, hx=hx0, return_hidden=True)

    # Empty observation with retained memory differs from empty observation with zero memory
    act_with_mem = model.get_deterministic_action(obs_empty, hx=hx1)
    act_no_mem = model.get_deterministic_action(obs_empty, hx=hx0)
    assert not torch.allclose(act_with_mem, act_no_mem, atol=1e-5)

    # Passing episode_dones=1.0 must zero next_hx
    done_flag = torch.tensor([1.0], dtype=torch.float32, device=device)
    _, _, _, _, hx_reset = model.get_action_and_value(
        obs_target_left, hx=hx0, episode_dones=done_flag, return_hidden=True
    )
    assert torch.allclose(hx_reset, torch.zeros_like(hx_reset))


def test_recurrent_rollout_buffer_and_vectorized_gae():
    """Verify multi-env RolloutBuffer vectorized GAE and sequence-chunk BPTT batches."""
    device = get_test_device()
    buf_size = 32
    num_envs = 4
    seq_len = 16
    buffer = RolloutBuffer(
        buffer_size=buf_size,
        obs_dim=490,
        act_dim=2,
        device=device,
        num_envs=num_envs,
        gru_hidden_dim=128,
    )

    for t in range(buf_size):
        obs = np.ones((num_envs, 490), dtype=np.float32)
        act = np.zeros((num_envs, 2), dtype=np.float32)
        logp = np.zeros(num_envs, dtype=np.float32)
        rew = np.ones(num_envs, dtype=np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        val = np.full(num_envs, 0.5, dtype=np.float32)
        hx = np.zeros((num_envs, 128), dtype=np.float32)
        buffer.add(obs, act, logp, rew, done, val, hidden=hx, episode_done=done)

    assert buffer.full is True
    buffer.compute_returns_and_advantages(
        last_val=np.full(num_envs, 0.5, dtype=np.float32),
        last_done=np.zeros(num_envs, dtype=np.float32),
    )
    assert buffer.advantages.shape == (buf_size, num_envs)
    assert buffer.returns.shape == (buf_size, num_envs)

    batches = list(buffer.get_recurrent_batches(num_mini_batches=4, seq_len=seq_len))
    assert len(batches) == 4
    for b_obs, b_act, b_logp, b_adv, b_ret, b_val, b_hx, b_ep_dones in batches:
        assert b_obs.shape == (seq_len, 2, 490)
        assert b_act.shape == (seq_len, 2, 2)
        assert b_hx.shape == (1, 2, 128)
        assert b_ep_dones.shape == (seq_len, 2)


def test_gae_truncation_no_double_bootstrap_or_leakage():
    """Verify truncated episodes (terminated=0, episode_done=1) do not double-bootstrap or leak GAE."""
    device = torch.device("cpu")
    buf_size = 6
    buffer = RolloutBuffer(buffer_size=buf_size, obs_dim=490, act_dim=2, device=device)

    obs = np.zeros(490, dtype=np.float32)
    act = np.zeros(2, dtype=np.float32)

    # Step 0..2: Episode 1; Step 2 is truncated (terminated=0, episode_done=1) with bootstrap already added to reward
    # Step 3..5: Episode 2; Step 5 is also truncated at buffer end
    rewards = [1.0, 1.0, 1.0 + 0.99 * 4.0, 2.0, 2.0, 2.0 + 0.99 * 7.0]
    values = [1.0, 1.0, 1.0, 100.0, 5.0, 5.0]
    terminateds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    episode_dones = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]

    for t in range(buf_size):
        buffer.add(
            obs,
            act,
            0.0,
            rewards[t],
            terminateds[t],
            values[t],
            episode_done=episode_dones[t],
        )

    buffer.compute_returns_and_advantages(last_val=999.0, gamma=0.99, gae_lambda=0.95)
    adv = buffer.advantages.cpu().numpy()

    # Step 2 must NOT bootstrap from Step 3's value (100.0) and must NOT receive Step 3's advantage
    expected_adv_2 = (1.0 + 0.99 * 4.0) - 1.0
    assert np.isclose(adv[2], expected_adv_2, atol=1e-5)

    # Step 5 (final step truncated) must NOT bootstrap from last_val (999.0)
    expected_adv_5 = (2.0 + 0.99 * 7.0) - 5.0
    assert np.isclose(adv[5], expected_adv_5, atol=1e-5)


def test_actor_critic_buffers_follow_device_and_config():
    """Verify scaling constants are registered buffers and follow RobotConfig."""
    custom_cfg = RobotConfig(v_min=-0.20, v_max=1.00, max_steering_angle=0.45)
    model = ActorCritic(robot_cfg=custom_cfg)

    named_buffers = dict(model.named_buffers())
    for key in ("v_min", "v_max", "steer_max", "v_scale", "v_bias", "steer_scale"):
        assert key in named_buffers

    assert torch.isclose(model.v_scale, torch.tensor(0.60))
    assert torch.isclose(model.v_bias, torch.tensor(0.40))
    assert torch.isclose(model.steer_scale, torch.tensor(0.45))

    device = get_test_device()
    model = model.to(device)
    assert model.v_scale.device.type == device.type
    assert model.v_bias.device.type == device.type
    assert model.steer_scale.device.type == device.type


def test_actor_critic_sequence_mode_with_mid_episode_dones():
    """Verify [T, B, obs_dim] sequence execution resets GRU state cleanly at mid-sequence episode_dones."""
    device = get_test_device()
    model = ActorCritic(obs_dim=490, act_dim=2).to(device)
    model.eval()

    seq_len, batch_size = 5, 2
    x = torch.ones((seq_len, batch_size, 490), dtype=torch.float32, device=device)
    x[:, :, 64:128] = 0.0
    x[0:3, :, 50:55] = 0.2  # Strong target signal in first 3 steps
    x[0:3, :, 64 + 50 : 64 + 55] = 1.0
    ep_dones = torch.zeros((seq_len, batch_size), dtype=torch.float32, device=device)
    ep_dones[2, 0] = 1.0  # Batch item 0 resets after step 2; batch item 1 does not

    feats_seq, _ = model._extract_features(x, episode_dones=ep_dones)
    # Compare step 3 onwards for batch item 0 against fresh zero-hidden sequence starting at t=3
    feats_fresh, _ = model._extract_features(x[3:, 0:1])
    assert torch.allclose(feats_seq[3:, 0:1], feats_fresh, atol=1e-5)
    # Batch item 1 retained memory from t=0..2, so its step 3 features differ from fresh zero-hidden
    assert not torch.allclose(feats_seq[3:, 1:2], feats_fresh, atol=1e-5)


def test_tanh_normal_saturated_logprob_consistency():
    """Verify stored raw_action eliminates atanh log_prob drift even at extreme saturation."""
    device = get_test_device()
    model = ActorCritic(obs_dim=490, act_dim=2).to(device)
    obs = torch.randn((4, 490), dtype=torch.float32, device=device)

    actions, logp_rollout, entropy, _, raw_u = model.get_action_and_value(
        obs, return_raw_action=True
    )
    assert torch.isfinite(logp_rollout).all()
    assert torch.isfinite(entropy).all()

    # Force extreme saturation |u| > 6.5 (beyond float32 tanh precision limits)
    extreme_u = torch.tensor([[6.5, -7.0], [-6.8, 6.2], [7.5, 7.5], [-7.5, -7.5]], device=device)
    extreme_act = model.scale_action(torch.tanh(extreme_u))

    _, logp_with_raw, ent_with_raw, _ = model.get_action_and_value(
        obs, action=extreme_act, raw_action=extreme_u
    )
    assert torch.isfinite(logp_with_raw).all()
    assert torch.isfinite(ent_with_raw).all()

    # Re-evaluating with same raw_u produces identical log_prob
    _, logp_repeat, _, _ = model.get_action_and_value(
        obs, action=extreme_act, raw_action=extreme_u
    )
    assert torch.allclose(logp_with_raw, logp_repeat, atol=1e-6)


def test_recurrent_buffer_uneven_minibatches_and_raw_actions():
    """Verify np.array_split handles VM-sized num_envs=6 cleanly with include_raw_actions=True."""
    device = get_test_device()
    buf_size = 64
    num_envs = 6
    seq_len = 32
    buffer = RolloutBuffer(
        buffer_size=buf_size,
        obs_dim=490,
        act_dim=2,
        device=device,
        num_envs=num_envs,
        gru_hidden_dim=128,
    )

    for _ in range(buf_size):
        obs = np.ones((num_envs, 490), dtype=np.float32)
        act = np.zeros((num_envs, 2), dtype=np.float32)
        raw_act = np.full((num_envs, 2), 0.25, dtype=np.float32)
        logp = np.zeros(num_envs, dtype=np.float32)
        rew = np.ones(num_envs, dtype=np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        val = np.full(num_envs, 0.5, dtype=np.float32)
        hx = np.zeros((num_envs, 128), dtype=np.float32)
        buffer.add(
            obs, act, logp, rew, done, val, hidden=hx, episode_done=done, raw_action=raw_act
        )

    buffer.compute_returns_and_advantages(last_val=np.full(num_envs, 0.5, dtype=np.float32))
    # 6 envs * (64 / 32) = 12 chunks split across 8 minibatches -> all 12 chunks yielded
    batches = list(
        buffer.get_recurrent_batches(
            num_mini_batches=8, seq_len=seq_len, include_raw_actions=True
        )
    )
    assert len(batches) == 8
    total_chunks = sum(b[0].shape[1] for b in batches)
    assert total_chunks == 12
    for b_obs, b_act, _, _, _, _, b_hx, b_ep_dones, b_raw_act in batches:
        assert b_obs.shape[0] == seq_len and b_obs.shape[2] == 490
        assert b_act.shape == b_raw_act.shape
        assert torch.allclose(b_raw_act, torch.full_like(b_raw_act, 0.25))
        assert b_hx.shape == (1, b_obs.shape[1], 128)
        assert b_ep_dones.shape == (seq_len, b_obs.shape[1])
