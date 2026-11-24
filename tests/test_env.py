import gymnasium as gym
import mujoco
import numpy as np
import torch

import ackerman_follow.envs  # noqa: F401
from ackerman_follow.ppo.actor_critic import ActorCritic
from scripts.eval_policy import load_policy, run_evaluation
from scripts.train_ppo import _EnvFactory


def test_env_spaces_and_gym_make():
    env = gym.make("AckermanFollow-v0")
    try:
        assert env.observation_space.shape == (490,)
        assert env.action_space.shape == (2,)
        obs, info = env.reset(seed=42)
        assert obs.shape == (490,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert info["stage"] == 1
        assert "target_dist" in info and "target_in_fov" in info
    finally:
        env.close()


def test_ray_ordering_right_to_left_and_lidar_ccw():
    env = gym.make("AckermanFollow-v0")
    try:
        base = env.unwrapped
        base.reset(seed=7, options={"stage": 1})

        # Place robot cleanly at origin facing +X (yaw = 0)
        base.data.qpos[base._robot_free_qpos : base._robot_free_qpos + 3] = [0.0, 0.0, 0.0]
        base.data.qpos[base._robot_free_qpos + 3 : base._robot_free_qpos + 7] = [1.0, 0.0, 0.0, 0.0]

        # 1. Target far away (> 5m) -> open arena walls at ±8m are outside camera (5m) and LiDAR (3m)
        base.target.x = 6.5
        base.target.y = 0.0
        base._sync_human_mocap()
        mujoco.mj_forward(base.model, base.data)
        obs_empty, min_lidar_empty = base._get_obs()
        assert np.allclose(obs_empty[:64], 1.0, atol=1e-5)
        assert np.allclose(obs_empty[64:128], 0.0, atol=1e-5)
        assert np.allclose(obs_empty[128:488], 1.0, atol=1e-5)
        assert np.isclose(min_lidar_empty, base.cfg.lidar_range_max, atol=1e-5)

        # 2. Target on robot's RIGHT (-Y, ~ -25 deg from front) within 2.0m
        base.target.x = 1.6
        base.target.y = -0.8
        base.target.yaw = 0.0
        base._sync_human_mocap()
        mujoco.mj_forward(base.model, base.data)
        obs_right, _ = base._get_obs()
        cam_r = obs_right[:64]
        mask_r = obs_right[64:128]
        lidar_r = obs_right[128:488]
        # Camera rays 0..20 (right) must detect the target, while 43..63 (left) stay empty
        assert np.min(cam_r[:21]) < 0.5
        assert np.allclose(cam_r[43:], 1.0, atol=1e-5)
        assert np.max(mask_r[:21]) == 1.0
        assert np.allclose(mask_r[43:], 0.0, atol=1e-5)
        # LiDAR CCW: right sector (~315..359 deg) detects target, left sector (~45..135 deg) is empty
        assert np.min(lidar_r[315:]) < 0.8
        assert np.allclose(lidar_r[45:135], 1.0, atol=1e-5)

        # 3. Target on robot's LEFT (+Y, ~ +25 deg from front) within 2.0m
        base.target.x = 1.6
        base.target.y = 0.8
        base._sync_human_mocap()
        mujoco.mj_forward(base.model, base.data)
        obs_left, _ = base._get_obs()
        cam_l = obs_left[:64]
        mask_l = obs_left[64:128]
        lidar_l = obs_left[128:488]
        assert np.min(cam_l[43:]) < 0.5
        assert np.allclose(cam_l[:21], 1.0, atol=1e-5)
        assert np.max(mask_l[43:]) == 1.0
        assert np.allclose(mask_l[:21], 0.0, atol=1e-5)
        # LiDAR index ~20..40 (+Y left front) detects target, right sector (~225..315) is empty
        assert np.min(lidar_l[10:45]) < 0.8
        assert np.allclose(lidar_l[225:315], 1.0, atol=1e-5)
    finally:
        env.close()


def test_ackermann_turning_and_inner_wheel_clearance():
    env = gym.make("AckermanFollow-v0")
    try:
        base = env.unwrapped
        env.reset(seed=10, options={"stage": 1})

        # Command max left steer (+0.553953 rad) and forward drive (0.60 m/s) for 25 steps (1.25s)
        action = np.array([0.60, base.cfg.max_steering_angle], dtype=np.float32)
        for _ in range(25):
            obs, _, terminated, _, _ = env.step(action)
            assert not terminated
            assert np.isfinite(obs).all()

        fl_angle = float(base.data.qpos[base._fl_steer_qpos])
        fr_angle = float(base.data.qpos[base._fr_steer_qpos])
        _, _, yaw, v_fwd, _ = base._get_robot_pose_and_vel()

        # Inner wheel (FL) turns sharper (~0.70-0.73 rad / >40 deg) than outer wheel (FR ~0.41-0.44 rad)
        assert fl_angle > 0.65
        assert fr_angle > 0.38
        assert fl_angle > fr_angle + 0.20
        assert yaw > 0.5
        assert v_fwd > 0.4
    finally:
        env.close()


def test_stage_switching_and_collision_termination():
    env = gym.make("AckermanFollow-v0")
    try:
        for stg in (1, 2, 3, 4):
            obs, info = env.reset(seed=100 + stg, options={"stage": stg})
            assert info["stage"] == stg
            assert obs.shape == (490,)

        # In Stage 4, place robot directly moving into pillar_3 at (3.40, 0.15) or drive into Stage 2 L-wall
        base = env.unwrapped
        base.reset(seed=5, options={"stage": 2})
        # Drive straight forward into Stage 2 L-wall (s2_lwall_a at x=2.80, y in [-2.3, 0.5])
        base.data.qpos[base._robot_free_qpos : base._robot_free_qpos + 3] = [1.8, -0.3, 0.0]
        base.data.qpos[base._robot_free_qpos + 3 : base._robot_free_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(base.model, base.data)

        collided = False
        for _ in range(60):
            _, reward, terminated, _, info = env.step(np.array([0.80, 0.0], dtype=np.float32))
            if terminated:
                collided = True
                assert info["collision"] is True
                assert reward == -25.0
                break
        assert collided
    finally:
        env.close()


def test_human_overstep_does_not_terminate_and_stage2_fov_spawn():
    env = gym.make("AckermanFollow-v0")
    try:
        base = env.unwrapped

        # 1. Stage 2 spawns must stay inside camera FOV across varied seeds and have 900-step horizon
        dists = []
        for s in range(20):
            _, info = env.reset(seed=200 + s, options={"stage": 2})
            assert base.max_steps == 900
            assert info["target_in_fov"] is True
            dists.append(info["target_dist"])
        assert max(dists) - min(dists) > 1.0  # spans close and near-corner urgent spawns

        # 2. Human target stepping over robot must NOT terminate or physically stop robot
        env.reset(seed=42, options={"stage": 1})
        base.data.qpos[base._robot_free_qpos : base._robot_free_qpos + 3] = [0.0, 0.0, 0.0]
        base.data.qpos[base._robot_free_qpos + 3 : base._robot_free_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        base.target.x = 0.6
        base.target.y = 0.0
        base.target.speed = 0.0
        base._sync_human_mocap()
        mujoco.mj_forward(base.model, base.data)

        for _ in range(30):
            obs, _, terminated, _, info = env.step(np.array([0.80, 0.0], dtype=np.float32))
            assert not terminated
            assert info["collision"] is False
            assert np.isfinite(obs).all()

        rx, _, _, _, _ = base._get_robot_pose_and_vel()
        assert rx > 0.9  # robot drove cleanly past the human's position without collision

        # 3. Reversing away when far (> 1.0 m) earns less reward than driving forward toward target
        env.reset(seed=99, options={"stage": 3})
        _, rew_rev, _, _, _ = env.step(np.array([-0.40, 0.0], dtype=np.float32))
        env.reset(seed=99, options={"stage": 3})
        _, rew_fwd, _, _, _ = env.step(np.array([0.80, 0.0], dtype=np.float32))
        assert rew_fwd > rew_rev

        # 4. Stage 3 & Stage 4 trajectories never intersect walls/pillars across full 600-step episodes
        env.reset(seed=12, options={"stage": 3})
        for _ in range(600):
            base._step_human_target()
            tx, ty = base.target.x, base.target.y
            # Ensure target never intersects Stage 3 wall boxes
            assert not (0.30 <= tx <= 5.90 and 0.98 <= ty <= 1.22)  # s3_corr_left_1
            assert not (0.30 <= tx <= 3.60 and -1.22 <= ty <= -0.98)  # s3_corr_right_1
            assert not (3.48 <= tx <= 3.72 and -3.30 <= ty <= -1.00)  # s3_corr_left_2
            assert not (5.68 <= tx <= 5.92 and -3.30 <= ty <= 1.20)  # s3_corr_right_2

        env.reset(seed=14, options={"stage": 4})
        pillars = [
            (1.80, 0.85), (2.30, -0.95), (3.40, 0.15), (4.30, 1.45),
            (4.60, -1.25), (5.70, 0.35), (-1.20, 1.60), (-1.40, -1.60),
        ]
        for _ in range(600):
            base._step_human_target()
            tx, ty = base.target.x, base.target.y
            for px, py in pillars:
                assert np.hypot(tx - px, ty - py) > 0.45

        # 5. Bearing-gated reverse penalty: reversing toward a target behind has smaller penalty than away in front
        def _eval_rev_delta(tx: float) -> float:
            env.reset(seed=99, options={"stage": 1})
            base.data.qpos[base._robot_free_qpos : base._robot_free_qpos + 3] = [0.0, 0.0, 0.0]
            base.data.qpos[base._robot_free_qpos + 3 : base._robot_free_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
            base.target.x, base.target.y, base.target.speed = tx, 0.0, 0.0
            base._sync_human_mocap()
            mujoco.mj_forward(base.model, base.data)
            dist0, bear0, _ = base._compute_target_metrics()
            base._prev_dist_err = abs(dist0 - base.cfg.target_follow_distance)
            base._prev_abs_bearing = abs(bear0)
            _, rew_rev, _, _, _ = env.step(np.array([-0.40, 0.0], dtype=np.float32))

            env.reset(seed=99, options={"stage": 1})
            base.data.qpos[base._robot_free_qpos : base._robot_free_qpos + 3] = [0.0, 0.0, 0.0]
            base.data.qpos[base._robot_free_qpos + 3 : base._robot_free_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
            base.target.x, base.target.y, base.target.speed = tx, 0.0, 0.0
            base._sync_human_mocap()
            mujoco.mj_forward(base.model, base.data)
            dist0, bear0, _ = base._compute_target_metrics()
            base._prev_dist_err = abs(dist0 - base.cfg.target_follow_distance)
            base._prev_abs_bearing = abs(bear0)
            _, rew_stop, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
            return rew_rev - rew_stop

        assert _eval_rev_delta(-2.0) > _eval_rev_delta(2.0)
    finally:
        env.close()


def test_async_vector_env_spawn_rollout():
    factories = [
        _EnvFactory(
            auto_curriculum=True,
            start_stage=1,
            min_episodes_per_stage=40,
            stage_thresholds={1: 850.0, 2: 700.0, 3: 600.0},
            replay_previous_ratio=0.30,
            worker_threads=1,
        )
        for _ in range(2)
    ]
    vec_env = gym.vector.AsyncVectorEnv(
        factories,
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
        context="spawn",
    )
    try:
        obs, infos = vec_env.reset(seed=42)
        assert obs.shape == (2, 490)
        assert list(vec_env.get_attr("stage")) == [1, 1]

        vec_env.call("set_unlocked_stage", 3)
        assert list(vec_env.get_attr("unlocked_stage")) == [3, 3]

        actions = np.array([[0.4, 0.1], [0.4, -0.1]], dtype=np.float32)
        next_obs, rewards, terminateds, truncateds, step_infos = vec_env.step(actions)
        assert next_obs.shape == (2, 490)
        assert rewards.shape == (2,)
        assert terminateds.shape == (2,)
        assert truncateds.shape == (2,)
        assert "target_dist" in step_infos
    finally:
        vec_env.close()


def test_eval_policy_loads_checkpoint_and_runs(tmp_path):
    ckpt_path = tmp_path / "test_ckpt.pt"
    model = ActorCritic()
    torch.save({"step": 100, "unlocked_stage": 2, "model_state_dict": model.state_dict()}, ckpt_path)

    loaded_model, meta = load_policy(ckpt_path, device=torch.device("cpu"))
    assert meta["unlocked_stage"] == 2
    results = run_evaluation(
        model=loaded_model,
        stages=(1, 2),
        n_episodes=1,
        device=torch.device("cpu"),
        seed=42,
        render=False,
    )
    assert set(results.keys()) == {1, 2}
    for stg_res in results.values():
        assert stg_res["mean_length"] >= 1.0
        assert np.isfinite(stg_res["mean_reward"])


def test_stage5_dead_end_road_walk_past_and_cap_collision():
    env = gym.make("AckermanFollow-v0")
    try:
        base = env.unwrapped
        obs, info = env.reset(seed=501, options={"stage": 5})
        assert info["stage"] == 5
        assert base.max_steps == 700
        assert obs.shape == (490,)

        # Human walks down dead end (> 3.5 m), turns 180 deg, and returns past car (< 0.0 m) cleanly
        max_x, min_x = -999.0, 999.0
        for _ in range(700):
            base._step_human_target()
            tx, ty = base.target.x, base.target.y
            max_x = max(max_x, tx)
            min_x = min(min_x, tx)
            # Never intersects dead-end side walls or end cap
            assert not (-1.00 <= tx <= 4.90 and 1.18 <= abs(ty) <= 1.42)
            assert not (4.68 <= tx <= 4.92 and abs(ty) <= 1.40)

        assert max_x > 3.5
        assert min_x < 0.0

        # Driving robot forward into dead-end cap at x=4.80 triggers collision termination
        env.reset(seed=502, options={"stage": 5})
        base.data.qpos[base._robot_free_qpos : base._robot_free_qpos + 3] = [3.8, 0.0, 0.0]
        base.data.qpos[base._robot_free_qpos + 3 : base._robot_free_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(base.model, base.data)

        collided = False
        for _ in range(60):
            _, reward, terminated, _, step_info = env.step(np.array([0.80, 0.0], dtype=np.float32))
            if terminated:
                collided = True
                assert step_info["collision"] is True
                assert reward == -25.0
                break
        assert collided
    finally:
        env.close()
