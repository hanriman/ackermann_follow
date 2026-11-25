import pickle

import numpy as np

from ackermann_follow.config import configure_cpu_threads, get_robot_config
from ackermann_follow.envs.human_target import BipedalHumanTarget
from scripts.train_ppo import _EnvFactory


def test_robot_config_normalization_roundtrip():
    cfg = get_robot_config()

    cam_m = np.array([cfg.cam_range_min, 1.5, cfg.cam_range_max], dtype=np.float32)
    cam_norm = cfg.normalize_cam(cam_m)
    assert np.allclose(cam_norm, [0.0, (1.5 - cfg.cam_range_min) / (cfg.cam_range_max - cfg.cam_range_min), 1.0])
    assert np.allclose(cfg.denormalize_cam(cam_norm), cam_m, atol=1e-5)

    lidar_m = np.array([cfg.lidar_range_min, 1.55, cfg.lidar_range_max], dtype=np.float32)
    lidar_norm = cfg.normalize_lidar(lidar_m)
    assert np.allclose(cfg.denormalize_lidar(lidar_norm), lidar_m, atol=1e-5)

    v, steer = 0.35, -0.25
    norm_act = cfg.normalize_action(v, steer)
    assert np.allclose(norm_act, cfg.normalize_state(v, steer))
    v_rt, steer_rt = cfg.unscale_action(float(norm_act[0]), float(norm_act[1]))
    assert np.isclose(v_rt, v, atol=1e-5)
    assert np.isclose(steer_rt, steer, atol=1e-5)


def test_get_robot_config_cache_isolation(tmp_path):
    default_cfg = get_robot_config()
    custom_yaml = tmp_path / "custom_robot.yaml"
    custom_yaml.write_text(
        "kinematics:\n  max_linear_speed: 1.25\n  min_linear_speed: -0.50\n",
        encoding="utf-8",
    )

    custom_cfg = get_robot_config(custom_yaml)
    assert custom_cfg.v_max == 1.25
    assert custom_cfg.v_min == -0.50

    # Subsequent default call must NOT be polluted by custom_yaml
    default_again = get_robot_config()
    assert default_again.v_max == default_cfg.v_max
    assert default_again.v_min == default_cfg.v_min


def test_configure_cpu_threads():
    threads = configure_cpu_threads(num_threads=2, num_envs=6, vec_env="async")
    assert threads == 2
    auto_threads = configure_cpu_threads(num_threads="auto", num_envs=6, vec_env="async")
    assert auto_threads >= 1


def test_bipedal_human_target_seeded_determinism_and_gait():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    t_a = BipedalHumanTarget(nominal_speed=0.40, rng=rng_a)
    t_b = BipedalHumanTarget(nominal_speed=0.40, rng=rng_b)

    t_a.reset(x=1.0, y=0.5, yaw=0.2, speed=0.40)
    t_b.reset(x=1.0, y=0.5, yaw=0.2, speed=0.40)
    assert t_a.color_rgb == t_b.color_rgb
    assert np.isclose(t_a.gait_phase, t_b.gait_phase)

    for _ in range(10):
        t_a.step(dt=0.05)
        t_b.step(dt=0.05)

    assert np.isclose(t_a.x, t_b.x) and np.isclose(t_a.y, t_b.y)
    left, right = t_a.get_feet_positions()
    # Feet lateral midpoint should equal walker center (x, y)
    assert np.isclose(0.5 * (left.x + right.x), t_a.x, atol=1e-6)
    assert np.isclose(0.5 * (left.y + right.y), t_a.y, atol=1e-6)


def test_env_factory_is_picklable_for_spawn():
    factory = _EnvFactory(
        auto_curriculum=True,
        start_stage=1,
        min_episodes_per_stage=40,
        stage_thresholds={1: 850.0, 2: 700.0},
        replay_previous_ratio=0.30,
        worker_threads=1,
    )
    roundtripped = pickle.loads(pickle.dumps(factory))
    assert roundtripped.start_stage == 1
    assert roundtripped.worker_threads == 1
    assert roundtripped.stage_thresholds == {1: 850.0, 2: 700.0}
