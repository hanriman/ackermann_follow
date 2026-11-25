import gymnasium as gym
import mujoco
import numpy as np

import ackerman_follow.envs  # noqa: F401
from scripts.demo_3d import (
    HumanGameController,
    LiveVideoRecorder,
    _active_wall_boxes,
    _step_env_with_human_choice,
    append_3d_overlays,
    configure_robot_pip_camera,
    find_default_checkpoint,
    resolve_human_wall_collisions,
    update_camera_preset,
)


def test_find_default_checkpoint(tmp_path):
    assert find_default_checkpoint(tmp_path) is None

    exp1 = tmp_path / "exp_old" / "checkpoints"
    exp2 = tmp_path / "exp_new" / "checkpoints"
    exp1.mkdir(parents=True)
    exp2.mkdir(parents=True)

    old_latest = exp1 / "latest_model.pt"
    old_latest.write_bytes(b"old")
    assert find_default_checkpoint(tmp_path) == old_latest

    best = exp2 / "best_model.pt"
    best.write_bytes(b"best")
    assert find_default_checkpoint(tmp_path) == best


def test_resolve_human_wall_collisions_slides_along_wall():
    # Wall box centered at (2.0, 1.3) with half-extents (2.0, 0.1) -> inner face at y=1.2
    boxes = [(2.0, 1.3, 2.0, 0.1)]
    nx, ny = resolve_human_wall_collisions(
        x=1.5,
        y=1.15,
        prev_x=1.2,
        prev_y=0.90,
        boxes=boxes,
        radius=0.22,
    )
    # X advances freely along the wall while Y is clamped outside (1.2 - 0.22 = 0.98)
    assert np.isclose(nx, 1.5, atol=1e-5)
    assert ny <= 0.98 + 1e-5


def test_human_game_controller_and_step_with_wall_sliding():
    env = gym.make("AckermanFollow-v0")
    try:
        base = env.unwrapped
        obs, _ = base.reset(seed=11, options={"stage": 2, "force_front_spawn": True})
        boxes = _active_wall_boxes(base)
        ctrl = HumanGameController()

        # Walk forward (+X) & left (+Y) toward Stage 2 left corridor wall (y=1.30, half_y=0.10)
        start_x = base.target.x
        for _ in range(35):
            obs, _, terminated, _, info = _step_env_with_human_choice(
                env=base,
                action=np.array([0.4, 0.0], dtype=np.float32),
                human_mode=True,
                human_ctrl=ctrl,
                fwd_in=1.0,
                right_in=-1.0,
                cam_fwd_xy=(1.0, 0.0),
                sprint=True,
                slow=False,
                boxes=boxes,
            )
            assert not terminated
            assert obs.shape == (490,)
            assert base.observation_space.contains(obs)
            # Human must stay clear of the y=1.20 inner corridor wall face
            assert abs(base.target.y) <= 1.05
            assert "target_dist" in info

        assert base.target.x > start_x + 0.3

        # Holding Shift (sprint=True) speeds up past walk_speed (0.42 -> up to 1.05 m/s)
        assert base.target.speed > ctrl.walk_speed + 0.3

        # Releasing Shift returns smoothly back down to walk_speed
        for _ in range(25):
            _step_env_with_human_choice(
                env=base,
                action=np.array([0.0, 0.0], dtype=np.float32),
                human_mode=True,
                human_ctrl=ctrl,
                fwd_in=1.0,
                right_in=0.0,
                cam_fwd_xy=(1.0, 0.0),
                sprint=False,
                slow=False,
                boxes=boxes,
            )
        assert np.isclose(base.target.speed, ctrl.walk_speed, atol=0.02)

        # Releasing keys slows human back to a stop
        for _ in range(25):
            _step_env_with_human_choice(
                env=base,
                action=np.array([0.0, 0.0], dtype=np.float32),
                human_mode=True,
                human_ctrl=ctrl,
                fwd_in=0.0,
                right_in=0.0,
                cam_fwd_xy=(1.0, 0.0),
                sprint=False,
                slow=False,
                boxes=boxes,
            )
        assert ctrl.cur_speed == 0.0
    finally:
        env.close()


def test_3d_renderer_overlays_and_pip_camera():
    env = gym.make("AckermanFollow-v0")
    base = env.unwrapped
    obs, info = base.reset(seed=7, options={"stage": 5, "force_front_spawn": True})
    renderer = mujoco.Renderer(base.model, height=240, width=320)
    try:
        cam = mujoco.MjvCamera()
        pip_cam = mujoco.MjvCamera()

        for preset in (0, 1, 2, 3):
            update_camera_preset(
                cam=cam,
                preset_idx=preset,
                env=base,
                yaw_offset_deg=0.0,
                elev_offset_deg=0.0,
                dist_scale=1.0,
                pan_offset=np.zeros(3),
                sim_time=0.5,
            )
            renderer.update_scene(base.data, camera=cam)
            append_3d_overlays(
                renderer.scene,
                base,
                obs,
                info,
                robot_trail=[(0.0, 0.0), (0.2, 0.0)],
                human_trail=[(1.6, 0.0), (1.8, 0.0)],
            )
            frame = renderer.render()
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8
            assert renderer.scene.ngeom < renderer.scene.maxgeom

        configure_robot_pip_camera(pip_cam, base)
        renderer.update_scene(base.data, camera=pip_cam)
        pip_frame = renderer.render()
        assert pip_frame.shape == (240, 320, 3)
    finally:
        renderer.close()
        env.close()


def test_live_video_recorder_writes_mp4(tmp_path):
    rec = LiveVideoRecorder(fps=20)
    assert rec.is_recording is False

    out_mp4 = tmp_path / "live_test.mp4"
    # Odd dimensions are clamped to even for libx264 yuv420p
    rec.start(321, 241, out_mp4)
    assert rec.is_recording is True
    assert (rec.width, rec.height) == (320, 240)

    # Normal matching frame + mismatched-size frame (simulating window resize)
    frame_a = np.full((240, 320, 3), 120, dtype=np.uint8)
    frame_b = np.full((260, 340, 3), 200, dtype=np.uint8)
    rec.write_frame(frame_a, flip_vertical=True)
    rec.write_frame(frame_b, flip_vertical=True)

    saved_path, n_frames = rec.stop()
    assert rec.is_recording is False
    assert saved_path == out_mp4
    assert n_frames == 2
    assert out_mp4.exists() and out_mp4.stat().st_size > 0

