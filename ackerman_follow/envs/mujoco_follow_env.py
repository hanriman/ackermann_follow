"""MuJoCo Gymnasium environment (`AckermanFollow-v0`) for 4-wheel Ackermann target following."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ackerman_follow.config import RobotConfig, get_robot_config
from ackerman_follow.envs.human_target import BipedalHumanTarget

DEFAULT_XML_PATH = Path(__file__).resolve().parent.parent / "assets" / "ackbot.xml"


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert planar yaw angle (rad) to MuJoCo [w, x, y, z] quaternion."""
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _quat_to_yaw(quat: np.ndarray) -> float:
    """Extract planar yaw angle (rad) from MuJoCo [w, x, y, z] quaternion."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(angle: float) -> float:
    """Wrap angle into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class MuJoCoAckermanFollowEnv(gym.Env):
    """
    4-wheel Rear-Drive, Front-Ackermann target-following environment in MuJoCo.

    Observation (`426-D` float32):
        - `[0:64]`   Front depth camera rays (`85°` HFOV, right-to-left, normalized `[0, 1]`)
        - `[64:424]` Planar 2D LiDAR rays (`360°`, CCW from `+X`, normalized `[0, 1]`)
        - `[424:426]` Normalized proprioception `[v_norm, steer_norm]` in `[-1, 1]`

    Action (`2-D` float32):
        - `[v_cmd, steer_cmd]` in physical units (`m/s`, `rad`)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    DEFAULT_STAGE_MAX_STEPS = {1: 500, 2: 450, 3: 800, 4: 700, 5: 750, 6: 750, 7: 950}

    def __init__(
        self,
        render_mode: str | None = None,
        xml_path: str | Path | None = None,
        robot_cfg: RobotConfig | None = None,
        max_steps: int | None = None,
        stage: int = 1,
        auto_curriculum: bool = False,
        min_episodes_per_stage: int = 35,
        stage_thresholds: dict[int, float] | None = None,
        replay_previous_ratio: float = 0.45,
        sensor_noise: bool = False,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.cfg = robot_cfg if robot_cfg is not None else get_robot_config()
        self._fixed_max_steps = int(max_steps) if max_steps is not None else None
        self.max_steps = self._fixed_max_steps or self.DEFAULT_STAGE_MAX_STEPS[int(np.clip(stage, 1, 7))]

        # Curriculum & domain randomization configuration (synchronized via train_ppo.py)
        self.unlocked_stage = int(np.clip(stage, 1, 7))
        self.stage = self.unlocked_stage
        self.auto_curriculum = bool(auto_curriculum)
        self.min_episodes_per_stage = int(min_episodes_per_stage)
        self.stage_thresholds = stage_thresholds or {
            1: 540.0,
            2: 430.0,
            3: 680.0,
            4: 620.0,
            5: 680.0,
            6: 620.0,
        }
        self.replay_previous_ratio = float(replay_previous_ratio)
        self.sensor_noise = bool(sensor_noise)

        # Load MuJoCo model and data
        model_file = Path(xml_path) if xml_path is not None else DEFAULT_XML_PATH
        self.model = mujoco.MjModel.from_xml_path(str(model_file))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = max(1, int(round(self.cfg.dt / float(self.model.opt.timestep))))

        # Gymnasium spaces (64 cam depth + 64 cam target mask + 360 LiDAR + 2 state = 490)
        obs_low = np.concatenate(
            [
                np.zeros(
                    self.cfg.cam_samples + self.cfg.cam_mask_samples + self.cfg.lidar_samples,
                    dtype=np.float32,
                ),
                np.full(2, -1.0, dtype=np.float32),
            ]
        )
        obs_high = np.ones(self.cfg.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        act_low = np.array([self.cfg.v_min, -self.cfg.max_steering_angle], dtype=np.float32)
        act_high = np.array([self.cfg.v_max, self.cfg.max_steering_angle], dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # Cache MuJoCo entity IDs
        self._init_mj_ids()
        self._default_floor_friction = float(self.model.geom_friction[self._floor_gid, 0])

        # Precompute sensor ray directions in sensor local frames
        cam_angles = np.linspace(
            -0.5 * self.cfg.cam_fov_rad,
            0.5 * self.cfg.cam_fov_rad,
            self.cfg.cam_samples,
            dtype=np.float64,
        )
        self._cam_local_vecs = np.stack(
            [np.cos(cam_angles), np.sin(cam_angles), np.zeros_like(cam_angles)], axis=-1
        )
        lidar_angles = np.linspace(
            0.0, 2.0 * math.pi, self.cfg.lidar_samples, endpoint=False, dtype=np.float64
        )
        self._lidar_local_vecs = np.stack(
            [np.cos(lidar_angles), np.sin(lidar_angles), np.zeros_like(lidar_angles)], axis=-1
        )

        # Reusable C buffers for mj_multiRay (only sense group 0: world obstacles & human)
        self._ray_geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._cam_geomid = np.empty(self.cfg.cam_samples, dtype=np.int32)
        self._cam_dist = np.empty(self.cfg.cam_samples, dtype=np.float64)
        self._lidar_geomid = np.empty(self.cfg.lidar_samples, dtype=np.int32)
        self._lidar_dist = np.empty(self.cfg.lidar_samples, dtype=np.float64)

        # Bipedal target model & episode state
        self.target = BipedalHumanTarget(rng=self.np_random)
        self._base_target_speed = 0.32
        self._adaptive_pacing = True
        self._speed_mult = 1.0
        self._behavior_timer = 80
        self._allow_pauses = True
        self._allow_reversals = False
        self._waypoints: list[tuple[float, float]] = []
        self._wp_idx = 0
        self._wp_dir = 1
        self._wp_loop = True
        self._step_count = 0
        self._steer_cmd = 0.0
        self._prev_steer_cmd = 0.0
        self._prev_dist_err = 0.0
        self._prev_abs_bearing = 0.0
        self._last_min_lidar = self.cfg.lidar_range_max
        self._last_min_front_cam = self.cfg.cam_range_max
        self._last_cam_target_visible = True
        self._mask_dropout_steps = 0
        self._lidar_yaw_roll = 0

        # Lazy viewer / renderer handles (never allocated during headless training)
        self._viewer: Any = None
        self._renderer: mujoco.Renderer | None = None

    def _init_mj_ids(self) -> None:
        m = self.model
        self._chassis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self._cam_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cam_site")
        self._lidar_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")

        # Joint qpos / dof indices
        self._robot_free_qpos = int(
            m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")]
        )
        self._robot_free_dof = int(
            m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")]
        )
        self._fl_steer_qpos = int(
            m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "fl_steer")]
        )
        self._fr_steer_qpos = int(
            m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "fr_steer")]
        )

        # Actuator IDs
        self._act_steer_fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer_fl")
        self._act_steer_fr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer_fr")
        self._act_drive_rl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_rl")
        self._act_drive_rr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_rr")

        # Mocap human IDs & shirt geom ID for episode appearance updates
        self._mocap_leg_l = int(
            m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "human_leg_l")]
        )
        self._mocap_leg_r = int(
            m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "human_leg_r")]
        )
        self._mocap_torso = int(
            m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "human_torso")]
        )
        self._shirt_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "human_torso_geom")
        self._human_gids = np.array(
            [
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "human_leg_l_geom"),
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "human_leg_r_geom"),
                self._shirt_geom_id,
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "human_head_geom"),
            ],
            dtype=np.int32,
        )

        # Stage body & geom IDs for sub-millisecond curriculum switching
        self._stage_bodies: dict[int, int] = {}
        self._stage_geoms: dict[int, np.ndarray] = {}
        for stg, bname in (
            (2, "stage2_obstacles"),
            (3, "stage3_obstacles"),
            (4, "stage4_obstacles"),
            (5, "stage5_obstacles"),
            (7, "stage7_obstacles"),
        ):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
            self._stage_bodies[stg] = bid
            self._stage_geoms[stg] = np.flatnonzero(m.geom_bodyid == bid)

        # Contact classification sets (robot collision geoms vs group-0 obstacles/human)
        floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._robot_col_geoms = set(
            int(gid)
            for gid in range(m.ngeom)
            if m.geom_group[gid] == 1 and m.geom_contype[gid] != 0
        )
        self._floor_gid = int(floor_gid)

    def set_unlocked_stage(self, unlocked_stage: int) -> None:
        """Synchronize the maximum unlocked curriculum stage from the PPO trainer."""
        self.unlocked_stage = int(np.clip(unlocked_stage, 1, 7))

    def _apply_stage_geometry(self, stage: int) -> None:
        """Enable only the active stage's obstacle geoms (Stages 5 & 6 share dead-end walls)."""
        for stg, bid in self._stage_bodies.items():
            active = (stg == stage) or (stg == 5 and stage == 6)
            self.model.body_pos[bid, 2] = 0.0 if active else -5.0
            gids = self._stage_geoms[stg]
            self.model.geom_group[gids] = 0 if active else 3
            self.model.geom_contype[gids] = 2 if active else 0

    def _sample_stage(self, options: dict[str, Any] | None) -> int:
        if options and "stage" in options and options["stage"] is not None:
            return int(np.clip(int(options["stage"]), 1, 7))
        if not self.auto_curriculum or self.unlocked_stage <= 1:
            return self.unlocked_stage
        if float(self.np_random.random()) < self.replay_previous_ratio:
            return int(self.np_random.integers(1, self.unlocked_stage))
        return self.unlocked_stage

    def _sync_human_mocap(self) -> None:
        """Write current BipedalHumanTarget leg and torso poses to MuJoCo mocap bodies."""
        left_leg, right_leg = self.target.get_feet_positions()
        q_human = _yaw_to_quat(self.target.yaw)

        self.data.mocap_pos[self._mocap_leg_l] = [left_leg.x, left_leg.y, 0.43]
        self.data.mocap_quat[self._mocap_leg_l] = q_human

        self.data.mocap_pos[self._mocap_leg_r] = [right_leg.x, right_leg.y, 0.43]
        self.data.mocap_quat[self._mocap_leg_r] = q_human

        bob_z = 0.02 * abs(math.sin(self.target.gait_phase))
        self.data.mocap_pos[self._mocap_torso] = [self.target.x, self.target.y, 1.18 + bob_z]
        self.data.mocap_quat[self._mocap_torso] = q_human

    def _setup_scenario(self, stage: int, options: dict[str, Any] | None = None) -> None:
        """Initialize robot pose, human target spawn, lane waypoints, pacing, and friction."""
        rng = self.np_random
        force_front = bool(options.get("force_front_spawn", False)) if options else False
        force_lost = bool(options.get("force_lost_spawn", False)) if options else False

        self._adaptive_pacing = True
        self._allow_pauses = not force_front
        self._allow_reversals = (not force_front) and (stage in (1, 3, 4))
        self._speed_mult = 1.0
        self._behavior_timer = int(rng.integers(40, 101))
        self._wp_dir = 1
        self._wp_loop = True
        self._mask_dropout_steps = 0

        if self.sensor_noise:
            self._lidar_yaw_roll = int(rng.integers(-1, 2))
            self.model.geom_friction[self._floor_gid, 0] = self._default_floor_friction * float(
                rng.uniform(0.85, 1.15)
            )
        else:
            self._lidar_yaw_roll = 0
            self.model.geom_friction[self._floor_gid, 0] = self._default_floor_friction

        robot_yaw = float(rng.uniform(-0.20, 0.20))

        if stage == 1:
            # Stage 1: Free obstacle open arena (randomized robot & target spawns)
            robot_x = float(rng.uniform(-1.5, 1.5))
            robot_y = float(rng.uniform(-1.5, 1.5))
            robot_yaw = float(rng.uniform(-math.pi, math.pi))
            dist = float(rng.uniform(1.4, 3.8))
            bearing = float(rng.uniform(-0.45, 0.45))
            hx = float(np.clip(robot_x + dist * math.cos(robot_yaw + bearing), -5.5, 5.5))
            hy = float(np.clip(robot_y + dist * math.sin(robot_yaw + bearing), -5.5, 5.5))
            hyaw = _wrap_angle(robot_yaw + float(rng.uniform(-0.45, 0.45)))
            speed = float(rng.uniform(0.24, 0.46))
            self._waypoints = []
        elif stage == 2:
            # Stage 2: Straight corridor (2.40 m wide, center/left/right wall lines, NO turnaround)
            robot_x = float(rng.uniform(-5.8, -4.6))
            robot_y = float(rng.uniform(-0.55, 0.55))
            robot_yaw = float(rng.uniform(-0.15, 0.15))
            lane_y = float(rng.choice([-0.75, 0.0, 0.75]))
            dist = float(rng.uniform(1.4, 3.6))
            bearing = float(rng.uniform(-0.22, 0.22))
            hx = float(np.clip(robot_x + dist * math.cos(robot_yaw + bearing), -4.4, -0.2))
            hy = float(np.clip(lane_y + float(rng.uniform(-0.10, 0.10)), -0.85, 0.85))
            hyaw = float(rng.uniform(-0.12, 0.12))
            speed = float(rng.uniform(0.26, 0.44))
            self._waypoints = [(2.0, lane_y), (6.1, lane_y)]
            self._wp_loop = False
        elif stage == 3:
            # Stage 3: L-turn (open outer side) with far pre-corner spawns & lane offset diversity
            robot_x = float(rng.uniform(-0.25, 0.35))
            robot_y = float(rng.uniform(0.85, 1.55))
            robot_yaw = float(rng.uniform(-0.18, 0.18))
            dist = float(rng.uniform(1.4, 4.5))
            bearing = float(rng.uniform(-0.22, 0.22))
            hx = float(np.clip(robot_x + dist * math.cos(robot_yaw + bearing), 1.25, 4.75))
            hy = float(np.clip(robot_y + dist * math.sin(robot_yaw + bearing), 0.85, 1.65))
            hyaw = float(rng.uniform(-0.20, 0.20))
            if (not force_front) and rng.random() < 0.50:
                self._adaptive_pacing = False
                speed = float(rng.uniform(0.40, 0.58))
            else:
                speed = float(rng.uniform(0.32, 0.46))
            lane_shift = float(rng.choice([0.0, 0.25, 0.55]))
            self._waypoints = [
                (5.25 + lane_shift, 1.15 + lane_shift),
                (5.25 + lane_shift, -3.05 - lane_shift),
                (1.90 - lane_shift, -3.05 - lane_shift),
                (1.90 - lane_shift, 1.15 + lane_shift),
            ]
        elif stage == 4:
            # Stage 4: L-turn inside corridor (both walls) with center/left/right lines & far spawns
            robot_x = float(rng.uniform(-0.30, 0.25))
            robot_y = float(rng.uniform(-0.45, 0.45))
            robot_yaw = float(rng.uniform(-0.18, 0.18))
            lane_offset = float(rng.choice([-0.45, 0.0, 0.45]))
            dist = float(rng.uniform(1.4, 4.0))
            bearing = float(rng.uniform(-0.20, 0.20))
            hx = float(np.clip(robot_x + dist * math.cos(robot_yaw + bearing), 1.20, 4.25))
            hy = float(
                np.clip(-lane_offset + float(rng.uniform(-0.10, 0.10)), -0.65, 0.65)
            )
            hyaw = float(rng.uniform(-0.15, 0.15))
            if (not force_front) and rng.random() < 0.50:
                self._adaptive_pacing = False
                speed = float(rng.uniform(0.38, 0.54))
            else:
                speed = float(rng.uniform(0.28, 0.42))
            self._waypoints = [
                (4.70 + lane_offset, -lane_offset),
                (4.70 + lane_offset, -4.10),
                (6.70, -4.10),
                (6.70, 2.00),
                (-0.50, 2.00),
                (-0.50, -lane_offset),
            ]
        elif stage == 5:
            # Stage 5: Dead-end turnaround where target walks near right or left wall
            robot_x = float(rng.uniform(-0.40, 0.25))
            robot_y = float(rng.uniform(-0.45, 0.45))
            robot_yaw = float(rng.uniform(-0.18, 0.18))
            dist = float(rng.uniform(1.4, 2.4))
            bearing = float(rng.uniform(-0.20, 0.20))
            hx = robot_x + dist * math.cos(robot_yaw + bearing)
            hy = float(np.clip(robot_y + dist * math.sin(robot_yaw + bearing), -0.55, 0.55))
            hyaw = float(rng.uniform(-0.15, 0.15))
            speed = float(rng.uniform(0.30, 0.44))
            pass_side = float(rng.choice([-1.0, 1.0]))
            self._waypoints = [
                (3.85, -0.75 * pass_side),
                (4.20, 0.00),
                (3.85, 0.72 * pass_side),
                (1.60, 0.72 * pass_side),
                (-2.20, 0.45 * pass_side),
                (-4.50, -1.20 * pass_side),
                (-2.20, -2.20 * pass_side),
                (-1.40, 0.00),
            ]
        elif stage == 6:
            # Stage 6: Dead-end turnaround where target walks straight down the middle
            robot_x = float(rng.uniform(-0.40, 0.25))
            robot_y = float(rng.uniform(-0.40, 0.40))
            robot_yaw = float(rng.uniform(-0.18, 0.18))
            dist = float(rng.uniform(1.4, 2.4))
            bearing = float(rng.uniform(-0.20, 0.20))
            hx = robot_x + dist * math.cos(robot_yaw + bearing)
            hy = float(np.clip(float(rng.uniform(-0.15, 0.15)), -0.25, 0.25))
            hyaw = float(rng.uniform(-0.15, 0.15))
            speed = float(rng.uniform(0.30, 0.44))
            exit_side = float(rng.choice([-1.0, 1.0]))
            self._waypoints = [
                (3.85, 0.00),
                (4.20, 0.00),
                (1.60, 0.12 * exit_side),
                (-2.20, 0.00),
                (-4.50, -1.20 * exit_side),
                (-2.20, -2.20 * exit_side),
                (-1.40, 0.00),
            ]
        else:
            # Stage 7: L-turn into dead-end wing (wall-hug or middle wing + far pre-turn spawns)
            robot_x = float(rng.uniform(-0.35, 0.25))
            robot_y = float(rng.uniform(-0.40, 0.40))
            robot_yaw = float(rng.uniform(-0.18, 0.18))
            dist = float(rng.uniform(1.4, 4.5))
            bearing = float(rng.uniform(-0.20, 0.20))
            hx = float(np.clip(robot_x + dist * math.cos(robot_yaw + bearing), 1.25, 4.75))
            hy = float(np.clip(robot_y + dist * math.sin(robot_yaw + bearing), -0.65, 0.65))
            hyaw = float(rng.uniform(-0.15, 0.15))
            if (not force_front) and rng.random() < 0.50:
                self._adaptive_pacing = False
                speed = float(rng.uniform(0.40, 0.56))
            else:
                speed = float(rng.uniform(0.30, 0.44))
            lane_mode = float(rng.choice([-0.42, 0.0, 0.42]))
            if lane_mode == 0.0:
                x_in = 4.50
                x_out = 4.50 + 0.22 * float(rng.choice([-1.0, 1.0]))
            else:
                x_in = 4.50 - lane_mode
                x_out = 4.50 + lane_mode
            self._waypoints = [
                (x_in, 0.00),
                (x_in, -3.45),
                (4.50, -3.60),
                (x_out, -3.45),
                (x_out, 0.35),
                (1.50, 0.35),
                (0.00, 0.00),
            ]

        # Mid-episode "Target Lost / Behind Car" recovery starts (25% in Stages 1, 5, 6)
        if force_lost or (not force_front and stage in (1, 5, 6) and rng.random() < 0.25):
            yaw_offset = float(rng.choice([-1.0, 1.0]) * rng.uniform(1.10, math.pi))
            robot_yaw = _wrap_angle(robot_yaw + yaw_offset)

        self.data.qpos[self._robot_free_qpos : self._robot_free_qpos + 3] = [
            robot_x,
            robot_y,
            0.0,
        ]
        self.data.qpos[self._robot_free_qpos + 3 : self._robot_free_qpos + 7] = _yaw_to_quat(
            robot_yaw
        )

        self._wp_idx = 0
        self._base_target_speed = speed
        self.target.reset(x=hx, y=hy, yaw=hyaw, speed=speed, rng=rng)
        self.model.geom_rgba[self._shirt_geom_id, :3] = self.target.color_rgb
        self._sync_human_mocap()

    def _step_human_target(self) -> None:
        """Advance human target with variable speeds, pauses, and sudden reversals."""
        self._behavior_timer -= 1
        if self._behavior_timer <= 0:
            rng = self.np_random
            if self._allow_pauses and rng.random() < 0.15:
                self._speed_mult = 0.0
                self._behavior_timer = int(rng.integers(30, 56))
            elif self._allow_reversals and rng.random() < 0.14:
                self._speed_mult = float(rng.uniform(0.75, 1.25))
                self._behavior_timer = int(rng.integers(45, 95))
                if self._waypoints:
                    self._wp_dir = -self._wp_dir
                    self._wp_idx = (self._wp_idx + self._wp_dir) % len(self._waypoints)
                else:
                    self.target.yaw = _wrap_angle(
                        self.target.yaw + math.pi + float(rng.uniform(-0.35, 0.35))
                    )
            else:
                self._speed_mult = float(rng.uniform(0.65, 1.35))
                self._behavior_timer = int(rng.integers(40, 101))

        if self.stage >= 3 and self._adaptive_pacing:
            rx, ry, _, _, _ = self._get_robot_pose_and_vel()
            dist = math.hypot(self.target.x - rx, self.target.y - ry)
            pace = float(np.clip(2.4 / max(2.4, dist), 0.55, 1.0))
            raw_speed = self._base_target_speed * pace * self._speed_mult
        else:
            raw_speed = self._base_target_speed * self._speed_mult

        self.target.speed = 0.0 if self._speed_mult == 0.0 else float(np.clip(raw_speed, 0.22, 0.65))

        if self._waypoints:
            wx, wy = self._waypoints[self._wp_idx]
            dx = wx - self.target.x
            dy = wy - self.target.y
            if math.hypot(dx, dy) < 0.45:
                if self._wp_loop:
                    self._wp_idx = (self._wp_idx + self._wp_dir) % len(self._waypoints)
                    wx, wy = self._waypoints[self._wp_idx]
                    dx = wx - self.target.x
                    dy = wy - self.target.y
                elif self._wp_idx < len(self._waypoints) - 1:
                    self._wp_idx += 1
                    wx, wy = self._waypoints[self._wp_idx]
                    dx = wx - self.target.x
                    dy = wy - self.target.y
                else:
                    self.target.speed = 0.0
                    self.target.omega = 0.0
            if self.target.speed > 0.0:
                desired_yaw = math.atan2(dy, dx)
                err = _wrap_angle(desired_yaw - self.target.yaw)
                self.target.omega = float(np.clip(2.0 * err, -1.2, 1.2))
        else:
            # Stage 1 smooth sinusoidal wandering with gentle arena boundary turn-back
            t_sec = self._step_count * self.cfg.dt
            wander = 0.35 * math.sin(0.45 * t_sec + 0.5)
            if abs(self.target.x) > 5.5 or abs(self.target.y) > 5.5:
                to_center = math.atan2(-self.target.y, -self.target.x)
                err = _wrap_angle(to_center - self.target.yaw)
                self.target.omega = float(np.clip(1.8 * err, -1.2, 1.2))
            else:
                self.target.omega = wander

        self.target.step(self.cfg.dt)
        # Clamp within outer arena bounds
        self.target.x = float(np.clip(self.target.x, -7.2, 7.2))
        self.target.y = float(np.clip(self.target.y, -7.2, 7.2))
        self._sync_human_mocap()

    def _get_robot_pose_and_vel(self) -> tuple[float, float, float, float, float]:
        """Return (x, y, yaw, v_forward, steer_center)."""
        pos = self.data.qpos[self._robot_free_qpos : self._robot_free_qpos + 3]
        quat = self.data.qpos[self._robot_free_qpos + 3 : self._robot_free_qpos + 7]
        yaw = _quat_to_yaw(quat)

        vx = float(self.data.qvel[self._robot_free_dof])
        vy = float(self.data.qvel[self._robot_free_dof + 1])
        v_fwd = vx * math.cos(yaw) + vy * math.sin(yaw)

        delta_fl = float(self.data.qpos[self._fl_steer_qpos])
        delta_fr = float(self.data.qpos[self._fr_steer_qpos])
        steer_center = 0.5 * (delta_fl + delta_fr)
        return float(pos[0]), float(pos[1]), yaw, v_fwd, steer_center

    def _get_obs(self) -> tuple[np.ndarray, float]:
        """Cast camera + LiDAR rays via mj_multiRay and build normalized 490-D observation."""
        cam_pnt = np.ascontiguousarray(self.data.site_xpos[self._cam_site_id], dtype=np.float64)
        cam_rot = self.data.site_xmat[self._cam_site_id].reshape(3, 3)
        cam_vecs = np.ascontiguousarray(self._cam_local_vecs @ cam_rot.T, dtype=np.float64).ravel()

        mujoco.mj_multiRay(
            self.model,
            self.data,
            cam_pnt,
            cam_vecs,
            self._ray_geomgroup,
            True,
            -1,
            self._cam_geomid,
            self._cam_dist,
            None,
            self.cfg.cam_samples,
            self.cfg.cam_range_max,
        )
        cam_meters = np.where(
            (self._cam_dist < 0.0) | (self._cam_dist > self.cfg.cam_range_max),
            self.cfg.cam_range_max,
            self._cam_dist,
        )

        # RGB-D target-detection mask: fill horizontal bounding-box span [min_hit .. max_hit]
        raw_hits = (
            np.isin(self._cam_geomid, self._human_gids)
            & (self._cam_dist >= 0.0)
            & (self._cam_dist <= self.cfg.cam_range_max)
        )
        self._last_cam_target_visible = bool(np.any(raw_hits))
        non_human_cam = np.where(raw_hits, self.cfg.cam_range_max, cam_meters)
        self._last_min_front_cam = float(np.min(non_human_cam))

        cam_mask = np.zeros(self.cfg.cam_mask_samples, dtype=np.float32)
        if self._last_cam_target_visible:
            hit_indices = np.flatnonzero(raw_hits)
            i0, i1 = int(hit_indices[0]), int(hit_indices[-1])
            target_depth_max = float(np.max(cam_meters[hit_indices])) + 0.35
            for idx in range(i0, i1 + 1):
                if raw_hits[idx] or cam_meters[idx] <= target_depth_max:
                    cam_mask[idx] = 1.0

        lidar_pnt = np.ascontiguousarray(self.data.site_xpos[self._lidar_site_id], dtype=np.float64)
        lidar_rot = self.data.site_xmat[self._lidar_site_id].reshape(3, 3)
        lidar_vecs = np.ascontiguousarray(
            self._lidar_local_vecs @ lidar_rot.T, dtype=np.float64
        ).ravel()

        mujoco.mj_multiRay(
            self.model,
            self.data,
            lidar_pnt,
            lidar_vecs,
            self._ray_geomgroup,
            True,
            -1,
            self._lidar_geomid,
            self._lidar_dist,
            None,
            self.cfg.lidar_samples,
            self.cfg.lidar_range_max,
        )
        lidar_meters = np.where(
            (self._lidar_dist < 0.0) | (self._lidar_dist > self.cfg.lidar_range_max),
            self.cfg.lidar_range_max,
            self._lidar_dist,
        )
        # Measure obstacle clearance strictly against static walls/pillars (ignoring human legs)
        obstacle_lidar_meters = np.where(
            np.isin(self._lidar_geomid, self._human_gids),
            self.cfg.lidar_range_max,
            lidar_meters,
        )
        min_lidar = float(np.min(obstacle_lidar_meters))

        _, _, _, v_fwd, steer_center = self._get_robot_pose_and_vel()

        if self.sensor_noise:
            rng = self.np_random
            cam_meters = cam_meters + rng.normal(0.0, 0.025, size=cam_meters.shape)
            cam_drop = rng.random(cam_meters.shape) < 0.02
            cam_meters = np.where(cam_drop, self.cfg.cam_range_max, cam_meters)

            if self._mask_dropout_steps > 0:
                cam_mask.fill(0.0)
                self._mask_dropout_steps -= 1
            elif self._last_cam_target_visible and rng.random() < 0.06:
                cam_mask.fill(0.0)
                self._mask_dropout_steps = int(rng.integers(0, 3))
            elif self._last_cam_target_visible and rng.random() < 0.15:
                shift = int(rng.choice([-1, 1]))
                shifted = np.roll(cam_mask, shift)
                if shift > 0:
                    shifted[:shift] = 0.0
                else:
                    shifted[shift:] = 0.0
                cam_mask = shifted

            if self._lidar_yaw_roll != 0:
                lidar_meters = np.roll(lidar_meters, self._lidar_yaw_roll)
            lidar_meters = lidar_meters + rng.normal(0.0, 0.020, size=lidar_meters.shape)
            lidar_drop = rng.random(lidar_meters.shape) < 0.03
            lidar_meters = np.where(lidar_drop, self.cfg.lidar_range_max, lidar_meters)

            v_fwd = v_fwd + float(rng.normal(0.0, 0.02))
            steer_center = steer_center + float(rng.normal(0.0, 0.02))

        cam_norm = self.cfg.normalize_cam(cam_meters).astype(np.float32)
        lidar_norm = self.cfg.normalize_lidar(lidar_meters).astype(np.float32)
        state_norm = self.cfg.normalize_state(v_fwd, steer_center)

        obs = np.concatenate([cam_norm, cam_mask, lidar_norm, state_norm], axis=0).astype(
            np.float32
        )
        return obs, min_lidar

    def _compute_target_metrics(self) -> tuple[float, float, bool]:
        """Return (distance_m, relative_bearing_rad, target_in_fov)."""
        rx, ry, ryaw, _, _ = self._get_robot_pose_and_vel()
        dx = self.target.x - rx
        dy = self.target.y - ry
        dist = math.hypot(dx, dy)
        bearing = _wrap_angle(math.atan2(dy, dx) - ryaw)
        in_fov = abs(bearing) <= (0.5 * self.cfg.cam_fov_rad) and dist <= self.cfg.cam_range_max
        return dist, bearing, in_fov

    def _check_collision(self) -> bool:
        """Return True if any robot collision geom contacts a non-floor world/human geom."""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == self._floor_gid or g2 == self._floor_gid:
                continue
            if (g1 in self._robot_col_geoms and g2 not in self._robot_col_geoms) or (
                g2 in self._robot_col_geoms and g1 not in self._robot_col_geoms
            ):
                return True
        return False

    def _apply_ackermann_control(self, v_cmd: float, steer_target: float) -> None:
        """Apply rate-limited Ackermann front wheel steering and differential rear-wheel drive."""
        v_cmd = float(np.clip(v_cmd, self.cfg.v_min, self.cfg.v_max))
        steer_target = float(
            np.clip(steer_target, -self.cfg.max_steering_angle, self.cfg.max_steering_angle)
        )

        max_delta = self.cfg.max_steering_rate * self.cfg.dt
        self._steer_cmd = float(
            np.clip(steer_target, self._steer_cmd - max_delta, self._steer_cmd + max_delta)
        )

        L = self.cfg.wheelbase
        W = self.cfg.track_width
        r_w = self.cfg.wheel_radius

        if abs(self._steer_cmd) < 1e-4:
            delta_fl = self._steer_cmd
            delta_fr = self._steer_cmd
            omega_rl = v_cmd / r_w
            omega_rr = v_cmd / r_w
        else:
            tan_d = math.tan(self._steer_cmd)
            turn_r = L / tan_d
            delta_fl = math.atan(L / (turn_r - 0.5 * W))
            delta_fr = math.atan(L / (turn_r + 0.5 * W))
            omega_rl = (v_cmd * (1.0 - (W * tan_d) / (2.0 * L))) / r_w
            omega_rr = (v_cmd * (1.0 + (W * tan_d) / (2.0 * L))) / r_w

        self.data.ctrl[self._act_steer_fl] = delta_fl
        self.data.ctrl[self._act_steer_fr] = delta_fr
        self.data.ctrl[self._act_drive_rl] = omega_rl
        self.data.ctrl[self._act_drive_rr] = omega_rr

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.stage = self._sample_stage(options)
        self.max_steps = (
            self._fixed_max_steps
            if self._fixed_max_steps is not None
            else self.DEFAULT_STAGE_MAX_STEPS[self.stage]
        )
        self._apply_stage_geometry(self.stage)
        self._setup_scenario(self.stage, options=options)

        self._step_count = 0
        self._steer_cmd = 0.0
        self._prev_steer_cmd = 0.0

        mujoco.mj_forward(self.model, self.data)
        obs, min_lidar = self._get_obs()
        self._last_min_lidar = min_lidar

        dist, bearing, geom_in_fov = self._compute_target_metrics()
        in_fov = geom_in_fov and (self._last_cam_target_visible or dist <= 1.15)
        self._prev_dist_err = abs(dist - self.cfg.target_follow_distance)
        self._prev_abs_bearing = abs(bearing)

        info = {
            "stage": self.stage,
            "unlocked_stage": self.unlocked_stage,
            "target_dist": dist,
            "target_bearing": bearing,
            "target_in_fov": in_fov,
            "collision": False,
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        act = np.asarray(action, dtype=np.float32).reshape(2)
        v_cmd = float(act[0])
        steer_target = float(act[1])

        self._prev_steer_cmd = self._steer_cmd
        self._apply_ackermann_control(v_cmd, steer_target)
        self._step_human_target()

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        self._step_count += 1

        obs, min_lidar = self._get_obs()
        self._last_min_lidar = min_lidar
        dist, bearing, geom_in_fov = self._compute_target_metrics()
        in_fov = geom_in_fov and (self._last_cam_target_visible or dist <= 1.15)
        collision = self._check_collision()

        terminated = collision
        truncated = (not terminated) and (self._step_count >= self.max_steps)

        if collision:
            reward = -25.0
        else:
            _, _, _, v_fwd, _ = self._get_robot_pose_and_vel()
            dist_err = abs(dist - self.cfg.target_follow_distance)
            dist_reward = 1.2 * math.exp(-((dist_err / 0.65) ** 2))
            # Gate proximity bonus while keeping dense cos(bearing) cue at 3-5 m / behind-car angles
            dist_scale = math.exp(-((dist_err / 1.40) ** 2))
            align_reward = 0.15 * math.cos(bearing) + dist_scale * (
                0.35 * math.cos(bearing) + (0.40 if in_fov else 0.0)
            )

            # Potential-based distance and bearing progress shaping
            dist_progress = 0.6 * (self._prev_dist_err - dist_err)
            turn_progress = 0.4 * (self._prev_abs_bearing - abs(bearing))
            self._prev_dist_err = dist_err
            self._prev_abs_bearing = abs(bearing)

            max_delta = max(1e-6, self.cfg.max_steering_rate * self.cfg.dt)
            steer_smooth_pen = -0.05 * ((self._steer_cmd - self._prev_steer_cmd) / max_delta) ** 2

            # Penalize reversing away only when target is ahead (allow K-turns when target is behind)
            rev_frac = max(0.0, -v_cmd / abs(self.cfg.v_min))
            far_rev_mult = (
                5.0
                if (dist > (self.cfg.target_follow_distance + 0.15) and abs(bearing) < 1.2)
                else 1.0
            )
            reverse_pen = -0.08 * far_rev_mult * rev_frac

            # Anti-freeze penalty: discourage standing still when target is lost out of FOV
            idle_lost_pen = -0.18 if (not in_fov and abs(v_fwd) < 0.10) else 0.0

            # Protect front wheels (~0.39m from rear LiDAR) with steep close-wall gradient
            lidar_lin = max(0.0, (0.62 - min_lidar) / 0.62)
            lidar_quad = max(0.0, (0.52 - min_lidar) / 0.52) ** 2
            front_lin = max(0.0, (0.50 - self._last_min_front_cam) / 0.50)
            front_quad = max(0.0, (0.42 - self._last_min_front_cam) / 0.42) ** 2
            clearance_pen = -(0.55 * lidar_lin + 1.50 * lidar_quad + 0.45 * front_lin + 1.00 * front_quad)
            personal_space_pen = -0.25 * max(0.0, (0.35 - dist) / 0.35)

            reward = (
                dist_reward
                + align_reward
                + dist_progress
                + turn_progress
                + steer_smooth_pen
                + reverse_pen
                + idle_lost_pen
                + clearance_pen
                + personal_space_pen
            )

        if self.render_mode == "human":
            self.render()

        info = {
            "stage": self.stage,
            "unlocked_stage": self.unlocked_stage,
            "target_dist": dist,
            "target_bearing": bearing,
            "target_in_fov": in_fov,
            "min_lidar": min_lidar,
            "collision": collision,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self._viewer.is_running():
                self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
