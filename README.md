# Ackerman Follow (MuJoCo)

Clean MuJoCo-native workspace for 4-wheel Ackermann target following with Recurrent Spatial PPO (`426-D` pure sensor input: `64` front camera depth rays + `360` planar LiDAR rays + `2` proprioception states).

## Directory Structure

```text
ackerman_follow/
├── spec.txt                        # Canonical physical & sensor specifications
├── configs/
│   ├── robot_params.yaml           # Chassis, Ackermann steering, camera & LiDAR params
│   └── ppo_follow.yaml             # Recurrent PPO & 4-stage curriculum hyperparameters
├── ackerman_follow/
│   ├── config.py                   # RobotConfig dataclass & select_device()
│   ├── assets/                     # MuJoCo MJCF XML models (robot, human, arenas)
│   ├── envs/
│   │   ├── __init__.py             # Gymnasium registration for AckermanFollow-v0
│   │   └── human_target.py         # BipedalHumanTarget gait & clothing color model
│   └── ppo/
│       ├── actor_critic.py         # CoordConv + Polar Soft-Argmin + GRU + TanhNormal
│       └── buffer.py               # Multi-env recurrent RolloutBuffer (GAE + BPTT)
├── scripts/
│   └── train_ppo.py                # Vectorized Recurrent PPO trainer (SyncVectorEnv)
└── tests/
    └── test_ppo.py                 # Policy & recurrent buffer unit tests
```

## Next Steps (MuJoCo Implementation)

1. **MJCF Model (`ackerman_follow/assets/ackbot.xml`):**
   - 4-wheel rear-drive, front-Ackermann chassis matching `spec.txt` / `configs/robot_params.yaml` (`wheelbase = 0.30 m`, `track = 0.30 m`, `max_steering_angle = 0.553953 rad`, `v ∈ [-0.40, 0.80] m/s`).
   - Dynamic human walker body (two alternating leg/foot geoms or capsule walker) + arena walls / Stage 2 L-wall / Stage 3 corridor / Stage 4 pillars.
2. **MuJoCo Gymnasium Env (`ackerman_follow/envs/mujoco_follow_env.py`):**
   - Implement `AckermanFollow-v0` using `mujoco.mj_multiRay` for fast vectorized `64`-ray front depth + `360`-ray LiDAR (`3.00 m` max range) and MuJoCo native viewer (`mujoco.viewer`).
   - Port the 13-term continuous 360° bearing, potential-based turn shaping, Ackermann recovery, and 4-stage curriculum logic.
3. **Verify & Train:**
   ```bash
   uv sync --extra dev
   uv run --extra dev pytest tests/ -v
   uv run python scripts/train_ppo.py --exp-name mujoco_recurrent_ppo
   ```
