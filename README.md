# Ackerman Follow (MuJoCo)

Clean MuJoCo-native workspace for 4-wheel Ackermann target following with Recurrent Spatial PPO (`490-D` sensor input: `64` front camera depth rays + `64` RGB-D target-detection mask rays + `360` planar LiDAR rays + `2` proprioception states).

## Directory Structure

```text
ackerman_follow/
├── spec.txt                        # Canonical physical & sensor specifications
├── docs/
│   ├── architecture.md             # Sensor contract, Spatial-GRU PPO & Ackermann math
│   ├── hardware_and_dev_setup.md   # Linux VM (6c/12GB) & MacBook Pro M4 Pro (24GB) guide
│   └── mujoco_env_roadmap.md       # Interface contract & MJCF checklist for AckermanFollow-v0
├── configs/
│   ├── robot_params.yaml           # Chassis, Ackermann steering, camera & LiDAR params
│   └── ppo_follow.yaml             # Recurrent PPO, curriculum & runtime hyperparameters
├── ackerman_follow/
│   ├── config.py                   # RobotConfig, CPU thread guard & select_device()
│   ├── assets/
│   │   └── ackbot.xml              # MuJoCo MJCF model (AckBot, human walker, 7-stage arenas)
│   ├── envs/
│   │   ├── __init__.py             # Gymnasium registration for AckermanFollow-v0
│   │   ├── human_target.py         # BipedalHumanTarget gait & clothing color model
│   │   └── mujoco_follow_env.py    # MuJoCoAckermanFollowEnv (mj_multiRay, 7-stage curriculum, sensory noise)
│   └── ppo/
│       ├── actor_critic.py         # CoordConv + Polar Soft-Argmin + GRU + TanhNormal
│       └── buffer.py               # Multi-env recurrent RolloutBuffer (GAE + BPTT)
├── scripts/
│   ├── train_ppo.py                # Vectorized Recurrent PPO trainer (Sync/AsyncVectorEnv)
│   ├── eval_policy.py              # Deterministic 7-stage evaluator & MuJoCo viewer
│   ├── record_video.py             # Sequential MP4 recorder (Stages 1..7)
│   └── record_custom_scenarios.py  # Custom edge-case showcase recorder (5 scenarios)
└── tests/
    ├── test_ppo.py                 # Policy, TanhNormal & recurrent GAE/buffer unit tests
    ├── test_config_and_target.py   # RobotConfig, BipedalHumanTarget & spawn-pickle tests
    └── test_env.py                 # MuJoCo MJCF, ray ordering, Ackermann & AsyncVectorEnv tests
```

## 7-Stage Curriculum

1. **Stage 1 — Free Obstacle:** Open `16 m × 16 m` arena with randomized spawns, `25%` behind-car lost starts, pauses, speed shifts, and sudden reversals.
2. **Stage 2 — Straight Corridor (No Turnaround):** `2.40 m` corridor with center, left-wall, and right-wall target walking paths.
3. **Stage 3 — L-Turn Outside Corridor:** Open-outer L-wall (`s3_lwall_a/b`) with far pre-corner spawns (`1.4–4.5 m`) and fast target escapes around the blind corner.
4. **Stage 4 — L-Turn Inside Corridor:** Closed `2.0 m` 90° corridor with lane offsets, far spawns, and sudden reversals.
5. **Stage 5 — Dead-End Turnaround (Wall-Hug):** `2.40 m` dead-end road where the target hugs the left or right wall, turns `180°`, and walks past the robot.
6. **Stage 6 — Dead-End Turnaround (Middle Walk):** `2.40 m` dead-end road where the target walks down the center (`y ≈ 0`), forcing the robot to yield and K-turn.
7. **Stage 7 — L-Turn → Dead-End:** Combined open approach → L-turn → dead-end wing → `180°` exit.

## Documentation

* [Architecture & Sensor Contract](docs/architecture.md)
* [Dual-Machine Hardware & Dev Setup (VM 6c/12GB & M4 Pro 24GB)](docs/hardware_and_dev_setup.md)
* [MuJoCo Environment Roadmap](docs/mujoco_env_roadmap.md)

## Quickstart

```bash
# Install dependencies & verify all unit tests + lint
uv sync --extra dev
uv run --extra dev pytest tests/ -v
uv run --extra dev ruff check ackerman_follow scripts tests

# Train on headless Linux VM (6 CPU cores, 12 GB RAM)
uv run python scripts/train_ppo.py --exp-name vm_ppo --device cpu --vec-env async --num-envs 6 --torch-threads 2

# Train on MacBook Pro M4 Pro (24 GB)
uv run python scripts/train_ppo.py --exp-name m4_ppo --device auto --vec-env async --num-envs 8

# Evaluate all stages or launch interactive MuJoCo viewer on macOS
uv run python scripts/eval_policy.py --checkpoint <path/to/best_model.pt> --stage 7 --episodes 5
uv run mjpython scripts/eval_policy.py --checkpoint <path/to/best_model.pt> --stage 7 --render

# Record evaluation & custom scenario showcase MP4s
uv run python scripts/record_video.py --checkpoint <path/to/best_model.pt> --stages 1 2 3 4 5 6 7
uv run python scripts/record_custom_scenarios.py --checkpoint <path/to/best_model.pt> --output outputs/video/ackbot_custom_scenarios.mp4
```
