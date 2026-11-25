# Ackermann Follow (MuJoCo)

Clean MuJoCo-native workspace for 4-wheel Ackermann target following with modular RL & World-Model algorithms (`gru_ppo`, `lewm`, `lewm_rl_modelfree`, `lewm_tdmpc2`) over a `490-D` sensor input (`64` front camera depth rays + `64` RGB-D target-detection mask rays + `360` planar LiDAR rays + `2` proprioception states).

## Directory Structure

```text
ackermann_follow/
├── spec.txt                        # Canonical physical & sensor specifications
├── docs/
│   ├── architecture.md             # Sensor contract, Spatial-GRU PPO & Ackermann math
│   ├── hardware_and_dev_setup.md   # Linux VM (6c/12GB) & MacBook Pro M4 Pro (24GB) guide
│   └── mujoco_env_roadmap.md       # Interface contract & MJCF checklist for AckermannFollow-v0
├── configs/
│   ├── robot_params.yaml           # Chassis, Ackermann steering, camera & LiDAR params
│   ├── curriculum.yaml             # Shared 7-stage curriculum thresholds & replay ratio
│   ├── ppo_follow.yaml             # Legacy PPO config (still supported by train_ppo.py)
│   └── algos/                      # Per-algorithm configs (gru_ppo, lewm, lewm_rl_modelfree, lewm_tdmpc2)
├── ackermann_follow/
│   ├── config.py                   # RobotConfig, CPU thread guard & select_device()
│   ├── assets/ackbot.xml           # MuJoCo MJCF model (AckBot, human walker, 7-stage arenas)
│   ├── envs/                       # AckermannFollow-v0 & BipedalHumanTarget
│   ├── models/                     # SpatialSensorEncoder, heads, & world_models/lewm_jepa.py
│   ├── planners/                   # Batched MPPI latent trajectory optimizer
│   ├── buffers/                    # On-policy RolloutBuffer & off-policy SequenceReplayBuffer
│   ├── algos/                      # BaseAgent, registry, gru_ppo, lewm, lewm_rl_modelfree, lewm_tdmpc2
│   ├── engine/                     # Shared vec_env, curriculum, runner, evaluator, logging, checkpoint
│   └── ppo/                        # Backward-compatible ActorCritic & RolloutBuffer shims
├── scripts/
│   ├── train.py                    # Unified multi-algo trainer (--algo gru_ppo|lewm|lewm_rl_modelfree|lewm_tdmpc2)
│   ├── train_ppo.py                # Backward-compatible PPO trainer
│   ├── collect_data.py             # Multi-stage dataset collector (.npz) for LeWM pretraining
│   ├── eval_policy.py              # Deterministic 7-stage evaluator (works for all algos)
│   ├── demo_3d.py                  # Interactive 3D demo show & playable WASD human-target game
│   ├── record_video.py             # Sequential MP4 recorder (Stages 1..7)
│   └── record_custom_scenarios.py  # Custom edge-case showcase recorder (5 scenarios)
└── tests/
    ├── test_ppo.py                 # Policy, TanhNormal & recurrent GAE/buffer unit tests
    ├── test_config_and_target.py   # RobotConfig, BipedalHumanTarget & spawn-pickle tests
    ├── test_env.py                 # MuJoCo MJCF, ray ordering, Ackermann & AsyncVectorEnv tests
    ├── test_demo_3d.py             # 3D demo, human WASD wall sliding & live recording tests
    └── test_modular_algos.py       # Encoders, SequenceReplay, LeWM-JEPA, MPPI & all-algo integration tests
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
uv run --extra dev ruff check ackermann_follow scripts tests

# Train with unified multi-algorithm runner
uv run python scripts/train.py --algo gru_ppo --exp-name m4_ppo --device auto --vec-env async --num-envs 8
uv run python scripts/train.py --algo lewm --exp-name m4_lewm --device auto
uv run python scripts/train.py --algo lewm_rl_modelfree --exp-name m4_lewm_ppo --device auto
uv run python scripts/train.py --algo lewm_tdmpc2 --exp-name m4_tdmpc2 --device auto

# Collect offline multi-stage dataset and pretrain LeWM (JEPA)
uv run python scripts/collect_data.py --steps 50000 --output outputs/datasets/mixed_7stage.npz
uv run python scripts/train.py --algo lewm --dataset outputs/datasets/mixed_7stage.npz --exp-name lewm_offline

# Legacy PPO training on headless Linux VM (6 CPU cores, 12 GB RAM)
uv run python scripts/train_ppo.py --exp-name vm_ppo --device cpu --vec-env async --num-envs 6 --torch-threads 2

# Evaluate all stages or launch interactive MuJoCo viewer on macOS
uv run python scripts/eval_policy.py --checkpoint <path/to/best_model.pt> --stage 7 --episodes 5
uv run mjpython scripts/eval_policy.py --checkpoint <path/to/best_model.pt> --stage 7 --render

# Launch 3D Demo Show (choose Auto showcase or WASD Human Game Control)
uv run python scripts/demo_3d.py
uv run python scripts/demo_3d.py --mode human --stage 7
uv run python scripts/demo_3d.py --record outputs/video/ackbot_3d_demo.mp4 --stages 1 3 5 7

# Record evaluation & custom scenario showcase MP4s
uv run python scripts/record_video.py --checkpoint <path/to/best_model.pt> --stages 1 2 3 4 5 6 7
uv run python scripts/record_custom_scenarios.py --checkpoint <path/to/best_model.pt> --output outputs/video/ackbot_custom_scenarios.mp4
```
