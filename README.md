# Ackerman Follow (MuJoCo)

Clean MuJoCo-native workspace for 4-wheel Ackermann target following with Recurrent Spatial PPO (`426-D` pure sensor input: `64` front camera depth rays + `360` planar LiDAR rays + `2` proprioception states).

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
│   │   └── ackbot.xml              # MuJoCo MJCF model (AckBot, human walker, 4-stage arenas)
│   ├── envs/
│   │   ├── __init__.py             # Gymnasium registration for AckermanFollow-v0
│   │   ├── human_target.py         # BipedalHumanTarget gait & clothing color model
│   │   └── mujoco_follow_env.py    # MuJoCoAckermanFollowEnv (mj_multiRay + curriculum)
│   └── ppo/
│       ├── actor_critic.py         # CoordConv + Polar Soft-Argmin + GRU + TanhNormal
│       └── buffer.py               # Multi-env recurrent RolloutBuffer (GAE + BPTT)
├── scripts/
│   ├── train_ppo.py                # Vectorized Recurrent PPO trainer (Sync/AsyncVectorEnv)
│   └── eval_policy.py              # Deterministic multi-stage evaluator & MuJoCo viewer
└── tests/
    ├── test_ppo.py                 # Policy, TanhNormal & recurrent GAE/buffer unit tests
    ├── test_config_and_target.py   # RobotConfig, BipedalHumanTarget & spawn-pickle tests
    └── test_env.py                 # MuJoCo MJCF, ray ordering, Ackermann & AsyncVectorEnv tests
```

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
```
