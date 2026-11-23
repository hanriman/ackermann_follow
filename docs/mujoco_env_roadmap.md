# MuJoCo Environment (`AckermanFollow-v0`) Implementation Roadmap

This document captures the interface contract and physical modeling checklist required to connect `ackerman_follow/assets/ackbot.xml` and `ackerman_follow/envs/mujoco_follow_env.py` to `scripts/train_ppo.py`.

---

## 1. Gymnasium Interface Expected by `scripts/train_ppo.py`

* **Environment ID:** `AckermanFollow-v0` registered in `ackerman_follow/envs/__init__.py`.
* **Constructor Keyword Arguments:**
  * `auto_curriculum: bool`
  * `stage: int` (`1..4`)
  * `min_episodes_per_stage: int`
  * `stage_thresholds: dict[int, float]`
  * `replay_previous_ratio: float`
* **Methods & Attributes:**
  * `env.unwrapped.stage -> int`: active curriculum stage for the current episode (accessible via `env.get_attr("stage")` in `SyncVectorEnv` and `AsyncVectorEnv`).
  * `env.unwrapped.set_unlocked_stage(unlocked_stage: int) -> None`: called via `env.call("set_unlocked_stage", current_unlocked)` upon stage graduation.
  * `env.reset(seed=..., options={"stage": stg})`: evaluation in `evaluate_policy()` explicitly forces stages `1..4` via `options={"stage": stg}`.
* **Observation & Action Spaces:**
  * `observation_space`: `Box(low=-1.0, high=1.0, shape=(426,), dtype=np.float32)` (ranges `[0, 1]` for `[0:424]` and `[-1, 1]` for `[424:426]`).
  * `action_space`: `Box(low=[v_min, -max_steering_angle], high=[v_max, max_steering_angle], shape=(2,), dtype=np.float32)`.
* **`info` Dict Keys Consumed by Trainer/Evaluator:**
  * `target_dist: float`
  * `target_in_fov: bool`
  * `stage: int` (recommended in both `reset` and `step` infos so `AsyncVectorEnv` never needs extra `get_attr` RPCs)

---

## 2. MJCF Model Checklist (`ackerman_follow/assets/ackbot.xml`)

* Match `configs/robot_params.yaml` (`wheelbase = 0.30 m`, `track = 0.30 m`, `wheel_radius = 0.06 m`, `chassis_mass = 5.0 kg`, `4 × 0.40 kg` wheels).
* **Steering Hinge Clearance:** Allow at least `±0.74 rad` (`±42.4°`) on front wheel steering hinges so the inner wheel reaches `41.84°` at `R_min = 0.485 m` without hitting joint limits or chassis corner geoms.
* **Sensor Sites & Ray Ordering (`mujoco.mj_multiRay`):**
  * **Front Camera (`64` rays):** Mounted at `(0.15, 0.00, 0.25) m`, `85°` HFOV, rays ordered **right-to-left** (`-42.5°` to `+42.5°`), clipped and normalized to `[0.15, 5.00] m`.
  * **Rear 2D LiDAR (`360` rays):** Mounted at `(-0.15, 0.00, 0.30) m` (`4 cm` above Level 3 top deck so beams clear the robot chassis), `360°` CCW from `+X`, clipped and normalized to `[0.10, 3.00] m`.
  * Exclude the robot's own body/geom group or set `bodyexclude` on `mj_multiRay` so rays never self-intersect the sensor mast.

---

## 3. Target-vs-Obstacle Consideration for Stages 2–4

`BipedalHumanTarget` (`ackerman_follow/envs/human_target.py`) simulates alternating 2-foot gait motion and clothing colors (`CLOTHING_PALETTE`). Note that the current `426-D` observation contains only **range** channels (no RGB/semantic target mask), so:
* In **Stage 1 (open arena)**, proximity `soft-argmin` directly tracks the human.
* In **Stages 2–4 (L-walls, corridors, cylindrical pillars)**, static obstacles closer than the target will dominate raw proximity `soft-argmin`, requiring the GRU temporal memory to separate moving legs from static geometry—or a future observation extension adding a per-ray semantic/target-likelihood channel if desired.
