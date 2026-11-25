# MuJoCo Environment (`AckermanFollow-v0`) Specification & Contract

This document captures the implemented interface contract and physical modeling details connecting `ackerman_follow/assets/ackbot.xml` and `ackerman_follow/envs/mujoco_follow_env.py` to `scripts/train_ppo.py`.

---

## 1. Gymnasium Interface (`AckermanFollow-v0`)

* **Environment ID:** `AckermanFollow-v0` registered in `ackerman_follow/envs/__init__.py`.
* **Constructor Keyword Arguments:**
  * `auto_curriculum: bool`
  * `stage: int` (`1..7`)
  * `min_episodes_per_stage: int`
  * `stage_thresholds: dict[int, float]`
  * `replay_previous_ratio: float`
  * `sensor_noise: bool` (enables LiDAR/camera/mask/state noise + floor friction jitter during training; `False` by default for deterministic evaluation)
* **Methods & Attributes:**
  * `env.unwrapped.stage -> int`: active curriculum stage for the current episode (`env.get_attr("stage")` in `SyncVectorEnv` / `AsyncVectorEnv`).
  * `env.unwrapped.set_unlocked_stage(unlocked_stage: int) -> None`: called via `env.call("set_unlocked_stage", current_unlocked)` upon stage graduation.
  * `env.reset(seed=..., options={"stage": stg, ...})`: supports deterministic stage selection plus optional `force_front_spawn` / `force_lost_spawn`.
* **Observation & Action Spaces:**
  * `observation_space`: `Box(low=-1.0, high=1.0, shape=(490,), dtype=np.float32)` (`[0, 1]` for `[0:488]` camera depth + RGB target mask + 2D LiDAR, `[-1, 1]` for `[488:490]` proprioception `[v_norm, steer_norm]`).
  * `action_space`: `Box(low=[v_min, -max_steering_angle], high=[v_max, max_steering_angle], shape=(2,), dtype=np.float32)`.
* **`info` Dict Keys:** `stage`, `unlocked_stage`, `target_dist`, `target_bearing`, `target_in_fov`, `min_lidar`, `collision`.

---

## 2. MJCF Model (`ackerman_follow/assets/ackbot.xml`)

* Matches `configs/robot_params.yaml` (`wheelbase = 0.30 m`, `track = 0.30 m`, `wheel_radius = 0.06 m`, `chassis_mass = 5.0 kg`, `4 × 0.40 kg` wheels).
* **Steering Hinge Clearance:** Front wheel steering hinges allow `±0.75 rad` (`~43°`) so the inner wheel reaches `41.84°` at `R_min = 0.485 m` without hitting joint limits or chassis corners.
* **Sensor Sites & Ray Ordering (`mujoco.mj_multiRay`):**
  * **Front Camera (`64` depth + `64` RGB mask rays):** Mounted at `(0.15, 0.00, 0.25) m`, `85°` HFOV, ordered **right-to-left** (`-42.5°` to `+42.5°`), clipped and normalized to `[0.15, 5.00] m`. The RGB target mask fills the detected human horizontal span `[i0..i1]`.
  * **Rear 2D LiDAR (`360` rays):** Mounted at `(-0.15, 0.00, 0.30) m` above the Level 3 top deck, `360°` CCW from `+X`, clipped and normalized to `[0.10, 3.00] m`.
  * Floor sits in geom group `2` and robot collision geoms sit in group `1`; rays query group `0` (walls, obstacles, human walker) so beams never graze the floor or chassis.

---

## 3. 7-Stage Arenas, Reward & Sensory Randomization

* **Stages `1..7`:** Open arena (`500` steps) → Straight corridor (`450`) → Open-outer L-wall (`800`) → Walled L-corridor (`700`) → Dead-end wall-hug turnaround (`750`) → Dead-end middle walk turnaround (`750`) → Combined L-turn into dead-end (`950`). Stages 5 & 6 share `stage5_obstacles` (`2.40 m` clear road).
* **Reward & Safety Terms:** Distance Gaussian (`0.80 m` goal), dense + visibility-gated bearing alignment (with occluded `in_fov`), potential-based progress shaping, steering smoothness, bearing-gated reverse penalty, `-0.18` anti-freeze penalty when idling out of FOV, and steep quadratic + linear clearance penalties (`0.62 m` LiDAR / `0.50 m` front camera non-human obstacles) to protect front corners before wheel impact.
* **Training Domain Randomization (`sensor_noise=True`):** Gaussian range noise + sparse ray dropouts on LiDAR/camera, `±1°` LiDAR yaw roll, `1–3`-step RGB mask blink-outs + column jitter, state noise, and `±15%` floor friction variation per episode.
