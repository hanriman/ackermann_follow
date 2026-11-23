# Dual-Machine Hardware & Development Setup

This project is actively developed and trained across two target machines:

1. **Linux KVM VM (Headless Compute):** 6 CPU cores (AMD EPYC, x86_64, AVX2), ~12 GB RAM, CPU-only.
2. **MacBook Pro M4 Pro (Local Dev & Viewer):** Apple Silicon (arm64), 24 GB unified memory, Metal (MPS) GPU + native display.

---

## 1. Environment & Dependency Sync

Both machines use `uv` with Python 3.12 (`requires-python = ">=3.12,<3.13"` in `pyproject.toml`):

```bash
# Install runtime + dev tools (pytest, ruff)
uv sync --extra dev

# Verify unit tests & lint
uv run --extra dev pytest tests/ -v
uv run --extra dev ruff check ackerman_follow scripts tests
```

Syncing via Git keeps code, `configs/*.yaml`, and `uv.lock` aligned across x86_64 Linux and arm64 macOS without committing `.venv/`, `outputs/`, or local logs.

---

## 2. Recommended Runtime Settings per Machine

### A. Linux VM (6 CPU Cores, 12 GB RAM, Headless)

* **Device:** `cpu` (automatically chosen by `select_device("auto")`).
* **Parallel Envs (`--num-envs`):** Use `4` or `6` with `--vec-env async` once `AckermanFollow-v0` is implemented (`SyncVectorEnv` is fine for quick smoke checks).
* **Why `4` or `6` envs fits GAE/BPTT cleanly:** With `n_steps = 512` and `seq_len = 32`, each env produces `16` sequence chunks:
  * `num_envs = 4` → `64` chunks/iter → `8` minibatches of `8` chunks (`256` transitions/minibatch).
  * `num_envs = 6` → `96` chunks/iter → `8` minibatches of `12` chunks (`384` transitions/minibatch).
  * `num_envs = 8` → `128` chunks/iter → `8` minibatches of `16` chunks (`512` transitions/minibatch).
* **CPU Threading (`configure_cpu_threads`):** Automatically sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` for spawned MuJoCo worker processes when `--vec-env async` is enabled, and caps main-process PyTorch intra-op threads (`--torch-threads` or auto `<= 4`) to avoid 6-core oversubscription.
* **Memory Budget:** `6` MuJoCo worker processes + `RolloutBuffer` + PyTorch Adam state use ~`1.5–2.5 GB` RSS total, leaving plenty of headroom inside `12 GB` RAM.
* **Headless MuJoCo:** `mujoco.mj_multiRay` runs purely on CPU geometry and requires no OpenGL context. Only set `MUJOCO_GL=egl` or `osmesa` if you later add offscreen RGB pixel rendering (`mujoco.Renderer`).

Example command (once `AckermanFollow-v0` is implemented):
```bash
uv run python scripts/train_ppo.py \
  --exp-name vm_recurrent_ppo \
  --device cpu \
  --vec-env async \
  --num-envs 6 \
  --torch-threads 2
```

---

### B. MacBook Pro M4 Pro (24 GB Unified Memory, Apple Silicon)

* **Vectorized Envs:** `scripts/train_ppo.py` uses a top-level picklable `_EnvFactory` and `AsyncVectorEnv(..., context="spawn")`, which is required on macOS (`fork` is unsafe with Accelerate/Objective-C/Metal).
* **CPU vs. MPS (`--device cpu` vs. `--device mps`):**
  * During **rollout collection**, batch size is small (`num_envs = 8` observations of `426-D`), so CPU often avoids MPS kernel-launch synchronization latency.
  * During **PPO BPTT updates** (`8` minibatches × `seq_len=32` × `16` chunks = `512` sequences through 1D Conv + GRU), MPS can accelerate the backward pass.
  * Benchmark both locally with a short `--total-steps 16384` run and compare `train/SPS`.
* **Interactive MuJoCo Viewer:** On macOS, `mujoco.viewer.launch_passive` requires Python's Cocoa main-thread wrapper (`mjpython`):
  ```bash
  uv run mjpython scripts/<your_eval_or_viewer_script>.py
  ```

Example commands:
```bash
# Multi-core rollouts + MPS or CPU policy updates
uv run python scripts/train_ppo.py \
  --exp-name m4pro_recurrent_ppo \
  --device auto \
  --vec-env async \
  --num-envs 8
```

---

## 3. Output Layout & Monitoring

Training runs write artifacts to `outputs/experiments/exp_<timestamp>_<exp_name>/`:
* `checkpoints/`: `best_model.pt`, `graduated_stage_<k>.pt`, `latest_model.pt` (milestone & latest skipped if `evaluation.save_best_only: true`)
* `logs/`: `train.log`, `metrics.csv`
* `tensorboard/`: TensorBoard event files

View TensorBoard on either machine (or tunnel port `6006` from the VM via `ssh -L 6006:localhost:6006 <vm>`):
```bash
uv run tensorboard --logdir outputs/experiments
```
