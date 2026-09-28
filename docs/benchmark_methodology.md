# Benchmark methodology

Two kinds of benchmark live in `benchmarks/`:

| Suite | What it measures | Where it runs |
|---|---|---|
| `benchmark_pipeline.py`, `benchmark_kernels.py` | Scheduler overhead with mock models; Triton kernels vs PyTorch | pytest-benchmark; kernels need a GPU |
| `benchmarks/sim/` | Closed-loop task success **and** runtime behaviour on LIBERO / ManiSkill with a real VLA | Remote GPU host (ADR-020) |

This page covers the second. The question it answers: *what does running a policy through this runtime change, compared with running the model the way its authors do, measured where it matters — task success in closed loop?*

## Execution modes

The same policy, the same episodes (same task order, same reset seeds as `TaskSuccessRateValidator`), four ways of executing it:

| Mode | Path | What it tells you |
|---|---|---|
| `native` | The model's own monolithic inference (LeRobot `predict_action_chunk`), blocking: sim pauses, a chunk is computed, its first `n_action_steps` actions execute | The baseline. What the model's authors report |
| `sync` | Same blocking loop through the runtime's split components: `VLMBackend.encode()` → `ActionHeadRuntime.predict()` | Must match `native` success. Latency delta = split overhead; VLM vs head latency share |
| `async_sim` | `AsyncScheduler` + `ChunkBuffer`, simulated clock: an inference started at step `s` lands at step `s + ceil(latency × control_hz)`; meanwhile the robot executes the queued tail | Success under real latency, blend-policy ablation (ADR-014), condition-staleness calibration via injected latency |
| `realtime` | `FleetScheduler` with N robots, each env stepped by a wall-clock control loop at the env's control rate | Deployment behaviour: buffer underruns, missed control ticks, fleet contention on one backbone |

Why both `async_sim` and `realtime`: the simulated clock makes latency effects deterministic and lets you inject latencies the hardware at hand cannot produce (a slow edge device, a cloud round trip). The wall-clock mode measures what the pipeline actually does, but depends on the simulator stepping faster than the control period — check `sim_step_p95_ms` against `1000 / control_hz`.

Blocking modes pause the sim during inference, so their success ignores latency; `blocked_fraction` reports how long a real robot would stand still instead.

## Metrics

| Field | Definition |
|---|---|
| `success_rate`, `success_ci95` | Episodes solved; 95 % Wilson interval (with 10 episodes/task the interval is wide — compare intervals, not points) |
| `latency_p50_ms`, `latency_p95_ms` | Wall clock, observation → chunk available |
| `vlm_p50_ms`, `head_p50_ms` | Per-module latency (`sync` mode) |
| `throughput_inferences_per_s` | Inferences over run wall time (fleet: all robots) |
| `blocked_fraction` | Share of episode time the robot waits for inference |
| `underrun_rate` | Share of control steps sent while the chunk buffer was empty (hold action instead of a planned one) |
| `overrun_rate`, `tick_lag_p95_ms` | `realtime`: control ticks that missed their deadline, and by how much |
| `jerk_rms` | RMS third difference of the executed arm trajectory (gripper excluded) |
| `boundary_ratio` | Mean step size where a new chunk takes effect ÷ median step elsewhere; 1.0 = seamless |
| `gpu_peak_mem_gb` | `torch.cuda.max_memory_allocated()` over the run |

## Policies and environments

| Policy | Notes |
|---|---|
| `mock` | CPU. Latency-injecting mock backbone + built-in `FlowMatchingHead` with random weights. Plumbing only |
| `smolvla` | LeRobot SmolVLA split at the prefix-KV boundary: `encode()` returns a `per_layer_kv` condition, the head runs the expert's denoising loop over it. Prefix-state → coupled (ADR-018). Default checkpoint `HuggingFaceVLA/smolvla_libero` |

| Env | Notes |
|---|---|
| `libero` | openpi / LeRobot protocol: suite init states, 10 settle steps, views rotated 180°, 8-D state (eef pos + axis-angle + gripper qpos), suite step limits (spatial 220 … libero_10 520) |
| `maniskill` | ManiSkill3, one scene per robot (`num_envs=1`), language from a built-in table. No public SmolVLA checkpoint: success needs your own fine-tuned checkpoint (`--policy-opt repo=...`); latency, underrun and fleet numbers are meaningful with any checkpoint |

`test_smolvla_split_matches_native` must pass before any `sync` / `async_sim` number is trusted: it checks the split reproduces LeRobot's own chunk with shared noise. A gap between `native` and `sync` success otherwise means an adapter bug, not a runtime effect.

## Running

All GPU work goes through the remote host configured in `.env`:

```bash
python scripts/remote.py sync
python scripts/remote.py run -- bash scripts/setup_sim.sh          # LIBERO + ManiSkill + LeRobot
python scripts/remote.py test benchmarks/benchmark_sim.py -m gpu -v # adapter alignment + env smoke

# LIBERO-Object: baseline vs runtime, blend-policy ablation
python scripts/remote.py run -- python -m benchmarks.sim.run \
    --env libero --env-opt suite=libero_object --policy smolvla \
    --modes native,sync,async_sim --blend-policies replace,trim,linear_overlap,ensemble \
    --episodes 10

# Condition-staleness calibration (TODO 0.3)
python scripts/remote.py run -- python -m benchmarks.sim.run \
    --env libero --env-opt suite=libero_object --policy smolvla \
    --modes async_sim --inject-latency-ms 0,150,500,1000,2000 --episodes 10

# Fleet scaling
python scripts/remote.py run -- python -m benchmarks.sim.run \
    --env libero --env-opt suite=libero_object --policy smolvla \
    --modes realtime --robots 1,2,4 --episodes 4

python scripts/remote.py fetch benchmarks/results/sim
```

Locally, the harness itself is covered on CPU by `tests/unit/test_sim_benchmark.py`, and
`python -m benchmarks.sim.run --env mock --policy mock --modes all --episodes 2` runs every mode end to end.

Each run writes `benchmarks/results/sim/<timestamp>_<env>_<policy>/`: `meta.json` (git commit, GPU, package versions, capabilities), one JSON per configuration with per-episode records, and `summary.md` — a table ready for `BENCHMARKS.md`.

## Reproducibility

- Episode enumeration uses `--seed` (default 42) exactly like the quantization validator; LIBERO initial states come from the suite's init files.
- `torch.manual_seed(reset_seed)` before each episode fixes the flow-matching noise in the blocking and `async_sim` modes. `realtime` is inherently timing-dependent.
- `--warmup` inferences (default 3) run before timing starts, and the CUDA peak-memory counter is reset after them.
