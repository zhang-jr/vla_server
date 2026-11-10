# Benchmark methodology

Two kinds of benchmark live in `benchmarks/`:

| Suite | What it measures | Where it runs |
|---|---|---|
| `benchmark_pipeline.py`, `benchmark_kernels.py` | Scheduler overhead with mock models; Triton kernels vs PyTorch | pytest-benchmark; kernels need a GPU |
| `benchmarks/sim/` | Closed-loop task success **and** runtime behaviour on LIBERO / ManiSkill with a real VLA | Remote GPU host (ADR-020) |
| `benchmarks/latency/` | Model-level end-to-end latency / sustained FPS of a VLA's native implementation, against a requirement | Remote GPU host, `docker/Dockerfile` image |
| `benchmarks/closed_loop/` | Real-robot end-to-end latency / FPS: camera frame → policy server → arm command (ARX X5) | Robot host (ROS2 Foxy) + GPU container on the same machine |

Most of this page covers `benchmarks/sim/`; the [model latency benchmark](#model-latency-benchmark) section at the end covers `benchmarks/latency/`. The question it answers: *what does running a policy through this runtime change, compared with running the model the way its authors do, measured where it matters — task success in closed loop?*

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

## Model latency benchmark

`python -m benchmarks.latency.run` answers *does model X, run the way its authors run it, meet "latency ≤ L ms at P95 and ≥ F inferences/s" on this GPU?* — no simulator, no arm, one robot.

**What one timed call covers.** Raw observation → actions on the host: uint8 camera frames at sensor resolution (`--image-hw`, default 480×640), float state and instruction go in; resize / normalize / tokenize, every forward pass (VLM, action decoding or denoising loop), unnormalize and the device→host copy happen inside the call. Camera capture, transport to the GPU host and the arm driver are *not* included — that is the [closed-loop benchmark](#closed-loop-benchmark-arx-x5)'s job.

**Protocol.**
- `--warmup` calls (default 10) first: CUDA context, cuDNN autotune, JAX / `torch.compile` compilation. The first one is reported as `cold_start_ms`.
- Then back-to-back calls until **both** `--iterations` (default 100) and `--duration` seconds (default 60) are reached. Latency percentiles come from every call; FPS = calls / elapsed, so it is a sustained closed-loop rate (one robot, the next frame is processed when the previous result returns), including thermal throttling and allocator / GC jitter.
- Verdict: P`--latency-percentile` (default 95) ≤ `--max-latency-ms` (500) **and** FPS ≥ `--min-fps` (3).
- Stage timings (`preprocess`, `generate` / `predict_chunk`, …) synchronize the device at each stage boundary; the call total is timed independently.
- GPU memory: NVML device usage sampled every 10 ms during the measured phase, minus the usage before the model loaded (`peak_delta_gb`), plus `torch.cuda.max_memory_allocated` where torch is the framework.
- Seeds fixed (`--seed 42`); frames are deterministic synthetic images. Compute of these models does not depend on pixel content (fixed token counts, fixed denoising steps).

**Models** (`benchmarks/latency/models.py`; options via `--opt <model>.<key>=<value>`):

| model | env | native path | per call |
|---|---|---|---|
| `openvla` | `openvla` | HF `openvla/openvla-7b`, `predict_action`, bf16 | 1 action (no chunking: control rate = inference rate) |
| `openvla_oft` | `oft` | `openvla-oft` repo, `get_vla_action`, L1 head | chunk of 8 (LIBERO checkpoint) |
| `pi0`, `pi05`, `smolvla` | `lerobot` | LeRobot processors + `predict_action_chunk` | chunk of 50 |
| `openpi` | `openpi` | Physical Intelligence `openpi` (JAX) `Policy.infer` | chunk (config dependent) |
| `mock` | any | sleeps | CPU plumbing check |

Each model runs in its own process with `/opt/venvs/<env>/bin/python` (override: `VLA_BENCH_PYTHON_<ENV>`), because the stacks pin incompatible `transformers` versions.

**Environment.** `docker/Dockerfile` builds all envs on CUDA 12.8 / torch 2.7 (needed for RTX 5090, sm_120; RTX 4090 runs the same image). Build and run instructions are in its header. π0 / π0.5 tokenizers come from the gated `google/paligemma-3b-pt-224` repo, so the container needs `HF_TOKEN` of an account that accepted its license.

```bash
python scripts/remote.py check
python scripts/remote.py sync
python scripts/remote.py run -- python -m benchmarks.latency.run --models openvla,openvla_oft,pi0,pi05,smolvla
python scripts/remote.py run -- python -m benchmarks.latency.run --models pi05     --opt pi05.cfg.dtype=bfloat16 --opt pi05.cfg.num_inference_steps=5   # sensitivity runs
python scripts/remote.py fetch benchmarks/results/latency
```

Each run writes `benchmarks/results/latency/<timestamp>/`: `meta.json`, `<model>.json` (environment, package versions, model description, every sample), `<model>.log`, and `summary.md` — the verdict table for `BENCHMARKS.md`. Locally, `tests/unit/test_latency_benchmark.py` covers the harness on CPU.

## Closed-loop benchmark (ARX X5)

`benchmarks/closed_loop/` measures the same requirement on the real robot path: camera frame → observation → TCP → policy server (GPU) → actions → `/arm_cmd`. It times the pipeline, not the policy's task skill: the models are not fine-tuned for this arm, so **model actions are logged and never sent to the arm**.

**Processes.**
- `policy_server` (GPU container, the model's own venv; re-execs into `/opt/venvs/<env>` itself) loads a script-1 adapter, warms it up and serves one client. `--camera-map SLOT=CAMERA` fills the model's camera slots from the client's named cameras (`SLOT=-` leaves a slot out; LeRobot pads it with a masked empty image, so compute is unchanged); state is zero-padded / truncated to the model's size.
- `arx_x5` (robot host, ROS2 Foxy, system Python 3.8 — no venv, stdlib + numpy only) subscribes to the camera colour topics and `/arm_status`, and drives the loop. Wire format: `protocol.py`, a JSON header plus raw numpy blobs over TCP (no msgpack on the host).

**Modes.**
- `dry-run`: no `/arm_cmd` publisher is even created; arm status optional (zeros without it). `command` is the time to the would-be publish.
- `hold`: needs `--operator-present`, a live `/arm_status` and a subscriber on `/arm_cmd` (`X5Controller`). The arm's pose is averaged over `--hold-capture` s (refused if it is moving) and that pose is published with `mode=5` (joint position control; gripper = status gripper × `--gripper-scale`, 5 as in `X5Controller`'s follower mode) after every inference, so the real command path is timed while the arm stays put. Publishing stops as soon as a joint leaves the held pose by more than `--max-deviation` rad (0.05) or the status is older than `--max-status-age` s (0.2); the controller then keeps the last, held target.

**Timing.** All timestamps share the host wall clock (camera header stamps, client, and the container). Segments per step: `frame_age` (oldest camera stamp → snapshot), `client_obs` (decode + pack), `transport` (round trip minus server time), `server_unpack`, `server_infer` (the script-1 timed call), `command`, `loop` (snapshot → command) and `e2e` (oldest camera stamp → command). The verdict uses `e2e` at the requirement percentile and sustained steps/s over ≥ `--duration` s, back to back. A median `frame_age` below −5 ms or above 1 s is flagged: the camera stamps are then not on the host clock.

```bash
# robot host: cameras (colour only, 640x480@30; serials from `rs-enumerate-devices -s`)
WRIST_SERIAL=... THIRD_SERIAL=... bash benchmarks/closed_loop/start_cameras.sh
# container: policy server (π0.5 bf16, single arm: right wrist slot left empty)
python -m benchmarks.closed_loop.policy_server --model pi05 --opt pi05.cfg.dtype=bfloat16 \
    --opt pi05.repo=/models/pi05_base --opt pi05.tokenizer=/models/paligemma-3b-pt-224 \
    --camera-map base_0_rgb=third --camera-map left_wrist_0_rgb=wrist --camera-map right_wrist_0_rgb=-
# robot host: the loop (docker bridge address of the container)
source benchmarks/closed_loop/host_env.sh
python3 -m benchmarks.closed_loop.arx_x5 --server 172.17.0.2:8765 \
    --camera wrist=/cam/wrist/color/image_raw --camera third=/cam/third/color/image_raw --mode dry-run
```

Each run writes `<out>/<timestamp>-<mode>/`: `meta.json`, `steps.jsonl` (every step's timestamps), `result.json` (statistics, camera / arm rates, server description, hold target) and `summary.md`. `tests/unit/test_closed_loop.py` covers the protocol, camera mapping, hold guard and both loop modes on CPU with a mock model.
