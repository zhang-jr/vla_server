# Quantization

A VLA policy does not compress uniformly. The vision-language backbone tolerates aggressive quantization the way any large transformer does. An iterative action head does not, and the reason is structural rather than incidental. The runtime therefore treats precision as a per-module policy, enforces the constraint in the type system, and validates the result against task success rather than loss.

Source: `vla_runtime/quantization/`

## Why the action head is different

Flow matching and diffusion heads produce an action chunk by running many steps, each conditioned on the output of the last. Quantization error at step *k* becomes part of the input at step *k+1*, so error compounds along the trajectory instead of averaging out. A small per-step deviation that would be invisible in a single classification decision becomes a visibly wrong end-effector path by the end of a chunk.

The backbone has the opposite profile. It runs once per frame, its output is a feature vector the action head conditions on, and small perturbations there are absorbed rather than amplified. This is why the runtime allows four-bit backbone weights alongside a float16 action head, and refuses the reverse.

## Configuration

`QuantizationConfig` in `vla_runtime/quantization/config.py` is a Pydantic model with one field per module.

```python
class QuantizationConfig(BaseModel):
    vision_encoder: Literal["fp16", "w8a8", "w4a8"] = "w8a8"
    llm_backbone:   Literal["fp16", "w4a16", "w8a16", "gptq", "gguf_q4"] = "w4a16"
    action_head:    Literal["fp16", "w8a16"] = "fp16"
```

The allowed schemes differ by module, and the `Literal` types are the first line of enforcement — an int4 or GPTQ action head is not a validation failure but a type error, so it cannot be expressed at all.

| Module | Schemes | Notes |
|---|---|---|
| Vision encoder | `fp16`, `w8a8`, `w4a8` | `w8a8` is the default; `w4a8` is aggressive |
| LLM backbone | `fp16`, `w4a16`, `w8a16`, `gptq`, `gguf_q4` | Where compression actually pays; `gguf_q4` targets edge deployment |
| Action head | `fp16`, `w8a16` | Nothing below eight bits is available |

`validate_for_head_type(head_type)` adds the type-dependent rule: a flow matching or diffusion head must be `fp16`, and anything else raises `PrecisionDegradationError`. `w8a16` remains available for discrete-token heads, which make one decision per token and tolerate it.

Three presets cover the common deployment shapes:

```python
QuantizationConfig.full_precision()     # fp16 everywhere — the comparison baseline
QuantizationConfig.cloud_aggressive()   # w8a8 vision, w4a16 backbone, fp16 head
QuantizationConfig.edge_balanced()      # w8a8 vision, gguf_q4 backbone, fp16 head
```

`edge_balanced()` pairs with `LlamaCppBackend` for embedded deployment — see [VLM backends](./vlm-backends.md).

## Calibration

Source: `vla_runtime/quantization/calibration.py`

`VLACalibrationPipeline` calibrates the three modules independently, each with its own calibrator, and returns a `CalibrationResult` carrying the quantized modules plus per-module timing and sample counts.

```python
pipeline = VLACalibrationPipeline(CalibrationConfig(num_samples=512, seed=42))
result = pipeline.calibrate(model, quant_config, samples, head_type="flow_matching")
print(result.summary())
```

`CalibrationConfig` controls scope and reproducibility:

| Field | Default | Purpose |
|---|---|---|
| `num_samples` | 512 | Calibration samples consumed |
| `batch_size` | 8 | Forward-pass batch size |
| `seed` | 42 | Fixed seed for shuffling |
| `max_seq_len` | 1024 | Token length cap during calibration |
| `device` | `"cuda"` | Where calibration runs |
| `verbose` | `False` | Emit per-layer quantization error statistics |

Reproducibility is a hard requirement rather than a convenience. Calibration is stochastic in its sample selection, and a quantized checkpoint that cannot be regenerated is one that cannot be debugged when its success rate drops — so the seed, the sample set, and the task suite are all pinned.

Calibration samples are `CalibrationSample` records holding an image, an instruction, and optionally a robot state. The state is needed only for action head calibration, since that is the only module conditioned on it.

Precision constraints are checked at the entry point, before any compute is spent: calibrating a flow matching head at anything other than float16 fails immediately rather than after several hundred forward passes.

## Validation

Source: `vla_runtime/quantization/validation.py`

Layer-wise mean squared error and perplexity are poor proxies for whether a quantized VLA still works. A joint-angle deviation of 0.01 radians is an unremarkable number in isolation, and after fifty chunk steps it is a gripper that closes on nothing. The acceptance criterion is therefore end-to-end task success rate in simulation.

`TaskSuccessRateValidator` runs episodes against a baseline and a quantized policy and compares the resulting rates.

```python
validator = TaskSuccessRateValidator(SimTaskConfig(mode="smoke", threshold_drop_pp=2.0))
baseline = validator.run_baseline(fp16_policy, env)
result = validator.validate(quantized_policy, env, baseline.success_rate,
                            quant_config_summary=quant_config.summary())
print(result.summary())
```

Both calls take the policy first and the environment second, and the same suite and task set must be used for the baseline and the comparison — otherwise the difference measures the task mix rather than the quantization. `validate()` raises `PrecisionDegradationError` when the drop against the baseline exceeds `threshold_drop_pp`. `ValidationResult` carries the aggregate rate, `per_task_rates()`, and per-episode records — a fleet-wide average can hide a quantization scheme that broke one task category entirely, so the per-task breakdown is usually where the real answer is.

`SimTaskConfig` controls cost and reproducibility:

| Field | Default | Purpose |
|---|---|---|
| `suite` | `"libero"` | Which simulation suite to run |
| `mode` | `"smoke"` | `smoke` runs a subset; `full` runs the complete suite |
| `num_tasks_smoke` / `num_tasks_full` | 10 / 50 | Task count per mode |
| `num_episodes_per_task` | 5 | More episodes, tighter estimate |
| `max_steps_per_episode` | 400 | Episode timeout in simulation steps |
| `seed` | 42 | Fixed task and episode seeding |
| `threshold_drop_pp` | 2.0 | Maximum tolerated drop in percentage points |

The two modes exist because a full suite is too slow to gate every change. `smoke` is cheap enough to run routinely; `full` is the one that decides whether a configuration ships.

### Testing without a simulator

`MockSimEnvironment` and `MockPolicy` implement the `SimEnvironment` and `PolicyCallable` interfaces with deterministic synthetic outcomes, so the validation logic — thresholds, aggregation, error paths — can be exercised in unit tests without installing a simulator. `validate_quantization_config()` is a one-shot helper wrapping the whole flow for quick checks.

To integrate a real simulator, implement `SimEnvironment`:

```python
class SimEnvironment(abc.ABC):
    def reset(self, task_name: str, episode_idx: int, seed: int) -> dict: ...
    def step(self, action: Tensor) -> tuple[dict, float, bool, dict]: ...
    def task_names(self) -> list[str]: ...
    def close(self) -> None: ...
```

The explicit `seed` on `reset()` is what makes an episode replayable — a failing episode can be reproduced exactly rather than sampled for again.

## Compiled engines

`TensorRTBackend` carries the same precision constraint at compile time: it refuses to build an int8 engine for a flow matching or diffusion head, raising `PrecisionDegradationError` in the constructor. An aggressive engine configuration fails at build rather than producing a policy that runs fast and misses. See [Action heads](./action-heads.md).

## Related documents

- [Action heads](./action-heads.md) — why the three paradigms differ in precision tolerance
- [VLM backends](./vlm-backends.md) — engine-level quantization options per adapter
- [Observability](./observability.md) — `PrecisionDegradationError` and its trace context
