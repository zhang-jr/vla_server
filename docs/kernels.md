# Kernels

The action head runs its inner loop tens of times per prediction, so the cost of that loop is what determines whether a paradigm fits inside a control cycle. The kernels are Triton implementations of the operations in that loop, each paired with a PyTorch fallback that is correctness-equivalent.

Source: `kernels/`

Kernels live outside `vla_runtime/` and contain no business logic. They take tensors and return tensors; scheduling, error context, and tracing all belong to the callers in `vla_runtime/action_head/`.

## The dispatch pattern

Every kernel module follows the same shape: a `@triton.jit` kernel guarded behind an availability check, a Python dispatch function that selects between it and a PyTorch path, and a `warmup()` entry point.

```python
def flow_matching_euler_step(z_t, velocity, dt):
    if not _triton_available or not z_t.is_cuda:
        return z_t + velocity * dt        # correctness-equivalent, not optimised
    out = torch.empty_like(z_t)
    _euler_step_kernel[(B,)](z_t, velocity, out, dt, D=D)
    return out
```

The fallback is what makes the whole runtime testable on CPU. Unit tests exercise the real integration and denoising math rather than mocking it out, which means a change to the arithmetic is caught by the test suite without a GPU — while the Triton path remains the one that runs in deployment. The two paths are required to agree numerically; the fallback simply does not fuse.

Dispatch functions assert their preconditions — matching shapes, contiguous inputs, head dimension within the kernel's limit — because a Triton kernel given a non-contiguous tensor produces wrong numbers rather than an error, and wrong actions are considerably harder to diagnose than a failed assertion.

## Available kernels

### Flow matching integration

`kernels/flow_matching_step.py`

```python
flow_matching_euler_step(z_t, velocity, dt) -> Tensor
flow_matching_rk4_step(z_t, velocity_fn, t, dt) -> Tensor
```

The Euler step fuses the update `z + v * dt` into a single pass. RK4 takes a callable for the velocity field because it evaluates it four times per step at different points, and coordinates those evaluations rather than fusing them.

### Diffusion denoising

`kernels/diffusion_step.py`

```python
ddpm_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev, sigma_t) -> Tensor
ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev) -> Tensor
```

Both recover the clean sample from the predicted noise and re-noise to the previous timestep in one fused pass. `ddim_step` is `ddpm_step` with `sigma_t = 0`, which is exactly what makes DDIM deterministic.

### Fused attention

`kernels/fused_attention.py`

```python
fused_mha(q, k, v, scale=None) -> Tensor
```

Tiled multi-head attention over `[B, H, N, D]`. The reason it matters here is memory rather than arithmetic: materializing an `N × N` attention matrix for a vision encoder sequence — a few hundred patch tokens plus a class token — costs a quadratic allocation per layer per frame, at frame rate. Tiling keeps the intermediate in on-chip memory and never writes it out. Head dimension is capped at 128. The fallback is `torch.nn.functional.scaled_dot_product_attention`.

### Fused LayerNorm and Linear

`kernels/layernorm_linear.py`

```python
fused_layernorm_linear(x, gamma, beta, weight, bias=None, eps=1e-5) -> Tensor
```

`x` is `[M, H]` where `M` is batch times sequence length, `gamma` and `beta` are the LayerNorm parameters, and `weight` is `[OUT, H]`.

Fuses normalization into the following projection. Separately, each transformer block writes the normalized activations to memory and immediately reads them back; fusing removes that round trip. The saving is per block per layer, so it accumulates across a deep backbone.

## Autotuning

The block-size kernels use `@triton.autotune` with several configurations keyed on the feature dimension, so the tile shape is selected by measurement rather than by a heuristic. Action dimension varies widely across robots — a six-joint arm and a dual-arm humanoid differ by an order of magnitude — and a block size tuned for one is a poor fit for the other. Keying the autotune cache on `D` means each robot geometry gets its own tuned configuration, chosen once and reused.

## Warmup

Every kernel module exposes `warmup(device)`, which runs a dummy launch to force compilation. The first launch of a Triton kernel includes compilation, which can exceed a control cycle by orders of magnitude. Calling `warmup()` during startup keeps that cost out of the first real request, where it would otherwise appear as a latency spike and trigger a spurious degradation path. `warmup()` is a no-op when Triton is unavailable.

## Adding a kernel

Follow the existing pattern: a `@triton.jit` kernel behind the availability guard, a dispatch function that falls back to PyTorch on CPU or when Triton is missing, precondition assertions in the dispatch function, and a `warmup()`. Compilation and launch failures should surface as `KernelError`, which is what the TensorRT path in `vla_runtime/action_head/tensorrt_backend.py` raises and which carries trace context — see [Observability](./observability.md).

Kernel changes should come with a `benchmarks/benchmark_kernels.py` comparison against the PyTorch path across a range of action dimensions, since a kernel that wins at one size frequently loses at another.

## Related documents

- [Action heads](./action-heads.md) — the heads that call these kernels
- [Architecture](./architecture.md) — where kernel cost lands in the latency budget
- [Observability](./observability.md) — `KernelError` and its trace context
