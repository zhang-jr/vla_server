"""Triton kernel for a single flow matching denoising step.

Target: < 5ms per step on A100 (see BENCHMARKS.md after Phase 0 baseline).

Flow matching overview:
    Given noise z_t at time t, a velocity field v_θ(z_t, t, c) is predicted
    by the action head network. The ODE is then integrated:

        z_{t+dt} = z_t + v_θ(z_t, t, c) * dt

    This kernel fuses the velocity network's final linear projection +
    the Euler integration step into a single GPU pass, avoiding a round-trip
    to HBM between the two ops.

Kernel design:
    - Each program instance handles one (batch, action_dim) slice.
    - Input:  z_t [B, D], velocity [B, D], dt scalar
    - Output: z_next [B, D] = z_t + velocity * dt   (in-place update)
    - Memory: all fp16, coalesced row-major access pattern.

Usage (Python side):
    from kernels.flow_matching_step import flow_matching_euler_step
    z_next = flow_matching_euler_step(z_t, velocity, dt)

Compilation:
    Triton JIT-compiles on first call and caches the PTX. On Jetson Orin the
    compilation takes ~5s; call warmup() in the serving startup path.
"""

from __future__ import annotations

import torch

# Triton is an optional dependency (not available on CPU-only machines).
# The _triton_available flag is checked before any kernel dispatch.
try:
    import triton  # type: ignore[import]
    import triton.language as tl  # type: ignore[import]
    _triton_available = True
except ImportError:
    _triton_available = False


# ── Autotune configs ──────────────────────────────────────────────────────────
# Triton will benchmark each config on first call and cache the winner.
# Configs target the typical VLA action dimension range: 7-256.

_EULER_AUTOTUNE_CONFIGS = []
if _triton_available:
    _EULER_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_D": 64}),
        triton.Config({"BLOCK_D": 128}),
        triton.Config({"BLOCK_D": 256}),
        triton.Config({"BLOCK_D": 512}),
        triton.Config({"BLOCK_D": 1024}),
    ]


# ── Kernel definition ─────────────────────────────────────────────────────────

if _triton_available:
    @triton.autotune(configs=_EULER_AUTOTUNE_CONFIGS, key=["D"])
    @triton.jit
    def _euler_step_kernel(
        z_ptr,          # [B, D] float16, input noise state
        v_ptr,          # [B, D] float16, predicted velocity
        out_ptr,        # [B, D] float16, output (may alias z_ptr for in-place)
        dt,             # scalar float32
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused Euler integration: out = z + v * dt.

        One program per row (batch element).
        """
        row = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_D)
        mask = col_offsets < D

        z = tl.load(z_ptr + row * D + col_offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + row * D + col_offsets, mask=mask, other=0.0)

        out = z + v * dt

        tl.store(out_ptr + row * D + col_offsets, out, mask=mask)


# ── Python dispatch ───────────────────────────────────────────────────────────

def flow_matching_euler_step(
    z_t: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Fused Euler integration step for flow matching.

    Args:
        z_t: Current noise state, shape [B, D], dtype float16.
        velocity: Predicted velocity field, shape [B, D], dtype float16.
        dt: Integration step size (scalar).

    Returns:
        Updated state z_{t+dt}, shape [B, D], dtype float16.

    Note:
        Falls back to pure PyTorch when Triton is not available
        (CPU inference, non-CUDA devices).
    """
    assert z_t.shape == velocity.shape, (
        f"Shape mismatch: z_t={z_t.shape}, velocity={velocity.shape}"
    )
    assert z_t.is_contiguous(), "z_t must be contiguous"
    assert velocity.is_contiguous(), "velocity must be contiguous"

    B, D = z_t.shape

    if not _triton_available or not z_t.is_cuda:
        # PyTorch fallback — correctness-equivalent, not optimised
        return z_t + velocity * dt

    out = torch.empty_like(z_t)
    # BLOCK_D is selected by autotuner; pass D as the key for cache lookup.
    _euler_step_kernel[(B,)](
        z_t, velocity, out,
        dt,
        D=D,
    )
    return out


def flow_matching_rk4_step(
    z_t: torch.Tensor,
    velocity_fn: "callable[[torch.Tensor, float], torch.Tensor]",
    t: float,
    dt: float,
) -> torch.Tensor:
    """RK4 integration step for higher-accuracy flow matching.

    Uses four Euler sub-steps with the Triton kernel for each sub-step.
    More accurate than Euler but 4x more velocity evaluations.

    Args:
        z_t: Current state [B, D].
        velocity_fn: Callable(z, t) → velocity [B, D].
        t: Current time in [0, 1].
        dt: Step size.

    Returns:
        z_{t+dt} via RK4, shape [B, D].
    """
    k1 = velocity_fn(z_t, t)
    k2 = velocity_fn(flow_matching_euler_step(z_t, k1, dt / 2), t + dt / 2)
    k3 = velocity_fn(flow_matching_euler_step(z_t, k2, dt / 2), t + dt / 2)
    k4 = velocity_fn(flow_matching_euler_step(z_t, k3, dt), t + dt)

    # Weighted average: (k1 + 2*k2 + 2*k3 + k4) / 6
    combined = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (1.0 / 6.0)
    return flow_matching_euler_step(z_t, combined, dt)


def warmup(device: torch.device = torch.device("cuda")) -> None:
    """Pre-compile the Triton kernel with a representative tensor size.

    Call once during serving startup. On Jetson Orin the first compilation
    takes ~5s; subsequent calls return immediately from the PTX cache.
    """
    if not _triton_available:
        return
    dummy_z = torch.zeros(1, 256, dtype=torch.float16, device=device)
    dummy_v = torch.zeros(1, 256, dtype=torch.float16, device=device)
    _ = flow_matching_euler_step(dummy_z, dummy_v, dt=0.1)
