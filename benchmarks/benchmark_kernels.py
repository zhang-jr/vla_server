"""Kernel micro-benchmarks — pytest-benchmark suite.

Compares:
  1. Triton fused Euler step vs PyTorch fallback
  2. Triton DDIM step vs PyTorch fallback
  3. Fused LayerNorm+Linear vs sequential torch ops
  4. Fused MHA vs torch.nn.functional.scaled_dot_product_attention

Run::

    pytest benchmarks/benchmark_kernels.py \
        --benchmark-only \
        --benchmark-json=benchmarks/results/kernels.json \
        -v

Regression guard:
    A GitHub CI step compares results against the stored baseline in
    BENCHMARKS.md. If any kernel regresses > 5%, the PR is blocked
    (see CLAUDE.md coding standards).

Requirements:
    - CUDA GPU available
    - triton installed
    - pytest-benchmark installed

CPU-only machines: all Triton benchmarks are skipped automatically via the
@pytest.mark.skipif decorator.
"""

from __future__ import annotations

import pytest
import torch

# ── Fixtures ──────────────────────────────────────────────────────────────────

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if CUDA_AVAILABLE else "cpu")


@pytest.fixture
def euler_tensors(device):
    """Standard flow matching tensors: batch=4, action_dim=128 (typical VLA)."""
    B, D = 4, 128
    z = torch.randn(B, D, dtype=torch.float16, device=device).contiguous()
    v = torch.randn(B, D, dtype=torch.float16, device=device).contiguous()
    return z, v, 0.05  # z_t, velocity, dt


@pytest.fixture
def diffusion_tensors(device):
    """Standard diffusion tensors: batch=4, action_dim=128."""
    B, D = 4, 128
    x = torch.randn(B, D, dtype=torch.float16, device=device).contiguous()
    noise = torch.randn(B, D, dtype=torch.float16, device=device).contiguous()
    return x, noise, 0.9, 0.85, 0.01


@pytest.fixture
def layernorm_linear_tensors(device):
    """Vision encoder dims: seq=257, hidden=1024, out=4096."""
    M, H, OUT = 257, 1024, 4096
    x = torch.randn(M, H, dtype=torch.float16, device=device).contiguous()
    g = torch.ones(H, dtype=torch.float32, device=device)
    b = torch.zeros(H, dtype=torch.float32, device=device)
    w = torch.randn(OUT, H, dtype=torch.float16, device=device).contiguous()
    return x, g, b, w


@pytest.fixture
def mha_tensors(device):
    """Vision encoder MHA: B=1, H=16, N=257, D=64."""
    B, H, N, D = 1, 16, 257, 64
    q = torch.randn(B, H, N, D, dtype=torch.float16, device=device).contiguous()
    k = torch.randn(B, H, N, D, dtype=torch.float16, device=device).contiguous()
    v = torch.randn(B, H, N, D, dtype=torch.float16, device=device).contiguous()
    return q, k, v


# ── Euler step benchmarks ─────────────────────────────────────────────────────

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_benchmark_euler_triton(benchmark, euler_tensors):
    """Triton fused Euler step — should be faster than PyTorch for D ≥ 64."""
    from kernels.flow_matching_step import flow_matching_euler_step

    z, v, dt = euler_tensors
    # Warmup: trigger Triton compilation before benchmarking
    _ = flow_matching_euler_step(z, v, dt)
    torch.cuda.synchronize()

    def run():
        out = flow_matching_euler_step(z, v, dt)
        torch.cuda.synchronize()
        return out

    result = benchmark(run)
    assert result is not None


def test_benchmark_euler_pytorch(benchmark, euler_tensors, device):
    """PyTorch fallback Euler step — reference baseline."""
    z, v, dt = euler_tensors

    def run():
        out = z + v * dt
        if device.type == "cuda":
            torch.cuda.synchronize()
        return out

    result = benchmark(run)
    assert result is not None


# ── DDIM step benchmarks ──────────────────────────────────────────────────────

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_benchmark_ddim_triton(benchmark, diffusion_tensors):
    """Triton fused DDIM step."""
    from kernels.diffusion_step import ddim_step

    x, noise, ab_t, ab_prev, _sigma = diffusion_tensors
    # Warmup
    _ = ddim_step(x, noise, ab_t, ab_prev)
    torch.cuda.synchronize()

    def run():
        out = ddim_step(x, noise, ab_t, ab_prev)
        torch.cuda.synchronize()
        return out

    result = benchmark(run)
    assert result is not None


def test_benchmark_ddim_pytorch(benchmark, diffusion_tensors, device):
    """PyTorch reference DDIM step."""
    import math
    x, noise, ab_t, ab_prev, _ = diffusion_tensors
    sqrt_ab_t = math.sqrt(ab_t)
    sqrt_1m_ab_t = math.sqrt(1.0 - ab_t)
    sqrt_ab_prev = math.sqrt(ab_prev)

    def run():
        x0 = (x.float() - sqrt_1m_ab_t * noise.float()) / (sqrt_ab_t + 1e-8)
        out = (sqrt_ab_prev * x0).half()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return out

    result = benchmark(run)
    assert result is not None


# ── Fused LayerNorm + Linear benchmarks ───────────────────────────────────────

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_benchmark_layernorm_linear_fused(benchmark, layernorm_linear_tensors):
    """Fused LayerNorm+Linear — target: ≤ 0.8× sequential latency."""
    from kernels.layernorm_linear import fused_layernorm_linear

    x, g, b, w = layernorm_linear_tensors
    # Warmup
    _ = fused_layernorm_linear(x, g, b, w)
    torch.cuda.synchronize()

    def run():
        out = fused_layernorm_linear(x, g, b, w)
        torch.cuda.synchronize()
        return out

    result = benchmark(run)
    assert result is not None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_benchmark_layernorm_linear_sequential(benchmark, layernorm_linear_tensors):
    """Sequential LayerNorm then Linear — reference for fused kernel comparison."""
    import torch.nn.functional as F

    x, g, b, w = layernorm_linear_tensors
    H = x.shape[1]

    def run():
        norm = F.layer_norm(x.float(), (H,), g, b, eps=1e-5).half()
        out = F.linear(norm, w)
        torch.cuda.synchronize()
        return out

    result = benchmark(run)
    assert result is not None


# ── Fused MHA benchmarks ──────────────────────────────────────────────────────

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_benchmark_fused_mha(benchmark, mha_tensors):
    """Triton fused MHA for vision encoder."""
    from kernels.fused_attention import fused_mha

    q, k, v = mha_tensors
    # Warmup
    _ = fused_mha(q, k, v)
    torch.cuda.synchronize()

    def run():
        out = fused_mha(q, k, v)
        torch.cuda.synchronize()
        return out

    result = benchmark(run)
    assert result is not None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_benchmark_sdpa_reference(benchmark, mha_tensors):
    """PyTorch SDPA (FlashAttention-2 if available) — reference."""
    import torch.nn.functional as F

    q, k, v = mha_tensors

    def run():
        out = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        return out

    result = benchmark(run)
    assert result is not None


# ── Parametric sweep ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("action_dim", [7, 32, 64, 128, 256])
def test_benchmark_euler_action_dim_sweep(benchmark, action_dim, device):
    """Euler step across typical VLA action dimensions."""
    from kernels.flow_matching_step import flow_matching_euler_step

    B = 4
    z = torch.randn(B, action_dim, dtype=torch.float16, device=device).contiguous()
    v = torch.randn(B, action_dim, dtype=torch.float16, device=device).contiguous()

    # Warmup
    _ = flow_matching_euler_step(z, v, 0.05)
    torch.cuda.synchronize()

    def run():
        out = flow_matching_euler_step(z, v, 0.05)
        torch.cuda.synchronize()
        return out

    result = benchmark(run)
    assert result is not None
