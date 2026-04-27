"""Fused LayerNorm + Linear kernel for VLM transformer blocks.

Motivation:
    In every transformer block, the pattern:
        y = LayerNorm(x)     # HBM write
        z = Linear(y, W)     # HBM read + matmul
    causes a round-trip to HBM between normalisation and the projection.
    This kernel eliminates that round-trip by fusing both ops into a single
    GPU pass, keeping the normalised values in SRAM registers.

Design:
    - Input x: [B*N, hidden_dim] (batch × sequence flattened)
    - Weight W: [out_features, hidden_dim]  (row-major)
    - Bias b: [out_features] (optional)
    - LayerNorm params: gamma [hidden_dim], beta [hidden_dim]
    - Output z: [B*N, out_features]

    Each kernel program handles one row of x:
        1. Load row → compute mean + variance → normalise → scale+shift
        2. Dot with W row-by-row in tiles → accumulate output

    FP32 accumulation within each tile; fp16 store.

Usage::

    from kernels.layernorm_linear import fused_layernorm_linear
    z = fused_layernorm_linear(x, gamma, beta, weight, bias, eps=1e-5)

Benchmark target: ≤ 0.8× latency of separate LayerNorm + Linear (torch).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import]
    import triton.language as tl  # type: ignore[import]
    _triton_available = True
except ImportError:
    _triton_available = False


# ── Autotune configs ──────────────────────────────────────────────────────────

_LN_LINEAR_CONFIGS = []
if _triton_available:
    _LN_LINEAR_CONFIGS = [
        triton.Config({"BLOCK_H": 128, "BLOCK_OUT": 64}, num_warps=4),
        triton.Config({"BLOCK_H": 256, "BLOCK_OUT": 64}, num_warps=4),
        triton.Config({"BLOCK_H": 512, "BLOCK_OUT": 64}, num_warps=8),
        triton.Config({"BLOCK_H": 512, "BLOCK_OUT": 128}, num_warps=8),
        triton.Config({"BLOCK_H": 1024, "BLOCK_OUT": 64}, num_warps=8),
    ]


# ── Kernel definition ─────────────────────────────────────────────────────────

if _triton_available:
    @triton.autotune(configs=_LN_LINEAR_CONFIGS, key=["H", "OUT"])
    @triton.jit
    def _layernorm_linear_kernel(
        x_ptr,       # [M, H]  float16  input
        g_ptr,       # [H]     float32  LN gamma
        b_ptr,       # [H]     float32  LN beta
        w_ptr,       # [OUT, H] float16 linear weight (row-major)
        bias_ptr,    # [OUT]   float32  linear bias (may be NULL)
        out_ptr,     # [M, OUT] float16 output
        eps,         # float32 layernorm epsilon
        M, H, OUT,
        has_bias: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        """One program per row m of the [M, H] input.

        Computes: out[m, :] = Linear(LayerNorm(x[m, :]))
        """
        row = tl.program_id(0)
        x_base = row * H

        # ── Step 1: compute mean and variance over H ──────────────────────────
        mean_acc = tl.zeros((1,), dtype=tl.float32)
        var_acc  = tl.zeros((1,), dtype=tl.float32)
        count = tl.zeros((1,), dtype=tl.float32)

        for h_start in range(0, H, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            mask = h_offsets < H
            x_val = tl.load(x_ptr + x_base + h_offsets, mask=mask, other=0.0).to(tl.float32)
            mean_acc += tl.sum(tl.where(mask, x_val, 0.0))
            var_acc  += tl.sum(tl.where(mask, x_val * x_val, 0.0))
            count    += tl.sum(tl.where(mask, 1.0, 0.0).to(tl.float32))

        mean = mean_acc / count
        var  = var_acc / count - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        # ── Step 2: normalise and project ─────────────────────────────────────
        # For each output tile, accumulate dot(norm_x, W[out_tile, :])
        for out_start in range(0, OUT, BLOCK_OUT):
            out_offsets = out_start + tl.arange(0, BLOCK_OUT)
            out_mask = out_offsets < OUT

            # Accumulate dot product in fp32
            acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

            for h_start in range(0, H, BLOCK_H):
                h_offsets = h_start + tl.arange(0, BLOCK_H)
                h_mask = h_offsets < H

                # Load x, normalise, scale
                x_val = tl.load(
                    x_ptr + x_base + h_offsets, mask=h_mask, other=0.0
                ).to(tl.float32)
                g_val = tl.load(g_ptr + h_offsets, mask=h_mask, other=1.0)
                b_val = tl.load(b_ptr + h_offsets, mask=h_mask, other=0.0)
                norm_x = (x_val - mean) * rstd * g_val + b_val  # [BLOCK_H]

                # Load weight tile: [BLOCK_OUT, BLOCK_H]
                w_offsets = (out_offsets[:, None] * H + h_offsets[None, :])
                w_val = tl.load(
                    w_ptr + w_offsets,
                    mask=(out_mask[:, None] & h_mask[None, :]),
                    other=0.0,
                ).to(tl.float32)

                acc += tl.sum(norm_x[None, :] * w_val, axis=1)

            if has_bias:
                bias_val = tl.load(bias_ptr + out_offsets, mask=out_mask, other=0.0)
                acc += bias_val

            tl.store(
                out_ptr + row * OUT + out_offsets,
                acc.to(tl.float16),
                mask=out_mask,
            )


# ── Python dispatch ───────────────────────────────────────────────────────────

def fused_layernorm_linear(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused LayerNorm → Linear without intermediate HBM write.

    Args:
        x: Input activations, shape [M, H], float16.
            M = batch * sequence_length, H = hidden_dim.
        gamma: LayerNorm scale, shape [H], float32.
        beta: LayerNorm shift, shape [H], float32.
        weight: Linear weight matrix, shape [OUT, H], float16.
        bias: Linear bias, shape [OUT], float32. Optional.
        eps: LayerNorm epsilon for numerical stability.

    Returns:
        Output activations, shape [M, OUT], float16.

    Note:
        Falls back to sequential F.layer_norm + F.linear when Triton is
        unavailable (CPU inference or non-CUDA environment).
    """
    assert x.ndim == 2, f"Expected 2-D input, got shape {x.shape}"
    M, H = x.shape
    OUT = weight.shape[0]
    assert weight.shape == (OUT, H), f"Weight shape mismatch: {weight.shape}"
    assert gamma.shape == (H,)
    assert beta.shape == (H,)

    if not _triton_available or not x.is_cuda:
        # PyTorch reference path
        norm = F.layer_norm(x.float(), (H,), gamma, beta, eps=eps).to(x.dtype)
        return F.linear(norm, weight, bias)

    x = x.contiguous()
    gamma_f32 = gamma.float().contiguous()
    beta_f32 = beta.float().contiguous()
    weight = weight.contiguous()
    out = torch.empty(M, OUT, dtype=torch.float16, device=x.device)

    bias_ptr = bias if bias is not None else x  # dummy; kernel checks has_bias

    _layernorm_linear_kernel[(M,)](
        x, gamma_f32, beta_f32, weight, bias_ptr, out,
        eps,
        M, H, OUT,
        has_bias=(bias is not None),
    )
    return out


def warmup(
    device: torch.device = torch.device("cuda"),
    hidden_dim: int = 1024,
    out_features: int = 4096,
    seq_len: int = 257,
) -> None:
    """Pre-compile fused_layernorm_linear for typical vision encoder dims."""
    if not _triton_available:
        return
    x = torch.zeros(seq_len, hidden_dim, dtype=torch.float16, device=device)
    g = torch.ones(hidden_dim, dtype=torch.float32, device=device)
    b = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
    w = torch.zeros(out_features, hidden_dim, dtype=torch.float16, device=device)
    _ = fused_layernorm_linear(x, g, b, w)
