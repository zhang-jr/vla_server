"""Fused multi-head attention kernel for the VLA vision encoder.

Target: replace the naive O(N²) attention materialisation in the vision
encoder with a tiled FlashAttention-style kernel that stays within SRAM.

Design:
    - Tile size (BLOCK_Q × BLOCK_K) tuned via @triton.autotune.
    - FP16 Q/K/V input, FP32 accumulation inside each tile.
    - Causal mask is disabled by default (ViT uses bidirectional attention).
    - Output dtype matches input (fp16).
    - Supports MHA with any num_heads; head_dim must be ≤ 128 for now.

Memory savings vs naive:
    Naive: O(B × H × N²) attention matrix in HBM.
    Fused: O(B × H × N × head_dim) — only Q/K/V and output tiles in SRAM.
    For N=256 (14×14+[CLS]) and H=16: naive ≈ 16 MB/batch, fused ≈ 3 MB/batch.

Usage::

    from kernels.fused_attention import fused_mha
    out = fused_mha(q, k, v, scale=head_dim**-0.5)

Compilation:
    warmup() pre-compiles for the typical 256-token vision sequence.
"""

from __future__ import annotations

import math

import torch

try:
    import triton  # type: ignore[import]
    import triton.language as tl  # type: ignore[import]
    _triton_available = True
except ImportError:
    _triton_available = False


# ── Autotune configs ──────────────────────────────────────────────────────────

_MHA_AUTOTUNE_CONFIGS = []
if _triton_available:
    _MHA_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_Q": 16, "BLOCK_K": 16}, num_warps=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_Q": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_Q": 64, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_Q": 128, "BLOCK_K": 64}, num_warps=8),
    ]


# ── Kernel definition ─────────────────────────────────────────────────────────

if _triton_available:
    @triton.autotune(configs=_MHA_AUTOTUNE_CONFIGS, key=["N", "HEAD_DIM"])
    @triton.jit
    def _fused_mha_kernel(
        q_ptr,      # [B, H, N, D]  float16
        k_ptr,      # [B, H, N, D]  float16
        v_ptr,      # [B, H, N, D]  float16
        out_ptr,    # [B, H, N, D]  float16
        scale,      # float32 = head_dim ** -0.5
        B, H, N,
        HEAD_DIM: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Tiled FlashAttention inner loop.

        One program handles one (batch, head, query-tile).
        """
        batch_head = tl.program_id(0)   # flattened B*H
        q_tile    = tl.program_id(1)   # which BLOCK_Q tile of queries

        b = batch_head // H
        h = batch_head % H

        # Pointers to the (b, h) slice of Q, K, V
        base_qkv = (b * H + h) * N * HEAD_DIM

        q_start = q_tile * BLOCK_Q
        q_offsets = q_start + tl.arange(0, BLOCK_Q)    # [BLOCK_Q]
        d_offsets = tl.arange(0, HEAD_DIM)              # [HEAD_DIM]

        q_mask = q_offsets < N

        # Load Q tile: [BLOCK_Q, HEAD_DIM]
        q = tl.load(
            q_ptr + base_qkv + q_offsets[:, None] * HEAD_DIM + d_offsets[None, :],
            mask=q_mask[:, None],
            other=0.0,
        ).to(tl.float32) * scale

        # Accumulator: [BLOCK_Q, HEAD_DIM] (weighted sum of V)
        acc = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)
        # Softmax state: log-sum-exp for numerically stable online softmax
        m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)

        # Iterate over K/V tiles
        for k_start in range(0, N, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            kv_mask = k_offsets < N

            # Load K tile: [BLOCK_K, HEAD_DIM]
            k = tl.load(
                k_ptr + base_qkv + k_offsets[:, None] * HEAD_DIM + d_offsets[None, :],
                mask=kv_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            # Attention scores: [BLOCK_Q, BLOCK_K]
            scores = tl.dot(q, tl.trans(k))

            # Mask out-of-bounds K positions
            scores = tl.where(kv_mask[None, :], scores, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            l_i = alpha * l_i + tl.sum(p, axis=1)
            m_i = m_new

            # Load V tile: [BLOCK_K, HEAD_DIM]
            v = tl.load(
                v_ptr + base_qkv + k_offsets[:, None] * HEAD_DIM + d_offsets[None, :],
                mask=kv_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            acc = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)

        # Normalise
        out = acc / (l_i[:, None] + 1e-8)

        # Store output
        tl.store(
            out_ptr + base_qkv + q_offsets[:, None] * HEAD_DIM + d_offsets[None, :],
            out.to(tl.float16),
            mask=q_mask[:, None],
        )


# ── Python dispatch ───────────────────────────────────────────────────────────

def fused_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused multi-head attention for vision encoder sequences.

    Args:
        q: Query tensor, shape [B, H, N, D], float16.
        k: Key tensor, shape [B, H, N, D], float16.
        v: Value tensor, shape [B, H, N, D], float16.
        scale: Attention scale (default: D**-0.5).

    Returns:
        Attention output, shape [B, H, N, D], float16.

    Note:
        Falls back to torch.nn.functional.scaled_dot_product_attention when
        Triton is not available or tensors are on CPU.
    """
    assert q.shape == k.shape == v.shape, "Q/K/V shapes must match"
    B, H, N, D = q.shape
    assert D <= 128, f"head_dim={D} exceeds kernel maximum of 128"

    if scale is None:
        scale = D ** -0.5

    if not _triton_available or not q.is_cuda:
        import torch.nn.functional as F
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    # Ensure contiguous BHND layout
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)
    grid = (B * H, math.ceil(N / 16))  # autotuner adjusts BLOCK_Q

    _fused_mha_kernel[grid](
        q, k, v, out,
        scale,
        B, H, N,
        HEAD_DIM=D,
    )
    return out


def warmup(
    device: torch.device = torch.device("cuda"),
    seq_len: int = 257,   # 14×14 + [CLS] token
    num_heads: int = 16,
    head_dim: int = 64,
    batch_size: int = 1,
) -> None:
    """Pre-compile the fused MHA kernel for the typical vision encoder config.

    Args:
        device: CUDA device to compile on.
        seq_len: Expected vision sequence length (default: 256 patches + CLS).
        num_heads: Number of attention heads.
        head_dim: Per-head dimension.
        batch_size: Batch size for compilation.
    """
    if not _triton_available:
        return
    dummy = torch.zeros(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float16, device=device,
    )
    _ = fused_mha(dummy, dummy, dummy)
