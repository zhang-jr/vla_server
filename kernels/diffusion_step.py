"""Triton kernel for a diffusion denoising step (DDPM / DDIM).

Target: < 5ms per step on A100.

Diffusion denoising:
    Given a noisy sample x_t, noise prediction ε_θ(x_t, t, c), and the DDPM
    noise schedule {α_t}, the denoised sample is computed as:

        x_{t-1} = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t  +  σ_t * noise

    where σ_t = 0 for DDIM (deterministic).

    This kernel fuses the noise subtraction, scale-divide, and optional
    stochastic noise addition into one CUDA pass.

Usage:
    from kernels.diffusion_step import ddpm_step, ddim_step
    x_prev = ddpm_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev, sigma_t)
    x_prev = ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton  # type: ignore[import]
    import triton.language as tl  # type: ignore[import]
    _triton_available = True
except ImportError:
    _triton_available = False


# ── Autotune configs ──────────────────────────────────────────────────────────
_DDPM_AUTOTUNE_CONFIGS = []
if _triton_available:
    _DDPM_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_D": 64}),
        triton.Config({"BLOCK_D": 128}),
        triton.Config({"BLOCK_D": 256}),
        triton.Config({"BLOCK_D": 512}),
        triton.Config({"BLOCK_D": 1024}),
    ]


# ── Kernel definition ─────────────────────────────────────────────────────────

if _triton_available:
    @triton.autotune(configs=_DDPM_AUTOTUNE_CONFIGS, key=["D"])
    @triton.jit
    def _ddpm_step_kernel(
        x_t_ptr,            # [B, D] float16, noisy sample at step t
        noise_pred_ptr,     # [B, D] float16, predicted noise ε_θ
        out_ptr,            # [B, D] float16, output x_{t-1}
        noise_ptr,          # [B, D] float16, Gaussian noise (may be zeros for DDIM)
        sqrt_alpha_bar_t,       # float32 scalar √ᾱ_t
        sqrt_one_minus_ab_t,    # float32 scalar √(1-ᾱ_t)
        sqrt_alpha_bar_prev,    # float32 scalar √ᾱ_{t-1}
        sigma_t,                # float32 scalar, 0 for DDIM
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK_D)
        mask = col < D

        x = tl.load(x_t_ptr + row * D + col, mask=mask, other=0.0).to(tl.float32)
        eps = tl.load(noise_pred_ptr + row * D + col, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(noise_ptr + row * D + col, mask=mask, other=0.0).to(tl.float32)

        # x0_pred = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        x0_pred = (x - sqrt_one_minus_ab_t * eps) / (sqrt_alpha_bar_t + 1e-8)

        # x_{t-1} = √ᾱ_{t-1} * x0_pred + σ_t * z
        x_prev = sqrt_alpha_bar_prev * x0_pred + sigma_t * z

        tl.store(out_ptr + row * D + col, x_prev.to(tl.float16), mask=mask)


# ── Public API ────────────────────────────────────────────────────────────────

def ddpm_step(
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    alpha_bar_t: float,
    alpha_bar_prev: float,
    sigma_t: float,
) -> torch.Tensor:
    """DDPM denoising step with stochastic noise injection.

    Args:
        x_t: Noisy sample at step t, shape [B, D], float16.
        noise_pred: Predicted noise ε_θ(x_t, t, c), shape [B, D], float16.
        alpha_bar_t: Cumulative noise schedule at step t (ᾱ_t).
        alpha_bar_prev: Cumulative noise schedule at step t-1 (ᾱ_{t-1}).
        sigma_t: Stochastic noise scale (σ_t > 0 for DDPM, 0 for DDIM).

    Returns:
        Denoised sample x_{t-1}, shape [B, D], float16.
    """
    assert x_t.shape == noise_pred.shape
    B, D = x_t.shape

    sqrt_ab_t = math.sqrt(alpha_bar_t)
    sqrt_1m_ab_t = math.sqrt(1.0 - alpha_bar_t)
    sqrt_ab_prev = math.sqrt(alpha_bar_prev)

    if not _triton_available or not x_t.is_cuda:
        # PyTorch fallback
        x0_pred = (x_t.float() - sqrt_1m_ab_t * noise_pred.float()) / (sqrt_ab_t + 1e-8)
        noise = torch.randn_like(x_t).float() if sigma_t > 0 else torch.zeros_like(x_t).float()
        return (sqrt_ab_prev * x0_pred + sigma_t * noise).half()

    noise = torch.randn_like(x_t) if sigma_t > 0 else torch.zeros_like(x_t)
    out = torch.empty_like(x_t)
    # BLOCK_D selected by autotuner keyed on D.
    _ddpm_step_kernel[(B,)](
        x_t, noise_pred, out, noise,
        sqrt_ab_t, sqrt_1m_ab_t, sqrt_ab_prev, sigma_t,
        D=D,
    )
    return out


def ddim_step(
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    alpha_bar_t: float,
    alpha_bar_prev: float,
) -> torch.Tensor:
    """Deterministic DDIM denoising step (σ_t = 0).

    Fewer steps required vs DDPM for the same quality.
    Preferred for real-time VLA where denoising budget is tight.

    Args:
        x_t: Noisy sample at step t, [B, D], float16.
        noise_pred: Predicted noise, [B, D], float16.
        alpha_bar_t: ᾱ_t.
        alpha_bar_prev: ᾱ_{t-1}.

    Returns:
        x_{t-1}, shape [B, D], float16.
    """
    return ddpm_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev, sigma_t=0.0)


def warmup(device: torch.device = torch.device("cuda")) -> None:
    """Pre-compile Triton diffusion kernel."""
    if not _triton_available:
        return
    dummy = torch.zeros(1, 256, dtype=torch.float16, device=device)
    _ = ddim_step(dummy, dummy, alpha_bar_t=0.9, alpha_bar_prev=0.85)
