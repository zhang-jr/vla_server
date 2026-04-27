"""Minimal end-to-end example: OpenVLA inference via VLA Runtime.

Demonstrates:
1. QuantizationConfig setup (cloud aggressive preset)
2. VLM backend initialization (vLLM adapter)
3. Action head setup (discrete token, OFT disabled for simplicity)
4. AsyncScheduler submission and streaming
5. Reading actions from the double buffer

This script uses MockVLMBackend instead of real vLLM so it runs without GPU.
To use real OpenVLA:
  1. Replace MockVLMBackend with VLLMBackend(model_path="openvla/openvla-7b")
  2. Set use_oft=True for parallel decoding (requires GPU)
  3. Run on a machine with an A100 or H100

Usage:
    python examples/openvla_minimal.py
"""

from __future__ import annotations

import asyncio
import uuid
import logging

import torch

from vla_runtime import configure_logging, get_tracer
from vla_runtime.action_head.discrete_token import DiscreteTokenHead
from vla_runtime.action_head.flow_matching import FlowMatchingHead
from vla_runtime.quantization.config import QuantizationConfig
from vla_runtime.scheduler.async_scheduler import AsyncScheduler
from vla_runtime.scheduler.budget_manager import BudgetConfig
from vla_runtime.vlm.base import ActionRequest, Embedding, TokenChunk

configure_logging(level=logging.INFO)
_tracer = get_tracer("example.openvla_minimal")


# ── Mock backend (no GPU required) ────────────────────────────────────────────

class MockOpenVLABackend:
    """Stand-in for VLLMBackend("openvla/openvla-7b").

    Replace with real VLLMBackend for production use.
    """

    @property
    def model_id(self) -> str:
        return "mock:openvla-7b"

    @property
    def latency_budget_ms(self) -> float:
        return 200.0

    @property
    def kv_cache_enabled(self) -> bool:
        return True

    async def encode(self, image: torch.Tensor, text: str) -> Embedding:
        # Simulate 50ms VLM encode latency
        await asyncio.sleep(0.05)
        return Embedding(
            data=torch.randn(512, dtype=torch.float16),
            model_id=self.model_id,
        )

    async def generate(self, image, text):
        n_tokens = 7 * 8   # action_dim * n_dct_coeffs
        yield TokenChunk(token_ids=[128] * n_tokens, is_final=True)

    async def warmup(self) -> None:
        print("[VLM] Warming up OpenVLA backend...")
        await asyncio.sleep(0.01)
        print("[VLM] Warmup complete.")

    async def shutdown(self) -> None:
        print("[VLM] Shutting down.")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 60)
    print("VLA Runtime — OpenVLA Minimal Example")
    print("=" * 60)

    # 1. Quantization config
    quant_cfg = QuantizationConfig.cloud_aggressive()
    print(f"\n[Config] Quantization: {quant_cfg.summary()}")

    # Validate: discrete_token head allows w8a16, but we use fp16 here
    quant_cfg.validate_for_head_type("discrete_token")
    print("[Config] Quantization config validated OK")

    # 2. Build pipeline components
    vlm_backend = MockOpenVLABackend()

    action_head = DiscreteTokenHead(
        action_dim=7,
        chunk_size=16,
        embedding_dim=512,
        vocab_size=256,
        n_dct_coeffs=8,
        use_oft=False,    # set True for OFT parallel decoding (GPU required)
        frequency_hz=15.0,
    )
    print(f"\n[ActionHead] Type: {action_head.head_type}, freq={action_head.frequency_hz}Hz")
    print(f"[ActionHead] Chunk size: {action_head.chunk_size}, action dim: {action_head.action_dim}")

    # 3. Create scheduler
    scheduler = AsyncScheduler(
        vlm_backend=vlm_backend,
        action_head=action_head,
        budget_config=BudgetConfig(
            vlm_budget_ms=200.0,
            action_head_budget_ms=50.0,
            total_budget_ms=300.0,
        ),
        max_stale_frames=5,
        robot_id="franka_panda_0",
    )

    await scheduler.start()
    print("\n[Scheduler] Started. Running inference...")

    # 4. Simulate 3 inference cycles
    for cycle in range(3):
        request = ActionRequest(
            request_id=str(uuid.uuid4())[:8],
            image=torch.rand(3, 224, 224),
            instruction="pick up the red cube and place it in the bin",
            robot_state=torch.zeros(14),   # 7 joints + 7 velocities
            frame_id=cycle,
        )

        handle = await scheduler.submit(request)

        print(f"\n[Cycle {cycle}] Request {request.request_id} submitted")

        async for chunk in scheduler.stream(handle):
            print(
                f"[Cycle {cycle}] Chunk received: "
                f"shape={list(chunk.actions.shape)}, "
                f"head={chunk.head_type}, "
                f"first_action={chunk.step(0).tolist()[:3]}..."
            )

    # 5. Print stats
    stats = scheduler.stats()
    print("\n[Stats] Scheduler snapshot:")
    print(f"  Buffer: {stats['buffer']}")
    for mod, info in stats['budget'].items():
        if info['n_samples'] > 0:
            print(
                f"  Budget [{mod}]: p50={info['p50_ms']}ms "
                f"p95={info['p95_ms']}ms "
                f"violation_rate={info['violation_rate']:.1%}"
            )

    await scheduler.stop()
    print("\n[Done] Scheduler stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
