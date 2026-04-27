"""Fixed-window KV cache manager for VLA Runtime.

Design (CLAUDE.md ADR-011):
    VLA context is short (image patches + short instruction, typically < 1024
    tokens). PagedAttention is overkill — the overhead of page tables exceeds
    the benefit for such short sequences. Instead, we allocate a fixed-size
    contiguous buffer per request and release it immediately on completion.

Memory model:
    - Each request gets a pre-allocated KV slot: [num_layers, 2, max_seq_len, num_heads, head_dim]
      (factor 2 = keys + values)
    - Slots are managed in a free-list pool; allocation is O(1) amortised.
    - When the running context exceeds max_context_tokens, the oldest tokens
      are evicted (FIFO sliding window), preserving image tokens at position 0.

Eviction policy — "vision-anchor":
    Image tokens are never evicted (position 0..num_image_tokens-1 are pinned).
    Text tokens are evicted FIFO when the window is full.
    This ensures vision context is always fresh, consistent with how
    VLA models condition action generation.

Memory estimate (example):
    Config: 4B LLM, 32 layers, 32 heads, head_dim=128, max_seq=512, FP16
    Per-slot: 32 × 2 × 512 × 32 × 128 × 2 bytes = 256 MB
    Pool of 8 slots (8 concurrent robots): 2 GB

Usage::

    manager = KVCacheManager(KVCacheConfig(
        num_layers=32, num_heads=32, head_dim=128,
        max_context_tokens=512, num_image_tokens=196,
        pool_size=8,
    ))
    slot = manager.allocate("req-001")
    # ... pass slot.k_cache / slot.v_cache to VLM attention layers ...
    manager.free("req-001")
    print(manager.stats())
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import torch

from vla_runtime.errors import EdgeResourceError, ErrorContext
from vla_runtime.observability.metrics import get_metrics
from vla_runtime.observability.tracer import get_tracer

_tracer = get_tracer("vlm.kv_cache")
_metrics = get_metrics()


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class KVCacheConfig:
    """KV cache allocation parameters.

    Attributes:
        num_layers: Number of transformer layers in the VLM backbone.
        num_heads: Number of attention heads per layer.
        head_dim: Dimension of each attention head.
        max_context_tokens: Maximum sequence length per slot.
        num_image_tokens: Number of image patch tokens (pinned — never evicted).
        pool_size: Number of pre-allocated slots in the free-list pool.
            Should be ≥ max concurrent robots.
        dtype: Cache tensor dtype (float16 recommended for memory efficiency).
        device: CUDA device to allocate cache buffers on.
        eviction_policy: Currently only "sliding_window" is supported.
            "none" disables eviction (raises on overflow).
    """

    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    max_context_tokens: int = 512
    num_image_tokens: int = 196
    pool_size: int = 8
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    eviction_policy: Literal["sliding_window", "none"] = "sliding_window"

    @property
    def bytes_per_slot(self) -> int:
        """Memory footprint in bytes for one KV slot."""
        element_size = 2 if self.dtype == torch.float16 else 4
        return (
            self.num_layers * 2  # K and V
            * self.max_context_tokens
            * self.num_heads
            * self.head_dim
            * element_size
        )

    @property
    def pool_memory_gb(self) -> float:
        return self.bytes_per_slot * self.pool_size / (1024 ** 3)


# ── KV slot ───────────────────────────────────────────────────────────────────

@dataclass
class KVCacheSlot:
    """A pre-allocated KV cache buffer for one request.

    Attributes:
        slot_id: Internal pool slot index.
        request_id: The request currently using this slot ("" if free).
        k_cache: Key cache tensor, shape [num_layers, max_seq, num_heads, head_dim].
        v_cache: Value cache tensor, shape [num_layers, max_seq, num_heads, head_dim].
        seq_len: Number of tokens currently stored.
        image_token_end: Tokens [0, image_token_end) are pinned and never evicted.
        allocated_at: Timestamp of most recent allocation (seconds since epoch).
    """

    slot_id: int
    request_id: str
    k_cache: torch.Tensor      # [L, S, H, D]
    v_cache: torch.Tensor      # [L, S, H, D]
    seq_len: int = 0
    image_token_end: int = 0
    allocated_at: float = field(default_factory=time.time)

    @property
    def is_free(self) -> bool:
        return self.request_id == ""

    def write_kv(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Append new K/V tokens at the current sequence position.

        Args:
            layer_idx: Which layer's cache to write.
            keys: New key tokens, shape [new_tokens, num_heads, head_dim].
            values: New value tokens, shape [new_tokens, num_heads, head_dim].
        """
        new_tokens = keys.shape[0]
        available = self.k_cache.shape[1] - self.seq_len
        if new_tokens > available:
            raise OverflowError(
                f"KV slot {self.slot_id} overflow: need {new_tokens} tokens, "
                f"only {available} remaining (seq_len={self.seq_len})"
            )
        end = self.seq_len + new_tokens
        self.k_cache[layer_idx, self.seq_len:end] = keys
        self.v_cache[layer_idx, self.seq_len:end] = values
        self.seq_len = end

    def active_keys(self, layer_idx: int) -> torch.Tensor:
        """Return the valid key slice for this layer."""
        return self.k_cache[layer_idx, : self.seq_len]

    def active_values(self, layer_idx: int) -> torch.Tensor:
        """Return the valid value slice for this layer."""
        return self.v_cache[layer_idx, : self.seq_len]

    def reset(self) -> None:
        """Clear the slot without deallocating the underlying tensors."""
        self.seq_len = 0
        self.image_token_end = 0
        self.request_id = ""


# ── KV cache manager ──────────────────────────────────────────────────────────

class KVCacheManager:
    """Fixed-window KV cache pool manager.

    Pre-allocates `pool_size` contiguous KV slots at startup. Requests check
    out a slot and return it on completion. If the pool is exhausted, an
    EdgeResourceError is raised (caller should degrade to a no-cache path).

    Thread-safe: uses a reentrant lock for slot allocation/deallocation.

    Args:
        config: KV cache configuration.
    """

    def __init__(self, config: KVCacheConfig | None = None) -> None:
        self._config = config or KVCacheConfig()
        self._lock = threading.RLock()
        self._slots: list[KVCacheSlot] = []
        self._active: dict[str, KVCacheSlot] = {}   # request_id → slot
        self._alloc_count = 0
        self._eviction_count = 0

        if torch.cuda.is_available() or self._config.device == "cpu":
            self._pre_allocate()
        else:
            _tracer.event("kv_cache_deferred", reason="cuda_not_available")

    # ── Pool lifecycle ────────────────────────────────────────────────────────

    def _pre_allocate(self) -> None:
        """Allocate all pool slots at startup."""
        cfg = self._config
        total_gb = cfg.pool_memory_gb
        _tracer.event(
            "kv_cache_pre_allocate",
            num_slots=cfg.pool_size,
            bytes_per_slot=cfg.bytes_per_slot,
            total_gb=round(total_gb, 2),
            device=cfg.device,
        )

        for i in range(cfg.pool_size):
            k = torch.zeros(
                cfg.num_layers,
                cfg.max_context_tokens,
                cfg.num_heads,
                cfg.head_dim,
                dtype=cfg.dtype,
                device=cfg.device,
            )
            v = torch.zeros_like(k)
            self._slots.append(
                KVCacheSlot(slot_id=i, request_id="", k_cache=k, v_cache=v)
            )

        _metrics.gauge("kv_cache_pool_size", cfg.pool_size)
        _metrics.gauge("kv_cache_pool_memory_gb", total_gb)

    # ── Public API ────────────────────────────────────────────────────────────

    def allocate(
        self,
        request_id: str,
        num_image_tokens: int | None = None,
    ) -> KVCacheSlot:
        """Allocate a free KV slot for a request.

        Args:
            request_id: Unique request identifier.
            num_image_tokens: Number of image patch tokens to pin.
                Defaults to config.num_image_tokens.

        Returns:
            A reset KVCacheSlot ready for writing.

        Raises:
            EdgeResourceError: All pool slots are in use.
        """
        with self._lock:
            if request_id in self._active:
                return self._active[request_id]

            free_slot = next((s for s in self._slots if s.is_free), None)
            if free_slot is None:
                ctx = ErrorContext(
                    trace_id=_tracer.trace_id,
                    module_name="vlm.kv_cache",
                    request_id=request_id,
                )
                bytes_per_slot = self._config.bytes_per_slot
                raise EdgeResourceError(
                    required_mb=bytes_per_slot / (1024 ** 2),
                    available_mb=0.0,
                    context=ctx,
                )

            free_slot.reset()
            free_slot.request_id = request_id
            free_slot.image_token_end = (
                num_image_tokens
                if num_image_tokens is not None
                else self._config.num_image_tokens
            )
            free_slot.allocated_at = time.time()
            self._active[request_id] = free_slot
            self._alloc_count += 1

            _metrics.increment("kv_cache_alloc")
            _metrics.gauge("kv_cache_active_slots", len(self._active))
            _tracer.event(
                "kv_slot_allocated",
                request_id=request_id,
                slot_id=free_slot.slot_id,
            )
            return free_slot

    def free(self, request_id: str) -> None:
        """Return a slot to the free pool.

        Args:
            request_id: The request whose slot should be freed.
        """
        with self._lock:
            slot = self._active.pop(request_id, None)
            if slot is None:
                return
            slot.reset()
            _metrics.increment("kv_cache_free")
            _metrics.gauge("kv_cache_active_slots", len(self._active))
            _tracer.event(
                "kv_slot_freed",
                request_id=request_id,
                slot_id=slot.slot_id,
                seq_len=slot.seq_len,
            )

    def evict_oldest_text_tokens(
        self,
        slot: KVCacheSlot,
        num_to_evict: int,
    ) -> None:
        """Slide the text token window by evicting the oldest non-image tokens.

        Image tokens (positions < slot.image_token_end) are pinned and never
        evicted. Only text tokens beyond that boundary are shifted out.

        Args:
            slot: The slot to modify.
            num_to_evict: How many text tokens to discard from the front of
                the text region.
        """
        if self._config.eviction_policy == "none":
            raise OverflowError(
                f"KV slot {slot.slot_id} overflow; eviction_policy='none'"
            )

        pin = slot.image_token_end
        text_len = slot.seq_len - pin
        if num_to_evict >= text_len:
            # Evict all text tokens — keep only image tokens
            slot.seq_len = pin
            self._eviction_count += num_to_evict
            return

        # Shift text region left by num_to_evict positions
        keep_start = pin + num_to_evict
        new_text_len = text_len - num_to_evict

        # Shift in-place for all layers
        slot.k_cache[:, pin : pin + new_text_len] = slot.k_cache[
            :, keep_start : keep_start + new_text_len
        ]
        slot.v_cache[:, pin : pin + new_text_len] = slot.v_cache[
            :, keep_start : keep_start + new_text_len
        ]
        slot.seq_len = pin + new_text_len
        self._eviction_count += num_to_evict

        _metrics.increment("kv_cache_eviction", value=num_to_evict)

    def get(self, request_id: str) -> KVCacheSlot | None:
        """Retrieve the active slot for a request (None if not found)."""
        return self._active.get(request_id)

    # ── Memory accounting ─────────────────────────────────────────────────────

    def memory_usage_gb(self) -> float:
        """Total GPU memory used by the KV cache pool."""
        return self._config.pool_memory_gb

    def active_count(self) -> int:
        """Number of currently allocated slots."""
        with self._lock:
            return len(self._active)

    def stats(self) -> dict:
        """Return a snapshot of KV cache manager metrics."""
        with self._lock:
            return {
                "pool_size": self._config.pool_size,
                "active_slots": len(self._active),
                "free_slots": self._config.pool_size - len(self._active),
                "total_allocs": self._alloc_count,
                "total_evictions": self._eviction_count,
                "pool_memory_gb": round(self._config.pool_memory_gb, 3),
                "max_context_tokens": self._config.max_context_tokens,
            }

    def shutdown(self) -> None:
        """Free all tensors and reset pool.

        Call during serving shutdown to release GPU memory explicitly rather
        than relying on GC timing.
        """
        with self._lock:
            for slot in self._slots:
                del slot.k_cache
                del slot.v_cache
            self._slots.clear()
            self._active.clear()
        _tracer.event("kv_cache_shutdown")
