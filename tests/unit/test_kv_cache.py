"""Unit tests for KVCacheManager — Phase 2 (ADR-011).

Tests verify:
  - Pre-allocation creates the correct number of slots
  - Allocate returns a valid KVCacheSlot with request_id set
  - Free returns slot to pool (can be reallocated)
  - Duplicate allocate for same request_id returns same slot
  - Pool exhaustion raises EdgeResourceError
  - KVCacheSlot.write_kv appends correctly
  - KVCacheSlot.write_kv raises OverflowError when full
  - evict_oldest_text_tokens slides window (preserves image tokens)
  - eviction_policy='none' raises OverflowError on eviction attempt
  - stats() returns correct accounting
  - shutdown() clears pool

All tests run on CPU (device="cpu", dtype=torch.float32).
"""

from __future__ import annotations

import pytest
import torch

from vla_runtime.errors import EdgeResourceError
from vla_runtime.vlm.kv_cache import KVCacheConfig, KVCacheManager


# ── Config helper ─────────────────────────────────────────────────────────────

def _small_cfg(**kwargs) -> KVCacheConfig:
    """Tiny config for fast CPU tests."""
    defaults = dict(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        max_context_tokens=16,
        num_image_tokens=4,
        pool_size=3,
        device="cpu",
        dtype=torch.float32,
    )
    defaults.update(kwargs)
    return KVCacheConfig(**defaults)


# ── Pre-allocation ────────────────────────────────────────────────────────────

def test_pre_allocate_correct_count():
    cfg = _small_cfg(pool_size=4)
    manager = KVCacheManager(cfg)
    assert len(manager._slots) == 4
    assert all(s.is_free for s in manager._slots)
    manager.shutdown()


def test_bytes_per_slot():
    cfg = _small_cfg(num_layers=2, num_heads=2, head_dim=8, max_context_tokens=16, dtype=torch.float32)
    # 2 layers × 2 (K+V) × 16 seq × 2 heads × 8 dim × 4 bytes = 4096 bytes
    expected = 2 * 2 * 16 * 2 * 8 * 4
    assert cfg.bytes_per_slot == expected


# ── Allocate / Free ───────────────────────────────────────────────────────────

def test_allocate_marks_slot_used():
    manager = KVCacheManager(_small_cfg())
    slot = manager.allocate("req-1")
    assert slot.request_id == "req-1"
    assert not slot.is_free
    assert manager.active_count() == 1
    manager.shutdown()


def test_free_returns_to_pool():
    manager = KVCacheManager(_small_cfg())
    manager.allocate("req-1")
    manager.free("req-1")
    assert manager.active_count() == 0
    # Slot is back in pool
    slot2 = manager.allocate("req-2")
    assert slot2.request_id == "req-2"
    manager.shutdown()


def test_allocate_duplicate_returns_same_slot():
    manager = KVCacheManager(_small_cfg())
    slot1 = manager.allocate("req-1")
    slot2 = manager.allocate("req-1")
    assert slot1 is slot2
    manager.shutdown()


def test_pool_exhaustion_raises():
    manager = KVCacheManager(_small_cfg(pool_size=2))
    manager.allocate("req-1")
    manager.allocate("req-2")
    with pytest.raises(EdgeResourceError):
        manager.allocate("req-3")
    manager.shutdown()


def test_free_nonexistent_is_noop():
    manager = KVCacheManager(_small_cfg())
    manager.free("ghost")  # should not raise
    assert manager.active_count() == 0
    manager.shutdown()


def test_get_active_slot():
    manager = KVCacheManager(_small_cfg())
    slot = manager.allocate("req-1")
    retrieved = manager.get("req-1")
    assert retrieved is slot
    manager.free("req-1")
    assert manager.get("req-1") is None
    manager.shutdown()


# ── KVCacheSlot.write_kv ──────────────────────────────────────────────────────

def test_write_kv_single_layer():
    manager = KVCacheManager(_small_cfg())
    slot = manager.allocate("req-1")

    new_k = torch.ones(3, 2, 8)  # 3 new tokens, 2 heads, 8 dim
    new_v = torch.ones(3, 2, 8) * 2.0

    slot.write_kv(layer_idx=0, keys=new_k, values=new_v)
    assert slot.seq_len == 3

    retrieved_k = slot.active_keys(0)
    retrieved_v = slot.active_values(0)
    assert retrieved_k.shape == (3, 2, 8)
    assert torch.allclose(retrieved_k, new_k)
    assert torch.allclose(retrieved_v, new_v)
    manager.shutdown()


def test_write_kv_accumulates():
    manager = KVCacheManager(_small_cfg(max_context_tokens=16))
    slot = manager.allocate("req-1")

    for _ in range(4):
        slot.write_kv(0, torch.zeros(2, 2, 8), torch.zeros(2, 2, 8))

    assert slot.seq_len == 8
    manager.shutdown()


def test_write_kv_overflow_raises():
    manager = KVCacheManager(_small_cfg(max_context_tokens=8))
    slot = manager.allocate("req-1")

    slot.write_kv(0, torch.zeros(8, 2, 8), torch.zeros(8, 2, 8))
    with pytest.raises(OverflowError):
        slot.write_kv(0, torch.zeros(1, 2, 8), torch.zeros(1, 2, 8))
    manager.shutdown()


def test_slot_reset_clears_state():
    manager = KVCacheManager(_small_cfg())
    slot = manager.allocate("req-1")
    slot.write_kv(0, torch.zeros(3, 2, 8), torch.zeros(3, 2, 8))
    assert slot.seq_len == 3

    manager.free("req-1")
    # Re-allocate same slot — seq_len must be 0
    slot2 = manager.allocate("req-2")
    assert slot2.seq_len == 0
    manager.shutdown()


# ── Eviction ──────────────────────────────────────────────────────────────────

def test_evict_preserves_image_tokens():
    """Image tokens (positions < image_token_end) must survive eviction."""
    manager = KVCacheManager(_small_cfg(max_context_tokens=12, num_image_tokens=4))
    slot = manager.allocate("req-1", num_image_tokens=4)

    # Write 4 image tokens then 6 text tokens
    img_keys = torch.ones(4, 2, 8) * 99.0   # distinctive value
    slot.write_kv(0, img_keys, img_keys)
    slot.write_kv(0, torch.zeros(6, 2, 8), torch.zeros(6, 2, 8))
    assert slot.seq_len == 10

    # Evict 3 text tokens
    manager.evict_oldest_text_tokens(slot, num_to_evict=3)
    assert slot.seq_len == 7  # 4 image + 3 remaining text

    # Image tokens should still be at positions 0-3
    remaining_k = slot.active_keys(0)
    assert torch.allclose(remaining_k[:4], img_keys)
    manager.shutdown()


def test_evict_all_text_tokens():
    """Evicting all text tokens should leave only image tokens."""
    manager = KVCacheManager(_small_cfg(max_context_tokens=10, num_image_tokens=3))
    slot = manager.allocate("req-1", num_image_tokens=3)

    slot.write_kv(0, torch.zeros(3, 2, 8), torch.zeros(3, 2, 8))  # image
    slot.write_kv(0, torch.zeros(5, 2, 8), torch.zeros(5, 2, 8))  # text
    assert slot.seq_len == 8

    manager.evict_oldest_text_tokens(slot, num_to_evict=10)  # more than available
    assert slot.seq_len == 3  # only image tokens remain
    manager.shutdown()


def test_eviction_policy_none_raises():
    manager = KVCacheManager(_small_cfg(eviction_policy="none"))
    slot = manager.allocate("req-1")
    # Write to near-full
    slot.write_kv(0, torch.zeros(4, 2, 8), torch.zeros(4, 2, 8))

    with pytest.raises(OverflowError):
        manager.evict_oldest_text_tokens(slot, num_to_evict=2)
    manager.shutdown()


# ── Stats ─────────────────────────────────────────────────────────────────────

def test_stats_correct_counts():
    manager = KVCacheManager(_small_cfg(pool_size=4))
    manager.allocate("req-1")
    manager.allocate("req-2")
    stats = manager.stats()

    assert stats["pool_size"] == 4
    assert stats["active_slots"] == 2
    assert stats["free_slots"] == 2
    assert stats["total_allocs"] == 2
    manager.shutdown()


def test_eviction_count_tracked():
    manager = KVCacheManager(_small_cfg(max_context_tokens=10, num_image_tokens=2))
    slot = manager.allocate("req-1", num_image_tokens=2)
    slot.write_kv(0, torch.zeros(2, 2, 8), torch.zeros(2, 2, 8))
    slot.write_kv(0, torch.zeros(4, 2, 8), torch.zeros(4, 2, 8))

    manager.evict_oldest_text_tokens(slot, num_to_evict=2)
    assert manager.stats()["total_evictions"] == 2
    manager.shutdown()


# ── Shutdown ──────────────────────────────────────────────────────────────────

def test_shutdown_clears_pool():
    manager = KVCacheManager(_small_cfg())
    manager.allocate("req-1")
    manager.shutdown()
    assert len(manager._slots) == 0
    assert len(manager._active) == 0


def test_memory_usage_gb():
    cfg = _small_cfg(pool_size=2)
    manager = KVCacheManager(cfg)
    assert manager.memory_usage_gb() == pytest.approx(cfg.pool_memory_gb, rel=1e-5)
    manager.shutdown()
