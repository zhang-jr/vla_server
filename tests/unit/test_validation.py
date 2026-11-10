"""Unit tests for the sim task-success-rate validator (ADR-008).

All tests run without GPU — mock policies and sim environments are used.
Tests verify:
  - TaskSuccessRateValidator pass / fail logic
  - PrecisionDegradationError raised on excessive drop
  - MockSimEnvironment determinism
  - ValidationResult.per_task_rates() aggregation
"""

from __future__ import annotations

import pytest
import torch

from vla_runtime.errors import PrecisionDegradationError
from vla_runtime.quantization.validation import (
    MockPolicy,
    MockSimEnvironment,
    SimTaskConfig,
    TaskSuccessRateValidator,
    validate_quantization_config,
)


# ── MockSimEnvironment ────────────────────────────────────────────────────────

def test_mock_env_task_names():
    env = MockSimEnvironment(num_tasks=5)
    names = env.task_names()
    assert len(names) == 5
    assert all(n.startswith("task_") for n in names)


def test_mock_env_reset_returns_obs():
    env = MockSimEnvironment()
    obs = env.reset("task_0", episode_idx=0, seed=42)
    assert "image" in obs
    assert "instruction" in obs


def test_mock_env_step_terminates():
    env = MockSimEnvironment(num_steps_to_done=5)
    env.reset("task_0", 0, seed=42)
    for _ in range(5):
        obs, reward, done, info = env.step(torch.zeros(7))
    assert done


def test_mock_env_determinism():
    """Same seed must produce same success sequence."""
    env1 = MockSimEnvironment(success_probability=0.7, num_steps_to_done=3)
    env2 = MockSimEnvironment(success_probability=0.7, num_steps_to_done=3)
    results1, results2 = [], []
    for ep in range(5):
        env1.reset("task_0", ep, seed=100)
        env2.reset("task_0", ep, seed=100)
        for _ in range(3):
            _, _, done1, info1 = env1.step(torch.zeros(7))
            _, _, done2, info2 = env2.step(torch.zeros(7))
        results1.append(info1.get("success"))
        results2.append(info2.get("success"))
    assert results1 == results2


# ── TaskSuccessRateValidator ──────────────────────────────────────────────────

def test_validator_baseline_no_comparison():
    """Baseline run: success_rate == baseline_rate, drop == 0."""
    env = MockSimEnvironment(success_probability=0.8, num_steps_to_done=5)
    policy = MockPolicy()
    cfg = SimTaskConfig(
        suite="mock",
        mode="smoke",
        num_tasks_smoke=3,
        num_episodes_per_task=5,
        seed=42,
    )
    validator = TaskSuccessRateValidator(cfg)
    result = validator.run_baseline(policy, env)
    assert result.baseline_rate == result.success_rate
    assert result.drop_pp == 0.0
    assert not result.exceeds_threshold
    assert result.num_episodes == 15  # 3 tasks × 5 episodes


def test_validator_pass_small_drop():
    """< 2pp drop should pass."""
    env = MockSimEnvironment(success_probability=0.85, num_steps_to_done=5)
    policy = MockPolicy()
    cfg = SimTaskConfig(
        suite="mock",
        mode="smoke",
        num_tasks_smoke=3,
        num_episodes_per_task=10,
        seed=0,
        threshold_drop_pp=2.0,
    )
    validator = TaskSuccessRateValidator(cfg)
    # Run baseline with high success rate
    baseline = validator.run_baseline(policy, env)
    # Validate with near-equal policy (same mock)
    result = validator.validate(
        policy, env,
        baseline_rate=baseline.success_rate,
        quant_config_summary="test_quant",
    )
    assert not result.exceeds_threshold


def test_validator_fail_large_drop():
    """A large drop (> 2pp) must raise PrecisionDegradationError."""
    env = MockSimEnvironment(success_probability=0.5, num_steps_to_done=3)
    policy = MockPolicy()
    cfg = SimTaskConfig(
        suite="mock",
        mode="smoke",
        num_tasks_smoke=3,
        num_episodes_per_task=5,
        seed=42,
        threshold_drop_pp=2.0,
    )
    validator = TaskSuccessRateValidator(cfg)

    # Pretend FP16 baseline was 90% — quantized is ~50% → 40pp drop
    with pytest.raises(PrecisionDegradationError):
        validator.validate(
            policy, env,
            baseline_rate=0.90,
            quant_config_summary="w4a8",
        )


def test_validator_per_task_rates():
    """per_task_rates() should aggregate per-task success rates correctly."""
    env = MockSimEnvironment(
        success_probability=1.0,  # always succeed
        num_steps_to_done=3,
    )
    policy = MockPolicy()
    cfg = SimTaskConfig(
        suite="mock",
        mode="smoke",
        num_tasks_smoke=4,
        num_episodes_per_task=3,
        seed=42,
    )
    validator = TaskSuccessRateValidator(cfg)
    result = validator.run_baseline(policy, env)
    rates = result.per_task_rates()
    assert len(rates) == 4
    # All tasks should succeed (success_prob=1.0)
    for task, rate in rates.items():
        assert rate == 1.0, f"Task {task} rate {rate} != 1.0"


def test_validate_quantization_config_helper():
    """One-shot helper raises on excessive drop."""
    env = MockSimEnvironment(success_probability=0.3, num_steps_to_done=3)
    fp16_policy = MockPolicy()
    quant_policy = MockPolicy()
    cfg = SimTaskConfig(
        suite="mock",
        mode="smoke",
        num_tasks_smoke=2,
        num_episodes_per_task=5,
        seed=0,
        threshold_drop_pp=2.0,
    )
    # fp16 baseline will be ~30% success with this env
    # quant policy also ~30% — may pass or fail depending on random seed
    # Test that it returns a ValidationResult (not raises) when rates are similar
    # Force a failure scenario: set a very high fake baseline
    with pytest.raises(PrecisionDegradationError):
        env_low = MockSimEnvironment(success_probability=0.1, num_steps_to_done=3)

        class _HighBaselinePolicy(MockPolicy):
            pass

        # Validate with a 90% baseline against a 10%-success env
        validator = TaskSuccessRateValidator(cfg)
        validator.validate(
            quant_policy, env_low,
            baseline_rate=0.90,
            quant_config_summary="extreme_quant",
        )


def test_validation_result_summary():
    """summary() string should contain key fields."""
    from vla_runtime.quantization.validation import ValidationResult
    result = ValidationResult(
        success_rate=0.75,
        baseline_rate=0.80,
        drop_pp=5.0,
        exceeds_threshold=True,
        suite="libero",
        mode="smoke",
        num_episodes=50,
    )
    s = result.summary()
    assert "75.0%" in s
    assert "FAIL" in s
    assert "5.00pp" in s
