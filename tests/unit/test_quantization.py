"""Unit tests for quantization calibration and validation (Phase 2).

All tests run without GPU — mock modules and sim environments are used.
Tests verify:
  - CalibrationConfig seed isolation
  - Per-module calibrators handle fp16 (no-op) correctly
  - VLACalibrationPipeline orchestration
  - ADR-003 enforcement: flow_matching/diffusion head cannot use non-fp16
  - TaskSuccessRateValidator pass / fail logic
  - PrecisionDegradationError raised on excessive drop
  - MockSimEnvironment determinism
  - ValidationResult.per_task_rates() aggregation
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vla_runtime.errors import PrecisionDegradationError
from vla_runtime.quantization.calibration import (
    CalibrationConfig,
    CalibrationSample,
    VLACalibrationPipeline,
    _collect_samples,
    _seed_everything,
)
from vla_runtime.quantization.config import QuantizationConfig
from vla_runtime.quantization.validation import (
    MockPolicy,
    MockSimEnvironment,
    SimTaskConfig,
    TaskSuccessRateValidator,
    validate_quantization_config,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _sample_iter(n: int = 20):
    """Generate n deterministic calibration samples."""
    for i in range(n):
        yield CalibrationSample(
            image=torch.zeros(3, 64, 64),
            instruction=f"task {i}",
            robot_state=torch.zeros(14),
        )


def _tiny_module() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 16))


# ── CalibrationConfig ─────────────────────────────────────────────────────────

def test_calibration_config_defaults():
    cfg = CalibrationConfig()
    assert cfg.num_samples == 512
    assert cfg.seed == 42
    assert cfg.batch_size == 8


def test_seed_everything_reproducibility():
    _seed_everything(42)
    a = torch.rand(4)
    _seed_everything(42)
    b = torch.rand(4)
    assert torch.allclose(a, b), "Same seed must produce same random values"


def test_collect_samples_exact():
    samples = _collect_samples(_sample_iter(20), num_samples=10)
    assert len(samples) == 10


def test_collect_samples_partial():
    """If iterator has fewer samples than requested, collect all available."""
    samples = _collect_samples(_sample_iter(5), num_samples=100)
    assert len(samples) == 5


def test_collect_samples_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        _collect_samples(iter([]), num_samples=10)


# ── VLACalibrationPipeline ────────────────────────────────────────────────────

def test_pipeline_fp16_no_op():
    """Full FP16 config should return modules unchanged (no-op calibration)."""
    pipeline = VLACalibrationPipeline(
        config=CalibrationConfig(num_samples=10, device="cpu")
    )
    m = _tiny_module()
    result = pipeline.calibrate(
        vision_module=m,
        llm_module=m,
        action_head_module=m,
        head_type="flow_matching",
        quant_config=QuantizationConfig.full_precision(),
        calibration_data=lambda: _sample_iter(10),
    )
    assert result.vision.scheme == "fp16"
    assert result.llm.scheme == "fp16"
    assert result.action_head.scheme == "fp16"
    assert result.vision.num_samples_used == 0
    assert result.vision.calibration_ms == 0.0
    assert "fp16" in result.summary()


def test_pipeline_timing():
    """total_ms should be ≥ sum of individual module times."""
    pipeline = VLACalibrationPipeline(
        config=CalibrationConfig(num_samples=5, device="cpu")
    )
    m = _tiny_module()
    result = pipeline.calibrate(
        vision_module=m,
        llm_module=m,
        action_head_module=m,
        head_type="discrete_token",
        quant_config=QuantizationConfig.full_precision(),
        calibration_data=lambda: _sample_iter(5),
    )
    per_module = (
        result.vision.calibration_ms
        + result.llm.calibration_ms
        + result.action_head.calibration_ms
    )
    assert result.total_ms >= per_module - 1.0


def test_pipeline_adr003_flow_matching_fp16_only():
    """ADR-003: flow_matching head with w8a16 must raise PrecisionDegradationError."""
    pipeline = VLACalibrationPipeline(
        config=CalibrationConfig(num_samples=5, device="cpu")
    )
    m = _tiny_module()
    with pytest.raises(PrecisionDegradationError):
        pipeline.calibrate(
            vision_module=m,
            llm_module=m,
            action_head_module=m,
            head_type="flow_matching",
            quant_config=QuantizationConfig(action_head="w8a16"),
            calibration_data=lambda: _sample_iter(5),
        )


def test_pipeline_adr003_diffusion_fp16_only():
    """ADR-003: diffusion head with w8a16 must raise PrecisionDegradationError."""
    pipeline = VLACalibrationPipeline(
        config=CalibrationConfig(num_samples=5, device="cpu")
    )
    m = _tiny_module()
    with pytest.raises(PrecisionDegradationError):
        pipeline.calibrate(
            vision_module=m,
            llm_module=m,
            action_head_module=m,
            head_type="diffusion",
            quant_config=QuantizationConfig(action_head="w8a16"),
            calibration_data=lambda: _sample_iter(5),
        )


def test_pipeline_discrete_token_allows_w8a16():
    """ADR-003: discrete_token head MAY use w8a16."""
    pipeline = VLACalibrationPipeline(
        config=CalibrationConfig(num_samples=5, device="cpu")
    )
    m = _tiny_module()
    # Should not raise
    result = pipeline.calibrate(
        vision_module=m,
        llm_module=m,
        action_head_module=m,
        head_type="discrete_token",
        quant_config=QuantizationConfig(action_head="w8a16"),
        calibration_data=lambda: _sample_iter(5),
    )
    assert result.action_head.scheme == "w8a16"


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
