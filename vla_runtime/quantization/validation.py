"""Sim task success rate validator for quantized VLA models.

Design constraints (CLAUDE.md ADR-008):
- Quantization validation MUST use sim task success rate — not loss / perplexity.
- PrecisionDegradationError is raised when quantized rate drops more than
  threshold_drop below the FP16 baseline (default 2 pp).
- Calibration set and sim task suite are fixed/reproducible (same seed).
- Supports two evaluation modes:
    smoke   — few tasks, fast (~5 min), run per-PR
    full    — complete suite (~60 min), run nightly

Sim backends supported:
    LIBERO  — 10 task suites with varying difficulty
    RoboCasa — broader household task distribution

Usage::

    validator = TaskSuccessRateValidator(
        config=SimTaskConfig(suite="libero", mode="smoke", seed=42),
    )
    baseline = validator.run_baseline(fp16_model, env)
    result = validator.validate(quantized_model, env, baseline_rate=baseline.success_rate)
    # raises PrecisionDegradationError if drop > 2 pp
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch

from vla_runtime.errors import ErrorContext, PrecisionDegradationError
from vla_runtime.observability.tracer import get_tracer

_tracer = get_tracer("quantization.validation")
_log = logging.getLogger(__name__)


# ── Sim task configuration ────────────────────────────────────────────────────

SimSuite = Literal["libero", "robocasa", "mock"]
ValidationMode = Literal["smoke", "full"]


@dataclass
class SimTaskConfig:
    """Configuration for sim task success rate evaluation.

    Attributes:
        suite: Which sim environment to use.
        mode: "smoke" runs a small subset; "full" runs the complete suite.
        num_tasks_smoke: Number of tasks in smoke mode.
        num_tasks_full: Number of tasks in full mode.
        num_episodes_per_task: Episodes per task (higher = more robust estimate).
        max_steps_per_episode: Timeout per episode in sim steps.
        seed: Fixed seed for task + episode reproducibility (ADR-008).
        threshold_drop_pp: Max allowed success rate drop vs FP16 baseline (pp).
        device: Inference device for policy evaluation.
        render: Whether to render frames (slow; for debugging only).
    """

    suite: SimSuite = "libero"
    mode: ValidationMode = "smoke"
    num_tasks_smoke: int = 10
    num_tasks_full: int = 50
    num_episodes_per_task: int = 5
    max_steps_per_episode: int = 400
    seed: int = 42
    threshold_drop_pp: float = 2.0
    device: str = "cuda"
    render: bool = False

    @property
    def num_tasks(self) -> int:
        return self.num_tasks_smoke if self.mode == "smoke" else self.num_tasks_full

    @property
    def total_episodes(self) -> int:
        return self.num_tasks * self.num_episodes_per_task


# ── Task episode result ───────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Result of a single sim episode.

    Attributes:
        task_name: Name of the task (e.g. "pick_apple_from_bowl").
        episode_idx: Episode index within the task.
        success: Whether the task was completed successfully.
        num_steps: Steps taken before success or timeout.
        total_reward: Accumulated reward (for diagnostic purposes).
    """

    task_name: str
    episode_idx: int
    success: bool
    num_steps: int
    total_reward: float = 0.0


@dataclass
class ValidationResult:
    """Result of a full validation run.

    Attributes:
        success_rate: Fraction of episodes that succeeded (0–1).
        baseline_rate: The FP16 baseline rate this was compared against.
        drop_pp: Absolute drop in success rate vs baseline, in percentage points.
        exceeds_threshold: True if drop_pp > config.threshold_drop_pp.
        episode_results: Per-episode breakdown.
        quant_config_summary: Human-readable quantization config string.
        elapsed_ms: Total wall-clock evaluation time.
        suite: Which sim suite was used.
        mode: "smoke" or "full".
        num_episodes: Total episodes evaluated.
    """

    success_rate: float
    baseline_rate: float
    drop_pp: float
    exceeds_threshold: bool
    episode_results: list[EpisodeResult] = field(default_factory=list)
    quant_config_summary: str = ""
    elapsed_ms: float = 0.0
    suite: SimSuite = "libero"
    mode: ValidationMode = "smoke"
    num_episodes: int = 0

    def per_task_rates(self) -> dict[str, float]:
        """Aggregate success rate per task name."""
        from collections import defaultdict
        totals: dict[str, list[bool]] = defaultdict(list)
        for ep in self.episode_results:
            totals[ep.task_name].append(ep.success)
        return {
            name: sum(results) / len(results)
            for name, results in totals.items()
        }

    def summary(self) -> str:
        return (
            f"{self.suite}/{self.mode} | "
            f"success={self.success_rate:.1%} | "
            f"baseline={self.baseline_rate:.1%} | "
            f"drop={self.drop_pp:+.2f}pp | "
            f"{'FAIL' if self.exceeds_threshold else 'PASS'}"
        )


# ── Sim environment interface ─────────────────────────────────────────────────

class SimEnvironment(abc.ABC):
    """Abstract sim environment interface.

    Concrete implementations wrap LIBERO, RoboCasa, or mock environments.
    """

    @abc.abstractmethod
    def reset(self, task_name: str, episode_idx: int, seed: int) -> dict[str, Any]:
        """Reset to a specific task + episode and return initial observation."""

    @abc.abstractmethod
    def step(self, action: torch.Tensor) -> tuple[dict[str, Any], float, bool, dict]:
        """Step the environment. Returns (obs, reward, done, info)."""

    @abc.abstractmethod
    def task_names(self) -> list[str]:
        """List all available task names in this suite."""

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up sim resources."""


class MockSimEnvironment(SimEnvironment):
    """Mock environment for unit testing — deterministic, GPU-free.

    Simulates a configurable success rate so that tests can verify the
    validator logic without a real LIBERO or RoboCasa installation.
    """

    def __init__(
        self,
        success_probability: float = 0.8,
        num_tasks: int = 5,
        num_steps_to_done: int = 10,
    ) -> None:
        self._success_prob = success_probability
        self._num_tasks = num_tasks
        self._steps_to_done = num_steps_to_done
        self._rng = None
        self._step_count = 0
        self._current_task = ""

    def reset(self, task_name: str, episode_idx: int, seed: int) -> dict[str, Any]:
        import random as rnd
        self._rng = rnd.Random(seed + episode_idx)
        self._step_count = 0
        self._current_task = task_name
        return {"image": torch.zeros(3, 84, 84), "instruction": task_name}

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, Any], float, bool, dict]:
        self._step_count += 1
        done = self._step_count >= self._steps_to_done
        success = done and (self._rng.random() < self._success_prob)  # type: ignore[union-attr]
        reward = 1.0 if success else 0.0
        obs = {"image": torch.zeros(3, 84, 84), "instruction": self._current_task}
        return obs, reward, done, {"success": success}

    def task_names(self) -> list[str]:
        return [f"task_{i}" for i in range(self._num_tasks)]

    def close(self) -> None:
        pass


# ── Policy interface ──────────────────────────────────────────────────────────

class PolicyCallable(abc.ABC):
    """Minimal policy interface for validator compatibility.

    In production, pass an AsyncScheduler or ActionHeadRuntime wrapped in
    a synchronous adapter. For testing, use MockPolicy.
    """

    @abc.abstractmethod
    def predict(self, obs: dict[str, Any]) -> torch.Tensor:
        """Map observation to action tensor."""


class MockPolicy(PolicyCallable):
    """Returns random actions — used to establish a mock baseline."""

    def __init__(self, action_dim: int = 7, chunk_size: int = 1) -> None:
        self._action_dim = action_dim
        self._chunk_size = chunk_size

    def predict(self, obs: dict[str, Any]) -> torch.Tensor:
        return torch.randn(self._action_dim)


# ── Validator ─────────────────────────────────────────────────────────────────

class TaskSuccessRateValidator:
    """Measures sim task success rate and compares against a FP16 baseline.

    Args:
        config: Sim task evaluation configuration.
    """

    def __init__(self, config: SimTaskConfig | None = None) -> None:
        self._config = config or SimTaskConfig()

    def run_baseline(
        self,
        policy: PolicyCallable,
        env: SimEnvironment,
        quant_config_summary: str = "fp16",
    ) -> ValidationResult:
        """Establish the FP16 baseline success rate.

        Args:
            policy: FP16 policy to evaluate.
            env: Sim environment.
            quant_config_summary: Human-readable label for the result.

        Returns:
            ValidationResult with baseline_rate == success_rate (no comparison).
        """
        return self._run_evaluation(
            policy=policy,
            env=env,
            baseline_rate=None,
            quant_config_summary=quant_config_summary,
        )

    def validate(
        self,
        policy: PolicyCallable,
        env: SimEnvironment,
        baseline_rate: float,
        quant_config_summary: str = "",
    ) -> ValidationResult:
        """Evaluate a quantized policy and compare against the FP16 baseline.

        Args:
            policy: Quantized policy to evaluate.
            env: Sim environment (same suite + tasks as baseline).
            baseline_rate: Success rate from run_baseline().
            quant_config_summary: Human-readable quantization config string.

        Returns:
            ValidationResult with pass/fail determination.

        Raises:
            PrecisionDegradationError: If the drop in success rate exceeds
                config.threshold_drop_pp.
        """
        result = self._run_evaluation(
            policy=policy,
            env=env,
            baseline_rate=baseline_rate,
            quant_config_summary=quant_config_summary,
        )

        if result.exceeds_threshold:
            ctx = ErrorContext(
                trace_id=_tracer.trace_id,
                module_name="quantization.validation",
                request_id="validation",
            )
            raise PrecisionDegradationError(
                baseline_rate=baseline_rate,
                quantized_rate=result.success_rate,
                threshold_drop=self._config.threshold_drop_pp / 100.0,
                context=ctx,
            )

        return result

    def _run_evaluation(
        self,
        policy: PolicyCallable,
        env: SimEnvironment,
        baseline_rate: float | None,
        quant_config_summary: str,
    ) -> ValidationResult:
        import random as rnd

        t0 = time.perf_counter()
        rng = rnd.Random(self._config.seed)

        task_names = env.task_names()[: self._config.num_tasks]
        episode_results: list[EpisodeResult] = []
        num_success = 0

        with _tracer.span(
            "sim_evaluation",
            suite=self._config.suite,
            mode=self._config.mode,
            num_tasks=len(task_names),
        ):
            for task_name in task_names:
                for ep_idx in range(self._config.num_episodes_per_task):
                    ep_seed = rng.randint(0, 2**31)
                    ep_result = self._run_episode(policy, env, task_name, ep_idx, ep_seed)
                    episode_results.append(ep_result)
                    if ep_result.success:
                        num_success += 1

                    _tracer.event(
                        "episode_done",
                        task=task_name,
                        episode=ep_idx,
                        success=ep_result.success,
                        steps=ep_result.num_steps,
                    )

        num_episodes = len(episode_results)
        success_rate = num_success / max(num_episodes, 1)
        effective_baseline = baseline_rate if baseline_rate is not None else success_rate
        drop_pp = (effective_baseline - success_rate) * 100.0
        exceeds = drop_pp > self._config.threshold_drop_pp

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ValidationResult(
            success_rate=success_rate,
            baseline_rate=effective_baseline,
            drop_pp=drop_pp,
            exceeds_threshold=exceeds,
            episode_results=episode_results,
            quant_config_summary=quant_config_summary,
            elapsed_ms=elapsed_ms,
            suite=self._config.suite,
            mode=self._config.mode,
            num_episodes=num_episodes,
        )
        _tracer.event("evaluation_complete", summary=result.summary())
        return result

    def _run_episode(
        self,
        policy: PolicyCallable,
        env: SimEnvironment,
        task_name: str,
        episode_idx: int,
        seed: int,
    ) -> EpisodeResult:
        obs = env.reset(task_name, episode_idx, seed)
        total_reward = 0.0
        success = False

        for step_idx in range(self._config.max_steps_per_episode):
            action = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                success = info.get("success", False)
                return EpisodeResult(
                    task_name=task_name,
                    episode_idx=episode_idx,
                    success=success,
                    num_steps=step_idx + 1,
                    total_reward=total_reward,
                )

        # Timeout
        return EpisodeResult(
            task_name=task_name,
            episode_idx=episode_idx,
            success=False,
            num_steps=self._config.max_steps_per_episode,
            total_reward=total_reward,
        )


# ── Convenience wrapper ───────────────────────────────────────────────────────

def validate_quantization_config(
    fp16_policy: PolicyCallable,
    quantized_policy: PolicyCallable,
    env: SimEnvironment,
    config: SimTaskConfig | None = None,
    quant_config_summary: str = "",
) -> ValidationResult:
    """One-shot helper: run baseline then validate quantized model.

    Args:
        fp16_policy: FP16 reference policy.
        quantized_policy: Quantized policy under test.
        env: Sim environment.
        config: Evaluation configuration.
        quant_config_summary: Human-readable quantization config string.

    Returns:
        ValidationResult (pass/fail included).

    Raises:
        PrecisionDegradationError: If drop exceeds threshold.
    """
    validator = TaskSuccessRateValidator(config=config)
    baseline = validator.run_baseline(fp16_policy, env, quant_config_summary="fp16_baseline")
    return validator.validate(
        quantized_policy, env,
        baseline_rate=baseline.success_rate,
        quant_config_summary=quant_config_summary,
    )
