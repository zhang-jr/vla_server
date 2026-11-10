"""Closed-loop simulation benchmarks: LIBERO / ManiSkill task success + runtime metrics.

Entry point::

    python -m benchmarks.sim.run --help

Modules:
    envs      — LIBERO / ManiSkill / mock environments behind one observation format
    policies  — policy registry (mock on CPU, SmolVLA via LeRobot on GPU)
    runners   — the four execution modes compared (native / sync / async_sim / realtime)
    metrics   — per-episode records and run summaries
"""
