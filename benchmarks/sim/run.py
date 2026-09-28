"""Closed-loop sim benchmark CLI — task success + runtime metrics per execution mode.

Examples (GPU runs go through the remote host, ADR-020)::

    # CPU plumbing check, any machine
    python -m benchmarks.sim.run --env mock --policy mock --modes all --episodes 2

    # LIBERO-Object, SmolVLA: native baseline vs runtime, blend-policy ablation
    python scripts/remote.py run -- python -m benchmarks.sim.run \\
        --env libero --env-opt suite=libero_object \\
        --policy smolvla --policy-opt repo=HuggingFaceVLA/smolvla_libero \\
        --modes native,sync,async_sim --blend-policies replace,trim,linear_overlap,ensemble \\
        --episodes 10

    # Condition-staleness calibration: injected latency sweep (TODO 0.3)
    ... --modes async_sim --inject-latency-ms 0,150,500,1000,2000

    # Fleet scaling on ManiSkill: N robots share one backbone
    ... --env maniskill --env-opt env_ids=PickCube-v1 \\
        --policy smolvla --policy-opt repo=<ckpt> \\
        --policy-opt cameras=base_camera=observation.images.image \\
        --modes realtime --robots 1,2,4

Outputs, under ``--out/<timestamp>_<env>_<policy>/``:
    summary.json   one summary per run configuration
    summary.md     the same as a markdown table (paste into BENCHMARKS.md)
    <label>.json   per-episode records
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from benchmarks.sim.envs import env_factory
from benchmarks.sim.metrics import EpisodeRecord, markdown_table, records_to_json, summarize
from benchmarks.sim.policies import make_policy
from benchmarks.sim.runners import MODES, EnvWorker, RunConfig, run_benchmark
from vla_runtime import configure_logging

ROOT = Path(__file__).resolve().parents[2]
BLEND_POLICIES = ("replace", "trim", "linear_overlap", "ensemble")


def _say(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _options(pairs: list[str]) -> dict[str, str]:
    """Repeated ``key=value`` flags (a value may itself contain '=' and ',')."""
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {pair!r}")
        out[key.strip()] = value.strip()
    return out


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "host": platform.node(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    for package in ("lerobot", "mani_skill", "libero", "robosuite", "mujoco"):
        try:
            info[package] = importlib.metadata.version(package.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info


def build_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Expand the mode × sweep flags into one RunConfig per run."""
    modes = list(MODES) if args.modes == "all" else _csv(args.modes)
    for mode in modes:
        if mode not in MODES:
            raise SystemExit(f"unknown mode {mode!r}; expected {', '.join(MODES)} or all")
    blends = _csv(args.blend_policies)
    for blend in blends:
        if blend not in BLEND_POLICIES:
            raise SystemExit(f"unknown blend policy {blend!r}; expected {', '.join(BLEND_POLICIES)}")
    latencies: list[float | None] = (
        [float(v) for v in _csv(args.inject_latency_ms)] if args.inject_latency_ms else [None]
    )
    robots = [int(v) for v in _csv(args.robots)]

    base = RunConfig(
        mode="sync",
        episodes_per_task=args.episodes,
        num_tasks=args.num_tasks,
        seed=args.seed,
        chunk_threshold=args.chunk_threshold,
        max_trim_steps=args.max_trim_steps,
        latency_scale=args.latency_scale,
        vlm_budget_ms=args.vlm_budget_ms,
        head_budget_ms=args.head_budget_ms,
        max_steps=args.max_steps,
        warmup=args.warmup,
    )
    configs: list[RunConfig] = []
    for mode in modes:
        if mode in ("native", "sync"):
            configs.append(replace(base, mode=mode))
        elif mode == "async_sim":
            configs += [
                replace(base, mode=mode, blend_policy=b, latency_ms=lat)
                for b in blends
                for lat in latencies
            ]
        else:
            configs += [
                replace(base, mode=mode, blend_policy=b, num_robots=n)
                for b in blends
                for n in robots
            ]
    return configs


async def _main(args: argparse.Namespace) -> int:
    configs = build_configs(args)
    factory = env_factory(args.env, _options(args.env_opt))
    worker = await EnvWorker.create(factory)
    spec = worker.env.spec
    bench = make_policy(args.policy, spec, _options(args.policy_opt))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / f"{stamp}_{spec.name.replace(':', '-').replace(',', '+')}_{args.policy}"
    if args.tag:
        out_dir = out_dir.with_name(out_dir.name + f"_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "env": args.env,
        "env_spec": spec.__dict__,
        "policy": bench.name,
        "policy_info": bench.info,
        "capabilities": bench.policy.capabilities.model_dump(),
        "frequency_mode": bench.policy.capabilities.effective_frequency_mode,
        "argv": sys.argv[1:],
        "environment": _environment_info(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    _say(f"[sim-bench] {spec.name} | {bench.name} | {len(configs)} run(s) -> {out_dir}")

    summaries: list[dict[str, Any]] = []
    try:
        for config in configs:
            if config.mode == "native" and bench.native is None:
                _say(f"[sim-bench] skip {config.label}: {bench.name} has no native path")
                continue

            def progress(record: EpisodeRecord, label: str = config.label) -> None:
                if args.quiet:
                    return
                lat = record.inference_ms
                mean = sum(lat) / len(lat) if lat else 0.0
                _say(
                    f"  [{label}] {record.robot_id} {record.task} ep{record.episode}: "
                    f"{'OK  ' if record.success else 'FAIL'} steps={record.steps} "
                    f"infer={len(lat)}x{mean:.0f}ms underrun={record.underrun_steps}"
                )

            _say(f"[sim-bench] run {config.label}")
            records, wall_s = await run_benchmark(
                bench, factory, config, worker=worker, on_episode=progress
            )
            summary = summarize(records, wall_s, extra={"run": config.label, **config.__dict__})
            summaries.append(summary)
            label_file = config.label.replace("/", "_") + ".json"
            (out_dir / label_file).write_text(
                json.dumps({"summary": summary, "episodes": records_to_json(records)}, indent=2),
                encoding="utf-8",
            )
            _say(
                f"[sim-bench] {config.label}: success={summary['success_rate']:.1%} "
                f"{summary['success_ci95']} p50={summary['latency_p50_ms']}ms "
                f"p95={summary['latency_p95_ms']}ms underrun={summary['underrun_rate']}"
            )
    finally:
        await worker.close()
        bench.close()
        (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        table = markdown_table(summaries)
        (out_dir / "summary.md").write_text(
            f"<!-- {spec.name} | {bench.name} | {meta['environment']} -->\n\n{table}\n",
            encoding="utf-8",
        )
    _say("")
    _say(table)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.sim.run",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", required=True, choices=("mock", "libero", "maniskill"))
    parser.add_argument("--env-opt", action="append", default=[], metavar="KEY=VALUE",
                        help="libero: suite, resolution, num_steps_wait, flip_images, task_ids; "
                             "maniskill: env_ids, control_mode, sim_backend, robot_uids, "
                             "resolution, max_episode_steps, gripper_dofs, state_mode")
    parser.add_argument("--policy", required=True, choices=("mock", "smolvla"))
    parser.add_argument("--policy-opt", action="append", default=[], metavar="KEY=VALUE",
                        help="smolvla: repo, device, cameras, n_action_steps, vlm_budget_ms; "
                             "mock: latency_ms, jitter_ms, chunk_size, n_action_steps, n_steps, "
                             "staleness_ms, embedding_dim")
    parser.add_argument("--modes", default="native,sync,async_sim",
                        help=f"comma list of {', '.join(MODES)}, or 'all'")
    parser.add_argument("--blend-policies", default="linear_overlap",
                        help=f"async modes sweep: {', '.join(BLEND_POLICIES)}")
    parser.add_argument("--inject-latency-ms", default="",
                        help="async_sim sweep of injected latencies, e.g. 0,150,500,1000,2000")
    parser.add_argument("--latency-scale", type=float, default=1.0,
                        help="async_sim: multiply measured latency (emulate slower hardware)")
    parser.add_argument("--robots", default="1", help="realtime sweep of fleet sizes, e.g. 1,2,4")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per task")
    parser.add_argument("--num-tasks", type=int, default=None, help="first N tasks (default all)")
    parser.add_argument("--max-steps", type=int, default=None, help="override task step limits")
    parser.add_argument("--chunk-threshold", type=float, default=0.5,
                        help="async: request next chunk when queued <= this fraction of a chunk")
    parser.add_argument("--max-trim-steps", type=int, default=None,
                        help="ChunkBuffer latency-trim ceiling (default: runtime default)")
    parser.add_argument("--vlm-budget-ms", type=float, default=2000.0)
    parser.add_argument("--head-budget-ms", type=float, default=1000.0)
    parser.add_argument("--warmup", type=int, default=3, help="untimed inferences before each run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "sim"))
    parser.add_argument("--tag", default="", help="suffix for the output directory name")
    parser.add_argument("--quiet", action="store_true", help="no per-episode lines")
    parser.add_argument("--verbose", action="store_true", help="runtime trace logs at INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level=logging.INFO if args.verbose else logging.WARNING)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
