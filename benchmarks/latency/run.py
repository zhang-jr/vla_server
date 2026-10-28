"""Model latency / throughput benchmark CLI — checks a latency + FPS requirement.

Each model runs in its own process with the interpreter of its env group (the
model stacks pin incompatible ``transformers``), resolved as
``$VLA_BENCH_PYTHON_<GROUP>`` → ``/opt/venvs/<group>/bin/python`` → this
interpreter. ``docker/Dockerfile`` builds those environments.

Examples (GPU runs go through the remote host, ADR-020)::

    # CPU plumbing check, any machine
    python -m benchmarks.latency.run --models mock --duration 2 --iterations 20

    # The requirement check: P95 ≤ 500 ms and ≥ 3 inferences/s, 480×640 frames
    python scripts/remote.py run -- python -m benchmarks.latency.run \\
        --models openvla,openvla_oft,pi0,pi05,smolvla

    # π0.5 in bf16 with fewer denoising steps; openpi's JAX reference
    ... --models pi05,openpi --opt pi05.cfg.dtype=bfloat16 --opt openpi.config=pi05_droid

Outputs, under ``--out/<timestamp>/``:
    meta.json      arguments and requirement
    <model>.json   environment, model description, samples and statistics
    <model>.log    the model process's console output
    summary.md     one row per model with the verdict (paste into BENCHMARKS.md)
    summary.json   the same, machine-readable
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.latency.harness import (
    GpuMemoryMonitor,
    MeasureConfig,
    Requirement,
    markdown_table,
    measure,
    percentile,
)
from benchmarks.latency.models import ENV_GROUPS, make_model

ROOT = Path(__file__).resolve().parents[2]
VENV_ROOT = Path("/opt/venvs")
PACKAGES = (
    "torch", "transformers", "tokenizers", "timm", "accelerate", "flash-attn",
    "lerobot", "openvla-oft", "openpi", "jax", "jaxlib", "numpy",
)


def _write_console(text: str) -> None:
    # The summary table is Chinese / emoji; a GBK or ASCII console must not crash the run.
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, "replace").decode(encoding))


def _say(message: str) -> None:
    _write_console(message + "\n")
    sys.stdout.flush()


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_model_options(pairs: list[str], models: list[str]) -> dict[str, dict[str, str]]:
    """``["pi05.cfg.dtype=bfloat16"]`` → ``{"pi05": {"cfg.dtype": "bfloat16"}}``."""
    options: dict[str, dict[str, str]] = {m: {} for m in models}
    for pair in pairs:
        target, sep, value = pair.partition("=")
        model, dot, key = target.partition(".")
        if not sep or not dot or not key:
            raise SystemExit(f"--opt {pair!r} is not <model>.<key>=<value>")
        if model not in options:
            raise SystemExit(f"--opt {pair!r} targets {model!r}, which is not in --models")
        options[model][key] = value
    return options


def python_for(group: str) -> str:
    """Interpreter for an env group: env override → /opt/venvs/<group> → this one."""
    override = os.environ.get(f"VLA_BENCH_PYTHON_{group.upper()}")
    if override:
        return override
    candidate = VENV_ROOT / group / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def environment_info(monitor: GpuMemoryMonitor) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **monitor.device_info(),
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["torch_device"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    packages: dict[str, str] = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    info["packages"] = packages
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info


# ── Worker: one model, in its own env ─────────────────────────────────────────


def run_worker(args: argparse.Namespace) -> int:
    """Load one model, measure it, write ``args.result``; never raises."""
    requirement = Requirement(args.max_latency_ms, args.latency_percentile, args.min_fps)
    result: dict[str, Any] = {"model": args.model, "status": "error", "describe": {}}
    try:
        options = parse_model_options(args.opt, [args.model])[args.model]
        result["options"] = options
        random.seed(args.seed)
        np.random.seed(args.seed)
        monitor = GpuMemoryMonitor()
        model = make_model(args.model, options)
        _say(f"[{args.model}] loading ...")
        load_start = time.perf_counter()
        model.load()
        result["load_s"] = round(time.perf_counter() - load_start, 1)
        torch = sys.modules.get("torch")
        if torch is not None:
            torch.manual_seed(args.seed)
        obs = model.make_observation(np.random.default_rng(args.seed), tuple(args.image_hw))
        result["describe"] = model.describe()
        result["describe"]["image_hw"] = list(args.image_hw)
        result["describe"]["cameras_fed"] = sorted(obs.images)
        config = MeasureConfig(args.warmup, args.iterations, args.duration)
        _say(
            f"[{args.model}] warmup {config.warmup}, then >={config.iterations} calls "
            f"and >={config.duration_s:g}s ..."
        )
        measurement = measure(model, obs, config, sync=getattr(model, "sync", None), monitor=monitor)
        result["environment"] = environment_info(monitor)
        result["measurement"] = measurement
        latency = percentile(measurement["samples_ms"], requirement.percentile)
        result["requirement_latency_ms"] = None if latency is None else round(latency, 1)
        result["verdict"] = requirement.judge(latency, measurement["fps"])
        result["status"] = "ok"
        lat = measurement["latency_ms"]
        _say(
            f"[{args.model}] P50 {lat['p50']} ms, P95 {lat['p95']} ms, "
            f"{measurement['fps']} inferences/s over {measurement['count']} calls -> {result['verdict']}"
        )
    except Exception as exc:  # recorded in the result file; the orchestrator moves on
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        _say(result["traceback"])
    Path(args.result).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


# ── Orchestrator ──────────────────────────────────────────────────────────────


def _worker_command(args: argparse.Namespace, model: str, result: Path) -> list[str]:
    cmd = [
        python_for(ENV_GROUPS[model]), "-m", "benchmarks.latency.run", "--worker",
        "--models", model, "--result", str(result),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
        "--duration", str(args.duration), "--image-hw", *map(str, args.image_hw),
        "--seed", str(args.seed),
        "--max-latency-ms", str(args.max_latency_ms),
        "--latency-percentile", str(args.latency_percentile),
        "--min-fps", str(args.min_fps),
    ]
    for pair in args.opt:
        if pair.split(".", 1)[0] == model:
            cmd += ["--opt", pair]
    return cmd


def _run_logged(cmd: list[str], log_path: Path) -> int:
    """Run with output streamed to the console and tee'd into ``log_path``."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with open(log_path, "w", encoding="utf-8") as log, subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            _write_console(line)
            log.write(line)
        sys.stdout.flush()
        return proc.wait()


def run_all(args: argparse.Namespace) -> int:
    models = _csv(args.models)
    for model in models:
        if model not in ENV_GROUPS:
            raise SystemExit(f"unknown model {model!r}; expected one of {', '.join(ENV_GROUPS)}")
    parse_model_options(args.opt, models)  # fail fast on malformed --opt
    requirement = Requirement(args.max_latency_ms, args.latency_percentile, args.min_fps)

    run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"argv": sys.argv, "args": vars(args), "requirement": vars(requirement)}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    results: list[dict[str, Any]] = []
    for model in models:
        result_path = run_dir / f"{model}.json"
        cmd = _worker_command(args, model, result_path)
        _say(f"=== {model} ({ENV_GROUPS[model]}: {cmd[0]}) ===")
        code = _run_logged(cmd, run_dir / f"{model}.log")
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {"model": model, "status": "error", "describe": {},
                      "error": f"worker exited with {code} before writing a result"}
        result.pop("traceback", None)
        if "measurement" in result:
            result["measurement"].pop("samples_ms", None)
        results.append(result)

    table = markdown_table(results, requirement)
    (run_dir / "summary.md").write_text(table, encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _say("\n" + table)
    _say(f"results: {run_dir}")
    return 0 if all(r.get("status") == "ok" for r in results) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", default="mock", help=f"comma-separated: {', '.join(ENV_GROUPS)}")
    parser.add_argument("--opt", action="append", default=[], metavar="MODEL.KEY=VALUE",
                        help="per-model option (repeatable), see benchmarks/latency/models.py")
    parser.add_argument("--warmup", type=int, default=10, help="untimed calls first (JIT, autotune, caches)")
    parser.add_argument("--iterations", type=int, default=100, help="minimum timed calls")
    parser.add_argument("--duration", type=float, default=60.0, help="minimum timed seconds (sustained FPS)")
    parser.add_argument("--image-hw", type=int, nargs=2, default=(480, 640), metavar=("H", "W"),
                        help="raw camera resolution fed to the model (default 480 640)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-latency-ms", type=float, default=500.0)
    parser.add_argument("--latency-percentile", type=float, default=95.0,
                        help="which latency percentile must meet --max-latency-ms")
    parser.add_argument("--min-fps", type=float, default=3.0, help="sustained inferences per second")
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "latency"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.result or len(_csv(args.models)) != 1:
            parser.error("--worker needs --result and exactly one model")
        args.model = _csv(args.models)[0]
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args(argv)
    return run_worker(args) if args.worker else run_all(args)


if __name__ == "__main__":
    sys.exit(main())
