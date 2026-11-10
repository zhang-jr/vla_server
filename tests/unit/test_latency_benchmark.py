"""CPU tests for the model latency benchmark harness (benchmarks/latency).

Mock model only: these check the timing loop, statistics, verdict, option
parsing and the per-model process orchestration. Real models run on the
remote GPU host (``python -m benchmarks.latency.run --models ...``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.latency import run as latency_run
from benchmarks.latency.harness import (
    MeasureConfig,
    Requirement,
    StageTimer,
    latency_stats,
    markdown_table,
    measure,
    percentile,
)
from benchmarks.latency.models import Options, make_model, synthetic_image


def test_percentile_matches_numpy_and_handles_empty():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 95) == pytest.approx(np.percentile(values, 95))
    assert percentile([], 50) is None
    stats = latency_stats([10.0, 20.0, 30.0])
    assert stats["p50"] == 20.0 and stats["max"] == 30.0


@pytest.mark.parametrize(
    ("latency", "fps", "verdict"),
    [
        (400.0, 3.5, "pass"),
        (500.0, 3.0, "pass"),          # bounds are inclusive
        (520.0, 3.5, "fail_latency"),
        (400.0, 2.9, "fail_fps"),
        (600.0, 1.0, "fail_both"),
        (None, 3.5, "error"),
    ],
)
def test_requirement_verdict(latency, fps, verdict):
    assert Requirement(500.0, 95.0, 3.0).judge(latency, fps) == verdict


def test_stage_timer_syncs_before_closing_each_stage():
    calls: list[str] = []
    timer = StageTimer(sync=lambda: calls.append("sync"))
    with timer.stage("a"):
        pass
    with timer.stage("a"):
        pass
    timer.record("inner", 1.5)
    assert calls == ["sync", "sync"]
    assert set(timer.stages) == {"a", "inner"} and timer.stages["inner"] == 1.5


def test_measure_runs_until_both_iteration_and_duration_floors():
    model = make_model("mock", {"model_ms": "5", "preprocess_ms": "1", "chunk": "8"})
    obs = model.make_observation(np.random.default_rng(0), (48, 64))
    out = measure(model, obs, MeasureConfig(warmup=2, iterations=10, duration_s=0.2))
    assert out["count"] >= 10 and out["elapsed_s"] >= 0.2
    assert len(out["warmup_ms"]) == 2
    assert set(out["stages_ms"]) == {"preprocess", "model"}
    assert out["action_shape"] == [8, 7]
    assert out["latency_ms"]["p50"] >= 6.0
    assert out["fps"] == pytest.approx(out["count"] / out["elapsed_s"], rel=0.05)


def test_synthetic_image_is_deterministic_uint8():
    a = synthetic_image(np.random.default_rng(1), (48, 64))
    b = synthetic_image(np.random.default_rng(1), (48, 64))
    assert a.dtype == np.uint8 and a.shape == (48, 64, 3)
    assert np.array_equal(a, b)


def test_options_reject_unknown_and_decode_prefixed_json():
    with pytest.raises(ValueError, match="unknown option"):
        make_model("mock", {"model_ms": "5", "typo": "1"})
    opts = Options("pi05", {"cfg.num_inference_steps": "5", "cfg.dtype": "bfloat16"})
    assert opts.prefixed("cfg.") == {"num_inference_steps": 5, "dtype": "bfloat16"}
    opts.check_consumed()
    with pytest.raises(ValueError, match="unknown model"):
        make_model("gpt", {})


def test_parse_model_options_routes_by_model_prefix():
    parsed = latency_run.parse_model_options(
        ["pi05.cfg.dtype=bfloat16", "openvla.attn=eager"], ["pi05", "openvla"]
    )
    assert parsed == {"pi05": {"cfg.dtype": "bfloat16"}, "openvla": {"attn": "eager"}}
    with pytest.raises(SystemExit, match="not in --models"):
        latency_run.parse_model_options(["pi0.repo=x"], ["pi05"])
    with pytest.raises(SystemExit, match="<model>.<key>=<value>"):
        latency_run.parse_model_options(["pi05=x"], ["pi05"])


def test_python_for_prefers_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VLA_BENCH_PYTHON_OFT", "/custom/python")
    assert latency_run.python_for("oft") == "/custom/python"
    monkeypatch.delenv("VLA_BENCH_PYTHON_OFT")
    monkeypatch.setattr(latency_run, "VENV_ROOT", tmp_path)
    assert latency_run.python_for("oft") == latency_run.sys.executable


def test_markdown_table_has_a_row_per_result_including_failures():
    requirement = Requirement()
    model = make_model("mock", {"model_ms": "1"})
    obs = model.make_observation(np.random.default_rng(0), (8, 8))
    m = measure(model, obs, MeasureConfig(warmup=1, iterations=3, duration_s=0.0))
    ok = {"model": "mock", "status": "ok", "describe": {"checkpoint": "mock"},
          "environment": {"gpu": "RTX 4090"}, "measurement": m, "verdict": "pass"}
    bad = {"model": "pi05", "status": "error", "describe": {}, "error": "OSError: gated repo"}
    table = markdown_table([ok, bad], requirement)
    assert "P95 ≤ 500 ms" in table
    assert "| mock | mock | RTX 4090 |" in table and "✅ 达标" in table
    assert "| pi05 |" in table and "gated repo" in table


def test_cli_runs_mock_end_to_end_in_a_worker_process(tmp_path, capsys):
    code = latency_run.main([
        "--models", "mock", "--warmup", "1", "--iterations", "5", "--duration", "0.1",
        "--image-hw", "16", "16", "--opt", "mock.model_ms=2", "--out", str(tmp_path),
    ])
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    assert {p.name for p in run_dir.iterdir()} >= {
        "meta.json", "mock.json", "mock.log", "summary.md", "summary.json",
    }
    (result,) = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok" and result["verdict"] == "pass"
    assert result["describe"]["image_hw"] == [16, 16]
    assert "samples_ms" not in result["measurement"]          # kept only in mock.json
    detail = json.loads((run_dir / "mock.json").read_text(encoding="utf-8"))
    assert len(detail["measurement"]["samples_ms"]) == detail["measurement"]["count"]


def test_cli_records_a_failing_model_and_returns_nonzero(tmp_path):
    code = latency_run.main([
        "--models", "mock", "--duration", "0", "--iterations", "1",
        "--opt", "mock.bogus=1", "--out", str(tmp_path),
    ])
    assert code == 1
    (run_dir,) = tmp_path.iterdir()
    (result,) = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "error" and "bogus" in result["error"]
    assert "⚠️ 运行失败" in (run_dir / "summary.md").read_text(encoding="utf-8")


def test_shadow_checkpoint_copies_small_files_and_links_large_ones(tmp_path, monkeypatch):
    from benchmarks.latency import models

    src = tmp_path / "ckpt"
    (src / "lora_adapter").mkdir(parents=True)
    (src / ".git").mkdir()
    (src / "config.json").write_text('{"a": 1}', encoding="utf-8")
    (src / "lora_adapter" / "adapter.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors").write_bytes(b"x" * 64)
    (src / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    monkeypatch.setattr(models, "_SHADOW_LINK_BYTES", 32)
    try:
        dst = models.shadow_checkpoint(str(src), root=str(tmp_path / "scratch"))
    except OSError as exc:  # Windows without symlink privilege
        pytest.skip(f"symlinks unavailable: {exc}")
    shadow = tmp_path / "scratch" / "vla_ckpt_shadow" / "ckpt"
    assert dst == str(shadow)
    assert (shadow / "model.safetensors").is_symlink()
    assert not (shadow / "config.json").is_symlink()
    assert (shadow / "lora_adapter" / "adapter.json").exists()
    assert not (shadow / ".git").exists()
    (shadow / "config.json").write_text("rewritten", encoding="utf-8")   # what get_vla does
    assert (src / "config.json").read_text(encoding="utf-8") == '{"a": 1}'
