from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_ssdg_sat_per_rx.py"
SPEC = importlib.util.spec_from_file_location("eval_ssdg_sat_per_rx_strict", SCRIPT)
evaluator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluator)


class _RecordingModel:
    def __init__(self):
        self.strict_values = []

    def load_state_dict(self, _state, *, strict):
        self.strict_values.append(strict)
        return [], []


def test_strict_checkpoint_loader_uses_torch_strict_true_and_reports_zero_mismatch():
    model = _RecordingModel()

    audit = evaluator._load_checkpoint_state(
        model,
        {"weight": torch.ones(1)},
        strict_reconstruction=True,
    )

    assert model.strict_values == [True]
    assert audit == {
        "checkpoint_load_strict": True,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "shape_mismatches": 0,
    }


def test_strict_reconstruction_never_calls_direct_builder_fallback(monkeypatch):
    fallback_calls = []

    def fail_exact(*_args, **_kwargs):
        raise RuntimeError("strict restore mismatch")

    monkeypatch.setattr(evaluator, "_build_exact_ssdg_context", fail_exact)
    monkeypatch.setattr(
        evaluator,
        "_build_direct_context",
        lambda *_args, **_kwargs: fallback_calls.append("fallback"),
    )

    with pytest.raises(RuntimeError, match="strict restore mismatch"):
        evaluator._build_evaluation_context(
            {"model": {}},
            object(),
            torch.device("cpu"),
            strict_reconstruction=True,
        )

    assert fallback_calls == []


def test_legacy_non_strict_reconstruction_preserves_direct_builder_fallback(monkeypatch):
    fallback_context = (
        object(),
        {},
        {},
        {},
        {},
        object(),
        {
            "checkpoint_load_strict": False,
            "missing_keys": 1,
            "unexpected_keys": 0,
            "shape_mismatches": 0,
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_build_exact_ssdg_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("legacy mismatch")),
    )
    monkeypatch.setattr(
        evaluator,
        "_build_direct_context",
        lambda *_args, **_kwargs: fallback_context,
    )

    context = evaluator._build_evaluation_context(
        {"model": {}},
        object(),
        torch.device("cpu"),
        strict_reconstruction=False,
    )

    assert context[-2] == "direct_builder_fallback"
    assert context[-1] == {
        "strict_requested": False,
        "checkpoint_load_strict": False,
        "missing_keys": 1,
        "unexpected_keys": 0,
        "shape_mismatches": 0,
        "fallback_used": True,
    }


def test_strict_reconstruction_failure_exits_before_metrics_are_written(tmp_path, monkeypatch):
    output_json = tmp_path / "metrics.json"
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(evaluator.torch, "load", lambda *_args, **_kwargs: {"model": {}})

    def fail_context(*_args, **_kwargs):
        raise RuntimeError("strict checkpoint restore failed")

    monkeypatch.setattr(evaluator, "_build_evaluation_context", fail_context)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--ckpt",
            str(checkpoint),
            "--output_json",
            str(output_json),
            "--device",
            "cpu",
            "--strict_reconstruction",
        ],
    )

    with pytest.raises(RuntimeError, match="strict checkpoint restore failed"):
        evaluator.main()

    assert not output_json.exists()
