from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "code" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_cvs_stage2_predictor import (  # noqa: E402
    _class_handle_predictions,
    _prepare_device,
)


def test_prediction_indices_are_exported_as_opaque_class_handles() -> None:
    registry = [
        {"class_index": 0, "class_handle": "cls_" + "0" * 64},
        {"class_index": 1, "class_handle": "cls_" + "1" * 64},
    ]
    streams = {
        "candidate_after": np.asarray([0, 1]),
        "candidate_before": np.asarray([0, 0]),
        "identity_after": np.asarray([0, 1]),
        "identity_before": np.asarray([0, 0]),
        "direct": np.asarray([0, 0]),
    }
    converted = _class_handle_predictions(streams, registry)
    assert converted["candidate_after"].tolist() == [
        registry[0]["class_handle"], registry[1]["class_handle"]
    ]
    assert all(value.startswith("cls_") for value in converted["direct"])


def test_prediction_indices_outside_registry_fail_closed() -> None:
    registry = [{"class_index": 0, "class_handle": "cls_" + "0" * 64}]
    streams = {name: np.asarray([0]) for name in (
        "candidate_after", "candidate_before", "identity_after",
        "identity_before", "direct",
    )}
    streams["candidate_after"] = np.asarray([1])
    with pytest.raises(ValueError, match="outside registry"):
        _class_handle_predictions(streams, registry)


def test_cuda_is_initialized_before_peak_memory_reset(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        "run_cvs_stage2_predictor.torch.cuda.is_available", lambda: True
    )
    monkeypatch.setattr(
        "run_cvs_stage2_predictor.torch.cuda.init", lambda: calls.append("init")
    )
    monkeypatch.setattr(
        "run_cvs_stage2_predictor.torch.cuda.reset_peak_memory_stats",
        lambda device: calls.append(("reset", str(device))),
    )

    device = _prepare_device("cuda:0")

    assert str(device) == "cuda:0"
    assert calls == ["init", ("reset", "cuda:0")]


def test_cpu_device_does_not_initialize_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        "run_cvs_stage2_predictor.torch.cuda.is_available", lambda: True
    )
    monkeypatch.setattr(
        "run_cvs_stage2_predictor.torch.cuda.init",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected CUDA init")),
    )

    assert str(_prepare_device("cpu")) == "cpu"
