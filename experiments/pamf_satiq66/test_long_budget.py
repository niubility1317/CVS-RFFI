"""Regression tests for the source-only stopping rule and fixed-budget compatibility."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.pamf_satiq66 import train


def test_long_budget_crosses_fixed_limit_and_stops_at_first_qualified_validation(tmp_path, monkeypatch):
    sampled = torch.zeros(4, 2, 256)
    meta = {"y": torch.tensor([0, 1, 0, 1]), "day": torch.tensor([0, 0, 1, 1])}
    adam = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(512, 66))
    results = iter([.97, .9799, .98])
    source = tmp_path / "source"
    out = tmp_path / "run"

    monkeypatch.setattr(train, "cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train, "check_contract", lambda *_: {"iq_rate_hz": [25000000], "counts": train.SOURCE_COUNTS})
    monkeypatch.setattr(train, "row_data", lambda _, role, __: (sampled, [role], {}))
    monkeypatch.setattr(train, "build_model", lambda *_: adam)
    monkeypatch.setattr(train, "batch", lambda *_: (sampled, meta))
    monkeypatch.setattr(train, "capture", lambda m, x: (m(x), None, None))
    monkeypatch.setattr(train, "Batches", lambda *_: SimpleNamespace(
        next=lambda: [0, 1, 2, 3], rng=np.random.default_rng(12)))

    def validate(*_):
        return {"accuracy": next(results), "macro_f1": .6, "records": 19800}

    monkeypatch.setattr(train, "validate", validate)
    train.train(source, out, {
        "runtime": {}, "model_seed": 11, "data_seed": 12, "augmentation_seed": 13,
        "epochs": 1, "steps_per_epoch": 1, "lr": .0003, "weight_decay": .0001,
        "long_budget": {"source_v_accuracy_target": .98, "checkpoint_every": 2},
    }, "S0", 0)

    reports = [json.loads(line) for line in (out / "epochs.jsonl").read_text().splitlines()]
    assert [r["v"]["accuracy"] for r in reports] == [.97, .9799, .98]
    assert json.loads((out / "complete.json").read_text())["updates"] == 3
    assert json.loads((out / "resolved_config.json").read_text())["role"] == "long_source_only"
    assert (out / "checkpoints" / "e00000002.pt").is_file()


@pytest.mark.parametrize("smoke", (False, True))
def test_supervised_phase_rejects_ssl_before_reading_any_data(tmp_path, smoke):
    with pytest.raises(ValueError, match="only S0 and S1"):
        train.train(tmp_path / "missing", tmp_path / "unused",
                    {"long_budget": {"source_v_accuracy_target": .98, "checkpoint_every": 25}},
                    "U1", 0, smoke=smoke, ssl_smoke=smoke)


def test_pseudo_label_probe_waits_for_both_supervised_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "check_contract", lambda *_: {"schema": "fixture"})
    statuses = {
        "resolved_config.json": {},
        "S0": {"v": {"accuracy": .98}, "target_reached": True},
        "S1": {"v": {"accuracy": .9799}, "target_reached": False},
    }
    monkeypatch.setattr(train, "read_json", lambda p: statuses[
        p.parent.name if p.name == "complete.json" else p.name])
    monkeypatch.setattr(train, "row_data", lambda *_: pytest.fail("U was read before the gate"))
    with pytest.raises(ValueError, match="S1 has not reached"):
        train.probe_source_u(tmp_path / "source", tmp_path,
                             {"long_budget": {"source_v_accuracy_target": .98}})
