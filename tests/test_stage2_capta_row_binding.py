from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import scripts.run_stage2_capta_p0 as runner


def test_script_is_directly_executable_from_repo_root() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_stage2_capta_p0.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "run-row" in completed.stdout


def test_smoke_parser_has_no_query_surface() -> None:
    args = runner.build_parser().parse_args(
        [
            "smoke",
            "--checkpoint",
            "checkpoint.pt",
            "--support-only",
            "support.npz",
            "--frozen-prototypes",
            "prototypes.npz",
            "--context",
            "context.json",
            "--output",
            "smoke.json",
        ]
    )

    assert args.command == "smoke"
    assert all(
        token not in name.lower()
        for name in vars(args)
        for token in ("query", "truth", "role", "source", "clean", "quota")
    )


def test_run_row_freezes_adaptation_before_query_open_and_records_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapted = {"done": False}
    capta_state = SimpleNamespace(
        prototype_class_ids=("old-a", "old-b"),
        source_weight=0.75,
        audit={"backward_count": 0, "trainable_parameter_count": 0},
    )

    def fake_adapt(_: object) -> tuple[object, dict[str, object], dict[str, object], object]:
        adapted["done"] = True
        return (
            model,
            {"checkpoint_load_strict": True, "checkpoint_load_audit": {}},
            {
                "protocol_schema": "p2_min_v1",
                "phase2_data_status": "VALIDATED_ONCE",
                "capsule_id": "capsule-fixed",
                "split_id": "split-fixed",
                "support_input_count": 10,
            },
            capta_state,
        )

    def fake_query(*_: object) -> tuple[np.ndarray, np.ndarray]:
        assert adapted["done"] is True
        assert all(not parameter.requires_grad for parameter in model.parameters())
        return (
            np.zeros((2, 2, 8), dtype=np.float32),
            np.asarray(["opaque-q0", "opaque-q1"]),
        )

    prediction = SimpleNamespace(
        predicted_class_ids=("old-a", "old-b"),
        source_scores=torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
        target_scores=torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        mixed_scores=torch.tensor([[3.25, 0.75], [0.75, 3.25]]),
        query_batch_state_updated=False,
    )
    monkeypatch.setattr(runner, "_adapt_from_whitelist", fake_adapt)
    monkeypatch.setattr(
        runner.late,
        "_read_row_binding",
        lambda _: {
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "capsule-fixed",
            "split_id": "split-fixed",
        },
    )
    monkeypatch.setattr(runner.late, "_load_query_received_iq", fake_query)
    monkeypatch.setattr(runner, "predict_query_read_only", lambda *args, **kwargs: prediction)
    output = tmp_path / "prediction.json"
    args = SimpleNamespace(
        query_package="query.npz",
        package_manifest="package.json",
        validated_row_manifest="validated.json",
        row_binding="row.json",
        output=output,
    )

    result = runner.run_row(args)

    assert result["state"] == "DA1_REG0"
    assert result["query_truth_loaded"] is False
    assert result["query_role_loaded"] is False
    assert result["query_batch_state_updated"] is False
    assert result["audit"]["backward_count"] == 0
    assert result["predictions"][0] == {
        "sample_index": 0,
        "query_token": "opaque-q0",
        "predicted_class_id": "old-a",
        "scores": [3.25, 0.75],
        "source_scores": [3.0, 1.0],
        "target_scores": [4.0, 0.0],
        "source_weight": 0.75,
    }
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == result
    with pytest.raises(FileExistsError):
        runner.run_row(args)


def test_run_row_rejects_individually_valid_support_query_row_mismatch_before_query_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capta_state = SimpleNamespace(source_weight=1.0, audit={})
    support_context = {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-support-a",
        "split_id": "split-support-a",
        "support_input_count": 10,
    }
    query_row = {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-query-b",
        "split_id": "split-query-b",
    }
    query_opened = {"value": False}
    prediction_started = {"value": False}

    monkeypatch.setattr(
        runner,
        "_adapt_from_whitelist",
        lambda _: (
            model,
            {"checkpoint_load_strict": True, "checkpoint_load_audit": {}},
            support_context,
            capta_state,
        ),
    )
    monkeypatch.setattr(runner.late, "_read_row_binding", lambda _: query_row)

    def forbidden_query_open(*_: object) -> tuple[np.ndarray, np.ndarray]:
        query_opened["value"] = True
        raise AssertionError("query must remain unopened for a mismatched row")

    def forbidden_prediction(*_: object, **__: object) -> object:
        prediction_started["value"] = True
        raise AssertionError("prediction must not start for a mismatched row")

    monkeypatch.setattr(runner.late, "_load_query_received_iq", forbidden_query_open)
    monkeypatch.setattr(runner, "predict_query_read_only", forbidden_prediction)
    args = SimpleNamespace(
        query_package="query-b.npz",
        package_manifest="package-b.json",
        validated_row_manifest="validated-b.json",
        row_binding="row-b.json",
        output=tmp_path / "must-not-exist.json",
    )

    with pytest.raises(ValueError, match="support/query row binding mismatch"):
        runner.run_row(args)

    assert query_opened["value"] is False
    assert prediction_started["value"] is False
    assert not Path(args.output).exists()


def test_baseline_delegates_to_the_same_frozen_decision_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"state": "DA0_REG0", "status": "PREDICTIONS_COMPLETE"}
    monkeypatch.setattr(runner.late, "run_baseline", lambda args: expected)

    assert runner.run_baseline(SimpleNamespace()) is expected
