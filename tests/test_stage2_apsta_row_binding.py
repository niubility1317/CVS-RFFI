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

import scripts.run_stage2_apsta_p1 as runner


def test_script_is_directly_executable_and_smoke_has_no_query_surface() -> None:
    script = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_stage2_apsta_p1.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    args = runner.build_parser().parse_args(
        [
            "smoke", "--checkpoint", "c.pt", "--support-only", "s.npz",
            "--frozen-prototypes", "p.npz", "--context", "x.json",
            "--output", "o.json",
        ]
    )
    assert all(
        token not in name.lower()
        for name in vars(args)
        for token in ("query", "truth", "role", "source", "clean", "quota")
    )


def test_run_row_binds_before_query_and_emits_teacher_student_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student = nn.Linear(2, 2)
    teacher = nn.Linear(2, 2)
    for model in (student, teacher):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    context = {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "cap-a",
        "split_id": "split-a",
        "support_input_count": 10,
    }
    audit = SimpleNamespace(backward_count=300, selected_step=30)
    adapted = {"done": False}

    def fake_adapt(_: object) -> tuple[object, object, dict[str, object], dict[str, object], object, tuple[str, ...], np.ndarray]:
        adapted["done"] = True
        return student, teacher, {"checkpoint_load_strict": True}, context, audit, ("a", "b"), np.eye(2)

    def fake_query(*_: object) -> tuple[np.ndarray, np.ndarray]:
        assert adapted["done"]
        return np.zeros((2, 2, 8), np.float32), np.asarray(["q0", "q1"])

    prediction = SimpleNamespace(
        predicted_class_ids=("a", "b"),
        student_scores=torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
        teacher_scores=torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
        query_state_updated=False,
    )
    monkeypatch.setattr(runner, "_adapt_from_whitelist", fake_adapt)
    monkeypatch.setattr(runner.late, "_read_row_binding", lambda _: dict(context))
    monkeypatch.setattr(runner.late, "_load_query_received_iq", fake_query)
    monkeypatch.setattr(runner, "predict_query_read_only", lambda *args, **kwargs: prediction)
    output = tmp_path / "pred.json"
    args = SimpleNamespace(
        row_binding="row.json", query_package="q.npz", package_manifest="p.json",
        validated_row_manifest="v.json", output=output,
    )
    result = runner.run_row(args)
    assert result["state"] == "DA1_REG0"
    assert result["query_truth_loaded"] is False
    assert result["query_batch_state_updated"] is False
    assert result["predictions"][0]["scores"] == [3.0, 1.0]
    assert result["predictions"][0]["student_scores"] == [3.0, 1.0]
    assert result["predictions"][0]["teacher_scores"] == [2.0, 1.0]
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_run_row_rejects_support_query_mismatch_before_query_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    student = nn.Linear(2, 2)
    teacher = nn.Linear(2, 2)
    for model in (student, teacher):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    context = {
        "protocol_schema": "p2_min_v1", "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "support", "split_id": "split-a", "support_input_count": 2,
    }
    monkeypatch.setattr(
        runner, "_adapt_from_whitelist",
        lambda _: (student, teacher, {}, context, SimpleNamespace(), ("a", "b"), np.eye(2)),
    )
    monkeypatch.setattr(
        runner.late, "_read_row_binding",
        lambda _: {**context, "capsule_id": "query"},
    )
    opened = {"value": False}

    def forbidden(*_: object) -> object:
        opened["value"] = True
        raise AssertionError("query must remain unopened")

    monkeypatch.setattr(runner.late, "_load_query_received_iq", forbidden)
    args = SimpleNamespace(
        row_binding="row.json", query_package="q.npz", package_manifest="p.json",
        validated_row_manifest="v.json", output=tmp_path / "no.json",
    )
    with pytest.raises(ValueError, match="support/query row binding mismatch"):
        runner.run_row(args)
    assert opened["value"] is False
