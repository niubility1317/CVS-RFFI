from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts.run_stage2_capta_target5_matrix import (
    MatrixConfigError,
    load_matrix_config,
    plan_tasks,
    run_matrix,
)


CANDIDATES = [
    "CAPTA_A1_SUPPORT_SHRINK",
    "CAPTA_A2_SHARED_SHIFT",
    "CAPTA_A3_R4_SUPPORT_SHIFT",
]


def _config(path: Path) -> Path:
    value = {
        "schema": "cvs.stage2b.capta_p0_target5_matrix.v1",
        "run_id": "capta-test-run",
        "checkpoint": "/frozen/checkpoint.pth",
        "candidates": CANDIDATES,
        "rank": 4,
        "prior_strength": 3.0,
        "rows": [
            {
                "row_id": "rx20_1_k5_new20_clear",
                "scenario": "leo_clear_weak",
                "work_dir": "/prior/work/clear",
                "query_package": "/input/predictor/query_leo_clear_weak.npz",
                "package_manifest": "/input/predictor/package_manifest.json",
                "validated_row_manifest": "/input/validated/rx20_1_k5_new20.json",
                "row_binding": "/prior/input/row_bindings/clear.row.json",
                "truth": "/input/scorer/truth_sidecar.json",
                "baseline_prediction": "/prior/predictions/clear_da0.json",
            }
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_plan_expands_each_row_to_three_nonoverwriting_candidate_pairs(
    tmp_path: Path,
) -> None:
    config = load_matrix_config(_config(tmp_path / "matrix.json"))

    tasks = plan_tasks(
        config,
        release_root=Path("/release/src"),
        output_root=Path("/run/results"),
        device="cuda:0",
    )

    assert len(tasks) == 3
    assert [task.candidate_id for task in tasks] == CANDIDATES
    assert len({task.prediction_output for task in tasks}) == 3
    assert len({task.score_output for task in tasks}) == 3
    assert tasks[2].prediction_command[:3] == (
        str(Path("/release/src") / "code" / "scripts" / "run_stage2_capta_p0.py"),
        "run-row",
        "--checkpoint",
    )
    assert "CAPTA_A3_R4_SUPPORT_SHIFT" in tasks[2].prediction_command
    assert tasks[2].score_command[0] == str(
        Path("/release/src") / "code" / "scripts" / "score_stage2_structured_late_block_pair.py"
    )
    assert "/prior/predictions/clear_da0.json" in tasks[2].score_command


def test_matrix_refuses_existing_output_root_before_any_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_matrix_config(_config(tmp_path / "matrix.json"))
    output_root = tmp_path / "results"
    output_root.mkdir()
    called = {"value": False}

    def fail_if_called(*args: object, **kwargs: object) -> object:
        called["value"] = True
        raise AssertionError("subprocess must not start")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    with pytest.raises(FileExistsError):
        run_matrix(
            config,
            release_root=tmp_path / "release",
            output_root=output_root,
            device="cpu",
        )
    assert called["value"] is False


def test_matrix_records_complete_predictions_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_matrix_config(_config(tmp_path / "matrix.json"))
    output_root = tmp_path / "results"

    def complete(command: list[str] | tuple[str, ...], **_: object) -> object:
        output_index = command.index("--output") + 1
        output = Path(command[output_index])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", complete)

    summary = run_matrix(
        config,
        release_root=tmp_path / "release",
        output_root=output_root,
        device="cpu",
    )

    assert summary["status"] == "ARTIFACTS_COMPLETE"
    assert summary["prediction_count"] == 3
    assert summary["score_count"] == 3
    assert summary["failed_count"] == 0
    persisted = json.loads(
        (output_root / "matrix_summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_config_rejects_duplicate_rows_and_non_target5_candidates(tmp_path: Path) -> None:
    path = _config(tmp_path / "matrix.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["rows"].append(dict(value["rows"][0]))
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(MatrixConfigError):
        load_matrix_config(path)
