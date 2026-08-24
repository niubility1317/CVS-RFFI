from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts.run_stage2_apsta_target5_matrix import (
    MatrixConfigError,
    load_matrix_config,
    plan_tasks,
    run_matrix,
)


def _config(tmp_path: Path) -> Path:
    rows = {
        "rows": [{
            "row_id": "rx20_1_k5_new20_clear", "scenario": "leo_clear_weak",
            "work_dir": "/work/clear", "query_package": "/input/q.npz",
            "package_manifest": "/input/package.json",
            "validated_row_manifest": "/input/validated.json",
            "row_binding": "/input/row.json", "truth": "/input/truth.json",
            "baseline_prediction": "/input/da0.json",
        }]
    }
    (tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    value = {
        "schema": "cvs.stage2b.apsta_p1_target5_matrix.v1",
        "run_id": "apsta-test", "checkpoint": "/frozen/model.pth",
        "candidate_id": "APSTA_P1_TIME_FUSION_ROBUST",
        "checkpoints": [0, 10, 30, 100, 300], "learning_rate": 0.0002,
        "anchor_strength": 3.0, "head_ce_weight": 0.25,
        "loo_mean_weight": 1.0, "tail_weight": 0.5,
        "tail_temperature": 0.5, "topology_weight": 0.25,
        "l2sp_weight": 0.001, "margin_epsilon": 0.0,
        "row_source_config": "rows.json",
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_plan_has_one_truth_last_pair_per_row(tmp_path: Path) -> None:
    config = load_matrix_config(_config(tmp_path))
    tasks = plan_tasks(config, release_root=Path("/release"), output_root=Path("/run"), device="cuda:0")
    assert len(tasks) == 1
    task = tasks[0]
    assert "run_stage2_apsta_p1.py" in task.prediction_command[0]
    assert task.prediction_command[1] == "run-row"
    assert task.prediction_command.index("--checkpoints") < task.prediction_command.index("--query-package")
    assert task.score_command[0].endswith("score_stage2_structured_late_block_pair.py")
    assert task.score_command.index("--truth") > task.score_command.index("--da1")


def test_matrix_refuses_output_collision(tmp_path: Path) -> None:
    config = load_matrix_config(_config(tmp_path))
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_matrix(config, release_root=tmp_path / "release", output_root=output, device="cpu")


def test_matrix_closes_one_prediction_and_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_matrix_config(_config(tmp_path))
    output = tmp_path / "output"

    def complete(command: list[str] | tuple[str, ...], **_: object) -> object:
        destination = Path(command[command.index("--output") + 1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    monkeypatch.setattr(subprocess, "run", complete)
    summary = run_matrix(config, release_root=tmp_path / "release", output_root=output, device="cpu")
    assert summary["status"] == "ARTIFACTS_COMPLETE"
    assert summary["prediction_count"] == summary["score_count"] == 1
    assert summary["candidate_count"] == 1


def test_config_rejects_trainable_head_or_nonformal_schedule(tmp_path: Path) -> None:
    path = _config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["candidate_id"] = "RANK4_TRAINABLE_HEAD"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(MatrixConfigError):
        load_matrix_config(path)

