from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_d92_continuous_session_analysis import (
    ContinuousSessionAnalysisError,
    analyze_continuous_session_run,
)


SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
SCHEDULES = ("batch_5", "singleton_forward", "singleton_reverse", "chunk_2_2_1")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.chmod(path, stat.S_IREAD)
    return _sha(path)


def _write_state(root: Path, *, lifecycle: str, session_index: int, state_sha: str, classes: tuple[str, ...]) -> dict[str, object]:
    root.mkdir(parents=True)
    tokens = np.asarray(("qid_old", "qid_new", "qid_future"))
    scenarios = np.asarray((root.parents[1].name, ) * 3)
    # REG0 competes only over the frozen old-class head; the future new-class
    # truth is intentionally present but is not scored until registration.
    prediction = np.asarray(("old_0", "new_0" if "new_0" in classes else "old_0", "old_0"))
    artifact = root / "prediction_artifact.npz"
    np.savez(artifact, query_tokens=tokens, scenarios=scenarios, predicted_class_handles=prediction)
    os.chmod(artifact, stat.S_IREAD)
    prediction_sha = _sha(artifact)
    fit_sha = _write_json(root / "fit_audit.json", {
        "lifecycle_state": lifecycle,
        "session_index": session_index,
        "state_sha256": state_sha,
        "registered_classes": list(classes),
        "old_class_count": 6,
    })
    resource_sha = _write_json(root / "resource_audit.json", {
        "registration_wall_time_ns": 10,
        "registration_incremental_peak_working_set_bytes": 20,
        "support_bytes": 30,
        "state_bytes": 40,
        "query_macs": len(classes) * 288,
        "head_latency_ns": 0,
    })
    token_sha = hashlib.sha256(json.dumps(tokens.tolist(), separators=(",", ":")).encode()).hexdigest()
    scenario_sha = hashlib.sha256(json.dumps(scenarios.tolist(), separators=(",", ":")).encode()).hexdigest()
    receipt_sha = _write_json(root / "execution_receipt.json", {
        "lifecycle_state": lifecycle,
        "session_index": session_index,
        "state_sha256": state_sha,
        "registered_classes": list(classes),
        "registered_class_count": len(classes),
        "prediction_artifact_sha256": prediction_sha,
        "fit_audit_sha256": fit_sha,
        "resource_audit_sha256": resource_sha,
        "query_token_sha256": token_sha,
        "query_scenario_sha256": scenario_sha,
    })
    members = [{"relative_path": path.name, "sha256": _sha(path), "size_bytes": path.stat().st_size} for path in sorted(root.iterdir())]
    commit_sha = _write_json(root / "COMMIT.json", {
        "members": members,
        "artifact_root_sha256": hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "prediction_artifact_sha256": prediction_sha,
        "execution_receipt_sha256": receipt_sha,
    })
    return {
        "output_root": str(root), "lifecycle_state": lifecycle, "session_index": session_index,
        "state_sha256": state_sha, "prediction_artifact_sha256": prediction_sha,
        "commit_sha256": commit_sha,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    output_root = tmp_path / "output"
    job_root = output_root / "jobs" / "outer0" / "full"
    classes0 = tuple(f"old_{index}" for index in range(6))
    classes1 = classes0 + ("new_0",)
    scenes: dict[str, object] = {}
    for scene in SCENES:
        baseline = _write_state(job_root / scene / "DA1_REG0", lifecycle="DA1_REG0", session_index=0, state_sha="a" * 64, classes=classes0)
        schedules: dict[str, object] = {}
        for schedule in SCHEDULES:
            final = _write_state(job_root / scene / schedule / "session_01", lifecycle="DA1_REG1_S1", session_index=1, state_sha="b" * 64, classes=classes1)
            schedules[schedule] = {"increments": [1], "arrival_order": [0], "sessions": [final]}
        scenes[scene] = {"DA1_REG0": baseline, "schedules": schedules}
    prediction_manifest = job_root / "prediction_manifest.json"
    _write_json(prediction_manifest, {"schema": "cvs.phase2.d92_e0_continuous_session.truth_free_prediction.v1", "scenes": scenes})
    matrix_manifest = tmp_path / "matrix_manifest.json"
    _write_json(matrix_manifest, {"jobs": [{"job_id": "job0", "outer_key": "outer0", "output_root": str(output_root / "jobs" / "outer0")} ]})
    truth_root = tmp_path / "truth"
    _write_json(truth_root / "outer0" / "truth_sidecar.json", {
        "schema": "cvs.phase2.query_truth_sidecar.v2",
        "rows": [
            {"query_token": "qid_old", "true_class_handle": "old_0", "transmitter_label": "old_0", "evaluation_role": "target_old"},
            {"query_token": "qid_new", "true_class_handle": "new_0", "transmitter_label": "new_0", "evaluation_role": "target_new"},
            {"query_token": "qid_future", "true_class_handle": "new_future", "transmitter_label": "new_future", "evaluation_role": "target_new"},
        ],
    })
    return matrix_manifest, output_root, truth_root, tmp_path / "analysis"


def test_truth_last_scores_registered_only_and_requires_terminal_equivalence(tmp_path: Path) -> None:
    manifest, output_root, truth_root, analysis_root = _fixture(tmp_path)

    result = analyze_continuous_session_run(
        manifest_path=manifest,
        output_root=output_root,
        truth_root=truth_root,
        analysis_root=analysis_root,
    )

    assert result["status"] == "ANALYZED_TRUTH_LAST"
    assert result["terminal_equivalence"]["outer0"]["leo_clear_weak"]["status"] == "STRICT_EQUAL"
    trajectory = result["trajectories"]["outer0"]["leo_clear_weak"]["batch_5"][0]
    assert trajectory["registered_new_accuracy"] == 1.0
    assert trajectory["unregistered_truth_count"] == 1
    assert trajectory["unregistered_truth_status"] == "UNREGISTERED_NOT_SCORED"
    assert (analysis_root / "continuous_session_analysis.json").is_file()


def test_truth_last_rejects_missing_prediction_token_in_truth(tmp_path: Path) -> None:
    manifest, output_root, truth_root, analysis_root = _fixture(tmp_path)
    truth_path = truth_root / "outer0" / "truth_sidecar.json"
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    payload["rows"] = payload["rows"][:-1]
    truth_path.chmod(stat.S_IWRITE)
    truth_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContinuousSessionAnalysisError, match="truth"):
        analyze_continuous_session_run(
            manifest_path=manifest,
            output_root=output_root,
            truth_root=truth_root,
            analysis_root=analysis_root,
        )
