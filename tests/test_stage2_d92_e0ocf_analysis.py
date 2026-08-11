from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_d92_e0d_slim import D92_E0D_ARMS
from cvsrffi.stage2_d92_e0ocf_analysis import (
    D92E0OCFAnalysisError,
    analyze_d92_e0ocf_hard12v3,
)
from cvsrffi.stage2_d92_e0ocf_hard12 import CANONICAL_SELECTION_SHA256


SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARMS = ("D92_FULL", "E0_FULL_ONLY", "E0_FIXED50", "E0_OCF25", "E0_OCF50")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_manifest(tmp_path: Path, *, ocf50_wins: bool = False) -> Path:
    output = tmp_path / "run"
    lock_path = tmp_path / "method_lock.json"
    _write_json(
        lock_path,
        {
            "schema": "cvs.phase2.d92_e0ocf.method_lock.v1",
            "only_promotion_candidate": "E0_OCF25",
            "strict_geometry_gate": {
                "mean_delta_old_floor_vs_full_only_min_exclusive": 0.0,
                "old_floor_nonnegative_vs_full_only_min": 8,
                "mean_delta_h_vs_full_only_min": 0.0,
                "mean_delta_old_balanced_vs_full_only_min": 0.0,
                "mean_delta_seen_new_vs_full_only_min": 0.0,
                "mean_delta_forgetting_vs_full_only_max": 0.0,
                "mean_delta_h_vs_d92_full_min": 0.005,
                "h_nonnegative_vs_d92_full_min": 8,
                "mean_delta_old_balanced_vs_d92_full_min": 0.0,
                "mean_delta_old_floor_vs_d92_full_min": 0.0,
                "mean_delta_seen_new_vs_d92_full_min": 0.0,
                "mean_delta_forgetting_vs_d92_full_max": 0.0,
                "median_wall_reduction_vs_d92_full_min": 0.6,
            },
        },
    )
    selected_rows = []
    jobs = []
    for outer_index in range(12):
        k_shot = 1 if outer_index < 2 else (5 if outer_index < 5 else 10)
        role = "liveness" if k_shot == 1 else "performance"
        outer_key = f"outer_{outer_index:02d}"
        selected_rows.append({"outer_key": outer_key, "outer_role": role, "k_shot": k_shot})
        before_fp = hashlib.sha256(f"before:{outer_key}".encode()).hexdigest()
        for arm in ARMS:
            root = output / "jobs" / outer_key / arm
            before = root / "diag" / "before" / "prediction_artifact.npz"
            after = root / "diag" / "after" / "prediction_artifact.npz"
            before.parent.mkdir(parents=True, exist_ok=True)
            after.parent.mkdir(parents=True, exist_ok=True)
            np.savez(before, query_tokens=np.asarray(["q0"]), scenarios=np.asarray([SCENES[0]]), predicted_class_handles=np.asarray(["old_0"]))
            np.savez(after, query_tokens=np.asarray(["q0"]), scenarios=np.asarray([SCENES[0]]), predicted_class_handles=np.asarray(["old_0"]))
            (before.parent / "COMMIT.json").write_text("{}", encoding="utf-8")
            (after.parent / "COMMIT.json").write_text("{}", encoding="utf-8")
            score = {
                "schema": "cvs.phase2.diag_cosine_dev_pair_score.v1",
                "candidate": D92_E0D_ARMS[arm].candidate_id,
                "before": {"by_scenario": {s: {"old_acc": 0.7} for s in SCENES}, "by_tx": {"old0": {"role": "target_old", "accuracy": 0.7}}},
                "after": {"by_scenario": {s: {"old_acc": 0.6, "seen_new_acc": 0.5, "h_old_new": 0.55, "old_to_new_rate": 0.2, "new_to_old_rate": 0.1} for s in SCENES}, "by_tx": {"old0": {"role": "target_old", "accuracy": 0.6}}},
                "old_forgetting_pp": 10.0,
                "per_old_class_floor_after": 0.4,
                "query_truth_joined_only_after_immutable_predictions": True,
                "query_truth_fed_back_to_predictor": False,
            }
            if role == "performance":
                score["after"]["by_scenario"] = {s: {"old_acc": 0.6, "seen_new_acc": 0.5, "h_old_new": 0.51 if arm == "D92_FULL" else (0.516 if arm == "E0_OCF25" else (0.60 if ocf50_wins and arm == "E0_OCF50" else 0.51)), "old_to_new_rate": 0.2, "new_to_old_rate": 0.1} for s in SCENES}
            score_path = root / "scorer" / "diag_cosine_score.json"
            score["before_prediction_sha256"] = _sha256(before)
            score["after_prediction_sha256"] = _sha256(after)
            _write_json(score_path, score)
            fit = [{
                "scenario": s,
                "after_total_component_fit_count": 3 if k_shot <= 2 else (8 * (k_shot + 1) if arm == "D92_FULL" else (2 if arm == "E0_FULL_ONLY" else 4)),
                "after_actual_component_inventory": {"actual_component_fit_count": 3 if k_shot <= 2 else (4 * (k_shot + 1) if arm == "D92_FULL" else (1 if arm == "E0_FULL_ONLY" else 2))},
                "after_registration_resource": {"registration_wall_time_ns": 100 if arm == "D92_FULL" else 20, "registration_incremental_peak_working_set_bytes": 1000 if arm == "D92_FULL" else 900},
                "query_macs": 7488,
                "state_bytes": 10000,
                "d92_e0d_ocf_active": arm in {"E0_OCF25", "E0_OCF50"} and k_shot > 2,
                "d92_e0d_ocf_lambda": (0.25 if arm == "E0_OCF25" else 0.50) if arm in {"E0_OCF25", "E0_OCF50"} and k_shot > 2 else None,
                "query_truth_access": False, "query_fit_access": False, "query_update_access": False, "query_selection_access": False, "query_role_oracle_access": False, "query_class_quota_access": False, "query_global_reassignment": False,
                "d92_e0d_ocf_support_alignment_macs_upper_bound": 10 if arm in {"E0_OCF25", "E0_OCF50"} and k_shot > 2 else 0,
                "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound": 100 if arm in {"E0_OCF25", "E0_OCF50"} and k_shot > 2 else 0,
                "before_state_fingerprint_sha256": before_fp,
                "after_state_fingerprint_sha256": before_fp if k_shot <= 2 else hashlib.sha256(f"after:{outer_key}:{arm}".encode()).hexdigest(),
            } for s in SCENES]
            _write_json(root / "diag" / "after" / "fit_audit.json", fit)
            receipt = {"schema": "cvs.phase2.d92_e0ocf_hard12v3.job_receipt.v1", "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE", "job_id": f"{outer_key}__arm_{arm.lower()}", "outer_key": outer_key, "outer_role": role, "k_shot": k_shot, "arm_id": arm, "candidate": D92_E0D_ARMS[arm].candidate_id, "before_prediction_sha256": _sha256(before), "after_prediction_sha256": _sha256(after), "score_sha256": _sha256(score_path), "truth_sidecar_exposed_to_predictor": False, "query_truth_joined_only_after_immutable_predictions": True, "query_truth_fed_back_to_predictor": False}
            _write_json(root / "job_receipt.json", receipt)
            jobs.append({"job_id": receipt["job_id"], "outer_key": outer_key, "outer_role": role, "k_shot": k_shot, "arm_id": arm, "candidate": receipt["candidate"], "output_root": str(root)})
    for shard in range(8):
        count = 60 if shard == 0 else 0
        _write_json(output / "summaries" / f"shard_{shard}.json", {"schema": "cvs.phase2.d92_e0ocf_hard12v3.shard_summary.v1", "status": "PASS", "shard_index": shard, "selected_job_count": count, "completed_job_count": count, "failed_job_count": 0, "performance_result_allowed": True})
    manifest = {"schema": "cvs.phase2.d92_e0ocf_hard12v3.matrix.v1", "status": "FROZEN_DEVELOPMENT_MATRIX", "claim_scope": "DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN", "protocol_schema": "p2_min_v1", "selection_sha256": CANONICAL_SELECTION_SHA256, "method_lock": str(lock_path), "method_lock_sha256": _sha256(lock_path), "output_root": str(output), "shard_count": 8, "outer_count": 12, "performance_outer_count": 10, "liveness_outer_count": 2, "job_count": 60, "scene_arm_count": 180, "primary_arm": "E0_OCF25", "arms": list(ARMS), "selected_rows": selected_rows, "jobs": jobs}
    manifest_path = output / "matrix_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_analysis_reports_confusion_formula_and_excludes_ocf50(tmp_path: Path) -> None:
    result = analyze_d92_e0ocf_hard12v3(_make_manifest(tmp_path, ocf50_wins=True))
    row = result["paired_rows"][0]
    assert row["E0_OCF25_old_to_old_rate"] == pytest.approx(0.2)
    assert row["E0_OCF25_old_to_new_rate"] == pytest.approx(0.2)
    assert row["E0_OCF25_new_to_old_rate"] == pytest.approx(0.1)
    assert result["promotion_candidate"] == "E0_OCF25"
    assert result["verdict"] != "PROMOTE_E0_OCF50_TO_TARGET125_CONFIRMATION"


def test_analysis_rejects_missing_complete_matrix(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["job_count"] = 59
    _write_json(path, manifest)
    with pytest.raises(D92E0OCFAnalysisError):
        analyze_d92_e0ocf_hard12v3(path)


def test_analysis_rejects_non_ocf_active_lambda_drift(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    job = next(row for row in manifest["jobs"] if row["arm_id"] == "E0_FULL_ONLY" and row["k_shot"] == 5)
    fit_path = Path(job["output_root"]) / "diag" / "after" / "fit_audit.json"
    rows = json.loads(fit_path.read_text(encoding="utf-8"))
    rows[0]["d92_e0d_ocf_active"] = True
    rows[0]["d92_e0d_ocf_lambda"] = 0.25
    _write_json(fit_path, rows)
    with pytest.raises(D92E0OCFAnalysisError, match="non-OCF arm"):
        analyze_d92_e0ocf_hard12v3(path)


def test_analysis_rejects_missing_ocf_support_cost(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    job = next(row for row in manifest["jobs"] if row["arm_id"] == "E0_OCF25" and row["k_shot"] == 5)
    fit_path = Path(job["output_root"]) / "diag" / "after" / "fit_audit.json"
    rows = json.loads(fit_path.read_text(encoding="utf-8"))
    rows[0].pop("d92_e0d_ocf_support_alignment_macs_upper_bound")
    _write_json(fit_path, rows)
    with pytest.raises(D92E0OCFAnalysisError, match="support-side cost"):
        analyze_d92_e0ocf_hard12v3(path)
