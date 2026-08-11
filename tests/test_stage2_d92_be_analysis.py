from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_d92_be_analysis import (
    D92BEAnalysisError,
    analyze_d92_be_hard12,
)
from cvsrffi.stage2_d92_be_slim import D92_BE_ARMS


SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARMS = ("FULL", "B0", "E0", "B0E0")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_payload(*, arm: str, passing: bool) -> dict:
    if arm == "B0E0":
        old_after = 0.61
        seen_new = 0.46
        h_value = 0.51 if passing else 0.50
        floor = 0.41
    else:
        old_after = 0.60
        seen_new = 0.45
        h_value = 0.50
        floor = 0.40
    before_by_scenario = {
        scene: {"old_acc": 0.70, "seen_new_acc": None, "h_old_new": None}
        for scene in SCENES
    }
    after_by_scenario = {
        scene: {
            "old_acc": old_after,
            "seen_new_acc": seen_new,
            "h_old_new": h_value,
        }
        for scene in SCENES
    }
    before_by_tx = {
        "old_0": {"role": "target_old", "accuracy": 0.70},
        "old_1": {"role": "target_old", "accuracy": 0.70},
    }
    after_by_tx = {
        "old_0": {"role": "target_old", "accuracy": old_after},
        "old_1": {"role": "target_old", "accuracy": old_after},
        "new_0": {"role": "target_new", "accuracy": seen_new},
    }
    return {
        "schema": "cvs.phase2.diag_cosine_dev_pair_score.v1",
        "candidate": D92_BE_ARMS[arm].candidate_id,
        "before": {
            "old_acc": 0.70,
            "seen_new_acc": None,
            "by_scenario": before_by_scenario,
            "by_tx": before_by_tx,
        },
        "after": {
            "old_acc": old_after,
            "seen_new_acc": seen_new,
            "h_old_new": h_value,
            "by_scenario": after_by_scenario,
            "by_tx": after_by_tx,
        },
        "old_forgetting_pp": 100.0 * (0.70 - old_after),
        "per_old_class_floor_after": floor,
        "query_truth_joined_only_after_immutable_predictions": True,
        "query_truth_fed_back_to_predictor": False,
    }


def _build_matrix(tmp_path: Path, *, passing: bool = True) -> Path:
    output = tmp_path / "matrix"
    lock_path = tmp_path / "method_lock.json"
    _write_json(
        lock_path,
        {
            "schema": "cvs.phase2.d92_be.method_lock.v1",
            "only_promotion_candidate": "B0E0",
            "strict_pareto_gate": {
                "mean_delta_h_min": 0.005,
                "nonnegative_delta_h_outer_min": 8,
                "mean_delta_old_balanced_min": 0.0,
                "mean_delta_seen_new_min": 0.0,
                "mean_delta_old_floor_min": 0.0,
                "mean_delta_forgetting_max": 0.0,
                "median_wall_reduction_min": 0.4,
                "median_incremental_peak_reduction_min": 0.4,
                "query_cost_increase_max": 0.0,
            },
        },
    )
    k_values = [1, 1, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10]
    selected_rows = []
    jobs = []
    for outer_index, k_shot in enumerate(k_values):
        outer_key = f"outer_{outer_index:02d}"
        role = "liveness" if k_shot == 1 else "performance"
        selected_rows.append(
            {
                "outer_key": outer_key,
                "outer_role": role,
                "k_shot": k_shot,
            }
        )
        for arm in ARMS:
            job_root = output / "jobs" / outer_key / arm
            before_path = job_root / "diag" / "before" / "prediction_artifact.npz"
            after_path = job_root / "diag" / "after" / "prediction_artifact.npz"
            before_path.parent.mkdir(parents=True, exist_ok=True)
            after_path.parent.mkdir(parents=True, exist_ok=True)
            common_predictions = np.asarray(["old_0", "new_0"])
            np.savez(
                before_path,
                query_tokens=np.asarray(["q0", "q1"]),
                scenarios=np.asarray([SCENES[0], SCENES[1]]),
                predicted_class_handles=common_predictions,
            )
            np.savez(
                after_path,
                query_tokens=np.asarray(["q0", "q1"]),
                scenarios=np.asarray([SCENES[0], SCENES[1]]),
                predicted_class_handles=common_predictions,
            )
            score_path = job_root / "scorer" / "diag_cosine_score.json"
            score = _score_payload(arm=arm, passing=passing)
            score["before_prediction_sha256"] = _sha256(before_path)
            score["after_prediction_sha256"] = _sha256(after_path)
            _write_json(score_path, score)
            expected_fit = 8 * (k_shot + 1) if arm in {"FULL", "B0"} else 4 + 4 * k_shot
            if k_shot <= 2:
                expected_fit = 8 * (k_shot + 1)
            wall = 50 if arm in {"E0", "B0E0"} and k_shot > 2 else 100
            peak = 500 if arm in {"E0", "B0E0"} and k_shot > 2 else 1000
            fit_rows = [
                {
                    "scenario": scene,
                    "after_total_component_fit_count": expected_fit,
                    "after_registration_resource": {
                        "registration_wall_time_ns": wall,
                        "registration_incremental_peak_working_set_bytes": peak,
                    },
                    "query_macs": 7488,
                }
                for scene in SCENES
            ]
            _write_json(job_root / "diag" / "after" / "fit_audit.json", fit_rows)
            receipt = {
                "schema": "cvs.phase2.d92_be_hard12.job_receipt.v1",
                "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE",
                "job_id": f"{outer_key}__arm_{arm.lower()}",
                "outer_key": outer_key,
                "arm_id": arm,
                "candidate": D92_BE_ARMS[arm].candidate_id,
                "before_prediction_sha256": _sha256(before_path),
                "after_prediction_sha256": _sha256(after_path),
                "score_sha256": _sha256(score_path),
                "truth_sidecar_exposed_to_predictor": False,
                "query_truth_joined_only_after_immutable_predictions": True,
                "query_truth_fed_back_to_predictor": False,
            }
            _write_json(job_root / "job_receipt.json", receipt)
            jobs.append(
                {
                    "job_id": receipt["job_id"],
                    "outer_key": outer_key,
                    "outer_role": role,
                    "k_shot": k_shot,
                    "arm_id": arm,
                    "candidate": D92_BE_ARMS[arm].candidate_id,
                    "output_root": str(job_root),
                    "scenarios": list(SCENES),
                }
            )
    for shard in range(8):
        _write_json(
            output / "summaries" / f"shard_{shard}.json",
            {"status": "PASS", "shard_index": shard, "performance_result_allowed": True},
        )
    manifest = {
        "schema": "cvs.phase2.d92_be_hard12.matrix.v1",
        "status": "FROZEN_DEVELOPMENT_MATRIX",
        "claim_scope": "DEVELOPMENT_ONLY_COVERAGE_CONSTRAINED_STRESS_SCREEN",
        "protocol_schema": "p2_min_v1",
        "method_lock": str(lock_path),
        "method_lock_sha256": _sha256(lock_path),
        "output_root": str(output),
        "shard_count": 8,
        "outer_count": 12,
        "performance_outer_count": 10,
        "liveness_outer_count": 2,
        "job_count": 48,
        "selected_rows": selected_rows,
        "jobs": jobs,
    }
    manifest_path = output / "matrix_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_analysis_promotes_only_when_all_strict_pareto_gates_pass(tmp_path: Path) -> None:
    result = analyze_d92_be_hard12(_build_matrix(tmp_path))
    assert result["verdict"] == "STRICT_PARETO_PROMOTE_TO_TARGET125_CONFIRMATION"
    assert result["all_gates_pass"] is True
    assert result["performance_outer_count"] == 10
    assert result["aggregate"]["B0E0"]["mean_h_old_new"] == pytest.approx(0.51)
    assert result["paired"]["B0E0_minus_FULL"]["mean_delta_h"] == pytest.approx(0.01)
    assert result["paired"]["B0E0_minus_FULL"]["median_wall_reduction"] == pytest.approx(0.5)
    assert result["gates"]["fit_count_exact"]["passed"] is True
    assert result["gates"]["da0_reg0_prediction_exact"]["passed"] is True
    assert result["gates"]["k1_exact_alias_liveness"]["passed"] is True


def test_analysis_rejects_when_h_gain_misses_registered_threshold(tmp_path: Path) -> None:
    result = analyze_d92_be_hard12(_build_matrix(tmp_path, passing=False))
    assert result["verdict"] == "NO_STRICT_PARETO_PROMOTION"
    assert result["all_gates_pass"] is False
    assert result["gates"]["mean_delta_h"]["passed"] is False


def test_analysis_rejects_cross_arm_da0_reg0_prediction_drift(tmp_path: Path) -> None:
    manifest_path = _build_matrix(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = next(row for row in manifest["jobs"] if row["outer_key"] == "outer_02" and row["arm_id"] == "B0E0")
    job_root = Path(job["output_root"])
    before_path = job_root / "diag" / "before" / "prediction_artifact.npz"
    np.savez(
        before_path,
        query_tokens=np.asarray(["q0", "q1"]),
        scenarios=np.asarray([SCENES[0], SCENES[1]]),
        predicted_class_handles=np.asarray(["old_1", "new_0"]),
    )
    score_path = job_root / "scorer" / "diag_cosine_score.json"
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score["before_prediction_sha256"] = _sha256(before_path)
    _write_json(score_path, score)
    receipt_path = job_root / "job_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["before_prediction_sha256"] = _sha256(before_path)
    receipt["score_sha256"] = _sha256(score_path)
    _write_json(receipt_path, receipt)
    with pytest.raises(D92BEAnalysisError, match="DA0_REG0 prediction drift"):
        analyze_d92_be_hard12(manifest_path)


def test_analysis_accepts_local_overrides_for_retrieved_remote_artifacts(tmp_path: Path) -> None:
    manifest_path = _build_matrix(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    local_run_root = Path(manifest["output_root"])
    local_method_lock = Path(manifest["method_lock"])
    manifest["output_root"] = "/remote/runs/d92_be"
    manifest["method_lock"] = "/remote/source/configs/method_lock.json"
    for job in manifest["jobs"]:
        job["output_root"] = f"/remote/runs/d92_be/jobs/{job['outer_key']}/{job['arm_id']}"
    _write_json(manifest_path, manifest)
    result = analyze_d92_be_hard12(
        manifest_path,
        run_root=local_run_root,
        method_lock_path=local_method_lock,
    )
    assert result["verdict"] == "STRICT_PARETO_PROMOTE_TO_TARGET125_CONFIRMATION"


def test_zero_mean_floor_delta_is_not_failed_by_binary_roundoff(tmp_path: Path) -> None:
    manifest_path = _build_matrix(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    performance = [
        row
        for row in manifest["selected_rows"]
        if row["outer_role"] == "performance"
    ]
    candidate_floors = [
        0.41666666666666663,
        0.38333333333333336,
        *([0.4] * 8),
    ]
    for selected, floor in zip(performance, candidate_floors):
        job = next(
            row
            for row in manifest["jobs"]
            if row["outer_key"] == selected["outer_key"] and row["arm_id"] == "B0E0"
        )
        job_root = Path(job["output_root"])
        score_path = job_root / "scorer" / "diag_cosine_score.json"
        score = json.loads(score_path.read_text(encoding="utf-8"))
        score["per_old_class_floor_after"] = floor
        _write_json(score_path, score)
        receipt_path = job_root / "job_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["score_sha256"] = _sha256(score_path)
        _write_json(receipt_path, receipt)
    result = analyze_d92_be_hard12(manifest_path)
    gate = result["gates"]["mean_delta_old_floor"]
    assert gate["observed"] == pytest.approx(0.0, abs=1e-15)
    assert gate["passed"] is True
