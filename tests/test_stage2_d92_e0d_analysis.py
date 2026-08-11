from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_d92_e0d_analysis import (
    D92E0DAnalysisError,
    analyze_d92_e0d_hard12v2,
)
from cvsrffi.stage2_d92_e0d_slim import D92_E0D_ARMS


SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARMS = (
    "D92_FULL",
    "E0_FUSION",
    "E0_FULL_ONLY",
    "E0_BLOCK_ONLY",
    "E0_FIXED50",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm_metrics(arm: str, *, candidate_h: float) -> tuple[float, float, float, float]:
    if arm == "E0_FULL_ONLY":
        return 0.61, 0.46, candidate_h, 0.41
    if arm == "E0_FUSION":
        return 0.60, 0.452, 0.502, 0.40
    if arm == "D92_FULL":
        return 0.60, 0.45, 0.50, 0.40
    return 0.595, 0.445, 0.495, 0.395


def _score_payload(*, arm: str, candidate_h: float, k_shot: int) -> dict:
    old_after, seen_new, h_value, floor = _arm_metrics(
        arm, candidate_h=candidate_h
    )
    if k_shot == 1:
        old_after, seen_new, h_value, floor = 0.60, 0.45, 0.50, 0.40
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
        "candidate": D92_E0D_ARMS[arm].candidate_id,
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


def _expected_fit_count(k_shot: int, arm: str) -> int:
    if k_shot <= 2:
        return 3
    if arm == "D92_FULL":
        return 8 * (k_shot + 1)
    if arm == "E0_FUSION":
        return 4 * (k_shot + 1)
    if arm == "E0_FIXED50":
        return 4
    return 2


def _resource_values(arm: str, *, k_shot: int) -> tuple[int, int]:
    if k_shot <= 2:
        return 100, 1000
    return {
        "D92_FULL": (100, 1000),
        "E0_FUSION": (60, 600),
        "E0_FULL_ONLY": (30, 500),
        "E0_BLOCK_ONLY": (25, 450),
        "E0_FIXED50": (35, 550),
    }[arm]


def _build_matrix(tmp_path: Path, *, candidate_h: float = 0.51) -> Path:
    output = tmp_path / "matrix"
    lock_path = tmp_path / "method_lock.json"
    _write_json(
        lock_path,
        {
            "schema": "cvs.phase2.d92_e0d.method_lock.v1",
            "only_promotion_candidate": "E0_FULL_ONLY",
            "strict_geometry_gate": {
                "mean_delta_h_vs_e0_fusion_min_exclusive": 0.0,
                "nonnegative_delta_h_vs_e0_fusion_outer_min": 8,
                "mean_delta_h_vs_d92_full_min": 0.005,
                "nonnegative_delta_h_vs_d92_full_outer_min": 8,
                "mean_delta_old_balanced_vs_d92_full_min": 0.0,
                "mean_delta_seen_new_vs_d92_full_min": 0.0,
                "mean_delta_old_floor_vs_d92_full_min": 0.0,
                "mean_delta_forgetting_vs_d92_full_max": 0.0,
                "median_wall_reduction_vs_e0_fusion_min": 0.4,
                "median_wall_reduction_vs_d92_full_min": 0.6,
                "median_incremental_peak_increase_vs_e0_fusion_max": 0.0,
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
            {"outer_key": outer_key, "outer_role": role, "k_shot": k_shot}
        )
        before_fingerprint = hashlib.sha256(f"before:{outer_key}".encode()).hexdigest()
        k1_after_fingerprint = hashlib.sha256(
            f"after:{outer_key}".encode()
        ).hexdigest()
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
            score = _score_payload(
                arm=arm, candidate_h=candidate_h, k_shot=k_shot
            )
            score["before_prediction_sha256"] = _sha256(before_path)
            score["after_prediction_sha256"] = _sha256(after_path)
            _write_json(score_path, score)
            wall, peak = _resource_values(arm, k_shot=k_shot)
            after_fingerprint = (
                k1_after_fingerprint
                if k_shot == 1
                else hashlib.sha256(f"after:{outer_key}:{arm}".encode()).hexdigest()
            )
            fit_rows = [
                {
                    "scenario": scene,
                    "after_total_component_fit_count": _expected_fit_count(
                        k_shot, arm
                    ),
                    "after_actual_component_inventory": {
                        "actual_component_fit_count": (
                            _expected_fit_count(k_shot, arm)
                            if k_shot <= 2
                            else _expected_fit_count(k_shot, arm) // 2
                        )
                    },
                    "after_registration_resource": {
                        "registration_wall_time_ns": wall,
                        "registration_incremental_peak_working_set_bytes": peak,
                    },
                    "query_macs": 7488,
                    "query_truth_access": False,
                    "query_fit_access": False,
                    "query_update_access": False,
                    "query_selection_access": False,
                    "query_role_oracle_access": False,
                    "query_class_quota_access": False,
                    "query_global_reassignment": False,
                    "before_state_fingerprint_sha256": before_fingerprint,
                    "after_state_fingerprint_sha256": after_fingerprint,
                }
                for scene in SCENES
            ]
            _write_json(job_root / "diag" / "after" / "fit_audit.json", fit_rows)
            receipt = {
                "schema": "cvs.phase2.d92_e0d_hard12v2.job_receipt.v1",
                "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE",
                "job_id": f"{outer_key}__arm_{arm.lower()}",
                "outer_key": outer_key,
                "arm_id": arm,
                "candidate": D92_E0D_ARMS[arm].candidate_id,
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
                    "candidate": D92_E0D_ARMS[arm].candidate_id,
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
        "schema": "cvs.phase2.d92_e0d_hard12v2.matrix.v1",
        "status": "FROZEN_DEVELOPMENT_MATRIX",
        "claim_scope": "DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN",
        "protocol_schema": "p2_min_v1",
        "selection_sha256": "2e3b3333a4a325bd0443a31065d3340d6a650a3e89620951a786637e6bce8d3a",
        "method_lock": str(lock_path),
        "method_lock_sha256": _sha256(lock_path),
        "output_root": str(output),
        "shard_count": 8,
        "outer_count": 12,
        "performance_outer_count": 10,
        "liveness_outer_count": 2,
        "job_count": 60,
        "selected_rows": selected_rows,
        "jobs": jobs,
    }
    manifest_path = output / "matrix_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_analysis_promotes_only_when_both_controls_and_resources_pass(
    tmp_path: Path,
) -> None:
    result = analyze_d92_e0d_hard12v2(_build_matrix(tmp_path))
    assert result["verdict"] == "D_GEOMETRY_PROMOTE_TO_TARGET125_CONFIRMATION"
    assert result["all_gates_pass"] is True
    assert result["job_count"] == 60
    assert result["aggregate"]["E0_FULL_ONLY"]["mean_h_old_new"] == pytest.approx(
        0.51
    )
    assert result["paired"]["E0_FULL_ONLY_minus_E0_FUSION"][
        "mean_delta_h"
    ] == pytest.approx(0.008)
    assert result["paired"]["E0_FULL_ONLY_minus_D92_FULL"][
        "mean_delta_h"
    ] == pytest.approx(0.01)
    assert result["gates"]["fit_count_exact"]["passed"] is True
    assert result["gates"]["da0_reg0_state_prediction_exact"]["passed"] is True
    assert result["gates"]["k1_state_prediction_exact_alias"]["passed"] is True
    assert result["gates"]["query_protocol_zero_access"]["passed"] is True


def test_analysis_rejects_when_full_only_does_not_beat_e0_fusion(
    tmp_path: Path,
) -> None:
    result = analyze_d92_e0d_hard12v2(
        _build_matrix(tmp_path, candidate_h=0.502)
    )
    assert result["verdict"] == "NO_D_GEOMETRY_PROMOTION"
    assert result["gates"]["mean_delta_h_vs_e0_fusion"]["passed"] is False


def test_analysis_rejects_cross_arm_before_state_drift(tmp_path: Path) -> None:
    manifest_path = _build_matrix(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = next(
        row
        for row in manifest["jobs"]
        if row["outer_key"] == "outer_02" and row["arm_id"] == "E0_FIXED50"
    )
    fit_path = Path(job["output_root"]) / "diag" / "after" / "fit_audit.json"
    fit_rows = json.loads(fit_path.read_text(encoding="utf-8"))
    fit_rows[0]["before_state_fingerprint_sha256"] = "0" * 64
    _write_json(fit_path, fit_rows)
    with pytest.raises(D92E0DAnalysisError, match="DA0_REG0 state drift"):
        analyze_d92_e0d_hard12v2(manifest_path)


def test_analysis_rejects_nonzero_query_selection_audit(tmp_path: Path) -> None:
    manifest_path = _build_matrix(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = next(row for row in manifest["jobs"] if row["arm_id"] == "E0_FULL_ONLY")
    fit_path = Path(job["output_root"]) / "diag" / "after" / "fit_audit.json"
    fit_rows = json.loads(fit_path.read_text(encoding="utf-8"))
    fit_rows[0]["query_selection_access"] = True
    _write_json(fit_path, fit_rows)
    with pytest.raises(D92E0DAnalysisError, match="query protocol audit drift"):
        analyze_d92_e0d_hard12v2(manifest_path)


def test_analysis_rejects_hidden_component_inventory_drift(tmp_path: Path) -> None:
    manifest_path = _build_matrix(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = next(
        row
        for row in manifest["jobs"]
        if row["outer_role"] == "performance"
        and row["arm_id"] == "E0_FULL_ONLY"
    )
    fit_path = Path(job["output_root"]) / "diag" / "after" / "fit_audit.json"
    fit_rows = json.loads(fit_path.read_text(encoding="utf-8"))
    fit_rows[0]["after_actual_component_inventory"][
        "actual_component_fit_count"
    ] = 99
    _write_json(fit_path, fit_rows)
    with pytest.raises(D92E0DAnalysisError, match="actual component inventory drift"):
        analyze_d92_e0d_hard12v2(manifest_path)


def test_k1_alias_uses_the_real_three_component_inventory(tmp_path: Path) -> None:
    result = analyze_d92_e0d_hard12v2(_build_matrix(tmp_path))
    k1_rows = [row for row in result["paired_rows"] if row["vs_d92_full_k_shot"] == 1]
    assert k1_rows == []
    assert result["gates"]["fit_count_exact"]["passed"] is True


def test_analysis_accepts_local_overrides_for_retrieved_remote_artifacts(
    tmp_path: Path,
) -> None:
    manifest_path = _build_matrix(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    local_run_root = Path(manifest["output_root"])
    local_method_lock = Path(manifest["method_lock"])
    manifest["output_root"] = "/remote/runs/d92_e0d"
    manifest["method_lock"] = "/remote/source/configs/method_lock.json"
    for job in manifest["jobs"]:
        job["output_root"] = (
            f"/remote/runs/d92_e0d/jobs/{job['outer_key']}/{job['arm_id']}"
        )
    _write_json(manifest_path, manifest)
    result = analyze_d92_e0d_hard12v2(
        manifest_path,
        run_root=local_run_root,
        method_lock_path=local_method_lock,
    )
    assert result["verdict"] == "D_GEOMETRY_PROMOTE_TO_TARGET125_CONFIRMATION"


def test_zero_guardrail_delta_is_not_failed_by_binary_roundoff(
    tmp_path: Path,
) -> None:
    manifest_path = _build_matrix(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    performance = [
        row for row in manifest["selected_rows"] if row["outer_role"] == "performance"
    ]
    candidate_floors = [0.41666666666666663, 0.38333333333333336, *([0.40] * 8)]
    for selected, floor in zip(performance, candidate_floors):
        job = next(
            row
            for row in manifest["jobs"]
            if row["outer_key"] == selected["outer_key"]
            and row["arm_id"] == "E0_FULL_ONLY"
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
    result = analyze_d92_e0d_hard12v2(manifest_path)
    gate = result["gates"]["mean_delta_old_floor_vs_d92_full"]
    assert gate["observed"] == pytest.approx(0.0, abs=1e-15)
    assert gate["passed"] is True
