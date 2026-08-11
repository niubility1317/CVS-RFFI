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
from cvsrffi.stage2_d92_e0ocf_hard12 import (
    CANONICAL_SELECTION_SHA256,
    build_hard12v3_manifest,
)


SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARMS = ("D92_FULL", "E0_FULL_ONLY", "E0_FIXED50", "E0_OCF25", "E0_OCF50")
CONTEXT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d131_d92_lite160_qtie_target125_20260804_r3\artifacts\prepared\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_e0ocf_5arm_hard12v3_v1.json")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_manifest(
    tmp_path: Path,
    *,
    ocf50_wins: bool = False,
    linux_manifest_paths: bool = False,
) -> Path:
    output = tmp_path / "run"
    lock_path = tmp_path / "method_lock.json"
    _write_json(
        lock_path,
        json.loads(METHOD_LOCK.read_text(encoding="utf-8")),
    )
    manifest = build_hard12v3_manifest(
        context_path=CONTEXT,
        method_lock_path=lock_path,
        output_root=output,
        require_package_files=False,
    )
    for job in manifest["jobs"]:
        for package in job["packages"].values():
            package["expected_seal_sha256"] = "a" * 64
    if linux_manifest_paths:
        manifest["context_path"] = "/srv/e0ocf/context.json"
        manifest["method_lock"] = "/srv/e0ocf/method-lock.json"
        manifest["output_root"] = "/srv/e0ocf/run"
        for job in manifest["jobs"]:
            job["output_root"] = (
                f"/srv/e0ocf/run/jobs/{job['outer_key']}/{job['arm_id']}"
            )
    manifest_path = output / "matrix_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)
    for job in manifest["jobs"]:
        k_shot = int(job["k_shot"])
        role = str(job["outer_role"])
        outer_key = str(job["outer_key"])
        arm = str(job["arm_id"])
        before_fp = hashlib.sha256(f"before:{outer_key}".encode()).hexdigest()
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
        ocf_active = arm in {"E0_OCF25", "E0_OCF50"} and k_shot > 2
        support_affine_macs = 2 * (6 * k_shot) * 6 * 288 if ocf_active else 0
        support_mix_macs = 5 * 6 * (288 + 1) if ocf_active else 0
        support_total_macs = support_affine_macs + support_mix_macs
        fit = [{
                "scenario": s,
                "after_total_component_fit_count": 3 if k_shot <= 2 else (8 * (k_shot + 1) if arm == "D92_FULL" else (2 if arm == "E0_FULL_ONLY" else 4)),
                "after_actual_component_inventory": {"actual_component_fit_count": 3 if k_shot <= 2 else (4 * (k_shot + 1) if arm == "D92_FULL" else (1 if arm == "E0_FULL_ONLY" else 2))},
                "after_registration_resource": {"registration_wall_time_ns": 100 if arm == "D92_FULL" else 20, "registration_incremental_peak_working_set_bytes": 1000 if arm == "D92_FULL" else 900},
                "query_macs": 7488,
                "state_bytes": 10000,
                "d92_e0d_ocf_active": ocf_active,
                "d92_e0d_ocf_lambda": (0.25 if arm == "E0_OCF25" else 0.50) if ocf_active else None,
                "query_truth_access": False, "query_fit_access": False, "query_update_access": False, "query_selection_access": False, "query_role_oracle_access": False, "query_class_quota_access": False, "query_global_reassignment": False,
                "d92_e0d_ocf_support_alignment_affine_macs_upper_bound": support_affine_macs,
                "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound": support_mix_macs,
                "d92_e0d_ocf_support_alignment_macs_upper_bound": support_total_macs,
                "d92_e0d_ocf_support_alignment_transient_bytes_upper_bound": 100 if ocf_active else 0,
                "before_state_fingerprint_sha256": before_fp,
                "after_state_fingerprint_sha256": before_fp if k_shot <= 2 else hashlib.sha256(f"after:{outer_key}:{arm}".encode()).hexdigest(),
            } for s in SCENES]
        _write_json(root / "diag" / "after" / "fit_audit.json", fit)
        receipt = {"schema": "cvs.phase2.d92_e0ocf_hard12v3.job_receipt.v1", "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE", "job_id": job["job_id"], "outer_key": outer_key, "outer_role": role, "k_shot": k_shot, "arm_id": arm, "candidate": D92_E0D_ARMS[arm].candidate_id, "role": job["role"], "matrix_manifest_sha256": manifest_sha256, "method_lock_sha256": manifest["method_lock_sha256"], "selection_sha256": CANONICAL_SELECTION_SHA256, "before_prediction_sha256": _sha256(before), "after_prediction_sha256": _sha256(after), "score_sha256": _sha256(score_path), "truth_sidecar_exposed_to_predictor": False, "query_truth_joined_only_after_immutable_predictions": True, "query_truth_fed_back_to_predictor": False}
        _write_json(root / "job_receipt.json", receipt)
    jobs_by_shard = {shard: [job["job_id"] for job in manifest["jobs"] if job["planned_shard_index"] == shard] for shard in range(8)}
    for shard, job_ids in jobs_by_shard.items():
        _write_json(output / "summaries" / f"shard_{shard}.json", {"schema": "cvs.phase2.d92_e0ocf_hard12v3.shard_summary.v1", "status": "PASS", "shard_index": shard, "selected_job_count": len(job_ids), "completed_job_count": len(job_ids), "failed_job_count": 0, "completed_job_ids": job_ids, "performance_result_allowed": True})
    return manifest_path


def test_analysis_reports_confusion_formula_and_excludes_ocf50(tmp_path: Path) -> None:
    result = analyze_d92_e0ocf_hard12v3(_make_manifest(tmp_path, ocf50_wins=True))
    row = result["paired_rows"][0]
    assert row["E0_OCF25_old_to_old_rate"] == pytest.approx(0.2)
    assert row["E0_OCF25_old_to_new_rate"] == pytest.approx(0.2)
    assert row["E0_OCF25_new_to_old_rate"] == pytest.approx(0.1)
    assert result["promotion_candidate"] == "E0_OCF25"
    assert result["verdict"] != "PROMOTE_E0_OCF50_TO_TARGET125_CONFIRMATION"
    assert result["aggregate"]["E0_OCF25"][
        "median_ocf_support_alignment_affine_macs"
    ] == 207360
    assert result["aggregate"]["E0_OCF25"][
        "median_ocf_support_alignment_contrast_mix_macs"
    ] == 8670
    assert result["aggregate"]["E0_OCF25"][
        "median_ocf_support_alignment_macs"
    ] == 216030


def test_analysis_accepts_linux_manifest_with_windows_local_overrides(
    tmp_path: Path,
) -> None:
    """Would fail if frozen POSIX paths were rebuilt with host path semantics."""

    manifest_path = _make_manifest(tmp_path, linux_manifest_paths=True)
    result = analyze_d92_e0ocf_hard12v3(
        manifest_path,
        run_root=manifest_path.parent,
        method_lock_path=tmp_path / "method_lock.json",
    )
    assert result["status"] == "ANALYZED"
    assert result["job_count"] == 60


def test_analysis_rejects_missing_complete_matrix(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["job_count"] = 59
    _write_json(path, manifest)
    with pytest.raises(D92E0OCFAnalysisError):
        analyze_d92_e0ocf_hard12v3(path)


def test_analysis_rejects_tampered_selected_outer_identity(tmp_path: Path) -> None:
    """Would fail if analysis bypassed the shared canonical manifest validator."""

    path = _make_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["selected_rows"][0]["receiver"] = "3-19"
    _write_json(path, manifest)
    with pytest.raises(D92E0OCFAnalysisError, match="manifest"):
        analyze_d92_e0ocf_hard12v3(path)


@pytest.mark.parametrize(
    "field",
    ("matrix_manifest_sha256", "method_lock_sha256", "selection_sha256"),
)
def test_analysis_rejects_tampered_job_receipt_identity_hash(
    tmp_path: Path, field: str
) -> None:
    """Would fail if a job receipt could detach from its frozen authorities."""

    path = _make_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    job = manifest["jobs"][0]
    receipt_path = Path(job["output_root"]) / "job_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = "0" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(D92E0OCFAnalysisError, match="receipt"):
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


@pytest.mark.parametrize(
    ("arm", "field"),
    (
        ("E0_OCF25", "d92_e0d_ocf_support_alignment_affine_macs_upper_bound"),
        ("E0_OCF25", "d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound"),
        ("E0_OCF25", "d92_e0d_ocf_support_alignment_macs_upper_bound"),
        ("E0_FULL_ONLY", "d92_e0d_ocf_support_alignment_affine_macs_upper_bound"),
    ),
)
def test_analysis_rejects_ocf_support_mac_component_drift(
    tmp_path: Path, arm: str, field: str
) -> None:
    """Would fail if formal analysis trusted unclosed OCF MAC components."""

    path = _make_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    job = next(
        row
        for row in manifest["jobs"]
        if row["arm_id"] == arm and row["k_shot"] == 5
    )
    fit_path = Path(job["output_root"]) / "diag" / "after" / "fit_audit.json"
    rows = json.loads(fit_path.read_text(encoding="utf-8"))
    rows[0][field] += 1
    _write_json(fit_path, rows)
    with pytest.raises(D92E0OCFAnalysisError, match="support alignment MAC"):
        analyze_d92_e0ocf_hard12v3(path)


def test_analysis_rejects_non_ocf_nonzero_support_cost(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    job = next(row for row in manifest["jobs"] if row["arm_id"] == "E0_FULL_ONLY" and row["k_shot"] == 5)
    fit_path = Path(job["output_root"]) / "diag" / "after" / "fit_audit.json"
    rows = json.loads(fit_path.read_text(encoding="utf-8"))
    rows[0]["d92_e0d_ocf_support_alignment_macs_upper_bound"] = 1
    _write_json(fit_path, rows)
    with pytest.raises(D92E0OCFAnalysisError, match="non-OCF support-side cost"):
        analyze_d92_e0ocf_hard12v3(path)


def test_analysis_rejects_non_ocf_nan_support_cost(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    job = next(row for row in manifest["jobs"] if row["arm_id"] == "E0_FULL_ONLY" and row["k_shot"] == 5)
    fit_path = Path(job["output_root"]) / "diag" / "after" / "fit_audit.json"
    rows = json.loads(fit_path.read_text(encoding="utf-8"))
    rows[0]["d92_e0d_ocf_support_alignment_transient_bytes_upper_bound"] = "NaN"
    _write_json(fit_path, rows)
    with pytest.raises(D92E0OCFAnalysisError):
        analyze_d92_e0ocf_hard12v3(path)


def test_analysis_rejects_redistributed_or_duplicate_shard_job_ids(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    summary_path = Path(manifest["output_root"]) / "summaries" / "shard_0.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summary["completed_job_ids"]) >= 2
    summary["completed_job_ids"][1] = summary["completed_job_ids"][0]
    _write_json(summary_path, summary)
    with pytest.raises(D92E0OCFAnalysisError, match="shard completed job IDs"):
        analyze_d92_e0ocf_hard12v3(path)
