from __future__ import annotations

import hashlib
import json
import copy
import csv
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi import stage2_d92_ccoc_hard9_k1_analysis as analysis  # noqa: E402


def test_analyzer_exports_frozen_verdicts() -> None:
    assert analysis.VERDICTS == (
        "REJECT_ROUTE",
        "REVISE_ONCE",
        "ADVANCE_TO_TARGET125_CANDIDATE",
    )


def _score(delta: float = 0.02) -> dict[str, object]:
    old = {f"tx_{i}": {"role": "target_old", "accuracy": 0.70 + delta} for i in range(6)}
    before_scene = {scene: {"query_count": 60} for scene in analysis.SCENES}
    after_scene = {
        scene: {
            "query_count": 80,
            "h_old_new": 0.70 + delta,
            "old_acc": 0.70 + delta,
            "seen_new_acc": 0.70 + delta,
            "new_to_old_rate": 0.10 - delta,
            "old_to_new_rate": 0.10 - delta,
        }
        for scene in analysis.SCENES
    }
    return {
        "candidate": analysis.CANDIDATE_ID,
        "truth_sidecar_sha256": "a" * 64,
        "before": {"old_acc": 0.70, "by_scenario": before_scene, "by_tx": old},
        "after": {
            "h_old_new": 0.70 + delta,
            "old_acc": 0.70 + delta,
            "seen_new_acc": 0.70 + delta,
            "by_tx": old,
            "by_scenario": after_scene,
            "query_macs": 3168,
            "after_state_bytes": 8583,
            "new_to_old_rate": 0.10 - delta,
            "old_to_new_rate": 0.10 - delta,
        },
        "old_forgetting_pp": -delta * 100.0,
    }


def test_score_metrics_and_strict_directions_are_explicitly_eight_metric() -> None:
    metrics = analysis.compute_score_metrics(_score())
    baseline = analysis.compute_score_metrics(_score(0.0))
    deltas = analysis.strict_pareto_deltas(metrics, baseline)
    assert metrics["old_class_count"] == 6
    assert all(deltas[name] > 0 for name in analysis.EIGHT_PARETO_METRICS[:5])
    assert deltas["average_forgetting"] < 0
    assert deltas["new_to_old_rate"] < 0
    assert deltas["old_to_new_rate"] < 0


def test_resource_gate_peak_is_absolute_and_512k_is_target_only() -> None:
    rows = [
        {
            "outer_key": "outer",
            "scenario": scene,
            "candidate_wall_ns": 100_000_000,
            "wall_ratio": 1.0,
            "candidate_peak_bytes": 729_088,
            "wall_hard_pass": True,
            "ratio_hard_pass": True,
            "peak_hard_pass": True,
            "wall_target_pass": True,
            "ratio_target_pass": True,
            "peak_target_pass": False,
            "query_state_exact": True,
        }
        for scene in analysis.SCENES
    ]
    result = analysis.evaluate_resource_gate(rows)
    assert result["hard_passed"] is True
    assert result["target_passed"] is False
    assert result["candidate_peak_max_bytes"] == 729_088


def test_decide_verdict_hard_failure_precedes_revision() -> None:
    base = {
        "complete_artifact_closure": True,
        "performance_outer_closure": True,
        "all_strict_pareto": True,
        "all_magnitude": False,
        "stability": True,
        "resource_integrity": True,
        "resource_hard": False,
        "resource_target": False,
    }
    assert analysis.decide_verdict(base) == "REJECT_ROUTE"
    base["resource_hard"] = True
    assert analysis.decide_verdict(base) == "REVISE_ONCE"
    base["all_magnitude"] = True
    base["resource_target"] = True
    assert analysis.decide_verdict(base) == "ADVANCE_TO_TARGET125_CANDIDATE"


def test_truth_binding_reads_actual_sidecar_bytes(tmp_path: Path) -> None:
    truth = tmp_path / "truth.json"
    truth.write_text("truth", encoding="utf-8")
    digest = hashlib.sha256(truth.read_bytes()).hexdigest()
    job = {"truth_sidecar": str(truth), "truth_sidecar_sha256": digest}
    receipt = {"truth_sidecar_sha256": digest}
    score = {"truth_sidecar_sha256": digest}
    assert analysis.validate_truth_binding(score, receipt, job, truth) == digest
    truth.write_text("drift", encoding="utf-8")
    import pytest

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError):
        analysis.validate_truth_binding(score, receipt, job, truth)


def test_output_package_is_exclusive_and_contains_all_seven_files(tmp_path: Path) -> None:
    result = {
        "schema": "cvs.phase2.d92_ccoc_hard9_k1.analysis.v1",
        "verdict": "REVISE_ONCE",
        "status": "ANALYZED",
        "gate_state": {},
        "gates": {},
        "aggregate": {},
        "paired_rows": [],
        "per_old_class_rows": [],
        "scenario_rows": [],
        "liveness_rows": [],
    }
    output_root = tmp_path / "analysis"
    paths = analysis.write_analysis_outputs(result, output_root)
    assert set(paths) == {
        "summary.json",
        "gates.json",
        "paired_rows.csv",
        "per_old_class_rows.csv",
        "scenario_rows.csv",
        "liveness_rows.csv",
        "analysis.md",
    }
    import pytest

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="overwrite"):
        analysis.write_analysis_outputs(result, output_root)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_commit(state_root: Path) -> None:
    members = []
    for name in ("execution_receipt.json", "fit_audit.json", "prediction_artifact.npz", "resource_audit.json"):
        path = state_root / name
        members.append({"relative_path": name, "sha256": _digest(path), "size_bytes": path.stat().st_size})
    _write_json(
        state_root / "COMMIT.json",
        {
            "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
            "members": members,
            "prediction_artifact_sha256": _digest(state_root / "prediction_artifact.npz"),
        },
    )


def _refresh_commit(state_root: Path) -> None:
    commit = json.loads((state_root / "COMMIT.json").read_text(encoding="utf-8"))
    members = []
    for name in ("execution_receipt.json", "fit_audit.json", "prediction_artifact.npz", "resource_audit.json"):
        path = state_root / name
        members.append({"relative_path": name, "sha256": _digest(path), "size_bytes": path.stat().st_size})
    commit["members"] = members
    commit["prediction_artifact_sha256"] = _digest(state_root / "prediction_artifact.npz")
    _write_json(state_root / "COMMIT.json", commit)


def _integration_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    from cvsrffi import stage2_d92_ccoc_hard9_k1 as matrix

    config = ROOT / "configs" / "stage2_d92_ccoc_hard9_k1_v1.json"
    lock_path = tmp_path / "method_lock.json"
    shutil.copyfile(config, lock_path)
    manifest = matrix.build_hard9_k1_manifest(config, require_package_files=False)
    run_root = tmp_path / "run"
    truth_root = tmp_path / "truth"
    for job in manifest["jobs"]:
        truth = truth_root / "jobs" / job["outer_key"] / "offline" / "scorer" / "truth_sidecar.json"
        truth.parent.mkdir(parents=True, exist_ok=True)
        truth.write_text("opaque truth", encoding="utf-8")
        job["truth_sidecar"] = str(truth)
        job["truth_sidecar_sha256"] = _digest(truth)
    manifest_path = tmp_path / "matrix_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = _digest(manifest_path)
    raw_root = Path(matrix.RAW_SCORE_ROOT)
    for job in manifest["jobs"]:
        outer = job["outer_key"]
        raw_path = raw_root / outer / "E0_FULL_ONLY" / "scorer" / "diag_cosine_score.json"
        baseline = json.loads(raw_path.read_text(encoding="utf-8"))
        candidate = copy.deepcopy(baseline)
        candidate["candidate"] = analysis.CANDIDATE_ID
        candidate["truth_sidecar_sha256"] = job["truth_sidecar_sha256"]
        candidate["before_prediction_sha256"] = "before-placeholder"
        candidate["after_prediction_sha256"] = "after-placeholder"
        candidate["query_truth_fed_back_to_predictor"] = False
        candidate["query_truth_joined_only_after_immutable_predictions"] = True
        candidate["old_forgetting_pp"] = float(baseline.get("old_forgetting_pp", 0.0)) - 2.0
        for field in ("h_old_new", "old_acc", "seen_new_acc"):
            candidate["after"][field] = min(0.99, float(candidate["after"][field]) + 0.05)
        for tx, row in candidate["after"]["by_tx"].items():
            if row.get("role") == "target_old":
                row["accuracy"] = float(row["accuracy"]) if float(row["accuracy"]) >= 0.95 else min(0.99, float(row["accuracy"]) + 0.05)
        for state_name in ("before", "after"):
            candidate[state_name]["by_tx"] = copy.deepcopy(candidate["before"]["by_tx"] if state_name == "before" else candidate["after"]["by_tx"])
        for scene in analysis.SCENES:
            row = candidate["after"]["by_scenario"][scene]
            for field in ("h_old_new", "old_acc", "seen_new_acc"):
                row[field] = min(0.99, float(row[field]) + 0.05)
            for field in ("new_to_old_rate", "old_to_new_rate"):
                row[field] = max(0.001, float(row[field]) - 0.02)
        job_root = run_root / "jobs" / outer / analysis.ARM_ID
        fit_rows = []
        for scene in analysis.SCENES:
            active = int(job["k_shot"]) > 2
            prefix = "d92_e0d_ccoc_"
            row = {
                "scenario": scene,
                "arm_id": analysis.ARM_ID,
                "candidate_id": analysis.CANDIDATE_ID,
                "after_state_postprocess_mode": None,
                "after_total_component_fit_count": 2 if active else 3,
                "after_actual_component_inventory": {"actual_component_fit_count": 1 if active else 3},
                "after_registered_d_mode_effective": "ccoc_full" if active else "d92_full_alias",
                "registered_class_count": 6 + int(job["new_class_count"]),
                "query_macs": int(job["e0_resource"]["scenes"][scene]["query_macs"]),
                "after_state_bytes": int(job["e0_resource"]["scenes"][scene]["state_bytes"]),
                "after_registration_resource": {"registration_wall_time_ns": int(job["e0_resource"]["scenes"][scene]["registration_wall_time_ns"]), "registration_incremental_peak_working_set_bytes": 1000},
                prefix + "active": active,
                prefix + "fallback_active": False,
                prefix + "fallback_reason": None if active else matrix.FIT_GATE["k1_alias"],
                prefix + "candidate_attempt_fit_count": 1 if active else 0,
                prefix + "fallback_reference_fit_count": 0,
                prefix + "candidate_statistic_receipt_available": active,
                prefix + "paired_e0_codec_state_equal": None,
                prefix + "g0_eligible": active,
                prefix + "g0_block_reason": None if active else matrix.FIT_GATE["k1_alias"],
                prefix + "query_rows_used": 0,
            }
            for field in matrix.QUERY_ZERO_FIELDS:
                row[field] = False
                row["d92_e0d_" + field] = False
                row[prefix + field] = False
            fit_rows.append(row)
        for state_name in ("before", "after"):
            state_root = job_root / "diag" / state_name
            state_root.mkdir(parents=True, exist_ok=True)
            _write_json(state_root / "execution_receipt.json", {"state": state_name})
            _write_json(state_root / "fit_audit.json", fit_rows)
            (state_root / "prediction_artifact.npz").write_bytes((state_name + outer).encode("utf-8"))
            _write_json(state_root / "resource_audit.json", {})
            _make_commit(state_root)
        before_sha = _digest(job_root / "diag" / "before" / "prediction_artifact.npz")
        after_sha = _digest(job_root / "diag" / "after" / "prediction_artifact.npz")
        candidate["before_prediction_sha256"] = before_sha
        candidate["after_prediction_sha256"] = after_sha
        _write_json(job_root / "scorer" / "diag_cosine_score.json", candidate)
        score_sha = _digest(job_root / "scorer" / "diag_cosine_score.json")
        binding_path = job_root / "score_binding.json"
        _write_json(binding_path, {
            "schema": "cvs.phase2.d92_ccoc_hard9_k1.score_binding.v1",
            "job_id": job["job_id"],
            "outer_key": outer,
            "outer_role": job["outer_role"],
            "arm_id": analysis.ARM_ID,
            "candidate": analysis.CANDIDATE_ID,
            "matrix_manifest_sha256": manifest_sha,
            "method_lock_sha256": manifest["method_lock_sha256"],
            "truth_sidecar": str(job["truth_sidecar"]),
            "truth_sidecar_sha256": job["truth_sidecar_sha256"],
            "before_prediction_sha256": before_sha,
            "after_prediction_sha256": after_sha,
            "score_command": ["python", "score"],
            "performance_result_allowed": False,
        })
        binding_sha = _digest(binding_path)
        score_evidence = {
            "job_id": job["job_id"],
            "outer_key": outer,
            "arm_id": analysis.ARM_ID,
            "candidate": analysis.CANDIDATE_ID,
            "matrix_manifest_sha256": manifest_sha,
            "method_lock_sha256": manifest["method_lock_sha256"],
            "score_artifact_sha256": score_sha,
            "truth_sidecar_sha256": job["truth_sidecar_sha256"],
            "before_prediction_sha256": before_sha,
            "after_prediction_sha256": after_sha,
        }
        _write_json(job_root / "job_receipt.json", {
            "schema": matrix.JOB_RECEIPT_SCHEMA,
            "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE",
            "job_id": job["job_id"], "outer_key": outer, "outer_role": job["outer_role"], "k_shot": job["k_shot"],
            "arm_id": analysis.ARM_ID, "candidate": analysis.CANDIDATE_ID,
            "matrix_manifest_sha256": manifest_sha, "method_lock_sha256": manifest["method_lock_sha256"], "selection_sha256": matrix.CANONICAL_SELECTION_SHA256,
            "before_prediction_sha256": before_sha, "after_prediction_sha256": after_sha, "score_sha256": score_sha,
            "truth_sidecar_sha256": job["truth_sidecar_sha256"],
            "truth_sidecar_sha256_before_score": job["truth_sidecar_sha256"],
            "truth_sidecar_sha256_after_score": job["truth_sidecar_sha256"],
            "score_binding": str(binding_path),
            "score_binding_sha256": binding_sha,
            "score_evidence": score_evidence,
            "truth_sidecar_exposed_to_predictor": False,
            "query_truth_joined_only_after_immutable_predictions": True, "query_truth_fed_back_to_predictor": False,
            "prediction_and_scorer_processes_isolated": True, "fresh_run_retry_authorized": False,
        })
    return manifest_path, lock_path, run_root, truth_root


def test_complete_fixture_has_9_performance_27_scene_and_60_old_rows(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )
    assert result["verdict"] == "ADVANCE_TO_TARGET125_CANDIDATE"
    assert len(result["paired_rows"]) == 10
    assert len(result["scenario_rows"]) == 30
    assert len([row for row in result["scenario_rows"] if row["outer_role"] == "performance"]) == 27
    assert len(result["per_old_class_rows"]) == 60
    assert len(result["liveness_rows"]) == 1


def test_single_scene_hard_peak_is_rejected_even_when_other_scenes_pass(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    job = json.loads(Path(manifest).read_text(encoding="utf-8"))
    outer = job["jobs"][0]["outer_key"]
    fit = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after" / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[1]["after_registration_resource"]["registration_incremental_peak_working_set_bytes"] = 1_048_577
    _write_json(fit, rows)
    _refresh_commit(fit.parent)

    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )

    assert result["verdict"] == "REJECT_ROUTE"
    assert result["gate_state"]["resource_hard"] is False


def test_baseline_order_is_joined_by_outer_key_not_position(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    source = Path(analysis.matrix.HISTORICAL_BASELINE_PATH)
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig", newline="")))
    rows[0], rows[-1] = rows[-1], rows[0]
    swapped = tmp_path / "paired_rows_swapped.csv"
    with swapped.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = analysis.analyze_d92_ccoc_hard9_k1(manifest, run_root=run_root, method_lock_path=lock, baseline_paired_rows_path=swapped, truth_sidecar_root=truth_root)
    assert result["verdict"] == "ADVANCE_TO_TARGET125_CANDIDATE"


def test_k1_cannot_enter_performance_aggregation(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    payload["jobs"][-1]["outer_role"] = "performance"
    _write_json(Path(manifest), payload)
    import pytest

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="K1"):
        analysis.analyze_d92_ccoc_hard9_k1(manifest, run_root=run_root, method_lock_path=lock, truth_sidecar_root=truth_root)


def test_duplicate_or_missing_scene_is_rejected_before_metric_aggregation(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    state_root = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after"
    fit = state_root / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[1]["scenario"] = rows[0]["scenario"]
    _write_json(fit, rows)
    _refresh_commit(state_root)

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="duplicate/missing"):
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )


def test_performance_fallback_receipt_is_rejected(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    state_root = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after"
    fit = state_root / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[0]["d92_e0d_ccoc_fallback_active"] = True
    _write_json(fit, rows)
    _refresh_commit(state_root)

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="active lifecycle"):
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )


def test_commit_and_truth_hash_drift_are_rejected_before_verdict(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    fit = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after" / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[0]["after_registration_resource"]["registration_wall_time_ns"] = 1
    _write_json(fit, rows)

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="commit member SHA drift"):
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )

    _refresh_commit(fit.parent)
    truth = truth_root / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json"
    truth.write_text("drifted truth", encoding="utf-8")
    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="truth sidecar hash binding drift"):
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )


def test_single_scene_wall_and_query_state_hard_failures_are_not_soft_revision(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    outer = payload["jobs"][0]["outer_key"]
    state_root = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after"
    fit = state_root / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[0]["after_registration_resource"]["registration_wall_time_ns"] = 150_000_001
    rows[1]["after_state_bytes"] += 1
    _write_json(fit, rows)
    _refresh_commit(state_root)
    result = analysis.analyze_d92_ccoc_hard9_k1(manifest, run_root=run_root, method_lock_path=lock, truth_sidecar_root=truth_root)
    assert result["verdict"] == "REJECT_ROUTE"
    assert result["gate_state"]["resource_hard"] is False


def test_truth_and_score_binding_drift_is_rejected(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    payload["jobs"][0]["truth_sidecar_sha256"] = "0" * 64
    _write_json(Path(manifest), payload)
    import pytest

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="matrix_manifest_sha256"):
        analysis.analyze_d92_ccoc_hard9_k1(manifest, run_root=run_root, method_lock_path=lock, truth_sidecar_root=truth_root)


def test_any_metric_tie_is_rejected_even_when_other_metrics_improve(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    outer = payload["jobs"][0]["outer_key"]
    raw = json.loads((Path(analysis.matrix.RAW_SCORE_ROOT) / outer / "E0_FULL_ONLY" / "scorer" / "diag_cosine_score.json").read_text(encoding="utf-8"))
    score_path = run_root / "jobs" / outer / analysis.ARM_ID / "scorer" / "diag_cosine_score.json"
    candidate = json.loads(score_path.read_text(encoding="utf-8"))
    candidate["after"]["h_old_new"] = raw["after"]["h_old_new"]
    _write_json(score_path, candidate)
    receipt_path = score_path.parent.parent / "job_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["score_sha256"] = _digest(score_path)
    receipt["score_evidence"]["score_artifact_sha256"] = receipt["score_sha256"]
    _write_json(receipt_path, receipt)
    result = analysis.analyze_d92_ccoc_hard9_k1(manifest, run_root=run_root, method_lock_path=lock, truth_sidecar_root=truth_root)
    assert result["verdict"] == "REJECT_ROUTE"
    assert result["gate_state"]["all_strict_pareto"] is False


@pytest.mark.parametrize("metric", analysis.EIGHT_PARETO_METRICS)
def test_each_eight_metric_tie_rejects_the_route(tmp_path: Path, metric: str) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    raw = json.loads(
        (
            Path(analysis.matrix.RAW_SCORE_ROOT)
            / outer
            / "E0_FULL_ONLY"
            / "scorer"
            / "diag_cosine_score.json"
        ).read_text(encoding="utf-8")
    )
    score_path = run_root / "jobs" / outer / analysis.ARM_ID / "scorer" / "diag_cosine_score.json"
    candidate = json.loads(score_path.read_text(encoding="utf-8"))
    if metric == "h_old_new":
        candidate["after"]["h_old_new"] = raw["after"]["h_old_new"]
    elif metric == "old_balanced_accuracy":
        for tx, row in candidate["after"]["by_tx"].items():
            if row.get("role") == "target_old":
                row["accuracy"] = raw["after"]["by_tx"][tx]["accuracy"]
    elif metric == "c_old_acc":
        candidate["after"]["old_acc"] = raw["after"]["old_acc"]
    elif metric == "old_floor":
        old_tx = min(
            (
                (tx, row["accuracy"])
                for tx, row in raw["after"]["by_tx"].items()
                if row.get("role") == "target_old"
            ),
            key=lambda item: item[1],
        )[0]
        candidate["after"]["by_tx"][old_tx]["accuracy"] = raw["after"]["by_tx"][old_tx]["accuracy"]
    elif metric == "seen_new_acc":
        candidate["after"]["seen_new_acc"] = raw["after"]["seen_new_acc"]
    elif metric == "average_forgetting":
        candidate["old_forgetting_pp"] = raw["old_forgetting_pp"]
    else:
        for scene in analysis.SCENES:
            candidate["after"]["by_scenario"][scene][metric] = raw["after"]["by_scenario"][scene][metric]
    _write_json(score_path, candidate)
    receipt_path = score_path.parent.parent / "job_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["score_sha256"] = _digest(score_path)
    receipt["score_evidence"]["score_artifact_sha256"] = receipt["score_sha256"]
    _write_json(receipt_path, receipt)

    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )

    assert result["verdict"] == "REJECT_ROUTE"
    assert result["gate_state"]["all_strict_pareto"] is False


def test_paired_e0_row_is_bound_to_its_same_outer_raw_score(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    outer = payload["jobs"][0]["outer_key"]
    source = Path(analysis.matrix.HISTORICAL_BASELINE_PATH)
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig", newline="")))
    for row in rows:
        if row["outer_key"] == outer:
            row["candidate_h_old_new"] = str(float(row["candidate_h_old_new"]) + 0.1)
            break
    else:  # pragma: no cover - frozen evidence must contain every selected outer
        raise AssertionError("selected outer missing from frozen paired evidence")
    paired = tmp_path / "paired_rows_drift.csv"
    with paired.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="paired E0"):
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            baseline_paired_rows_path=paired,
            truth_sidecar_root=truth_root,
        )


def test_score_binding_evidence_is_required_for_truth_last_closure(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    receipt_path = run_root / "jobs" / outer / analysis.ARM_ID / "job_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for field in ("score_binding", "score_binding_sha256", "score_evidence"):
        receipt.pop(field)
    _write_json(receipt_path, receipt)

    with pytest.raises(analysis.D92CCOCHard9K1AnalysisError, match="score binding"):
        analysis.analyze_d92_ccoc_hard9_k1(
            manifest,
            run_root=run_root,
            method_lock_path=lock,
            truth_sidecar_root=truth_root,
        )


def test_729088_peak_is_target_only_failure_and_revises_once(tmp_path: Path) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    outer = json.loads(Path(manifest).read_text(encoding="utf-8"))["jobs"][0]["outer_key"]
    state_root = run_root / "jobs" / outer / analysis.ARM_ID / "diag" / "after"
    fit = state_root / "fit_audit.json"
    rows = json.loads(fit.read_text(encoding="utf-8"))
    rows[0]["after_registration_resource"]["registration_incremental_peak_working_set_bytes"] = 729_088
    _write_json(fit, rows)
    _refresh_commit(state_root)

    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )

    assert result["verdict"] == "REVISE_ONCE"
    assert result["gate_state"]["resource_hard"] is True
    assert result["gate_state"]["resource_target"] is False


def test_markdown_reports_observed_reg0_old_metrics_and_na_reg0_new_metrics(
    tmp_path: Path,
) -> None:
    manifest, lock, run_root, truth_root = _integration_fixture(tmp_path)
    result = analysis.analyze_d92_ccoc_hard9_k1(
        manifest,
        run_root=run_root,
        method_lock_path=lock,
        truth_sidecar_root=truth_root,
    )

    markdown = analysis.render_analysis_markdown(result)

    assert "| DA0_REG0 | N/A | N/A | N/A | N/A |" in markdown
    assert (
        "| DA1_REG0 | "
        f"{result['aggregate']['candidate_mean_da1_reg0_old_acc']} | "
        f"{result['aggregate']['candidate_mean_da1_reg0_old_floor']} | N/A | N/A |"
    ) in markdown
    assert "| DA0_REG1 | N/A | N/A | N/A | N/A |" in markdown
    assert "| DA1_REG1 |" in markdown
