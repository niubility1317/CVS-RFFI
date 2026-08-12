from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from cvsrffi.stage2_d92_pareto_distill_hard11_analysis import (
    EIGHT_PARETO_METRICS,
    HISTORICAL_BASELINE_SHA256,
    HISTORICAL_PER_OLD_CLASS_SHA256,
    _fit_resource,
    compute_confusion_rates,
    compute_old_balanced_accuracy,
    compute_score_metrics,
    decide_verdict,
    evaluate_resource_gate,
    validate_truth_binding,
    validate_per_old_class_join,
    strict_pareto_deltas,
)
from cvsrffi.stage2_d92_pareto_distill_hard11 import SCENES
from scripts.run_d92_pareto_distill_hard11 import QUERY_ZERO_FIELDS


def _score() -> dict[str, object]:
    old = {f"old-{i}": {"accuracy": value, "count": 60, "role": "target_old"} for i, value in enumerate((0.50, 0.60, 0.70, 0.80, 0.90, 1.0))}
    new = {"new": {"accuracy": 0.75, "count": 360, "role": "target_new"}}
    return {
        "schema": "cvs.phase2.diag_cosine_dev_pair_score.v1",
        "candidate": "d92_e0_full_block_pareto_distill",
        "before": {"old_acc": 0.95, "by_tx": old},
        "after": {
            "h_old_new": 0.8,
            "old_acc": 0.75,
            "seen_new_acc": 0.75,
            "new_to_old_rate": 0.1,
            "old_to_new_rate": 0.05,
            "by_tx": {**old, **new},
        },
        "before_prediction_sha256": "a" * 64,
        "after_prediction_sha256": "b" * 64,
        "query_truth_joined_only_after_immutable_predictions": True,
        "query_truth_fed_back_to_predictor": False,
    }


def test_old_balanced_accuracy_is_equal_weighted_over_six_old_classes() -> None:
    score = _score()
    expected = sum((0.50, 0.60, 0.70, 0.80, 0.90, 1.0)) / 6.0
    assert compute_old_balanced_accuracy(score["after"]["by_tx"]) == pytest.approx(expected)
    metrics = compute_score_metrics(score)
    assert metrics["old_balanced_accuracy"] == pytest.approx(expected)
    assert metrics["c_old_acc"] == pytest.approx(0.75)
    assert metrics["old_floor"] == pytest.approx(0.50)
    assert metrics["old_class_count"] == 6
    assert set(metrics["old_class_accuracy"]) == {f"old-{i}" for i in range(6)}


def test_eight_strict_pareto_directions_are_explicit() -> None:
    assert EIGHT_PARETO_METRICS == (
        "h_old_new",
        "old_balanced_accuracy",
        "c_old_acc",
        "old_floor",
        "seen_new_acc",
        "average_forgetting",
        "new_to_old_rate",
        "old_to_new_rate",
    )
    candidate = {metric: 1.0 for metric in EIGHT_PARETO_METRICS}
    baseline = {metric: 0.0 for metric in EIGHT_PARETO_METRICS}
    candidate["average_forgetting"] = -1.0
    deltas = strict_pareto_deltas(candidate, baseline)
    assert deltas["h_old_new"] > 0
    assert deltas["average_forgetting"] < 0


def test_three_verdict_branches_have_no_weighted_compensation() -> None:
    advance = {
        "complete_artifact_closure": True,
        "performance_outer_closure": True,
        "all_strict_pareto": True,
        "all_magnitude": True,
        "stability": True,
        "resources": True,
    }
    assert decide_verdict(advance) == "ADVANCE_TO_TARGET125_CANDIDATE"
    revise = {**advance, "all_magnitude": False}
    assert decide_verdict(revise) == "REVISE_ONCE"
    assert decide_verdict({**advance, "stability": False}) == "REVISE_ONCE"
    assert decide_verdict({**advance, "resources": False}) == "REVISE_ONCE"
    reject = {**advance, "all_strict_pareto": False}
    assert decide_verdict(reject) == "REJECT_ROUTE"
    assert decide_verdict({
        "complete_artifact_closure": True,
        "performance_outer_closure": True,
        "all_strict_pareto": True,
        "all_magnitude": True,
    }) == "REJECT_ROUTE"


def test_frozen_baseline_hashes_are_exposed() -> None:
    assert HISTORICAL_BASELINE_SHA256 == "6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a"
    assert HISTORICAL_PER_OLD_CLASS_SHA256 == "c0fc1e02b66b01d06da68bdd824594f3281e601d72b32726fa1e97a1e49788e6"


def test_confusion_rates_use_real_old_and_new_query_counts() -> None:
    score = {
        "before": {"by_scenario": {"clear": {"query_count": 90}, "rain": {"query_count": 10}}},
        "after": {"by_scenario": {
            "clear": {"query_count": 100, "new_to_old_rate": 0.0, "old_to_new_rate": 0.0},
            "rain": {"query_count": 110, "new_to_old_rate": 1.0, "old_to_new_rate": 1.0},
        }},
    }
    rates = compute_confusion_rates(score)
    assert rates["new_to_old_rate"] == pytest.approx(100.0 / 110.0)
    assert rates["old_to_new_rate"] == pytest.approx(10.0 / 100.0)


def test_per_old_class_join_rejects_missing_or_tx_drift_and_never_fills_candidate() -> None:
    raw = {"after": {"by_tx": {
        "old-a": {"role": "target_old", "accuracy": 0.8},
        "old-b": {"role": "target_old", "accuracy": 0.6},
    }}}
    rows = [
        {"outer_key": "row", "tx": "old-a", "candidate_accuracy": "0.8", "baseline_accuracy": "0.7"},
        {"outer_key": "row", "tx": "old-b", "candidate_accuracy": "0.6", "baseline_accuracy": "0.7"},
    ]
    with pytest.raises(ValueError, match="per-old"):
        validate_per_old_class_join(rows[:1], raw, outer_key="row")
    with pytest.raises(ValueError, match="per-old"):
        validate_per_old_class_join(rows[:-1] + [{**rows[-1], "tx": "old-c"}], raw, outer_key="row")
    joined = validate_per_old_class_join(rows, raw, outer_key="row")
    assert joined["old-a"]["candidate_accuracy"] == pytest.approx(0.8)
    assert joined["old-a"]["e0_accuracy"] == pytest.approx(0.8)


def test_resource_gate_compares_matched_peak_and_wall_ratio_not_state_bytes() -> None:
    result = evaluate_resource_gate(
        candidate_rows=[{"registration_wall_time_ns": 120.0, "registration_incremental_peak_working_set_bytes": 1500.0}],
        baseline_rows=[{"registration_wall_time_ns": 100.0, "registration_incremental_peak_working_set_bytes": 1100.0, "state_bytes": 1.0}],
        query_state_exact=True,
    )
    assert result["passed"] is True
    assert result["wall_ratio_p90"] == pytest.approx(1.2)
    assert result["peak_delta_p90_bytes"] == pytest.approx(400.0)


def test_truth_binding_requires_manifest_receipt_score_and_actual_hash(tmp_path: Path) -> None:
    truth = tmp_path / "truth_sidecar.json"
    truth.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(truth.read_bytes()).hexdigest()
    score = _score()
    score["truth_sidecar_sha256"] = digest
    receipt = {"truth_sidecar_sha256": digest}
    job = {"truth_sidecar": str(truth), "truth_sidecar_sha256": digest}
    validate_truth_binding(score, receipt, job, truth)
    truth.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="truth"):
        validate_truth_binding(score, receipt, job, truth)


def test_truth_binding_accepts_a_hash_exact_cross_platform_retrieval_root(
    tmp_path: Path,
) -> None:
    outer = "rx_7_7__seed_713106__k_10__new_5"
    truth = (
        tmp_path
        / "truth_sidecars"
        / "jobs"
        / outer
        / "offline"
        / "scorer"
        / "truth_sidecar.json"
    )
    truth.parent.mkdir(parents=True)
    truth.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(truth.read_bytes()).hexdigest()
    score = _score()
    score["truth_sidecar_sha256"] = digest
    receipt = {"truth_sidecar_sha256": digest}
    job = {
        "outer_key": outer,
        "truth_sidecar": (
            "/home/user/project/runs/d92_registration/jobs/"
            f"{outer}/offline/scorer/truth_sidecar.json"
        ),
        "truth_sidecar_sha256": digest,
    }

    validate_truth_binding(score, receipt, job, truth)


def test_fit_resource_accepts_three_scene_pareto_distill_inventory(tmp_path: Path) -> None:
    rows = []
    for scenario in SCENES:
        rows.append({
            "scenario": scenario,
            **{field: False for field in QUERY_ZERO_FIELDS},
            "after_total_component_fit_count": 4,
            "after_actual_component_inventory": {"actual_component_fit_count": 2},
            "query_macs": 123,
            "after_state_bytes": 456,
            "after_registered_d_mode_effective": "pareto_distill",
            "d92_e0d_pareto_distill_covariance_estimation_count": 1,
            "d92_e0d_pareto_distill_robust_center_transform_count": 1,
            "d92_e0d_pareto_distill_full_solve_count": 1,
            "d92_e0d_pareto_distill_block_solve_count": 1,
            "d92_e0d_pareto_distill_loo_fit_count": 0,
            "d92_e0d_pareto_distill_fisher_fit_count": 0,
            "d92_e0d_pareto_distill_deployed_support_constraints_pass": True,
            "d92_e0d_pareto_distill_deployed_full_head_byte_exact": False,
            "d92_e0d_pareto_distill_persistent_state_bytes_delta": 0,
            "d92_e0d_pareto_distill_support_macs": 100,
            "d92_e0d_pareto_distill_support_transient_bytes": 200,
            "after_registration_resource": {
                "registration_wall_time_ns": 1000,
                "registration_incremental_peak_working_set_bytes": 2000,
            },
        })
    path = tmp_path / "job" / "diag" / "after" / "fit_audit.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows), encoding="utf-8")

    result = _fit_resource(tmp_path / "job", 10)

    assert result["fit_count"] == 4
    assert result["actual_fit_count"] == 2
    assert result["registered_d_mode"] == "pareto_distill"
