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
    evaluate_component_fit_reduction_gate,
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
    # A legacy aggregate resource failure is conservatively treated as a hard
    # miss; the split API below distinguishes hard failure from target-only
    # miss and returns REVISE_ONCE only for the latter.
    assert decide_verdict({**advance, "resources": False}) == "REJECT_ROUTE"
    reject = {**advance, "all_strict_pareto": False}
    assert decide_verdict(reject) == "REJECT_ROUTE"
    assert decide_verdict({
        "complete_artifact_closure": True,
        "performance_outer_closure": True,
        "all_strict_pareto": True,
        "all_magnitude": True,
    }) == "REJECT_ROUTE"


def test_split_resource_verdict_rejects_hard_miss_and_revises_target_only() -> None:
    base = {
        "complete_artifact_closure": True,
        "performance_outer_closure": True,
        "all_strict_pareto": True,
        "all_magnitude": True,
        "stability": True,
        "resource_integrity": True,
        "resource_hard": True,
        "resource_target": True,
        "compute_reduction": True,
    }
    assert decide_verdict({**base, "resource_hard": False}) == "REJECT_ROUTE"
    assert decide_verdict({**base, "resource_integrity": False}) == "REJECT_ROUTE"
    assert decide_verdict({**base, "compute_reduction": False}) == "REJECT_ROUTE"
    assert decide_verdict({**base, "resource_target": False}) == "REVISE_ONCE"


def test_component_fit_reduction_uses_frozen_d92_two_state_count_not_estimated_macs() -> None:
    result = evaluate_component_fit_reduction_gate([
        {"outer_key": "k5", "k_shot": 5, "fit_count": 4, "actual_fit_count": 2, "estimated_macs": 1},
        {"outer_key": "k10", "k_shot": 10, "fit_count": 4, "actual_fit_count": 2, "estimated_macs": 10**18},
        {"outer_key": "liveness", "outer_role": "liveness", "k_shot": 1, "fit_count": 3, "actual_fit_count": 3},
    ])
    assert result["passed"] is True
    assert len(result["rows"]) == 2
    assert result["rows"][0]["reduction_fraction_vs_d92"] == pytest.approx(1.0 - 4.0 / 48.0)
    assert result["rows"][1]["reduction_fraction_vs_d92"] == pytest.approx(1.0 - 4.0 / 88.0)

    failed = evaluate_component_fit_reduction_gate([
        {"outer_key": "k5", "k_shot": 5, "fit_count": 10, "actual_fit_count": 2},
    ])
    assert failed["passed"] is False


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
    assert result["hard_passed"] is True
    assert result["target_passed"] is True
    assert result["wall_ratio_p90"] == pytest.approx(1.2)
    assert result["wall_ratio_median"] == pytest.approx(1.2)
    assert result["peak_delta_p90_bytes"] == pytest.approx(400.0)
    assert result["peak_delta_max_bytes"] == pytest.approx(400.0)


def test_resource_gate_uses_ratio_median_and_separates_target_limits() -> None:
    result = evaluate_resource_gate(
        candidate_rows=[
            {"registration_wall_time_ns": 110_000_000.0, "registration_incremental_peak_working_set_bytes": 1200.0},
            {"registration_wall_time_ns": 120_000_000.0, "registration_incremental_peak_working_set_bytes": 1300.0},
            {"registration_wall_time_ns": 130_000_000.0, "registration_incremental_peak_working_set_bytes": 1400.0},
        ],
        baseline_rows=[
            {"registration_wall_time_ns": 100_000_000.0, "registration_incremental_peak_working_set_bytes": 1100.0},
            {"registration_wall_time_ns": 100_000_000.0, "registration_incremental_peak_working_set_bytes": 1100.0},
            {"registration_wall_time_ns": 100_000_000.0, "registration_incremental_peak_working_set_bytes": 1100.0},
        ],
        query_state_exact=True,
    )
    assert result["hard_passed"] is True
    assert result["target_passed"] is False
    assert result["wall_ratio_median"] == pytest.approx(1.2)

    hard_fail = evaluate_resource_gate(
        candidate_rows=[{"registration_wall_time_ns": 160_000_000.0, "registration_incremental_peak_working_set_bytes": 1200.0}],
        baseline_rows=[{"registration_wall_time_ns": 100_000_000.0, "registration_incremental_peak_working_set_bytes": 1100.0}],
        query_state_exact=True,
    )
    assert hard_fail["hard_passed"] is False
    assert hard_fail["target_passed"] is False

    peak_outlier = evaluate_resource_gate(
        candidate_rows=[
            {"registration_wall_time_ns": 100_000_000.0, "registration_incremental_peak_working_set_bytes": 1100.0},
            {"registration_wall_time_ns": 100_000_000.0, "registration_incremental_peak_working_set_bytes": 1100.0 + 512 * 1024 + 1},
            {"registration_wall_time_ns": 100_000_000.0, "registration_incremental_peak_working_set_bytes": 1100.0},
        ],
        baseline_rows=[
            {"registration_wall_time_ns": 100_000_000.0, "registration_incremental_peak_working_set_bytes": 1100.0},
            {"registration_wall_time_ns": 100_000_000.0, "registration_incremental_peak_working_set_bytes": 1100.0},
            {"registration_wall_time_ns": 100_000_000.0, "registration_incremental_peak_working_set_bytes": 1100.0},
        ],
        query_state_exact=True,
    )
    assert peak_outlier["hard_passed"] is False


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
            "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs": 0.25,
            "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum": 0.125,
            "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass": True,
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


@pytest.mark.parametrize(
    ("k_shot", "field", "value"),
    (
        (10, "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass", False),
        (10, "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum", 0.0),
        (10, "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs", 0.1),
        (1, "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum", 0.0),
    ),
)
def test_fit_resource_rejects_cross_group_quantum_receipt_drift(
    tmp_path: Path, k_shot: int, field: str, value: object
) -> None:
    active = k_shot > 2
    rows = []
    for scenario in SCENES:
        row = {
            "scenario": scenario,
            **{field_name: False for field_name in QUERY_ZERO_FIELDS},
            "after_total_component_fit_count": 4 if active else 3,
            "after_actual_component_inventory": {"actual_component_fit_count": 2 if active else 3},
            "query_macs": 123,
            "after_state_bytes": 456,
            "after_registered_d_mode_effective": "pareto_distill" if active else "d92_full_alias",
            "d92_e0d_pareto_distill_covariance_estimation_count": 1 if active else None,
            "d92_e0d_pareto_distill_robust_center_transform_count": 1 if active else None,
            "d92_e0d_pareto_distill_full_solve_count": 1 if active else None,
            "d92_e0d_pareto_distill_block_solve_count": 1 if active else None,
            "d92_e0d_pareto_distill_loo_fit_count": 0 if active else None,
            "d92_e0d_pareto_distill_fisher_fit_count": 0 if active else None,
            "d92_e0d_pareto_distill_deployed_support_constraints_pass": True if active else False,
            "d92_e0d_pareto_distill_deployed_full_head_byte_exact": False if active else True,
            "d92_e0d_pareto_distill_persistent_state_bytes_delta": 0 if active else None,
            "d92_e0d_pareto_distill_support_macs": 100 if active else None,
            "d92_e0d_pareto_distill_support_transient_bytes": 200 if active else None,
            "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs": 0.25 if active else None,
            "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum": 0.125 if active else None,
            "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass": True if active else None,
            "after_registration_resource": {
                "registration_wall_time_ns": 1000,
                "registration_incremental_peak_working_set_bytes": 2000,
            },
        }
        row[field] = value
        rows.append(row)
    path = tmp_path / "job" / "diag" / "after" / "fit_audit.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="quantum"):
        _fit_resource(tmp_path / "job", k_shot)
