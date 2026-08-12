from __future__ import annotations

from cvsrffi.stage2_d92_tpce_hard11_analysis import (
    EIGHT_PARETO_METRICS,
    compute_confusion_rates,
    compute_old_balanced_accuracy,
    compute_score_metrics,
    decide_verdict,
    evaluate_component_fit_reduction_gate,
    evaluate_resource_gate,
)


def test_analysis_exports_eight_metrics_and_confusion_sources() -> None:
    assert len(EIGHT_PARETO_METRICS) == 8
    score = {
        "before": {"old_acc": 0.6, "by_scenario": {"clear": {"query_count": 90}, "rain": {"query_count": 10}}, "by_tx": {f"o{i}": {"accuracy": 0.5, "role": "target_old"} for i in range(6)}},
        "after": {"by_scenario": {"clear": {"query_count": 100, "new_to_old_rate": 0.0, "old_to_new_rate": 0.0}, "rain": {"query_count": 110, "new_to_old_rate": 1.0, "old_to_new_rate": 1.0}}, "by_tx": {f"o{i}": {"accuracy": 0.5, "role": "target_old"} for i in range(6)}},
    }
    assert compute_confusion_rates(score)["new_to_old_rate"] == 100.0 / 110.0
    assert compute_old_balanced_accuracy(score["after"]["by_tx"]) == 0.5
    assert compute_score_metrics({**score, "after": {**score["after"], "h_old_new": 0.5, "old_acc": 0.5, "seen_new_acc": 0.5}})["old_floor"] == 0.5


def test_tpce_component_fit_proxy_uses_two_one_inventory() -> None:
    result = evaluate_component_fit_reduction_gate([
        {"outer_key": "k5", "k_shot": 5, "fit_count": 2, "actual_fit_count": 1},
        {"outer_key": "k10", "k_shot": 10, "fit_count": 2, "actual_fit_count": 1},
        {"outer_key": "k1", "outer_role": "liveness", "k_shot": 1, "fit_count": 3, "actual_fit_count": 3},
    ])
    assert result["passed"] is True
    assert all(row["actual_fit_count"] == 1 for row in result["rows"])


def test_verdict_keeps_hard_resource_failure_as_reject() -> None:
    base = {"complete_artifact_closure": True, "performance_outer_closure": True, "all_strict_pareto": True, "all_magnitude": True, "stability": True, "resource_integrity": True, "resource_hard": True, "resource_target": True, "compute_reduction": True}
    assert decide_verdict(base) == "ADVANCE_TO_TARGET125_CANDIDATE"
    assert decide_verdict({**base, "resource_hard": False}) == "REJECT_ROUTE"
    assert decide_verdict({**base, "resource_target": False}) == "REVISE_ONCE"


def test_resource_gate_uses_wall_p90_ratio_median_and_peak_delta() -> None:
    result = evaluate_resource_gate(
        [{"registration_wall_time_ns": 120.0, "registration_incremental_peak_working_set_bytes": 1500.0}],
        [{"registration_wall_time_ns": 100.0, "registration_incremental_peak_working_set_bytes": 1100.0}],
        query_state_exact=True,
    )
    assert result["hard_passed"] is True
    assert result["wall_ratio_median"] == 1.2
