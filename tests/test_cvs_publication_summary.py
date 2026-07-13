import pytest

from paper_reproduction.scripts.summarize_cvs_publication_stage2 import (
    SCENARIOS,
    _run_metric,
    summarize_groups,
)


def test_summary_keeps_joint_run_rows_and_reports_uncertainty() -> None:
    rows = [
        {"phase": "stage2b", "method": "x", "k_shot": 1, "accuracy": 0.4},
        {"phase": "stage2b", "method": "x", "k_shot": 1, "accuracy": 0.6},
    ]
    summary = summarize_groups(
        rows,
        group_fields=("phase", "method", "k_shot"),
        metric_keys=("accuracy",),
    )
    assert len(summary) == 1
    assert summary[0]["accuracy_n"] == 2
    assert summary[0]["accuracy_mean"] == 0.5
    assert summary[0]["accuracy_std"] > 0
    assert summary[0]["accuracy_ci95_low"] < 0.5 < summary[0]["accuracy_ci95_high"]


def test_run_metric_uses_scenario_fallback_for_missing_cvs_aggregate() -> None:
    scenario_metrics = {
        scenario: {"old_acc_before_increment": value}
        for scenario, value in zip(SCENARIOS, (0.6, 0.7, 0.8))
    }
    value, source = _run_metric(
        {"metrics": {}},
        scenario_metrics,
        "old_acc_before_increment",
        run_dir="example",
    )
    assert value == pytest.approx(0.7)
    assert source == "scenario_mean_fallback"


def test_run_metric_prefers_explicit_payload_mean() -> None:
    scenario_metrics = {scenario: {"old_acc": 0.1} for scenario in SCENARIOS}
    value, source = _run_metric(
        {"metrics": {"old_acc_mean": 0.4}},
        scenario_metrics,
        "old_acc",
        run_dir="example",
    )
    assert value == pytest.approx(0.4)
    assert source == "payload_mean"
