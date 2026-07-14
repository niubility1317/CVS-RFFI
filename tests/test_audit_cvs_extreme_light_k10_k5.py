from __future__ import annotations

from paper_reproduction.scripts.audit_cvs_extreme_light_k10_k5 import (
    scenario_metrics,
    validate_nested_ids,
)


def test_k5_support_is_nested_and_query_is_identical():
    support10 = [f"target_old|old-a|rx|day|1|{index}" for index in range(10)]
    support10 += [f"target_new|new-a|rx|day|1|{index}" for index in range(10)]
    checks = validate_nested_ids(
        support10[:5] + support10[10:15],
        support10,
        ["query-a", "query-b"],
        ["query-a", "query-b"],
    )
    assert all(checks.values())


def test_scenario_metrics_include_old_floor():
    rows = [
        {"true_label": "old-a", "predicted_label": "old-a"},
        {"true_label": "old-a", "predicted_label": "old-a"},
        {"true_label": "old-b", "predicted_label": "old-a"},
        {"true_label": "old-b", "predicted_label": "old-b"},
        {"true_label": "new-a", "predicted_label": "new-a"},
        {"true_label": "new-a", "predicted_label": "new-a"},
    ]
    metrics = scenario_metrics(rows, old_labels={"old-a", "old-b"}, new_labels={"new-a"})
    assert metrics["old_acc"] == 0.75
    assert metrics["min_old_class_acc"] == 0.5
    assert metrics["seen_new_acc"] == 1.0
