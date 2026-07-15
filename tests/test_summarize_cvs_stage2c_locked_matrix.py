from __future__ import annotations

import json

import pytest

from paper_reproduction.scripts.summarize_cvs_stage2c_locked_matrix import (
    clustered_paired_bootstrap,
    matched_k5_drop_summary,
    validate_nested_protocol,
)


def _tiny_rows():
    rows = []
    for receiver in ("r1", "r2"):
        for seed in (11, 12):
            query = [f"{receiver}:{seed}:q{i}" for i in range(4)]
            for k in (1, 5, 10, 20):
                support = [
                    f"{receiver}:{seed}:c{class_id}:s{shot}"
                    for class_id in range(2)
                    for shot in range(k)
                ]
                rows.append(
                    {
                        "candidate_id": "locked",
                        "candidate_lock_sha256": "a" * 64,
                        "receiver": receiver,
                        "seed": seed,
                        "scenario": "leo_clear_weak",
                        "new_class_count": 1,
                        "k_shot": k,
                        "registered_class_count": 2,
                        "query_per_tx": 2,
                        "old_tx_labels_json": json.dumps(["old"]),
                        "new_tx_labels_json": json.dumps(["new"]),
                        "support_query_view": "leo_weak_only",
                        "clean_support_query_rows": 0,
                        "support_ids_json": json.dumps(support),
                        "query_ids_json": json.dumps(query),
                    }
                )
    return rows


def test_nested_protocol_accepts_complete_role_free_cross_k_matrix() -> None:
    audit = validate_nested_protocol(
        _tiny_rows(),
        expected_receivers=("r1", "r2"),
        expected_scenarios=("leo_clear_weak",),
        expected_new_counts=(1,),
        expected_k=(1, 5, 10, 20),
        minimum_seeds=2,
    )
    assert audit["row_count"] == 16
    assert audit["nested_support_pass"] is True
    assert audit["query_identity_lock_pass"] is True


def test_nested_protocol_rejects_query_drift() -> None:
    rows = _tiny_rows()
    rows[-1]["query_ids_json"] = json.dumps(["drift0", "drift1", "drift2", "drift3"])
    with pytest.raises(ValueError, match="query IDs drift"):
        validate_nested_protocol(
            rows,
            expected_receivers=("r1", "r2"),
            expected_scenarios=("leo_clear_weak",),
            expected_new_counts=(1,),
            expected_k=(1, 5, 10, 20),
            minimum_seeds=2,
        )


def test_clustered_bootstrap_reports_positive_k1_delta() -> None:
    rows = []
    for receiver in ("r1", "r2"):
        for seed in (1, 2, 3):
            for index in range(20):
                rows.append(
                    {
                        "receiver": receiver,
                        "seed": seed,
                        "k_shot": 1,
                        "evaluation_role": "target_old",
                        "candidate_correct": 1,
                        "direct_correct": 0 if index < 4 else 1,
                    }
                )
    result = clustered_paired_bootstrap(rows, repetitions=1000, seed=7)
    assert result["delta"] == pytest.approx(0.2)
    assert result["ci95_lower"] > 0.0
    assert result["cluster_count"] == 6


def test_k5_drop_gate_uses_worst_exactly_matched_cell_not_mean() -> None:
    rows = []
    for receiver, k5_value in (("r1", 0.89), ("r2", 0.97)):
        for k_shot, value in ((5, k5_value), (10, 0.95)):
            rows.append(
                {
                    "receiver": receiver,
                    "seed": 1,
                    "scenario": "leo_clear_weak",
                    "new_class_count": 5,
                    "k_shot": k_shot,
                    "seen_new_acc": value,
                }
            )
    result = matched_k5_drop_summary(
        rows, new_class_count=5, metric="seen_new_acc"
    )
    assert result["mean_drop"] == pytest.approx(0.02)
    assert result["max_drop"] == pytest.approx(0.06)
