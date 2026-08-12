from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvsrffi.stage2_d92_newguard_hard11_analysis import (
    EIGHT_PARETO_METRICS,
    HISTORICAL_BASELINE_SHA256,
    HISTORICAL_PER_OLD_CLASS_SHA256,
    compute_old_balanced_accuracy,
    compute_score_metrics,
    decide_verdict,
    strict_pareto_deltas,
)


def _score() -> dict[str, object]:
    old = {f"old-{i}": {"accuracy": value, "count": 60, "role": "target_old"} for i, value in enumerate((0.50, 0.60, 0.70, 0.80, 0.90, 1.0))}
    new = {"new": {"accuracy": 0.75, "count": 360, "role": "target_new"}}
    return {
        "schema": "cvs.phase2.diag_cosine_dev_pair_score.v1",
        "candidate": "d92_e0_full_bidirectional_newguard_maxmin",
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
    reject = {**advance, "all_strict_pareto": False}
    assert decide_verdict(reject) == "REJECT_ROUTE"


def test_frozen_baseline_hashes_are_exposed() -> None:
    assert HISTORICAL_BASELINE_SHA256 == "6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a"
    assert HISTORICAL_PER_OLD_CLASS_SHA256 == "c0fc1e02b66b01d06da68bdd824594f3281e601d72b32726fa1e97a1e49788e6"
