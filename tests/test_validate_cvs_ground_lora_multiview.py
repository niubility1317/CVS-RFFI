from __future__ import annotations

import numpy as np
import pytest

from paper_reproduction.scripts.validate_cvs_ground_lora_multiview import (
    _expanded_indices,
    _fixed_metrics,
    build_locked_nested_k_source_scores,
    build_source_symmetric_head_lock,
    stratified_physical_split,
    validate_formal_scenarios,
    validate_receiver_holdout,
)


def test_source_receiver_holdout_must_be_disjoint() -> None:
    audit = validate_receiver_holdout("0,1,2,3,4,5", "6")
    assert audit["disjoint"] is True
    with pytest.raises(ValueError, match="overlap"):
        validate_receiver_holdout("0,1,2", "2,3")
    with pytest.raises(ValueError, match="formal target domain"):
        validate_receiver_holdout("0,1,2", "7")


def test_source_validation_scenarios_are_exactly_formal_leo_weak() -> None:
    assert validate_formal_scenarios(
        ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    )
    with pytest.raises(ValueError, match="exact formal leo_weak"):
        validate_formal_scenarios(
            ("leo_clear_weak", "leo_low_elev_weak", "legacy_clear")
        )


def test_stratified_split_keeps_each_class_on_both_sides() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    calibration, evaluation = stratified_physical_split(labels)
    assert set(calibration).isdisjoint(set(evaluation))
    assert set(labels[calibration]) == {0, 1}
    assert set(labels[evaluation]) == {0, 1}
    expanded = _expanded_indices(
        calibration, physical_count=len(labels), scenario_count=3
    )
    assert len(expanded) == 3 * len(calibration)
    assert int(expanded.max()) < 3 * len(labels)


def test_fixed_metrics_report_joint_class_floor() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scores = np.zeros((4, 5, 2), dtype=np.float32)
    scores[:, :, 0] = 1.0
    metrics = _fixed_metrics(scores, labels)
    assert metrics["fixed1"]["accuracy"] == pytest.approx(0.5)
    assert metrics["fixed1"]["min_class_accuracy"] == pytest.approx(0.0)


def test_source_head_lock_uses_three_base_views_and_no_target_rows() -> None:
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(3, dtype=np.int64), 4)
    physical_count = len(labels)
    blocks = []
    for scenario in range(3):
        values = rng.normal(0.0, 0.01, size=(physical_count, 5, 8)).astype(
            np.float32
        )
        values += np.eye(3, 8, dtype=np.float32)[labels, None, :]
        blocks.append(values)
    features = np.concatenate(blocks, axis=0)
    calibration, _ = stratified_physical_split(labels)
    lock = build_source_symmetric_head_lock(
        features,
        labels,
        calibration,
        physical_count=physical_count,
        scenario_count=3,
        source_mean=features[:, 0, :].mean(axis=0),
        source_std=np.maximum(features[:, 0, :].std(axis=0), 0.05),
    )
    assert lock["allowed_k"] == [1, 5, 10, 20]
    assert lock["support_receive_views_per_physical_sample"] == 3
    assert lock["target_support_used_for_selection"] is False
    assert lock["target_query_features_used"] is False
    assert {"use_alignment", "prototype_rule", "ridge"}.issubset(
        lock["selected"]
    )


def test_nested_k_source_scores_use_one_locked_head_rule_and_nested_support() -> None:
    rng = np.random.default_rng(17)
    labels = np.repeat(np.arange(2, dtype=np.int64), 44)
    physical_count = len(labels)
    blocks = []
    for _scenario in range(3):
        values = rng.normal(0.0, 0.01, size=(physical_count, 5, 8)).astype(
            np.float32
        )
        values += np.eye(2, 8, dtype=np.float32)[labels, None, :]
        blocks.append(values)
    features = np.concatenate(blocks, axis=0)
    calibration, evaluation = stratified_physical_split(labels)
    result = build_locked_nested_k_source_scores(
        features,
        labels,
        calibration,
        evaluation,
        physical_count=physical_count,
        scenario_count=3,
        selected={
            "use_alignment": False,
            "prototype_rule": "mean",
            "ridge": None,
        },
        source_mean=features[:, 0, :].mean(axis=0),
        source_std=np.maximum(features[:, 0, :].std(axis=0), 0.05),
    )
    assert result["calibration_scores"].shape[1:] == (5, 2)
    assert set(result["evaluation_scores_by_k"]) == {"1", "5", "10", "20"}
    support = result["support_indices_by_k"]
    assert set(support["1"]) < set(support["5"]) < set(support["10"]) < set(
        support["20"]
    )
    assert result["target_rows_used"] is False
