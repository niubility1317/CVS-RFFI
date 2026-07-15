from __future__ import annotations

import numpy as np

from paper_reproduction.cvs_aligned.adaptive_rxlight_tta import (
    AdaptiveTTAThresholds,
    apply_adaptive_rxlight_tta,
    apply_adaptive_rxlight_tta_lazy,
    calibrate_adaptive_rxlight_tta,
)


def _scores() -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(
        [
            [[5.0, 0.0], [4.8, 0.2], [4.7, 0.3], [4.9, 0.1], [4.6, 0.4]],
            [[1.1, 1.0], [3.0, 0.1], [2.8, 0.2], [2.9, 0.1], [2.7, 0.3]],
            [[1.1, 1.0], [0.1, 3.0], [0.2, 2.8], [3.2, 0.1], [3.1, 0.2]],
        ],
        dtype=np.float32,
    )
    return scores, np.asarray([0, 0, 0], dtype=np.int64)


def test_gate_stops_at_one_three_and_five_views() -> None:
    scores, _ = _scores()
    result = apply_adaptive_rxlight_tta(
        scores,
        AdaptiveTTAThresholds(
            base_stop_margin=1.0,
            shift3_stop_margin=0.5,
            shift3_max_disagreement=0.0,
        ),
    )
    assert result["view_budgets"].tolist() == [1, 3, 5]
    assert result["mean_backbone_forwards"] == 3.0
    assert result["trigger_rates"] == {
        "view1_rate": 1.0 / 3.0,
        "view3_rate": 1.0 / 3.0,
        "view5_rate": 1.0 / 3.0,
    }


def test_gate_is_invariant_to_other_query_rows() -> None:
    scores, _ = _scores()
    thresholds = AdaptiveTTAThresholds(1.0, 0.5, 0.0)
    original = apply_adaptive_rxlight_tta(scores, thresholds)
    augmented = apply_adaptive_rxlight_tta(
        np.concatenate([scores, scores[::-1] * 0.7], axis=0), thresholds
    )
    np.testing.assert_array_equal(
        original["predictions"], augmented["predictions"][: len(scores)]
    )
    np.testing.assert_array_equal(
        original["view_budgets"], augmented["view_budgets"][: len(scores)]
    )


def test_lazy_gate_requests_extra_views_only_for_low_confidence_rows() -> None:
    scores, _ = _scores()
    requested: dict[str, list[int]] = {}

    def shifts(indices: np.ndarray) -> np.ndarray:
        requested["shift"] = indices.tolist()
        return scores[indices, 1:3]

    def cfo(indices: np.ndarray) -> np.ndarray:
        requested["cfo"] = indices.tolist()
        return scores[indices, 3:5]

    thresholds = AdaptiveTTAThresholds(1.0, 0.5, 0.0)
    eager = apply_adaptive_rxlight_tta(scores, thresholds)
    lazy = apply_adaptive_rxlight_tta_lazy(scores[:, 0], shifts, cfo, thresholds)
    assert requested == {"shift": [1, 2], "cfo": [2]}
    assert lazy["shift_rows_requested"] == 2
    assert lazy["cfo_rows_requested"] == 1
    np.testing.assert_array_equal(lazy["view_budgets"], [1, 3, 5])
    np.testing.assert_array_equal(lazy["predictions"], eager["predictions"])
    np.testing.assert_allclose(lazy["scores"], eager["scores"], atol=1.0e-7)


def test_lazy_gate_does_not_call_providers_when_base_is_confident() -> None:
    scores, _ = _scores()

    def forbidden(_: np.ndarray) -> np.ndarray:
        raise AssertionError("extra-view provider must not run")

    result = apply_adaptive_rxlight_tta_lazy(
        scores[:1, 0], forbidden, forbidden, AdaptiveTTAThresholds(1.0, 0.5, 0.0)
    )
    assert result["view_budgets"].tolist() == [1]


def test_calibration_uses_labels_only_on_allowed_calibration_rows() -> None:
    scores, labels = _scores()
    result = calibrate_adaptive_rxlight_tta(
        scores,
        labels,
        base_margin_grid=[0.5, 1.0, 6.0],
        shift3_margin_grid=[0.0, 0.5, 6.0],
        disagreement_grid=[0.0, 1.0 / 3.0, 2.0 / 3.0],
        max_accuracy_drop_pp=0.0,
    )
    assert result["uses_query_labels"] is False
    assert result["uses_old_new_role"] is False
    assert result["uses_class_quota"] is False
    selected = result["selected"]
    assert selected["passes_accuracy_cap"]
    assert selected["mean_backbone_forwards"] < 5.0
    assert result["selection_policy"] == "accuracy_first_then_minimum_forwards"


def test_calibration_filters_compute_infeasible_candidates_before_ranking() -> None:
    scores, labels = _scores()
    result = calibrate_adaptive_rxlight_tta(
        scores,
        labels,
        base_margin_grid=[0.5, 1.0, 6.0],
        shift3_margin_grid=[0.0, 0.5, 6.0],
        disagreement_grid=[0.0, 1.0 / 3.0, 2.0 / 3.0],
        base_min_score_grid=[-1.0e9],
        shift3_min_score_grid=[-1.0e9],
        max_accuracy_drop_pp=100.0,
        max_mean_backbone_forwards=3.0,
        min_extra_view_rate=0.25,
    )
    selected = result["selected"]
    assert selected["passes_compute_constraints"] is True
    assert selected["mean_backbone_forwards"] <= 3.0
    assert selected["extra_view_rate"] >= 0.25


def test_low_absolute_similarity_triggers_more_views_despite_large_margin() -> None:
    scores = np.asarray(
        [[[0.20, 0.00], [1.0, 0.0], [0.9, 0.1], [1.0, 0.0], [0.9, 0.1]]],
        dtype=np.float32,
    )
    thresholds = AdaptiveTTAThresholds(
        base_stop_margin=0.1,
        shift3_stop_margin=0.1,
        shift3_max_disagreement=0.0,
        base_stop_min_score=0.5,
        shift3_stop_min_score=0.5,
    )
    result = apply_adaptive_rxlight_tta(scores, thresholds)
    assert result["base_margin"][0] >= thresholds.base_stop_margin
    assert result["base_top1_score"][0] < thresholds.base_stop_min_score
    assert result["view_budgets"].tolist() == [3]


def test_stability_lcb_fusion_suppresses_one_class_view_oscillation() -> None:
    scores = np.asarray(
        [[[1.0, 1.4], [1.0, 0.6], [1.0, 1.4], [1.0, 0.6], [1.0, 1.4]]],
        dtype=np.float32,
    )
    common = dict(
        base_stop_margin=10.0,
        shift3_stop_margin=10.0,
        shift3_max_disagreement=0.0,
    )
    mean_result = apply_adaptive_rxlight_tta(
        scores, AdaptiveTTAThresholds(**common, fusion_std_penalty=0.0)
    )
    stable_result = apply_adaptive_rxlight_tta(
        scores, AdaptiveTTAThresholds(**common, fusion_std_penalty=0.25)
    )
    assert mean_result["view_budgets"].tolist() == [5]
    assert mean_result["predictions"].tolist() == [1]
    assert stable_result["predictions"].tolist() == [0]


def test_lazy_and_eager_match_with_score_floor_and_stability_fusion() -> None:
    scores, _ = _scores()

    def shifts(indices: np.ndarray) -> np.ndarray:
        return scores[indices, 1:3]

    def cfo(indices: np.ndarray) -> np.ndarray:
        return scores[indices, 3:5]

    thresholds = AdaptiveTTAThresholds(
        1.0,
        0.5,
        0.0,
        base_stop_min_score=2.0,
        shift3_stop_min_score=1.0,
        fusion_std_penalty=0.1,
    )
    eager = apply_adaptive_rxlight_tta(scores, thresholds)
    lazy = apply_adaptive_rxlight_tta_lazy(scores[:, 0], shifts, cfo, thresholds)
    np.testing.assert_array_equal(lazy["view_budgets"], eager["view_budgets"])
    np.testing.assert_array_equal(lazy["predictions"], eager["predictions"])
    np.testing.assert_allclose(lazy["scores"], eager["scores"], atol=1.0e-7)
