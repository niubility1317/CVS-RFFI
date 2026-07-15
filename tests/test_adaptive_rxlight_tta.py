from __future__ import annotations

import numpy as np

from paper_reproduction.cvs_aligned.adaptive_rxlight_tta import (
    AdaptiveTTAThresholds,
    apply_adaptive_rxlight_tta,
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
