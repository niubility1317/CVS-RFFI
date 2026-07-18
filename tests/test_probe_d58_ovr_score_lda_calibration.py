from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d58_ovr_score_lda_calibration.py"
SPEC = importlib.util.spec_from_file_location("d58_probe_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d58 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d58)


SCORES = np.asarray(
    [
        [[2.0, 0.2, -0.3], [0.1, 1.5, 0.0], [-0.2, 0.4, 1.2]],
        [[1.4, -0.1, 0.3], [0.5, 1.8, -0.2], [0.2, 0.0, 1.6]],
        [[1.7, 0.3, 0.1], [-0.4, 1.1, 0.2], [0.0, -0.2, 1.4]],
    ],
    dtype=np.float64,
)


def _audit(scores: np.ndarray) -> dict[str, object]:
    k, c, _ = scores.shape
    return {
        "d56_full_inner_held_score_fp64": scores.tolist(),
        "d56_block_inner_held_score_fp64": scores.tolist(),
        "d56_held_true_class_by_fold": [list(range(c)) for _ in range(k)],
        "d56_held_prediction_by_fold": np.argmax(scores, axis=2).astype(int).tolist(),
        "d46_full_weight_by_class": [0.5] * c,
        "d46_block_weight_by_class": [0.5] * c,
        "d46_full_support_logit_rms": 1.0,
        "d46_block_support_logit_rms": 1.0,
    }


def test_closed_form_balanced_ovr_moments_and_positive_slopes() -> None:
    result = d58._ovr_score_lda_calibration(_audit(SCORES), 3, 3)
    assert result["status"] == "support_inner_held_ovr_score_lda_calibration_active"
    assert result["exact_fallback"] is False
    assert np.allclose(result["positive_mean"], [1.7, 1.4666666666666666, 1.4])
    assert np.all(result["positive_mean"] > result["negative_mean"])
    assert np.all(result["pooled_variance"] > 0.0)
    assert np.all(result["raw_slope"] > 0.0)
    assert np.isclose(np.mean(result["normalized_slope"]), 1.0)
    assert result["base_correct_count"] == 9
    assert result["calibrated_correct_count"] == 9


def test_per_class_score_translation_is_absorbed_by_intercept() -> None:
    baseline = d58._ovr_score_lda_calibration(_audit(SCORES), 3, 3)
    shift = np.asarray([3.0, -2.0, 0.7])
    actual = d58._ovr_score_lda_calibration(_audit(SCORES + shift), 3, 3)
    assert np.allclose(actual["normalized_slope"], baseline["normalized_slope"])
    assert np.allclose(
        actual["normalized_intercept"],
        baseline["normalized_intercept"] - baseline["normalized_slope"] * shift,
    )
    assert actual["calibrated_prediction_by_fold"] == baseline["calibrated_prediction_by_fold"]


def test_common_positive_score_scale_preserves_predictions() -> None:
    baseline = d58._ovr_score_lda_calibration(_audit(SCORES), 3, 3)
    actual = d58._ovr_score_lda_calibration(_audit(4.0 * SCORES), 3, 3)
    assert np.allclose(actual["normalized_slope"], baseline["normalized_slope"])
    assert np.allclose(actual["normalized_intercept"], 4.0 * baseline["normalized_intercept"])
    assert actual["calibrated_prediction_by_fold"] == baseline["calibrated_prediction_by_fold"]


def test_class_and_support_rank_permutations_are_equivariant() -> None:
    baseline = d58._ovr_score_lda_calibration(_audit(SCORES), 3, 3)
    order = np.asarray([2, 0, 1])
    permuted = SCORES[[2, 0, 1]][:, order, :][:, :, order]
    actual = d58._ovr_score_lda_calibration(_audit(permuted), 3, 3)
    assert np.allclose(actual["normalized_slope"], baseline["normalized_slope"][order])
    assert np.allclose(
        actual["normalized_intercept"], baseline["normalized_intercept"][order]
    )


def test_nonpositive_separation_falls_back_entire_fit() -> None:
    bad = SCORES.copy()
    bad[:, 0, 0] = -5.0
    result = d58._ovr_score_lda_calibration(_audit(bad), 3, 3)
    assert result["status"] == "nonpositive_or_degenerate_exact_d46_fallback"
    assert result["exact_fallback"] is True
    assert np.array_equal(result["normalized_slope"], np.ones(3))
    assert np.array_equal(result["normalized_intercept"], np.zeros(3))


@pytest.mark.parametrize("k", [1, 2])
def test_k1_k2_are_exact_d46_fallbacks_without_evidence(k: int) -> None:
    result = d58._ovr_score_lda_calibration({}, k, 3)
    assert result["status"] == "k1_k2_exact_d46_fallback"
    assert result["exact_fallback"] is True
    assert np.array_equal(result["normalized_slope"], np.ones(3))


def test_invalid_scores_and_weight_simplex_fail_closed() -> None:
    bad = _audit(SCORES)
    bad["d46_full_weight_by_class"] = [0.9, 0.9, 0.9]
    with pytest.raises(d58.D58ProbeError, match="evidence"):
        d58._ovr_score_lda_calibration(bad, 3, 3)
    bad_shape = _audit(SCORES)
    bad_shape["d56_full_inner_held_score_fp64"] = SCORES[:, :, :2].tolist()
    with pytest.raises(d58.D58ProbeError, match="evidence"):
        d58._ovr_score_lda_calibration(bad_shape, 3, 3)


def test_resource_account_reuses_d56_fits_and_counts_score_calibration() -> None:
    numeric, comparisons = d58._extra_resource(8, 6, 11, 288)
    expected_numeric = sum(
        8 * 8 * c * c + 24 * 8 * c + 12 * c + 2 * 289 * c
        for c in (6, 11)
    )
    expected_comparisons = sum(2 * 8 * c + 3 * c for c in (6, 11))
    assert numeric == expected_numeric
    assert comparisons == expected_comparisons
    assert d58._extra_resource(2, 6, 11, 288) == (0, 0)


def test_formula_has_no_tunable_or_role_specific_surface() -> None:
    assert "mu_pos-mu_neg" in d58.FORMULA
    assert "pooled_var" in d58.FORMULA
    assert "alpha" not in d58.FORMULA.lower()
    assert "threshold" not in d58.FORMULA.lower()
    assert "ridge" not in d58.FORMULA.lower()
