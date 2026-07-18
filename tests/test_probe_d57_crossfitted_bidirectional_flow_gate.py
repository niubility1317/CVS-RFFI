from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d57_crossfitted_bidirectional_flow_gate.py"
SPEC = importlib.util.spec_from_file_location("d57_probe_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d57 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d57)


ACTIVE_SCORES = np.asarray(
    [
        [[0.0356, 0.1835, -0.0666], [-0.0467, -0.2567, 0.1883], [0.1157, 0.1100, 0.2284]],
        [[0.2311, 0.0259, -0.0302], [-0.0244, -0.0291, 0.1814], [0.0667, -0.0070, 0.0643]],
        [[0.0914, 0.1923, 0.0608], [0.0081, 0.1546, -0.1331], [-0.0080, 0.1048, -0.0901]],
    ],
    dtype=np.float64,
)

ATOMIC_FALLBACK_SCORES = np.asarray(
    [
        [[0.1953, 0.0358, -0.0329], [-0.1069, 0.0542, -0.1190], [0.0072, 0.1608, 0.0152]],
        [[-0.0054, 0.0588, 0.0428], [0.0126, -0.1174, -0.0035], [0.0834, -0.1613, -0.1314]],
        [[-0.2504, -0.1547, -0.2210], [-0.0282, -0.0245, 0.0326], [0.0188, -0.0224, -0.3218]],
    ],
    dtype=np.float64,
)


def _audit(scores: np.ndarray) -> dict[str, object]:
    k, c, _ = scores.shape
    true_by_fold = [list(range(c)) for _ in range(k)]
    predictions = np.argmax(scores, axis=2).astype(np.int64).tolist()
    out_degree = np.zeros(c, dtype=np.int64)
    in_degree = np.zeros(c, dtype=np.int64)
    for true_fold, predicted_fold in zip(true_by_fold, predictions):
        for true_class, predicted_class in zip(true_fold, predicted_fold):
            if true_class != predicted_class:
                out_degree[true_class] += 1
                in_degree[predicted_class] += 1
    flow = (out_degree - in_degree).astype(np.float64) / (k * c)
    return {
        "d56_full_inner_held_score_fp64": scores.tolist(),
        "d56_block_inner_held_score_fp64": scores.tolist(),
        "d56_held_true_class_by_fold": true_by_fold,
        "d56_held_prediction_by_fold": predictions,
        "d46_full_weight_by_class": [0.5] * c,
        "d46_block_weight_by_class": [0.5] * c,
        "d46_full_support_logit_rms": 1.0,
        "d46_block_support_logit_rms": 1.0,
        "d56_centered_intercept_compensation_fp64": flow.tolist(),
    }


def test_active_gate_accepts_only_bidirectionally_safe_coordinate() -> None:
    result = d57._bidirectional_gate(_audit(ACTIVE_SCORES), 3, 3)
    assert result["status"] == "crossfitted_bidirectional_flow_gate_active"
    assert np.array_equal(result["initial_accept_mask"], [False, True, False])
    assert np.array_equal(result["final_accept_mask"], [False, True, False])
    assert np.array_equal(result["base_positive_correct"], [1, 1, 1])
    assert np.array_equal(result["coordinate_positive_correct"], [1, 1, 1])
    assert np.array_equal(result["base_false_positive"], [1, 3, 2])
    assert np.array_equal(result["coordinate_false_positive"], [2, 2, 2])
    assert np.all(result["joint_positive_correct"] >= result["base_positive_correct"])
    assert np.all(result["joint_false_positive"] <= result["base_false_positive"])
    assert abs(float(np.sum(result["delta_intercept"]))) < 1.0e-14


def test_joint_interaction_failure_atomically_restores_d46() -> None:
    result = d57._bidirectional_gate(_audit(ATOMIC_FALLBACK_SCORES), 3, 3)
    assert result["status"] == "joint_gate_atomic_d46_fallback"
    assert np.any(result["initial_accept_mask"])
    assert not np.any(result["final_accept_mask"])
    assert result["atomic_fallback"] is True
    assert result["exact_fallback"] is True
    assert np.array_equal(result["delta_intercept"], np.zeros(3))


@pytest.mark.parametrize("k", [1, 2])
def test_k1_k2_are_exact_d46_fallbacks_without_evidence(k: int) -> None:
    result = d57._bidirectional_gate({}, k, 3)
    assert result["status"] == "k1_k2_exact_d46_fallback"
    assert result["exact_fallback"] is True
    assert np.array_equal(result["delta_intercept"], np.zeros(3))


def test_class_permutation_is_equivariant() -> None:
    baseline = d57._bidirectional_gate(_audit(ACTIVE_SCORES), 3, 3)
    order = np.asarray([2, 0, 1])
    permuted_scores = ACTIVE_SCORES[:, order, :][:, :, order]
    actual = d57._bidirectional_gate(_audit(permuted_scores), 3, 3)
    assert np.array_equal(actual["final_accept_mask"], baseline["final_accept_mask"][order])
    assert np.allclose(actual["delta_intercept"], baseline["delta_intercept"][order])


def test_invalid_scores_and_weight_simplex_fail_closed() -> None:
    bad = _audit(ACTIVE_SCORES)
    bad["d46_full_weight_by_class"] = [0.9, 0.9, 0.9]
    with pytest.raises(d57.D57ProbeError, match="evidence"):
        d57._bidirectional_gate(bad, 3, 3)
    bad_shape = _audit(ACTIVE_SCORES)
    bad_shape["d56_full_inner_held_score_fp64"] = ACTIVE_SCORES[:, :, :2].tolist()
    with pytest.raises(d57.D57ProbeError, match="evidence"):
        d57._bidirectional_gate(bad_shape, 3, 3)


def test_resource_account_reuses_d56_fits_and_counts_only_gate_work() -> None:
    numeric, comparisons = d57._extra_resource(8, 6, 11, 288)
    assert numeric == sum(8 * 8 * c * c + 8 * 8 * c for c in (6, 11))
    assert comparisons == sum(4 * 8 * c * c for c in (6, 11))
    assert d57._extra_resource(2, 6, 11, 288) == (0, 0)


def test_formula_has_no_tunable_or_role_specific_surface() -> None:
    assert "positive_adjusted>=positive_base" in d57.FORMULA
    assert "fp_adjusted<=fp_base" in d57.FORMULA
    assert "CE" not in d57.FORMULA
    assert "alpha" not in d57.FORMULA.lower()
    assert "threshold" not in d57.FORMULA.lower()
