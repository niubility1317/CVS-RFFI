from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d77_ground_preconditioned_common_descent import (
    D77CommonDescentError,
    FW_ITERATIONS,
    fit_ground_preconditioned_common_descent,
    ground_reliability_preconditioner,
)


def _support() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7701)
    centers = np.asarray(
        [[2.0, 0.2, -0.4], [-1.0, 1.8, 0.3], [-1.2, -1.6, 0.8]],
        dtype=np.float64,
    )
    rows = np.concatenate(
        [center + rng.normal(scale=0.55, size=(4, 3)) for center in centers],
        axis=0,
    )
    labels = np.repeat(np.arange(3), 4)
    return rows.astype(np.float32), labels.astype(np.int64)


def _lda(rows, labels, class_count, k_shot):
    del k_shot
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    means = np.stack([np.mean(x[y == index], axis=0) for index in range(class_count)])
    coefficient = means
    intercept = -0.5 * np.sum(means * means, axis=1)
    return coefficient.astype(np.float32), intercept.astype(np.float32), {}


def _base(rows: np.ndarray, labels: np.ndarray, classes: int) -> np.ndarray:
    return _lda(rows, labels, classes, 4)[0]


def test_ground_preconditioner_is_positive_determinant_neutral_and_readonly() -> None:
    reliability = np.geomspace(0.03, 0.98, 160)
    scale = np.ones(288, dtype=np.float64)
    scale[:160] = np.sqrt(1.0 + reliability)
    first, audit = ground_reliability_preconditioner(scale)
    second, repeated = ground_reliability_preconditioner(scale.copy())
    np.testing.assert_array_equal(first, second)
    assert audit == repeated
    assert first.flags.writeable is False
    assert np.all(first > 0.0)
    assert np.exp(np.mean(np.log(first[:160]))) == pytest.approx(1.0, abs=1e-12)
    assert np.array_equal(first[160:], np.ones(128))
    assert audit["ground_class_score_access"] is False


def test_common_descent_is_deterministic_and_per_class_ce_safe() -> None:
    rows, labels = _support()
    metric = np.asarray([1.4, 0.7, 1.1], dtype=np.float64)
    base = _base(rows, labels, 3)
    delta, audit = fit_ground_preconditioned_common_descent(
        rows,
        labels,
        3,
        4,
        base_coefficient=base,
        preconditioner=metric,
        lda_fit=_lda,
    )
    repeated, second = fit_ground_preconditioned_common_descent(
        rows,
        labels,
        3,
        4,
        base_coefficient=base,
        preconditioner=metric,
        lda_fit=_lda,
    )
    np.testing.assert_array_equal(delta, repeated)
    assert audit == second
    assert audit["status"] == "ground_preconditioned_allclass_common_descent_active"
    assert audit["crossfit_fold_count"] == 4
    assert audit["crossfit_held_row_count"] == 12
    assert audit["frank_wolfe_iterations"] == FW_ITERATIONS
    assert len(audit["optimizer_objective_trace"]) == FW_ITERATIONS
    assert max(audit["oof_per_class_ce_delta"]) <= audit["ce_numeric_tolerance"]
    assert min(audit["oof_per_class_ce_delta"]) < -audit["ce_numeric_tolerance"]
    assert audit["query_rows_used"] == 0
    assert audit["ground_class_score_access"] is False


def test_class_permutation_is_equivariant() -> None:
    rows, labels = _support()
    metric = np.asarray([1.4, 0.7, 1.1], dtype=np.float64)
    base = _base(rows, labels, 3)
    reference, reference_audit = fit_ground_preconditioned_common_descent(
        rows,
        labels,
        3,
        4,
        base_coefficient=base,
        preconditioner=metric,
        lda_fit=_lda,
    )
    permutation = np.asarray([2, 0, 1])
    inverse = np.argsort(permutation)
    relabeled = permutation[labels]
    permuted_base = base[inverse]
    permuted, audit = fit_ground_preconditioned_common_descent(
        rows,
        relabeled,
        3,
        4,
        base_coefficient=permuted_base,
        preconditioner=metric,
        lda_fit=_lda,
    )
    np.testing.assert_allclose(permuted[permutation], reference, rtol=0.0, atol=2e-7)
    assert audit["oof_ce_delta_mean"] == pytest.approx(
        reference_audit["oof_ce_delta_mean"], abs=1e-12
    )


def test_k1_is_exact_fallback_and_asymmetric_support_fails() -> None:
    rows, labels = _support()
    one = np.asarray([0, 4, 8])
    base = _base(rows, labels, 3)
    delta, audit = fit_ground_preconditioned_common_descent(
        rows[one],
        labels[one],
        3,
        1,
        base_coefficient=base,
        preconditioner=np.ones(3),
        lda_fit=_lda,
    )
    assert audit["status"] == "k1_exact_d62_fallback"
    assert np.array_equal(delta, np.zeros_like(delta))
    with pytest.raises(D77CommonDescentError, match="exact symmetric support"):
        fit_ground_preconditioned_common_descent(
            rows[:-1],
            labels[:-1],
            3,
            4,
            base_coefficient=base,
            preconditioner=np.ones(3),
            lda_fit=_lda,
        )
