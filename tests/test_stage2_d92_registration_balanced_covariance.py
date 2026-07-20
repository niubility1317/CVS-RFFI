from __future__ import annotations

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_d92_registration_balanced_covariance import (
    build_registration_balanced_equal_lda,
)


def _support(classes: int, shots: int, seed: int = 92):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, d42.FEATURE_DIM))
    rows = means[labels] + 0.1 * rng.normal(
        size=(classes * shots, d42.FEATURE_DIM)
    )
    return rows.astype(np.float32), labels


def test_registered_fit_is_fixed_equal_task_covariance_and_compilable():
    rows, labels = _support(11, 5)
    fit = build_registration_balanced_equal_lda(
        d42, d42._fit_equal_prior_lda, arm="full"
    )
    coefficient, intercept, audit = fit(rows, labels, 11, 5)
    assert coefficient.shape == (11, 288)
    assert intercept.shape == (11,)
    assert np.isfinite(coefficient).all()
    assert np.isfinite(intercept).all()
    assert audit["d92_registration_balanced_active"] is True
    assert audit["d92_old_covariance_weight"] == 0.5
    assert audit["d92_new_covariance_weight"] == 0.5
    assert audit["d92_weight_scan_count"] == 0
    assert audit["d92_query_rows_used"] == 0
    assert audit["d92_query_role_oracle_access"] is False
    assert audit["d92_class_common_affine_omitted_before_fp32"] is True
    recentered_coefficient = coefficient.astype(np.float64)
    recentered_coefficient -= recentered_coefficient.mean(axis=0, keepdims=True)
    recentered_intercept = intercept.astype(np.float64)
    recentered_intercept -= recentered_intercept.mean()
    original_prediction = np.argmax(
        rows @ coefficient.T + intercept[None, :], axis=1
    )
    recentered_prediction = np.argmax(
        rows @ recentered_coefficient.astype(np.float32).T
        + recentered_intercept.astype(np.float32)[None, :],
        axis=1,
    )
    np.testing.assert_array_equal(recentered_prediction, original_prediction)
    state, _ = d42._compile_state(
        tuple(f"class{index}" for index in range(11)),
        6,
        np.zeros(288, dtype=np.float32),
        coefficient,
        intercept,
        audit["covariance_policy"],
        precision="fp32",
    )
    assert state.covariance_policy == "sklearn_lsqr_auto_shrinkage_equal_prior"


def test_before_and_k1_are_exact_baseline_fallbacks():
    for classes, shots, status in (
        (6, 5, "before_exact_d81"),
        (26, 1, "k1_k2_exact_d81_fallback"),
    ):
        rows, labels = _support(classes, shots, seed=classes + shots)
        fit = build_registration_balanced_equal_lda(
            d42, d42._fit_equal_prior_lda, arm="full"
        )
        coefficient, intercept, audit = fit(rows, labels, classes, shots)
        expected_coefficient, expected_intercept, _ = d42._fit_equal_prior_lda(
            rows, labels, classes, shots
        )
        np.testing.assert_array_equal(coefficient, expected_coefficient)
        np.testing.assert_array_equal(intercept, expected_intercept)
        assert audit["d92_status"] == status
        assert audit["d92_registration_balanced_active"] is False


def test_formula_is_equivariant_within_registration_groups():
    rows, labels = _support(16, 5, seed=101)
    fit = build_registration_balanced_equal_lda(
        d42, d42._fit_equal_prior_lda, arm="block3_centered"
    )
    coefficient, intercept, _ = fit(rows, labels, 16, 5)
    permutation = np.asarray(
        [2, 5, 0, 4, 1, 3, 9, 6, 8, 7, 10, 15, 12, 11, 14, 13]
    )
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    coefficient2, intercept2, _ = fit(rows, inverse[labels], 16, 5)
    np.testing.assert_allclose(coefficient2[inverse], coefficient, rtol=0.0, atol=2e-4)
    np.testing.assert_allclose(intercept2[inverse], intercept, rtol=0.0, atol=2e-4)
