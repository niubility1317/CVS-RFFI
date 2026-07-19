from __future__ import annotations

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_d83_ground_precision_loading import (
    build_ground_precision_loaded_equal_lda,
)


def _basis(seed: int = 7):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(160, 14))
    basis, _ = np.linalg.qr(raw)
    weights = np.arange(1, 15, dtype=np.float64)
    weights /= weights.sum()
    return basis, weights


def _support(classes: int = 4, shots: int = 5, seed: int = 11):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, 288))
    rows = means[labels] + 0.1 * rng.normal(size=(classes * shots, 288))
    return rows, labels


def test_loading_is_closed_form_parameter_free_and_spd():
    basis, weights = _basis()
    rows, labels = _support()
    fit = build_ground_precision_loaded_equal_lda(
        d42, basis, weights, arm="full"
    )
    coefficient, intercept, audit = fit(rows, labels, 4, 5)
    assert coefficient.shape == (4, 288)
    assert intercept.shape == (4,)
    assert audit["d83_loading_to_target_mean_variance_ratio"] == 0.2
    assert audit["d83_loading_mean_retained_direction"] == (
        audit["d83_target_z_mean_variance"] / 5
    )
    assert audit["d83_hyperparameter_count"] == 0
    assert audit["d83_loading_scan_count"] == 0
    assert audit["covariance_policy"] == "sklearn_lsqr_auto_shrinkage_equal_prior"
    assert audit["d83_covariance_policy"] == (
        "sklearn_lsqr_auto_plus_rank14_ground_loading"
    )
    assert audit["d83_posterior_eigenvalue_min"] > 0.0
    state, _ = d42._compile_state(
        tuple(f"class{index}" for index in range(4)),
        4,
        np.zeros(288, dtype=np.float32),
        coefficient,
        intercept,
        audit["covariance_policy"],
        precision="fp32",
    )
    assert state.covariance_policy == "sklearn_lsqr_auto_shrinkage_equal_prior"


def test_k1_has_exact_zero_loading():
    basis, weights = _basis()
    for shots in (1,):
        rows, labels = _support(classes=3, shots=shots, seed=13 + shots)
        fit = build_ground_precision_loaded_equal_lda(
            d42, basis, weights, arm="full"
        )
        _, _, audit = fit(rows, labels, 3, shots)
        assert audit["d83_k1_k2_exact_no_loading"] is True
        assert audit["d83_loading_scale"] == 0.0
        assert audit["d83_loading_trace"] == 0.0


def test_basis_sign_is_prediction_invariant():
    basis, weights = _basis()
    rows, labels = _support()
    fit = build_ground_precision_loaded_equal_lda(
        d42, basis, weights, arm="block3_centered"
    )
    fit2 = build_ground_precision_loaded_equal_lda(
        d42, basis * np.where(np.arange(14) % 2, -1.0, 1.0), weights,
        arm="block3_centered",
    )
    coefficient, intercept, _ = fit(rows, labels, 4, 5)
    coefficient2, intercept2, _ = fit2(rows, labels, 4, 5)
    np.testing.assert_allclose(coefficient2, coefficient, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(intercept2, intercept, rtol=0.0, atol=1e-6)


def test_loading_shape_is_class_permutation_equivariant():
    basis, weights = _basis()
    rows, labels = _support()
    fit = build_ground_precision_loaded_equal_lda(
        d42, basis, weights, arm="full"
    )
    coefficient, intercept, _ = fit(rows, labels, 4, 5)
    permutation = np.array([2, 0, 3, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(4)
    coefficient2, intercept2, _ = fit(rows, inverse[labels], 4, 5)
    np.testing.assert_allclose(coefficient2[inverse], coefficient, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(intercept2[inverse], intercept, rtol=0.0, atol=1e-5)
