from __future__ import annotations

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_d80_ground_commonmode_denoiser import (
    build_ground_prior_equal_lda,
    ground_classcentered_covariance,
)


def _ground(seed: int = 3):
    rng = np.random.default_rng(seed)
    prototypes = rng.normal(size=(7, 4, 160)) * 0.03
    prototypes += rng.normal(size=(1, 4, 160))
    scales = np.full((7, 4), 0.0025, dtype=np.float64)
    mask = np.ones((7, 4), dtype=np.uint8)
    return prototypes, scales, mask


def _support(classes: int, shots: int, seed: int = 5):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(classes, 288))
    labels = np.repeat(np.arange(classes), shots)
    rows = means[labels] + 0.08 * rng.normal(size=(classes * shots, 288))
    return rows, labels


def test_ground_covariance_is_class_and_domain_permutation_invariant():
    prototypes, scales, mask = _ground()
    covariance, audit = ground_classcentered_covariance(prototypes, scales, mask)
    domain_order = np.array([4, 2, 0, 6, 5, 3, 1])
    class_order = np.array([2, 0, 3, 1])
    permuted, permuted_audit = ground_classcentered_covariance(
        prototypes[domain_order][:, class_order],
        scales[domain_order][:, class_order],
        mask[domain_order][:, class_order],
    )
    np.testing.assert_allclose(covariance, permuted, rtol=0.0, atol=1e-15)
    assert audit["ground_residual_numerical_rank"] == permuted_audit[
        "ground_residual_numerical_rank"
    ]
    assert audit["ground_residual_numerical_rank"] <= 4 * (7 - 1)


def test_ground_covariance_has_quantization_floor_and_no_class_score_path():
    covariance, audit = ground_classcentered_covariance(*_ground())
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0
    assert audit["quantization_noise_floor"] > 0.0
    assert audit["ground_class_centers_discarded_after_residualization"] is True
    assert audit["ground_class_score_access"] is False
    assert audit["ground_component_update_access"] is False


def test_full_fit_uses_fixed_degrees_of_freedom_weight_and_is_finite():
    covariance, audit = ground_classcentered_covariance(*_ground())
    rows, labels = _support(3, 4)
    fit = build_ground_prior_equal_lda(d42, covariance, audit, arm="full")
    coefficient, intercept, evidence = fit(rows, labels, 3, 4)
    assert coefficient.shape == (3, 288)
    assert intercept.shape == (3,)
    assert np.isfinite(coefficient).all()
    assert np.isfinite(intercept).all()
    expected = 6.0 / (6.0 + 3.0 * 3.0)
    assert evidence["d80_ground_shrinkage_weight"] == expected
    assert evidence["d80_ground_class_score_access"] is False
    assert evidence["d80_query_rows_used"] == 0


def test_k1_uses_ground_shape_without_fake_target_covariance():
    covariance, audit = ground_classcentered_covariance(*_ground())
    rows, labels = _support(3, 1)
    fit = build_ground_prior_equal_lda(d42, covariance, audit, arm="full")
    coefficient, intercept, evidence = fit(rows, labels, 3, 1)
    assert np.isfinite(coefficient).all()
    assert np.isfinite(intercept).all()
    assert evidence["d80_target_covariance_fallback"] is True
    assert evidence["d80_target_degrees_of_freedom"] == 0
    assert evidence["d80_ground_shrinkage_weight"] == 1.0


def test_class_permutation_only_permutes_affine_rows():
    covariance, audit = ground_classcentered_covariance(*_ground())
    rows, labels = _support(3, 4)
    fit = build_ground_prior_equal_lda(d42, covariance, audit, arm="full")
    coefficient, intercept, _ = fit(rows, labels, 3, 4)
    permutation = np.array([2, 0, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(3)
    permuted_labels = inverse[labels]
    permuted_coefficient, permuted_intercept, _ = fit(
        rows, permuted_labels, 3, 4
    )
    np.testing.assert_allclose(
        permuted_coefficient, coefficient[permutation], rtol=0.0, atol=2e-5
    )
    np.testing.assert_allclose(
        permuted_intercept, intercept[permutation], rtol=0.0, atol=2e-5
    )


def test_block_fit_is_centered_and_positive_definite():
    covariance, audit = ground_classcentered_covariance(*_ground())
    rows, labels = _support(3, 4)
    fit = build_ground_prior_equal_lda(
        d42, covariance, audit, arm="block3_centered"
    )
    coefficient, intercept, evidence = fit(rows, labels, 3, 4)
    np.testing.assert_allclose(coefficient.mean(axis=0), 0.0, atol=1e-4)
    assert abs(float(intercept.mean())) < 2e-6
    assert evidence["d43_probe_arm"] == "block3_centered"
    assert evidence["d80_posterior_eigenvalue_min"] > 0.0
