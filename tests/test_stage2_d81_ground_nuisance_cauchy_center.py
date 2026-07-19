from __future__ import annotations

import numpy as np

from cvsrffi.stage2_d81_ground_nuisance_cauchy_center import (
    ground_nuisance_basis,
    translate_to_robust_centers,
)


def _covariance(seed: int = 7):
    rng = np.random.default_rng(seed)
    residual = rng.normal(size=(42, 160))
    raw = residual.T @ residual / len(residual)
    floor = 1.0e-6
    return raw + floor * np.eye(160), floor


def _support(classes: int = 4, shots: int = 5, seed: int = 11):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, 288))
    rows = means[labels] + 0.1 * rng.normal(size=(classes * shots, 288))
    rows[2, :160] += 2.0
    return rows, labels


def test_basis_rank_is_ceil_effective_rank_and_deterministic():
    covariance, floor = _covariance()
    basis, weights, audit = ground_nuisance_basis(covariance, floor)
    basis2, weights2, audit2 = ground_nuisance_basis(covariance, floor)
    assert audit["retained_rank"] == int(
        np.ceil(audit["participation_ratio_effective_rank"])
    )
    assert audit["rank_scan_count"] == 0
    assert audit["basis_sha256"] == audit2["basis_sha256"]
    np.testing.assert_array_equal(basis, basis2)
    np.testing.assert_array_equal(weights, weights2)


def test_translation_preserves_residuals_covariance_and_other_views():
    covariance, floor = _covariance()
    basis, weights, _ = ground_nuisance_basis(covariance, floor)
    rows, labels = _support()
    transformed, audit = translate_to_robust_centers(
        rows, labels, 4, 5, basis, weights
    )
    before_means = np.stack([rows[labels == c].mean(axis=0) for c in range(4)])
    after_means = np.stack(
        [transformed[labels == c].mean(axis=0) for c in range(4)]
    )
    np.testing.assert_allclose(
        rows - before_means[labels],
        transformed - after_means[labels],
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_array_equal(transformed[:, 160:], rows[:, 160:])
    assert audit["center_shift_l2_max"] > 0.0
    assert audit["within_class_residual_max_abs_error"] <= 2e-12


def test_k1_and_k2_are_exact_identity():
    covariance, floor = _covariance()
    basis, weights, _ = ground_nuisance_basis(covariance, floor)
    for shots in (1, 2):
        rows, labels = _support(classes=3, shots=shots, seed=13 + shots)
        transformed, audit = translate_to_robust_centers(
            rows, labels, 3, shots, basis, weights
        )
        np.testing.assert_array_equal(transformed, rows)
        assert audit["k1_k2_exact_identity"] is True


def test_class_permutation_only_permutes_audit_rows():
    covariance, floor = _covariance()
    basis, weights, _ = ground_nuisance_basis(covariance, floor)
    rows, labels = _support(classes=4, shots=5)
    transformed, audit = translate_to_robust_centers(
        rows, labels, 4, 5, basis, weights
    )
    permutation = np.array([2, 0, 3, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(4)
    transformed2, audit2 = translate_to_robust_centers(
        rows, inverse[labels], 4, 5, basis, weights
    )
    np.testing.assert_allclose(transformed2, transformed, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        audit2["center_shift_l2_by_class"],
        np.asarray(audit["center_shift_l2_by_class"])[permutation],
        rtol=0.0,
        atol=1e-15,
    )


def test_domain_basis_rotation_within_equal_eigenspace_is_score_invariant():
    covariance, floor = _covariance()
    basis, weights, _ = ground_nuisance_basis(covariance, floor)
    rows, labels = _support()
    transformed, _ = translate_to_robust_centers(
        rows, labels, 4, 5, basis, weights
    )
    signs = np.where(np.arange(basis.shape[1]) % 2 == 0, -1.0, 1.0)
    transformed2, _ = translate_to_robust_centers(
        rows, labels, 4, 5, basis * signs[None, :], weights
    )
    np.testing.assert_allclose(transformed2, transformed, rtol=0.0, atol=1e-15)
