from __future__ import annotations

import numpy as np

from cvsrffi.stage2_d87_ground_radius_sigma_margin import (
    fit_ground_radius_sigma_margin,
    ground_radius_sigma_geometry,
)


def _ground(seed: int = 8701):
    rng = np.random.default_rng(seed)
    identity = rng.normal(size=(6, 160))
    # Centered domain coordinates make the effective rank exactly 13.
    domain = rng.normal(scale=0.08, size=(14, 160))
    domain -= domain.mean(axis=0, keepdims=True)
    interaction = rng.normal(scale=0.005, size=(14, 6, 160))
    interaction -= interaction.mean(axis=0, keepdims=True)
    prototypes = identity[None, :, :] + domain[:, None, :] + interaction
    radius = rng.uniform(0.001, 0.02, size=(14, 6))
    return prototypes, radius


def _lda_fit(rows, labels, class_count, _k_shot):
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    means = np.stack([x[y == index].mean(axis=0) for index in range(class_count)])
    coefficient = means
    intercept = -0.5 * np.sum(means * means, axis=1)
    return coefficient, intercept, {"test_fit": True}


def _support(seed: int = 8702, classes: int = 4, shots: int = 5):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(scale=0.3, size=(classes, 288))
    rows = means[labels] + rng.normal(scale=0.15, size=(classes * shots, 288))
    return rows, labels


def test_geometry_has_exact_sigma_covariance_and_no_auxiliary_offset() -> None:
    prototypes, radius = _ground()
    basis, offsets, audit = ground_radius_sigma_geometry(
        prototypes, radius, feature_dim=288
    )
    order = np.asarray([9, 0, 4, 13, 1, 11, 2, 8, 3, 12, 5, 10, 6, 7])
    basis2, offsets2, audit2 = ground_radius_sigma_geometry(
        prototypes[order], radius[order], feature_dim=288
    )

    assert audit["effective_rank"] == 13
    assert audit["sigma_covariance_max_abs_error"] <= 1.0e-14
    np.testing.assert_allclose(basis.T @ basis, np.eye(13), atol=1.0e-10)
    np.testing.assert_array_equal(offsets[:, 160:], 0.0)
    covariance = offsets.T @ offsets / (2.0 * len(offsets))
    covariance2 = offsets2.T @ offsets2 / (2.0 * len(offsets2))
    np.testing.assert_allclose(covariance2, covariance, rtol=0.0, atol=1.0e-14)
    assert audit2["physical_sample_count_multiplier"] == 1


def test_sigma_fit_decreases_objective_and_compiles_centered_affine() -> None:
    prototypes, radius = _ground()
    basis, offsets, _ = ground_radius_sigma_geometry(prototypes, radius)
    rows, labels = _support()
    base_w, _, _ = _lda_fit(rows, labels, 4, 5)
    delta_w, delta_b, audit = fit_ground_radius_sigma_margin(
        rows,
        labels,
        4,
        5,
        base_coefficient=base_w,
        tangent_basis=basis,
        counterfactual_offsets=offsets,
        lda_fit=_lda_fit,
    )

    assert audit["optimizer_iterations"] == 20
    assert audit["final_objective"] <= audit["initial_objective"] + 1.0e-12
    assert audit["crossfit_fold_count"] == 5
    assert audit["crossfit_held_row_count"] == 20
    assert audit["counterfactual_views_count_as_physical_samples"] is False
    np.testing.assert_allclose(delta_w.mean(axis=0), 0.0, atol=1.0e-7)
    center = rows.mean(axis=0)
    np.testing.assert_allclose(delta_w @ center + delta_b, 0.0, atol=1.0e-6)
    np.testing.assert_array_equal(delta_w[:, 160:], 0.0)


def test_class_permutation_is_equivariant() -> None:
    prototypes, radius = _ground()
    basis, offsets, _ = ground_radius_sigma_geometry(prototypes, radius)
    rows, labels = _support(classes=3, shots=4)
    base_w, _, _ = _lda_fit(rows, labels, 3, 4)
    delta_w, delta_b, _ = fit_ground_radius_sigma_margin(
        rows,
        labels,
        3,
        4,
        base_coefficient=base_w,
        tangent_basis=basis,
        counterfactual_offsets=offsets,
        lda_fit=_lda_fit,
    )
    permutation = np.asarray([2, 0, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(3)
    permuted_labels = inverse[labels]
    permuted_w = base_w[permutation]
    delta_w2, delta_b2, _ = fit_ground_radius_sigma_margin(
        rows,
        permuted_labels,
        3,
        4,
        base_coefficient=permuted_w,
        tangent_basis=basis,
        counterfactual_offsets=offsets,
        lda_fit=_lda_fit,
    )
    np.testing.assert_allclose(delta_w2[inverse], delta_w, atol=1.0e-6)
    np.testing.assert_allclose(delta_b2[inverse], delta_b, atol=1.0e-6)


def test_k1_is_exact_d62_fallback() -> None:
    prototypes, radius = _ground()
    basis, offsets, _ = ground_radius_sigma_geometry(prototypes, radius)
    rows, labels = _support(classes=3, shots=1)
    base_w, _, _ = _lda_fit(rows, labels, 3, 1)
    delta_w, delta_b, audit = fit_ground_radius_sigma_margin(
        rows,
        labels,
        3,
        1,
        base_coefficient=base_w,
        tangent_basis=basis,
        counterfactual_offsets=offsets,
        lda_fit=_lda_fit,
    )
    np.testing.assert_array_equal(delta_w, 0.0)
    np.testing.assert_array_equal(delta_b, 0.0)
    assert audit["status"] == "k1_exact_d62_fallback"

