from __future__ import annotations

import numpy as np

from cvsrffi.stage2_d88_ground_sigma_pareto_guard import (
    fit_ground_sigma_pareto_guard,
    ground_radius_sigma_geometry,
    project_common_clean_descent,
)


def _ground(seed: int = 8801):
    rng = np.random.default_rng(seed)
    identity = rng.normal(size=(6, 160))
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


def _support(seed: int = 8802, classes: int = 4, shots: int = 5):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(scale=0.3, size=(classes, 288))
    rows = means[labels] + rng.normal(scale=0.15, size=(classes * shots, 288))
    return rows, labels


def test_cone_projection_satisfies_every_class_halfspace() -> None:
    rng = np.random.default_rng(8803)
    direction = rng.normal(size=(4, 7))
    direction -= direction.mean(axis=0, keepdims=True)
    gradients = rng.normal(size=(5, 4, 7))
    gradients -= gradients.mean(axis=1, keepdims=True)
    projected, audit = project_common_clean_descent(direction, gradients)

    violations = np.einsum("kcr,cr->k", gradients, projected)
    assert np.max(violations) <= max(
        1.0e-10, 16.0 * audit["projection_tolerance"]
    )
    np.testing.assert_allclose(projected.mean(axis=0), 0.0, atol=1.0e-12)


def test_fit_guards_every_class_clean_ce_and_compiles_affine() -> None:
    prototypes, radius = _ground()
    basis, offsets, _ = ground_radius_sigma_geometry(prototypes, radius)
    rows, labels = _support()
    base_w, _, _ = _lda_fit(rows, labels, 4, 5)
    delta_w, delta_b, audit = fit_ground_sigma_pareto_guard(
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
    assert audit["final_objective"] <= audit["initial_objective"] + 1.0e-10
    assert audit["all_class_clean_ce_nonincrease_verified"] is True
    assert audit["oof_clean_ce_delta_max_class"] <= (
        audit["clean_pareto_guard_tolerance"] + 1.0e-12
    )
    assert audit["crossfit_fold_count"] == 5
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
    delta_w, delta_b, audit = fit_ground_sigma_pareto_guard(
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
    delta_w2, delta_b2, audit2 = fit_ground_sigma_pareto_guard(
        rows,
        permuted_labels,
        3,
        4,
        base_coefficient=permuted_w,
        tangent_basis=basis,
        counterfactual_offsets=offsets,
        lda_fit=_lda_fit,
    )

    np.testing.assert_allclose(delta_w2[inverse], delta_w, atol=2.0e-6)
    np.testing.assert_allclose(delta_b2[inverse], delta_b, atol=2.0e-6)
    assert audit["all_class_clean_ce_nonincrease_verified"] is True
    assert audit2["all_class_clean_ce_nonincrease_verified"] is True


def test_k1_is_exact_d62_fallback() -> None:
    prototypes, radius = _ground()
    basis, offsets, _ = ground_radius_sigma_geometry(prototypes, radius)
    rows, labels = _support(classes=3, shots=1)
    base_w, _, _ = _lda_fit(rows, labels, 3, 1)
    delta_w, delta_b, audit = fit_ground_sigma_pareto_guard(
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
