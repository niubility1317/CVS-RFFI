from __future__ import annotations

import numpy as np

from cvsrffi.stage2_d86_ground_radius_counterfactual_center import (
    ground_radius_counterfactual_templates,
    translate_to_counterfactual_robust_centers,
)


def _ground(seed: int = 8601):
    rng = np.random.default_rng(seed)
    identity = rng.normal(size=(5, 160))
    domain = rng.normal(scale=0.1, size=(7, 160))
    interaction = rng.normal(scale=0.01, size=(7, 5, 160))
    prototypes = identity[None, :, :] + domain[:, None, :] + interaction
    radius = rng.uniform(0.001, 0.02, size=(7, 5))
    return prototypes, radius


def _support(shots: int = 4):
    labels = np.repeat(np.arange(3), shots)
    rows = np.zeros((3 * shots, 288), dtype=np.float64)
    if shots == 4:
        rows[:4, 0] = np.asarray([-0.1, -0.1, -0.1, 0.9])
        rows[4:8, 0] = np.asarray([1.9, 2.0, 2.0, 2.1])
        rows[8:12, 0] = np.asarray([-2.1, -2.0, -2.0, -1.9])
    else:
        rows[:, 0] = np.repeat(np.asarray([0.0, 2.0, -2.0]), shots)
    rows[:, 160:] = np.arange(len(rows) * 128, dtype=np.float64).reshape(len(rows), 128)
    return rows, labels


def test_ground_templates_bind_radius_without_ground_class_mapping() -> None:
    prototypes, radius = _ground()
    templates, amplitude, audit = ground_radius_counterfactual_templates(
        prototypes, radius
    )
    order = np.asarray([3, 0, 4, 1, 2])
    templates2, amplitude2, audit2 = ground_radius_counterfactual_templates(
        prototypes[:, order], radius[:, order]
    )

    np.testing.assert_allclose(templates2, templates, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(amplitude2, amplitude, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        amplitude,
        np.sqrt(2.0 * np.median(radius, axis=1)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.linalg.norm(templates, axis=0), 1.0, rtol=0.0, atol=1.0e-12
    )
    assert audit2["weight_sha256"] == audit["weight_sha256"]
    assert audit["ground_class_centers_discarded"] is True
    assert audit["ground_target_identity_mapping_access"] is False
    assert audit["radius_hyperparameter_count"] == 0


def test_borderline_support_gets_lower_weight_and_only_center_moves() -> None:
    rows, labels = _support()
    templates = np.zeros((160, 1), dtype=np.float64)
    templates[0, 0] = 1.0
    transformed, audit = translate_to_counterfactual_robust_centers(
        rows, labels, 3, 4, templates, np.asarray([0.4])
    )

    class_zero_weights = np.asarray(
        audit["normalized_cauchy_weight_by_class"][0]
    )
    assert class_zero_weights[-1] < np.min(class_zero_weights[:-1])
    assert audit["center_shift_l2_by_class"][0] > 0.0
    for class_index in range(3):
        selected = labels == class_index
        before = rows[selected] - rows[selected].mean(axis=0)
        after = transformed[selected] - transformed[selected].mean(axis=0)
        np.testing.assert_allclose(after, before, rtol=0.0, atol=2.0e-12)
    np.testing.assert_array_equal(transformed[:, 160:], rows[:, 160:])
    assert audit["query_rows_used"] == 0
    assert audit["hyperparameter_count"] == 0


def test_target_class_permutation_and_ground_sign_are_exactly_equivariant() -> None:
    rows, labels = _support()
    templates = np.zeros((160, 2), dtype=np.float64)
    templates[0, 0] = 1.0
    templates[1, 1] = 1.0
    amplitude = np.asarray([0.4, 0.2])
    transformed, _ = translate_to_counterfactual_robust_centers(
        rows, labels, 3, 4, templates, amplitude
    )
    transformed_sign, _ = translate_to_counterfactual_robust_centers(
        rows, labels, 3, 4, -templates, amplitude
    )
    permutation = np.asarray([2, 0, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(3)
    transformed_permuted, _ = translate_to_counterfactual_robust_centers(
        rows, inverse[labels], 3, 4, templates, amplitude
    )

    np.testing.assert_allclose(transformed_sign, transformed, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        transformed_permuted, transformed, rtol=0.0, atol=1.0e-14
    )


def test_k1_and_k2_are_exact_d62_fallbacks() -> None:
    templates = np.zeros((160, 1), dtype=np.float64)
    templates[0, 0] = 1.0
    for shots in (1, 2):
        rows, labels = _support(shots)
        transformed, audit = translate_to_counterfactual_robust_centers(
            rows, labels, 3, shots, templates, np.asarray([0.4])
        )
        np.testing.assert_array_equal(transformed, rows)
        assert audit["k1_k2_exact_identity"] is True

