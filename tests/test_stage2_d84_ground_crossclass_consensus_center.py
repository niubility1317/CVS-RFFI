from __future__ import annotations

import numpy as np

from cvsrffi.stage2_d84_ground_crossclass_consensus_center import (
    ground_crossclass_consensus_templates,
    translate_to_consensus_robust_centers,
)


def _ground(seed: int = 17):
    rng = np.random.default_rng(seed)
    domain = rng.normal(scale=0.2, size=(14, 160))
    identity = rng.normal(size=(6, 160))
    interaction = rng.normal(scale=0.05, size=(14, 6, 160))
    prototypes = identity[None, :, :] + domain[:, None, :] + interaction
    return prototypes, np.ones((14, 6), dtype=np.uint8)


def _support(seed: int = 23, classes: int = 4, shots: int = 8):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(size=(classes, 288))
    rows = means[labels] + 0.1 * rng.normal(size=(classes * shots, 288))
    return rows, labels


def test_templates_discard_ground_class_identity_and_are_permutation_invariant():
    prototypes, mask = _ground()
    templates, weights, audit = ground_crossclass_consensus_templates(
        prototypes, mask
    )
    permutation = np.array([3, 1, 5, 0, 4, 2])
    templates2, weights2, audit2 = ground_crossclass_consensus_templates(
        prototypes[:, permutation], mask[:, permutation]
    )
    np.testing.assert_allclose(templates2, templates, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(weights2, weights, rtol=0.0, atol=1e-14)
    assert audit2["template_sha256"] == audit["template_sha256"]
    assert audit["ground_class_centers_discarded"] is True
    assert audit["hyperparameter_count"] == 0


def test_center_translation_preserves_residuals_and_auxiliary_features():
    prototypes, mask = _ground()
    templates, weights, _ = ground_crossclass_consensus_templates(prototypes, mask)
    rows, labels = _support()
    transformed, audit = translate_to_consensus_robust_centers(
        rows, labels, 4, 8, templates, weights
    )
    for index in range(4):
        before = rows[labels == index] - rows[labels == index].mean(axis=0)
        after = transformed[labels == index] - transformed[labels == index].mean(axis=0)
        np.testing.assert_allclose(after, before, rtol=0.0, atol=2e-12)
    np.testing.assert_array_equal(transformed[:, 160:], rows[:, 160:])
    assert audit["d84_query_rows_used"] == 0
    assert audit["d84_old_new_role_specific_branch"] is False


def test_target_class_permutation_is_equivariant():
    prototypes, mask = _ground()
    templates, weights, _ = ground_crossclass_consensus_templates(prototypes, mask)
    rows, labels = _support()
    transformed, _ = translate_to_consensus_robust_centers(
        rows, labels, 4, 8, templates, weights
    )
    permutation = np.array([2, 0, 3, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(4)
    transformed2, _ = translate_to_consensus_robust_centers(
        rows, inverse[labels], 4, 8, templates, weights
    )
    np.testing.assert_allclose(transformed2, transformed, rtol=0.0, atol=1e-14)


def test_k1_is_exact_identity():
    prototypes, mask = _ground()
    templates, weights, _ = ground_crossclass_consensus_templates(prototypes, mask)
    rows, labels = _support(classes=3, shots=1)
    transformed, audit = translate_to_consensus_robust_centers(
        rows, labels, 3, 1, templates, weights
    )
    np.testing.assert_array_equal(transformed, rows)
    assert audit["k1_k2_exact_identity"] is True
