from __future__ import annotations

import numpy as np

from cvsrffi.stage2_d91_crossfit_consensus_sigma_margin import (
    fit_crossfit_consensus_sigma_margin,
    ground_radius_sigma_geometry,
)


def _ground(seed: int = 9101):
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
    return means, -0.5 * np.sum(means * means, axis=1), {"test_fit": True}


def _support(seed: int = 9102, classes: int = 4, shots: int = 5):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), shots)
    means = rng.normal(scale=0.3, size=(classes, 288))
    rows = means[labels] + rng.normal(scale=0.15, size=(classes * shots, 288))
    return rows, labels


def test_consensus_is_bounded_centered_and_query_free() -> None:
    prototypes, radius = _ground()
    basis, offsets, _ = ground_radius_sigma_geometry(prototypes, radius)
    rows, labels = _support()
    base_w, _, _ = _lda_fit(rows, labels, 4, 5)
    delta_w, delta_b, audit = fit_crossfit_consensus_sigma_margin(
        rows,
        labels,
        4,
        5,
        base_coefficient=base_w,
        tangent_basis=basis,
        counterfactual_offsets=offsets,
        lda_fit=_lda_fit,
    )
    assert 0.0 <= audit["consensus_factor"] <= 1.0
    assert audit["fold_gradient_count"] == 5
    assert audit["query_rows_used"] == 0
    assert audit["old_new_role_specific_branch"] is False
    assert audit["class_permutation_equivariant"] is True
    assert audit["residual_frobenius"] <= audit["d87_unshrunk_residual_frobenius"] + 1e-9
    np.testing.assert_allclose(delta_w.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(delta_w @ rows.mean(axis=0) + delta_b, 0.0, atol=1e-5)


def test_class_permutation_is_equivariant() -> None:
    prototypes, radius = _ground()
    basis, offsets, _ = ground_radius_sigma_geometry(prototypes, radius)
    rows, labels = _support(classes=3, shots=4)
    base_w, _, _ = _lda_fit(rows, labels, 3, 4)
    w1, b1, a1 = fit_crossfit_consensus_sigma_margin(
        rows, labels, 3, 4, base_coefficient=base_w,
        tangent_basis=basis, counterfactual_offsets=offsets, lda_fit=_lda_fit,
    )
    permutation = np.asarray([2, 0, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(3)
    w2, b2, a2 = fit_crossfit_consensus_sigma_margin(
        rows, inverse[labels], 3, 4, base_coefficient=base_w[permutation],
        tangent_basis=basis, counterfactual_offsets=offsets, lda_fit=_lda_fit,
    )
    np.testing.assert_allclose(w2[inverse], w1, atol=1e-6)
    np.testing.assert_allclose(b2[inverse], b1, atol=1e-6)
    np.testing.assert_allclose(a2["consensus_factor"], a1["consensus_factor"], atol=1e-12)


def test_k1_is_exact_fallback() -> None:
    prototypes, radius = _ground()
    basis, offsets, _ = ground_radius_sigma_geometry(prototypes, radius)
    rows, labels = _support(classes=3, shots=1)
    base_w, _, _ = _lda_fit(rows, labels, 3, 1)
    w, b, audit = fit_crossfit_consensus_sigma_margin(
        rows, labels, 3, 1, base_coefficient=base_w,
        tangent_basis=basis, counterfactual_offsets=offsets, lda_fit=_lda_fit,
    )
    np.testing.assert_array_equal(w, 0.0)
    np.testing.assert_array_equal(b, 0.0)
    assert audit["status"] == "k1_exact_d62_fallback"
