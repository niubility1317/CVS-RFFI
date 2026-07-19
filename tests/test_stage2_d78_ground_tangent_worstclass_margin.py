from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "code" / "cvsrffi" / "stage2_d78_ground_tangent_worstclass_margin.py"
SPEC = importlib.util.spec_from_file_location("d78_core_test", PATH)
assert SPEC is not None and SPEC.loader is not None
d78 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d78)


def _lda_fit(rows, labels, class_count, k_shot):
    del k_shot
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    means = np.stack([np.mean(x[y == c], axis=0) for c in range(class_count)])
    coefficient = means
    intercept = -0.5 * np.sum(means * means, axis=1)
    return coefficient, intercept, {}


def _support(seed=17):
    rng = np.random.default_rng(seed)
    classes, k_shot, dimension = 3, 4, 8
    centers = np.asarray(
        [
            [0.8, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.7, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
            [-0.5, -0.2, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rows = []
    labels = []
    for c in range(classes):
        for _ in range(k_shot):
            rows.append(centers[c] + rng.normal(scale=0.38, size=dimension))
            labels.append(c)
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, 2)))
    base = centers * 0.45
    return np.asarray(rows), np.asarray(labels), base, basis


def test_ground_tangent_is_rank_limited_orthonormal_and_permutation_invariant():
    rng = np.random.default_rng(123)
    prototypes = rng.normal(size=(14, 6, 160))
    mask = np.ones((14, 6), dtype=np.uint8)
    basis, audit = d78.ground_domain_tangent_basis(
        prototypes, mask, feature_dim=288
    )
    assert basis.shape == (288, 13)
    assert audit["tangent_rank"] == 13
    assert audit["ground_component_input_count"] == 84
    np.testing.assert_allclose(basis.T @ basis, np.eye(13), atol=1e-10, rtol=0)
    assert np.count_nonzero(basis[160:]) == 0

    permuted, permuted_audit = d78.ground_domain_tangent_basis(
        prototypes[np.arange(14)[::-1]][:, [2, 0, 5, 1, 4, 3]],
        mask[np.arange(14)[::-1]][:, [2, 0, 5, 1, 4, 3]],
        feature_dim=288,
    )
    np.testing.assert_allclose(
        basis @ basis.T, permuted @ permuted.T, atol=2e-12, rtol=0
    )
    assert audit["tangent_rank"] == permuted_audit["tangent_rank"]


def test_fit_is_deterministic_monotone_and_stays_in_ground_subspace():
    rows, labels, base, basis = _support()
    delta1, audit1 = d78.fit_ground_tangent_worstclass_margin(
        rows,
        labels,
        3,
        4,
        base_coefficient=base,
        tangent_basis=basis,
        lda_fit=_lda_fit,
    )
    delta2, audit2 = d78.fit_ground_tangent_worstclass_margin(
        rows,
        labels,
        3,
        4,
        base_coefficient=base,
        tangent_basis=basis,
        lda_fit=_lda_fit,
    )
    assert audit1["optimizer_iterations"] == 20
    assert len(audit1["optimizer_objective_trace"]) == 20
    assert all(
        row["objective_after"] <= row["objective_before"] + 1e-12
        for row in audit1["optimizer_objective_trace"]
    )
    assert audit1["final_objective"] <= audit1["initial_objective"]
    assert audit1["query_rows_used"] == 0
    assert audit1["ground_class_score_access"] is False
    np.testing.assert_array_equal(delta1, delta2)
    assert audit1["residual_sha256"] == audit2["residual_sha256"]
    np.testing.assert_allclose(np.mean(delta1, axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(delta1, (delta1 @ basis) @ basis.T, atol=2e-7)


def test_class_permutation_equivariance():
    rows, labels, base, basis = _support(29)
    delta, _ = d78.fit_ground_tangent_worstclass_margin(
        rows,
        labels,
        3,
        4,
        base_coefficient=base,
        tangent_basis=basis,
        lda_fit=_lda_fit,
    )
    permutation = np.asarray([2, 0, 1])
    inverse = np.argsort(permutation)
    permuted_labels = permutation[labels]
    permuted_base = base[inverse]
    permuted_delta, _ = d78.fit_ground_tangent_worstclass_margin(
        rows,
        permuted_labels,
        3,
        4,
        base_coefficient=permuted_base,
        tangent_basis=basis,
        lda_fit=_lda_fit,
    )
    np.testing.assert_allclose(permuted_delta[permutation], delta, atol=3e-7)


def test_k1_is_exact_fallback():
    rows, labels, base, basis = _support()
    chosen = np.asarray([0, 4, 8])
    delta, audit = d78.fit_ground_tangent_worstclass_margin(
        rows[chosen],
        labels[chosen],
        3,
        1,
        base_coefficient=base,
        tangent_basis=basis,
        lda_fit=_lda_fit,
    )
    assert audit["status"] == "k1_exact_d62_fallback"
    assert audit["optimizer_iterations"] == 0
    np.testing.assert_array_equal(delta, np.zeros_like(delta))
