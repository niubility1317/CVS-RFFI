from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d60_crossblock_spectral_stability_shrinkage.py"
SPEC = importlib.util.spec_from_file_location("probe_d60_test_target", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d60 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d60)


SLICES = (slice(0, 2), slice(2, 4), slice(4, 6))


def _rows(seed: int, classes: int = 3, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=1.1, size=(classes, 6))
    rows = np.concatenate(
        [means[c] + rng.normal(scale=0.55, size=(k, 6)) for c in range(classes)]
    )
    labels = np.repeat(np.arange(classes), k)
    return rows.astype(np.float64), labels.astype(np.int64)


def _full(rows: np.ndarray, labels: np.ndarray, classes: int = 3) -> np.ndarray:
    return d60._auto_covariance(rows, labels, classes)


def test_rankwise_partition_is_exact_once() -> None:
    _, labels = _rows(1, k=5)
    held, audit = d60._rankwise_partitions(labels, 3, 5)
    assert len(held) == 5
    assert all(len(item) == 3 for item in held)
    assert sorted(int(v) for item in held for v in item) == list(range(15))
    assert audit["held_support_row_exact_once_coverage"] is True
    assert audit["train_held_overlap_count"] == 0


def test_rankwise_partition_rejects_unequal_k() -> None:
    labels = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    with pytest.raises(d60.D60ProbeError, match="symmetric"):
        d60._rankwise_partitions(labels, 3, 2)


def test_active_covariance_is_spd_and_stability_bounded() -> None:
    rows, labels = _rows(2)
    covariance, audit = d60._stability_contracted_covariance(
        _full(rows, labels), rows, labels, 3, 5, SLICES
    )
    stability = np.asarray(audit["d60_stability_by_mode"])
    assert audit["d60_stability_active"] is True
    assert stability.shape == (6,)
    assert np.min(stability) >= 0.0 and np.max(stability) <= 1.0
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0
    assert audit["d60_fold_rayleigh_shape"] == [5, 6]


def test_contracted_normalized_eigenvalues_stay_between_block_and_full() -> None:
    rows, labels = _rows(3)
    _, audit = d60._stability_contracted_covariance(
        _full(rows, labels), rows, labels, 3, 5, SLICES
    )
    full_values = 1.0 + np.asarray(audit["d60_full_crossblock_eigenvalue_by_mode"])
    stability = np.asarray(audit["d60_stability_by_mode"])
    contracted = 1.0 + stability * (full_values - 1.0)
    lower = np.minimum(1.0, full_values)
    upper = np.maximum(1.0, full_values)
    assert np.all(contracted >= lower - 1.0e-12)
    assert np.all(contracted <= upper + 1.0e-12)
    assert np.min(contracted) > 0.0


def test_k2_uses_exact_d59_midpoint_fallback() -> None:
    rows, labels = _rows(4, k=2)
    rng = np.random.default_rng(404)
    factor = rng.normal(size=(6, 6))
    full = factor @ factor.T + np.eye(6)
    block = d60.d59._three_block_covariance(full, SLICES)
    expected, _ = d60.d59._spd_geometric_midpoint(block, full)
    actual, audit = d60._stability_contracted_covariance(
        full, rows, labels, 3, 2, SLICES
    )
    assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-11)
    assert audit["d60_stability_active"] is False
    assert audit["d60_boundary_status"] == "k2_exact_d59_midpoint_fallback"


def test_class_label_permutation_does_not_change_covariance() -> None:
    rows, labels = _rows(5)
    full = _full(rows, labels)
    covariance, _ = d60._stability_contracted_covariance(
        full, rows, labels, 3, 5, SLICES
    )
    permutation = np.array([2, 0, 1], dtype=np.int64)
    permuted_labels = permutation[labels]
    permuted_full = _full(rows, permuted_labels)
    permuted, _ = d60._stability_contracted_covariance(
        permuted_full, rows, permuted_labels, 3, 5, SLICES
    )
    assert np.allclose(permuted, covariance, rtol=0.0, atol=2.0e-10)


def test_within_class_row_permutation_does_not_change_fold_set_result() -> None:
    rows, labels = _rows(6)
    covariance, _ = d60._stability_contracted_covariance(
        _full(rows, labels), rows, labels, 3, 5, SLICES
    )
    order = np.concatenate(
        [np.flatnonzero(labels == c)[::-1] for c in range(3)]
    )
    rows2, labels2 = rows[order], labels[order]
    covariance2, _ = d60._stability_contracted_covariance(
        _full(rows2, labels2), rows2, labels2, 3, 5, SLICES
    )
    assert np.allclose(covariance2, covariance, rtol=0.0, atol=2.0e-10)


def test_fold_rayleigh_hash_changes_with_support() -> None:
    rows1, labels1 = _rows(7)
    rows2, labels2 = _rows(8)
    _, audit1 = d60._stability_contracted_covariance(
        _full(rows1, labels1), rows1, labels1, 3, 5, SLICES
    )
    _, audit2 = d60._stability_contracted_covariance(
        _full(rows2, labels2), rows2, labels2, 3, 5, SLICES
    )
    assert len(audit1["d60_fold_rayleigh_sha256"]) == 64
    assert audit1["d60_fold_rayleigh_sha256"] != audit2["d60_fold_rayleigh_sha256"]


def test_stability_formula_matches_mean_square_ratio() -> None:
    q = np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    ratio = np.mean(q, axis=0) ** 2 / np.mean(q**2, axis=0)
    assert np.allclose(ratio, [1.0, 1.0 / 9.0])
    assert np.all((ratio >= 0.0) & (ratio <= 1.0))


def test_auto_covariance_rejects_missing_class_order() -> None:
    rows, labels = _rows(9)
    with pytest.raises(ValueError):
        d60._auto_covariance(rows[labels != 2], labels[labels != 2], 3)


def _stub_d42() -> SimpleNamespace:
    def original_fit(rows, labels, class_count, k_shot):
        values = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        means = np.stack([values[y == c].mean(axis=0) for c in range(class_count)])
        return means.astype(np.float32), (-0.5 * np.sum(means**2, axis=1)).astype(
            np.float32
        ), {"unit_covariance_fallback": True, "covariance_equation_residual_max": 0.0}

    return SimpleNamespace(
        _fit_equal_prior_lda=original_fit,
        ENERGY_EPSILON=1.0e-12,
        BLOCK_SLICES=SLICES,
        FEATURE_DIM=6,
    )


def test_built_fit_uses_one_shared_affine_head() -> None:
    stub = _stub_d42()
    fit, original = d60.build_d60_fit(stub)
    try:
        rows, labels = _rows(10)
        coef, intercept, audit = fit(rows, labels, 3, 5)
    finally:
        d60.d43._structured_covariance = original
    assert coef.shape == (3, 6) and intercept.shape == (3,)
    assert np.isfinite(coef).all() and np.isfinite(intercept).all()
    assert audit["d60_stability_active"] is True
    assert audit["d60_class_shared_covariance"] is True
    assert audit["d60_class_logit_scale_or_intercept_calibration"] is False


def test_built_fit_k1_is_exact_centered_d42_fallback() -> None:
    stub = _stub_d42()
    fit, original = d60.build_d60_fit(stub)
    try:
        rows, labels = _rows(11, k=1)
        raw_coef, raw_intercept, _ = stub._fit_equal_prior_lda(rows, labels, 3, 1)
        coef, intercept, audit = fit(rows, labels, 3, 1)
    finally:
        d60.d43._structured_covariance = original
    expected_coef, expected_intercept = d60.d43._center_affine_scores(
        raw_coef, raw_intercept
    )
    assert np.array_equal(coef, expected_coef.astype(np.float32))
    assert np.array_equal(intercept, expected_intercept.astype(np.float32))
    assert audit["d60_stability_active"] is False


def test_formula_and_arm_are_locked_without_threshold() -> None:
    assert "mean(q_rj)^2/mean(q_rj^2)" in d60.FORMULA
    assert "threshold" not in d60.FORMULA.lower()
    assert d60.d43.ARM_STRUCTURES[d60.ARM] == d60.STRUCTURE


def test_spectral_resource_formula_is_deterministic() -> None:
    assert d60._spectral_macs(288) == 28 * 288**3
