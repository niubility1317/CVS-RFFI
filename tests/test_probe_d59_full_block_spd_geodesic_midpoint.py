from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d59_full_block_spd_geodesic_midpoint.py"
SPEC = importlib.util.spec_from_file_location("probe_d59_test_target", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d59 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d59)


def _spd(seed: int, dimension: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension))
    return matrix @ matrix.T + 0.7 * np.eye(dimension)


def _blocks(matrix: np.ndarray) -> np.ndarray:
    return d59._three_block_covariance(
        matrix, (slice(0, 2), slice(2, 4), slice(4, 6))
    )


def test_midpoint_is_spd_and_symmetric() -> None:
    full = _spd(1)
    block = _blocks(full)
    midpoint, audit = d59._spd_geometric_midpoint(block, full)
    assert np.allclose(midpoint, midpoint.T, rtol=0.0, atol=1.0e-12)
    assert np.min(np.linalg.eigvalsh(midpoint)) > 0.0
    assert audit["d59_midpoint_eigenvalue_min"] > 0.0


def test_midpoint_satisfies_riccati_identity() -> None:
    full = _spd(2)
    block = _blocks(full)
    midpoint, audit = d59._spd_geometric_midpoint(block, full)
    reconstructed = midpoint @ np.linalg.inv(block) @ midpoint
    relative = np.linalg.norm(reconstructed - full, ord="fro") / np.linalg.norm(
        full, ord="fro"
    )
    assert relative < d59.RICCATI_RELATIVE_TOLERANCE
    assert audit["d59_riccati_relative_frobenius_residual"] < d59.RICCATI_RELATIVE_TOLERANCE


def test_identical_endpoints_return_same_matrix() -> None:
    matrix = _spd(3)
    midpoint, audit = d59._spd_geometric_midpoint(matrix, matrix)
    assert np.allclose(midpoint, matrix, rtol=0.0, atol=2.0e-12)
    assert audit["d59_affine_distance_block_to_full"] < 1.0e-12


def test_geometric_midpoint_is_endpoint_symmetric() -> None:
    left = _spd(4)
    right = _spd(5)
    forward, _ = d59._spd_geometric_midpoint(left, right)
    reverse, _ = d59._spd_geometric_midpoint(right, left)
    assert np.allclose(forward, reverse, rtol=0.0, atol=2.0e-10)


def test_geometric_midpoint_is_positive_scale_equivariant() -> None:
    left = _spd(6)
    right = _spd(7)
    midpoint, _ = d59._spd_geometric_midpoint(left, right)
    scaled, _ = d59._spd_geometric_midpoint(3.25 * left, 3.25 * right)
    assert np.allclose(scaled, 3.25 * midpoint, rtol=0.0, atol=3.0e-10)


def test_geometric_midpoint_is_orthogonal_congruence_equivariant() -> None:
    rng = np.random.default_rng(8)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    left = _spd(9)
    right = _spd(10)
    midpoint, _ = d59._spd_geometric_midpoint(left, right)
    rotated, _ = d59._spd_geometric_midpoint(q @ left @ q.T, q @ right @ q.T)
    assert np.allclose(rotated, q @ midpoint @ q.T, rtol=0.0, atol=3.0e-10)


def test_midpoint_lies_at_half_affine_distance() -> None:
    full = _spd(11)
    block = _blocks(full)
    _, audit = d59._spd_geometric_midpoint(block, full)
    total = audit["d59_affine_distance_block_to_full"]
    assert audit["d59_affine_distance_block_to_midpoint"] == pytest.approx(
        total / 2.0, abs=2.0e-9
    )
    assert audit["d59_affine_distance_midpoint_to_full"] == pytest.approx(
        total / 2.0, abs=2.0e-9
    )


def test_three_blocks_preserve_diagonal_blocks_and_zero_cross_blocks() -> None:
    full = _spd(12)
    block = _blocks(full)
    assert np.array_equal(block[:2, :2], full[:2, :2])
    assert np.array_equal(block[2:4, 2:4], full[2:4, 2:4])
    assert np.array_equal(block[4:, 4:], full[4:, 4:])
    assert np.count_nonzero(block[:2, 2:]) == 0
    assert np.count_nonzero(block[2:4, :2]) == 0


def test_midpoint_retains_nonzero_cross_block_structure() -> None:
    full = _spd(13)
    block = _blocks(full)
    midpoint, _ = d59._spd_geometric_midpoint(block, full)
    assert np.linalg.norm(midpoint - _blocks(midpoint), ord="fro") > 0.0
    assert not np.allclose(midpoint, block)
    assert not np.allclose(midpoint, full)


@pytest.mark.parametrize(
    "bad",
    [
        np.eye(3)[:, :2],
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
    ],
)
def test_invalid_covariance_fails_closed(bad: np.ndarray) -> None:
    with pytest.raises(d59.D59ProbeError):
        d59._spd_geometric_midpoint(bad, bad)


def test_block_slices_must_partition_axis_exactly() -> None:
    full = _spd(14)
    with pytest.raises(d59.D59ProbeError, match="partition"):
        d59._three_block_covariance(full, (slice(0, 3), slice(2, 6)))
    with pytest.raises(d59.D59ProbeError, match="partition"):
        d59._three_block_covariance(full, (slice(0, 2), slice(3, 6)))


def test_array_hash_is_dtype_and_layout_canonical() -> None:
    matrix = _spd(15)
    assert d59._array_sha256(matrix) == d59._array_sha256(np.array(matrix, order="F"))
    assert d59._array_sha256(matrix) != d59._array_sha256(matrix + np.eye(6))


def test_extra_resource_is_deterministic_and_positive() -> None:
    expected = 40 * 288**3
    assert d59._extra_resource_per_active_fit(288) == expected
    with pytest.raises(d59.D59ProbeError):
        d59._extra_resource_per_active_fit(0)


def _stub_d42() -> SimpleNamespace:
    def original_fit(rows, labels, class_count, k_shot):
        values = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        means = np.stack([values[y == c].mean(axis=0) for c in range(class_count)])
        coef = means.astype(np.float32)
        intercept = (-0.5 * np.sum(means**2, axis=1)).astype(np.float32)
        return coef, intercept, {
            "unit_covariance_fallback": True,
            "covariance_equation_residual_max": 0.0,
        }

    return SimpleNamespace(
        _fit_equal_prior_lda=original_fit,
        ENERGY_EPSILON=1.0e-12,
        BLOCK_SLICES=(slice(0, 2), slice(2, 4), slice(4, 6)),
        FEATURE_DIM=6,
    )


def _balanced_rows(seed: int, classes: int = 3, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=1.4, size=(classes, 6))
    rows = np.concatenate(
        [means[c][None, :] + rng.normal(scale=0.4, size=(k, 6)) for c in range(classes)]
    )
    labels = np.repeat(np.arange(classes), k)
    return rows.astype(np.float64), labels.astype(np.int64)


def test_fit_builds_single_shared_midpoint_head() -> None:
    stub = _stub_d42()
    fit, original_structured = d59.build_d59_fit(stub)
    try:
        rows, labels = _balanced_rows(16)
        coef, intercept, audit = fit(rows, labels, 3, 4)
    finally:
        d59.d43._structured_covariance = original_structured
    assert coef.shape == (3, 6)
    assert intercept.shape == (3,)
    assert np.isfinite(coef).all() and np.isfinite(intercept).all()
    assert audit["d59_midpoint_active"] is True
    assert audit["d59_class_shared_covariance"] is True
    assert audit["d59_class_logit_scale_or_intercept_calibration"] is False
    assert audit["d59_single_affine_state_only"] is True


def test_fit_is_class_label_permutation_equivariant() -> None:
    stub = _stub_d42()
    fit, original_structured = d59.build_d59_fit(stub)
    try:
        rows, labels = _balanced_rows(17)
        coef, intercept, _ = fit(rows, labels, 3, 4)
        permutation = np.array([2, 0, 1], dtype=np.int64)
        permuted_labels = permutation[labels]
        permuted_coef, permuted_intercept, _ = fit(rows, permuted_labels, 3, 4)
    finally:
        d59.d43._structured_covariance = original_structured
    assert np.allclose(permuted_coef[permutation], coef, rtol=0.0, atol=2.0e-5)
    assert np.allclose(permuted_intercept[permutation], intercept, rtol=0.0, atol=2.0e-5)


def test_k1_uses_exact_d42_fallback_without_midpoint() -> None:
    stub = _stub_d42()
    fit, original_structured = d59.build_d59_fit(stub)
    try:
        rows, labels = _balanced_rows(18, classes=3, k=1)
        expected_coef, expected_intercept, _ = stub._fit_equal_prior_lda(
            rows, labels, 3, 1
        )
        coef, intercept, audit = fit(rows, labels, 3, 1)
    finally:
        d59.d43._structured_covariance = original_structured
    centered_coef, centered_intercept = d59.d43._center_affine_scores(
        expected_coef, expected_intercept
    )
    assert np.allclose(coef, centered_coef.astype(np.float32), rtol=0.0, atol=0.0)
    assert np.allclose(intercept, centered_intercept.astype(np.float32), rtol=0.0, atol=0.0)
    assert audit["d59_midpoint_active"] is False
    assert "fallback" in audit["d59_boundary_status"]


def test_formula_has_no_tunable_geodesic_weight() -> None:
    assert "0.5" not in d59.FORMULA
    assert d59.ARM == "full_block_spd_geodesic_midpoint"
    assert d59.STRUCTURE == "affine_invariant_spd_midpoint_full_auto_and_block3"


def test_arm_is_registered_with_d43_verifier_surface() -> None:
    assert d59.d43.ARM_STRUCTURES[d59.ARM] == d59.STRUCTURE
    assert d59.ARM in d59.d43.ARMS
