from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d52_bounded_base_relative_median_direction.py"
SPEC = importlib.util.spec_from_file_location("d52_probe_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d52 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d52)


def _support(k: int = 5, c: int = 3, d: int = 8, seed: int = 51):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(c, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows = []
    labels = []
    for index in range(c):
        samples = centers[index] + 0.15 * rng.normal(size=(k, d))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        rows.append(samples)
        labels.extend([index] * k)
    return np.concatenate(rows), np.asarray(labels, dtype=np.int64)


def _base(c: int = 3, d: int = 8, seed: int = 52) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.normal(size=(c, d))
    return value - value.mean(axis=0, keepdims=True)


def test_active_geometry_is_finite_and_intrinsically_bounded() -> None:
    rows, labels = _support()
    geometry = d52._geometry(rows, labels, 3, 5, _base())
    assert geometry["status"] == "bounded_base_relative_median_direction_active"
    assert geometry["correction"].shape == (3, 8)
    assert np.isfinite(geometry["correction"]).all()
    assert np.all(geometry["resultant"] > 0.0)
    assert np.all(geometry["resultant"] <= 1.0 + d52.UNIT_TOLERANCE)
    assert np.allclose(geometry["gamma"], 1.0 - geometry["resultant"])
    correction_norm = np.linalg.norm(geometry["correction"], axis=1)
    assert np.allclose(correction_norm, geometry["correction_bound"], atol=1.0e-12)
    assert np.allclose(
        geometry["correction_bound"],
        geometry["gamma"] * geometry["base_discriminant_norm"],
        atol=1.0e-12,
    )


@pytest.mark.parametrize("k", [1, 2])
def test_k1_k2_are_exact_zero_residual_fallbacks(k: int) -> None:
    rows, labels = _support(k=k)
    geometry = d52._geometry(rows, labels, 3, k, _base())
    assert geometry["exact_fallback"] is True
    assert np.array_equal(geometry["correction"], np.zeros_like(geometry["correction"]))


def test_rank_permutation_is_invariant() -> None:
    rows, labels = _support(k=8)
    base = _base()
    baseline = d52._geometry(rows, labels, 3, 8, base)
    order = np.concatenate(
        [np.asarray([5, 1, 7, 0, 3, 6, 2, 4]) + 8 * index for index in range(3)]
    )
    permuted = d52._geometry(rows[order], labels[order], 3, 8, base)
    assert np.allclose(permuted["correction"], baseline["correction"], atol=1.0e-14)
    assert np.allclose(permuted["resultant"], baseline["resultant"], atol=1.0e-14)


def test_class_permutation_is_equivariant() -> None:
    rows, labels = _support(k=5, c=4)
    base = _base(c=4)
    baseline = d52._geometry(rows, labels, 4, 5, base)
    order = np.asarray([2, 0, 3, 1])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    row_order = np.concatenate([np.flatnonzero(labels == old) for old in order])
    permuted_rows = rows[row_order]
    permuted_labels = inverse[labels[row_order]]
    permuted = d52._geometry(permuted_rows, permuted_labels, 4, 5, base[order])
    assert np.allclose(permuted["correction"], baseline["correction"][order], atol=1.0e-14)
    assert np.allclose(permuted["gamma"], baseline["gamma"][order], atol=1.0e-14)


def test_coordinate_median_rejects_single_outlier() -> None:
    rows, labels = _support(k=5, c=2, d=6)
    rows[4] = -rows[0]
    rows[4] /= np.linalg.norm(rows[4])
    geometry = d52._geometry(rows, labels, 2, 5, _base(c=2, d=6))
    assert not np.allclose(geometry["mean"][0], geometry["median"][0])
    assert np.linalg.norm(geometry["raw_direction"][0]) > 0.0


def test_nonunit_and_unequal_k_fail_closed() -> None:
    rows, labels = _support(k=4)
    bad = rows.copy()
    bad[0] *= 2.0
    with pytest.raises(d52.D52ProbeError, match="unit-sphere"):
        d52._geometry(bad, labels, 3, 4, _base())
    with pytest.raises(d52.D52ProbeError, match="cardinality"):
        d52._geometry(rows[:-1], labels[:-1], 3, 4, _base())
    with pytest.raises(d52.D52ProbeError, match="base coefficient"):
        d52._geometry(rows, labels, 3, 4, np.zeros((3, 7)))


def test_resource_upper_bound_is_deterministic_and_under_one_million() -> None:
    numeric, comparisons = d52._extra_resource(8, 6, 11, 288)
    assert numeric == 227_520
    assert 0 < numeric < 1_000_000
    assert comparisons == 117_504


def test_feature_dimension_comes_from_actual_state_not_resource_dict() -> None:
    state = SimpleNamespace(log_diag_fp32=np.zeros(288, dtype=np.float32))
    assert d52._state_dimension(state) == 288
    with pytest.raises(d52.D52ProbeError, match="feature dimension"):
        d52._state_dimension(SimpleNamespace(log_diag_fp32=np.zeros((2, 2))))


def test_formula_has_no_tunable_residual_coefficient() -> None:
    assert "(1-rho_c)" in d52.FORMULA
    assert "norm(Wc-mean(W))" in d52.FORMULA
    assert "support_rms" not in d52.FORMULA.lower()
    assert "alpha" not in d52.FORMULA.lower()
    assert "threshold" not in d52.FORMULA.lower()
