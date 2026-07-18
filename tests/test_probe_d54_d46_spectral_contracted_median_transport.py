from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d54_d46_spectral_contracted_median_transport.py"
SPEC = importlib.util.spec_from_file_location("d54_probe_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d54 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d54)


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


def test_active_geometry_is_finite_and_spectrally_contracted() -> None:
    rows, labels = _support()
    geometry = d54._geometry(rows, labels, 3, 5, _base())
    assert geometry["status"] == "spectral_contracted_median_transport_active"
    assert geometry["correction"].shape == (3, 8)
    assert np.isfinite(geometry["correction"]).all()
    assert np.all(geometry["resultant"] > 0.0)
    assert np.all(geometry["resultant"] <= 1.0 + d54.UNIT_TOLERANCE)
    assert np.allclose(geometry["gamma"], 1.0 - geometry["resultant"])
    assert geometry["transport_spectral_norm"] <= geometry["transport_bound"] + 1e-12
    assert np.allclose(
        geometry["correction"],
        geometry["gamma"][:, None] * geometry["transported_direction"],
        atol=1.0e-12,
    )
    global_bound = (
        float(np.max(geometry["gamma"]))
        * geometry["transport_bound"]
        * float(np.linalg.norm(_base(), ord=2))
    )
    assert float(np.linalg.norm(geometry["correction"], ord=2)) <= global_bound + 1e-12


@pytest.mark.parametrize("k", [1, 2])
def test_k1_k2_are_exact_zero_residual_fallbacks(k: int) -> None:
    rows, labels = _support(k=k)
    geometry = d54._geometry(rows, labels, 3, k, _base())
    assert geometry["exact_fallback"] is True
    assert np.array_equal(geometry["correction"], np.zeros_like(geometry["correction"]))


def test_k1_fallback_precedes_centered_mean_spectral_degeneracy() -> None:
    row = np.zeros(8, dtype=np.float64)
    row[0] = 1.0
    rows = np.stack([row, row, row])
    labels = np.arange(3, dtype=np.int64)
    geometry = d54._geometry(rows, labels, 3, 1, _base())
    assert geometry["exact_fallback"] is True
    assert geometry["tau"] == 0.0
    assert np.array_equal(geometry["correction"], np.zeros_like(rows))


def test_rank_permutation_is_invariant() -> None:
    rows, labels = _support(k=8)
    base = _base()
    baseline = d54._geometry(rows, labels, 3, 8, base)
    order = np.concatenate(
        [np.asarray([5, 1, 7, 0, 3, 6, 2, 4]) + 8 * index for index in range(3)]
    )
    permuted = d54._geometry(rows[order], labels[order], 3, 8, base)
    assert np.allclose(permuted["correction"], baseline["correction"], atol=1.0e-14)
    assert np.allclose(permuted["resultant"], baseline["resultant"], atol=1.0e-14)


def test_class_permutation_is_equivariant() -> None:
    rows, labels = _support(k=5, c=4)
    base = _base(c=4)
    baseline = d54._geometry(rows, labels, 4, 5, base)
    order = np.asarray([2, 0, 3, 1])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    row_order = np.concatenate([np.flatnonzero(labels == old) for old in order])
    permuted_rows = rows[row_order]
    permuted_labels = inverse[labels[row_order]]
    permuted = d54._geometry(permuted_rows, permuted_labels, 4, 5, base[order])
    assert np.allclose(permuted["correction"], baseline["correction"][order], atol=1.0e-14)
    assert np.allclose(permuted["gamma"], baseline["gamma"][order], atol=1.0e-14)


def test_coordinate_median_rejects_single_outlier() -> None:
    rows, labels = _support(k=5, c=2, d=6)
    rows[4] = -rows[0]
    rows[4] /= np.linalg.norm(rows[4])
    geometry = d54._geometry(rows, labels, 2, 5, _base(c=2, d=6))
    assert not np.allclose(geometry["mean"][0], geometry["median"][0])
    assert np.linalg.norm(geometry["raw_direction"][0]) > 0.0


def test_nonunit_and_unequal_k_fail_closed() -> None:
    rows, labels = _support(k=4)
    bad = rows.copy()
    bad[0] *= 2.0
    with pytest.raises(d54.D54ProbeError, match="unit-sphere"):
        d54._geometry(bad, labels, 3, 4, _base())
    with pytest.raises(d54.D54ProbeError, match="cardinality"):
        d54._geometry(rows[:-1], labels[:-1], 3, 4, _base())
    with pytest.raises(d54.D54ProbeError, match="base coefficient"):
        d54._geometry(rows, labels, 3, 4, np.zeros((3, 7)))


def test_resource_upper_bound_is_deterministic_and_under_one_million() -> None:
    numeric, comparisons = d54._extra_resource(8, 6, 11, 288)
    assert numeric == 430_272
    assert 0 < numeric < 1_000_000
    assert comparisons == 117_504


def test_feature_dimension_comes_from_actual_state_not_resource_dict() -> None:
    state = SimpleNamespace(log_diag_fp32=np.zeros(288, dtype=np.float32))
    assert d54._state_dimension(state) == 288
    with pytest.raises(d54.D54ProbeError, match="feature dimension"):
        d54._state_dimension(SimpleNamespace(log_diag_fp32=np.zeros((2, 2))))


def test_formula_has_no_tunable_residual_coefficient() -> None:
    assert "diag(1-rho)" in d54.FORMULA
    assert "U*M0.T/||M0||2^2" in d54.FORMULA
    assert "pinv" not in d54.FORMULA.lower()
    assert "ridge" not in d54.FORMULA.lower()
    assert "alpha" not in d54.FORMULA.lower()
    assert "threshold" not in d54.FORMULA.lower()
