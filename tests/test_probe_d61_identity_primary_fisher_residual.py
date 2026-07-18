from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d61_identity_primary_fisher_residual.py"
SPEC = importlib.util.spec_from_file_location("probe_d61_test_target", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d61 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d61)


def _rows(seed: int, classes: int = 4, k: int = 5, dimension: int = 9):
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=1.2, size=(classes, dimension))
    rows = np.concatenate(
        [means[index] + rng.normal(scale=0.5, size=(k, dimension)) for index in range(classes)]
    )
    labels = np.repeat(np.arange(classes), k)
    return rows.astype(np.float64), labels.astype(np.int64)


def test_fisher_transform_is_identity_primary_and_gain_bounded() -> None:
    rows, labels = _rows(1)
    transform, audit = d61._fisher_residual_transform(rows, labels, 4, 5)
    eigenvalues = np.linalg.eigvalsh(transform)
    gain = np.asarray(audit["d61_gain_by_mode"])
    assert audit["d61_fisher_active"] is True
    assert 1 <= audit["d61_machine_rank"] <= 3
    assert np.all((gain >= 0.0) & (gain <= 1.0))
    assert np.min(eigenvalues) >= 1.0 - 1.0e-12
    assert np.max(eigenvalues) <= 2.0 + 1.0e-12


def test_k1_is_exact_identity_fallback() -> None:
    rows, labels = _rows(2, k=1)
    transform, audit = d61._fisher_residual_transform(rows, labels, 4, 1)
    assert np.array_equal(transform, np.eye(rows.shape[1]))
    assert audit["d61_boundary_status"] == "k1_exact_d46_fallback"
    assert audit["d61_fisher_active"] is False


def test_class_id_permutation_does_not_change_transform() -> None:
    rows, labels = _rows(3)
    first, _ = d61._fisher_residual_transform(rows, labels, 4, 5)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    second, _ = d61._fisher_residual_transform(rows, permutation[labels], 4, 5)
    assert np.allclose(first, second, rtol=0.0, atol=2.0e-12)


def test_orthogonal_complement_remains_identity() -> None:
    rows, labels = _rows(4, classes=3, dimension=10)
    transform, audit = d61._fisher_residual_transform(rows, labels, 3, 5)
    eigenvalues = np.linalg.eigvalsh(transform)
    identity_count = int(np.sum(np.isclose(eigenvalues, 1.0, rtol=0.0, atol=2.0e-12)))
    assert identity_count >= rows.shape[1] - audit["d61_machine_rank"]


def _base_fit(rows, labels, class_count, k_shot):
    values = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    means = np.stack([values[y == index].mean(axis=0) for index in range(class_count)])
    intercept = -0.5 * np.sum(means**2, axis=1)
    return means.astype(np.float32), intercept.astype(np.float32), {
        "unit_covariance_fallback": bool(k_shot == 1),
        "covariance_equation_residual_max": 0.0,
    }


def test_component_wrapper_compiles_to_one_equivalent_affine_head() -> None:
    rows, labels = _rows(5)
    records = []
    fit = d61._wrap_component_fit(_base_fit, "unit", records)
    coefficient, intercept, audit = fit(rows, labels, 4, 5)
    transform, _ = d61._fisher_residual_transform(rows, labels, 4, 5)
    prime, expected_intercept, _ = _base_fit(rows @ transform, labels, 4, 5)
    expected = np.asarray(prime, dtype=np.float64) @ transform.T
    assert np.allclose(coefficient, expected.astype(np.float32), rtol=0.0, atol=2.0e-6)
    assert np.array_equal(intercept, expected_intercept)
    assert audit["d61_single_affine_state_only"] is True
    assert audit["d61_uses_held_or_query"] is False
    assert len(records) == 1


def test_each_component_call_refits_transform_from_its_own_rows() -> None:
    first_rows, first_labels = _rows(6)
    second_rows, second_labels = _rows(7)
    records = []
    fit = d61._wrap_component_fit(_base_fit, "unit", records)
    fit(first_rows, first_labels, 4, 5)
    fit(second_rows, second_labels, 4, 5)
    assert len(records) == 2
    assert records[0]["transform_sha256"] != records[1]["transform_sha256"]


def test_formula_has_no_rank_or_gain_scan() -> None:
    assert "b/(b+w)" in d61.FORMULA
    assert "machine_rank" in d61.RANK_POLICY
    assert d61._fisher_dense_macs(288, 36) == 36 * 8 * 288**3


def test_build_d61_fit_wraps_both_d46_components() -> None:
    stub = SimpleNamespace(
        _fit_equal_prior_lda=_base_fit,
        FEATURE_DIM=9,
        ENERGY_EPSILON=1.0e-12,
        BLOCK_SLICES=(slice(0, 3), slice(3, 6), slice(6, 9)),
    )
    fit, records = d61.build_d61_fit(stub)
    rows, labels = _rows(8, classes=3, k=3, dimension=9)
    coefficient, intercept, audit = fit(rows, labels, 3, 3)
    assert coefficient.shape == (3, 9)
    assert intercept.shape == (3,)
    assert np.isfinite(coefficient).all()
    assert audit["d61_component"] == "d46_full"
    assert {record["component"] for record in records} == {"d46_full", "d46_block3"}
    assert len(records) == 2 * (3 + 1)
