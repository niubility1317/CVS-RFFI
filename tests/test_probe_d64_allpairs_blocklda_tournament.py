from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d64_allpairs_blocklda_tournament.py"
SPEC = importlib.util.spec_from_file_location("probe_d64_test_target", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d64 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d64)


def _stub() -> SimpleNamespace:
    def base_fit(rows, labels, class_count, k_shot):
        values = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        means = np.stack([values[y == index].mean(axis=0) for index in range(class_count)])
        return (
            means.astype(np.float32),
            (-0.5 * np.sum(means**2, axis=1)).astype(np.float32),
            {"unit_covariance_fallback": True, "covariance_equation_residual_max": 0.0},
        )

    return SimpleNamespace(
        _fit_equal_prior_lda=base_fit,
        FEATURE_DIM=9,
        ENERGY_EPSILON=1.0e-12,
        BLOCK_SLICES=(slice(0, 3), slice(3, 6), slice(6, 9)),
    )


def _support(classes: int = 4, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(64)
    means = rng.normal(scale=2.0, size=(classes, 9))
    rows = np.concatenate(
        [means[index] + rng.normal(scale=0.2, size=(k, 9)) for index in range(classes)]
    )
    return rows, np.repeat(np.arange(classes), k)


def test_exact_pair_count_and_compiled_shape() -> None:
    rows, labels = _support()
    fit, records = d64.build_d64_fit(_stub())
    coefficient, intercept, audit = fit(rows, labels, 4, 4)
    assert coefficient.shape == (4, 9) and intercept.shape == (4,)
    assert audit["d64_pair_count"] == 6
    assert len(records) == 6
    assert audit["d64_single_affine_state_only"] is True
    assert audit["d64_pair_graph_persisted_for_query"] is False
    assert audit["covariance_policy"] == d64.STATE_COVARIANCE_POLICY
    assert audit["d43_covariance_structure"] == d64.STRUCTURE
    assert np.isfinite(coefficient).all() and np.isfinite(intercept).all()


def test_compiled_tournament_fits_separated_support() -> None:
    rows, labels = _support()
    fit, _ = d64.build_d64_fit(_stub())
    coefficient, intercept, audit = fit(rows, labels, 4, 4)
    predicted = np.argmax(rows @ coefficient.T + intercept[None, :], axis=1)
    assert np.mean(predicted == labels) >= 0.9
    assert audit["d64_compiled_support_accuracy"] >= 0.9


def test_pair_rms_removes_common_positive_score_scale() -> None:
    rows = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    def make_fit(scale):
        def pair_fit(_rows, _labels, _class_count, _k):
            coef = scale * np.asarray([[-1.0], [1.0]])
            return coef, np.zeros(2), {"unit_covariance_fallback": False}
        return pair_fit

    first = d64._normalized_pair_margin(make_fit(1.0), rows, labels, 2)
    second = d64._normalized_pair_margin(make_fit(7.0), rows, labels, 2)
    np.testing.assert_allclose(first[0], second[0])
    assert first[1] == pytest.approx(second[1])


def test_class_permutation_equivariance() -> None:
    rows, labels = _support(classes=4, k=4)
    first_fit, _ = d64.build_d64_fit(_stub())
    coef1, bias1, _ = first_fit(rows, labels, 4, 4)
    permutation = np.asarray([2, 0, 3, 1])
    second_fit, _ = d64.build_d64_fit(_stub())
    coef2, bias2, _ = second_fit(rows, permutation[labels], 4, 4)
    np.testing.assert_allclose(coef2[permutation], coef1, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(bias2[permutation], bias1, rtol=2e-5, atol=2e-5)


def test_support_row_permutation_invariance() -> None:
    rows, labels = _support()
    order = np.random.default_rng(640).permutation(len(rows))
    first_fit, _ = d64.build_d64_fit(_stub())
    second_fit, _ = d64.build_d64_fit(_stub())
    coef1, bias1, _ = first_fit(rows, labels, 4, 4)
    coef2, bias2, _ = second_fit(rows[order], labels[order], 4, 4)
    np.testing.assert_allclose(coef1, coef2, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(bias1, bias2, rtol=2e-5, atol=2e-5)


def test_rejects_non_symmetric_or_nonfinite_support() -> None:
    rows, labels = _support(classes=3, k=3)
    with pytest.raises(d64.D64ProbeError, match="symmetric"):
        d64._validate_symmetric_support(rows[:-1], labels[:-1], 3, 3)
    rows[0, 0] = np.nan
    with pytest.raises(d64.D64ProbeError, match="finite"):
        d64._validate_symmetric_support(rows, labels, 3, 3)


def test_formula_forbids_role_scene_and_tunable_gate() -> None:
    lowered = d64.FORMULA.lower()
    assert "every unordered class pair" in lowered and "rms" in lowered
    for forbidden in ("role", "scene", "threshold", "alpha", "temperature"):
        assert forbidden not in lowered
