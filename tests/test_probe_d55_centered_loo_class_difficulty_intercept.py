from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d55_centered_loo_class_difficulty_intercept.py"
SPEC = importlib.util.spec_from_file_location("d55_probe_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d55 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d55)


def _audit() -> dict[str, object]:
    return {
        "d46_full_inner_loo_ce_by_class": [0.2, 0.4, 0.8],
        "d46_block_inner_loo_ce_by_class": [0.3, 0.5, 0.6],
        "d46_full_weight_by_class": [0.75, 0.25, 0.5],
        "d46_block_weight_by_class": [0.25, 0.75, 0.5],
    }


def test_active_difficulty_is_weighted_and_centered() -> None:
    result = d55._difficulty(_audit(), 5, 3)
    expected = np.asarray([0.225, 0.475, 0.7])
    assert result["status"] == "centered_loo_class_difficulty_intercept_active"
    assert np.allclose(result["difficulty"], expected)
    assert np.allclose(result["delta_intercept"], expected - expected.mean())
    assert abs(float(np.sum(result["delta_intercept"]))) < 1.0e-14


@pytest.mark.parametrize("k", [1, 2])
def test_k1_k2_are_exact_d46_fallbacks_without_evidence(k: int) -> None:
    result = d55._difficulty({}, k, 3)
    assert result["exact_fallback"] is True
    assert np.array_equal(result["delta_intercept"], np.zeros(3))


def test_class_permutation_is_equivariant() -> None:
    baseline = d55._difficulty(_audit(), 5, 3)
    order = np.asarray([2, 0, 1])
    audit = _audit()
    permuted = {key: np.asarray(value)[order].tolist() for key, value in audit.items()}
    actual = d55._difficulty(permuted, 5, 3)
    assert np.allclose(actual["delta_intercept"], baseline["delta_intercept"][order])


def test_invalid_evidence_and_weight_simplex_fail_closed() -> None:
    bad = _audit()
    bad["d46_full_weight_by_class"] = [0.9, 0.9, 0.9]
    with pytest.raises(d55.D55ProbeError, match="simplex"):
        d55._difficulty(bad, 5, 3)
    with pytest.raises(d55.D55ProbeError, match="evidence"):
        d55._difficulty(_audit(), 5, 4)


def test_resource_account_is_small_and_deterministic() -> None:
    assert d55._extra_resource(8, 6, 11, 288) == (136, 0)


def test_formula_has_no_tunable_or_role_specific_surface() -> None:
    assert "w_g_c*CE_g_c" in d55.FORMULA
    assert "mean_j" in d55.FORMULA
    assert "alpha" not in d55.FORMULA.lower()
    assert "threshold" not in d55.FORMULA.lower()
