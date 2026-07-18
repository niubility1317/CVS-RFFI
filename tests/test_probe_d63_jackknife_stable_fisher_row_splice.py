from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d63_jackknife_stable_fisher_row_splice.py"
SPEC = importlib.util.spec_from_file_location("probe_d63_test_target", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d63 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d63)


def _perfect_scores(k: int = 4, classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    truth = np.tile(np.arange(classes), (k, 1))
    scores = np.zeros((k, classes, classes), dtype=np.float64)
    for fold in range(k):
        scores[fold, np.arange(classes), np.arange(classes)] = 2.0
    return scores, truth


def test_gate_rejects_no_strict_aggregate_gain() -> None:
    base, truth = _perfect_scores()
    gate = d63._jackknife_pareto_gate(base, base.copy(), truth)
    assert not np.any(gate["aggregate_initial_accept"])
    assert gate["exact_fallback"] is True


def test_gate_accepts_gain_that_is_safe_in_every_jackknife_subset() -> None:
    base, truth = _perfect_scores()
    base[0, 0] = [0.0, 1.0, 0.0]
    residual = base.copy()
    residual[0, 0, 0] = 2.0
    gate = d63._jackknife_pareto_gate(base, residual, truth)
    assert gate["aggregate_initial_accept"].tolist() == [True, False, False]
    assert gate["final_accept"].tolist() == [True, False, False]
    assert np.all(gate["jackknife_coordinate_safe"][:, 0])


def test_jackknife_rejects_aggregate_gain_that_hides_fold_harm() -> None:
    base, truth = _perfect_scores(k=4, classes=3)
    base[0, 0] = [0.0, 1.0, 0.0]
    base[1, 2] = [3.0, 2.0, 0.0]
    residual = base.copy()
    residual[0, 0, 0] = 2.0
    residual[1, 2, 0] = 0.0
    gate = d63._jackknife_pareto_gate(base, residual, truth)
    assert gate["aggregate_initial_accept"][0]
    assert not np.all(gate["jackknife_coordinate_safe"][:, 0])
    assert not gate["final_accept"][0]
    assert gate["status"] == "jackknife_coordinate_exact_d46_fallback"


def test_gate_rejects_invalid_or_too_short_evidence() -> None:
    base, truth = _perfect_scores(k=2)
    with pytest.raises(d63.D63ProbeError, match="evidence"):
        d63._jackknife_pareto_gate(base, base.copy(), truth)


def test_class_permutation_equivariance() -> None:
    rng = np.random.default_rng(63)
    truth = np.tile(np.arange(4), (5, 1))
    base = rng.normal(size=(5, 4, 4))
    residual = base + rng.normal(scale=0.4, size=base.shape)
    first = d63._jackknife_pareto_gate(base, residual, truth)
    permutation = np.asarray([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    second = d63._jackknife_pareto_gate(
        base[:, inverse][:, :, inverse],
        residual[:, inverse][:, :, inverse],
        permutation[truth][:, inverse],
    )
    np.testing.assert_array_equal(
        second["final_accept"], first["final_accept"][inverse]
    )


def test_formula_has_no_role_scene_or_tunable_threshold() -> None:
    lowered = d63.FORMULA.lower()
    assert "leave-one-fold" in lowered and "pareto" in lowered
    for forbidden in ("alpha", "threshold", "role", "scene"):
        assert forbidden not in lowered
