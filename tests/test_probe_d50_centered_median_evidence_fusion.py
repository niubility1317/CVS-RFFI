from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d50_centered_median_evidence_fusion.py"
SPEC = importlib.util.spec_from_file_location("d50_probe_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d50 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d50)


def _partition(values: np.ndarray) -> dict[str, object]:
    return {"held_ce_by_fold_and_class": np.asarray(values, dtype=np.float64).tolist()}


def _strategy(full: np.ndarray, block: np.ndarray):
    k, c = full.shape
    return d50._centered_median_strategy(
        full_per_class_ce=full.mean(axis=0),
        block_per_class_ce=block.mean(axis=0),
        full_partition=_partition(full),
        block_partition=_partition(block),
        k_shot=k,
        class_count=c,
    )


def test_k1_is_exact_equal_weight_fallback() -> None:
    weights, log_evidence, audit = d50._centered_median_strategy(
        full_per_class_ce=None,
        block_per_class_ce=None,
        full_partition=None,
        block_partition=None,
        k_shot=1,
        class_count=3,
    )
    assert np.array_equal(weights, np.full((3, 2), 0.5))
    assert log_evidence is None
    assert audit["d50_boundary_status"] == "k1_d45_equal_unit_fallback"


def test_centered_median_formula_and_d45_anchor_close() -> None:
    full = np.full((5, 3), 0.8)
    advantage = np.asarray(
        [
            [0.10, -0.10, 0.00],
            [0.12, -0.08, 0.01],
            [0.11, -0.09, 0.02],
            [0.09, -0.11, 0.01],
            [1.20, -0.07, 0.00],
        ]
    )
    weights, log_evidence, audit = _strategy(full, full + advantage)
    medians = np.median(advantage, axis=0)
    z0 = advantage.shape[1] * float(advantage.mean())
    expected_z = z0 + advantage.shape[0] * (medians - medians.mean())
    expected_weight = 1.0 / (1.0 + np.exp(-expected_z))
    assert np.allclose(weights[:, 0], expected_weight, rtol=0.0, atol=1.0e-15)
    assert np.allclose(log_evidence[:, 0], expected_z, rtol=0.0, atol=1.0e-15)
    assert np.allclose(log_evidence[:, 1], 0.0, rtol=0.0, atol=0.0)
    assert abs(float(np.mean(expected_z)) - z0) <= 1.0e-15
    assert audit["d50_post_log_odds_mean_anchor_error"] <= 1.0e-15


def test_median_resists_single_rank_outlier_without_threshold() -> None:
    full = np.full((5, 2), 0.7)
    advantage = np.asarray(
        [[0.10, -0.10], [0.10, -0.10], [0.10, -0.10], [0.10, -0.10], [9.0, -0.10]]
    )
    _, _, audit = _strategy(full, full + advantage)
    assert audit["d50_median_advantage_by_class"] == pytest.approx([0.10, -0.10])
    assert audit["d50_mean_advantage_by_class"][0] > 1.0
    assert audit["d50_no_temperature_clip_threshold_sign_gate_or_scan"] is True


def test_rank_permutation_is_invariant() -> None:
    rng = np.random.default_rng(50)
    full = rng.uniform(0.2, 1.4, size=(8, 4))
    block = full + rng.normal(0.0, 0.08, size=(8, 4))
    baseline = _strategy(full, block)[0]
    order = np.asarray([5, 1, 7, 0, 3, 6, 2, 4])
    permuted = _strategy(full[order], block[order])[0]
    assert np.allclose(permuted, baseline, rtol=0.0, atol=1.0e-15)


def test_class_permutation_is_equivariant() -> None:
    rng = np.random.default_rng(51)
    full = rng.uniform(0.2, 1.4, size=(8, 5))
    block = full + rng.normal(0.0, 0.08, size=(8, 5))
    baseline = _strategy(full, block)[0]
    order = np.asarray([3, 0, 4, 1, 2])
    permuted = _strategy(full[:, order], block[:, order])[0]
    assert np.allclose(permuted, baseline[order], rtol=0.0, atol=1.0e-15)


def test_k2_requires_equivalent_components_and_returns_half() -> None:
    full = np.full((2, 3), 0.5)
    weights, log_evidence, audit = _strategy(full, full.copy())
    assert np.allclose(weights, 0.5, rtol=0.0, atol=0.0)
    assert np.allclose(log_evidence, 0.0, rtol=0.0, atol=0.0)
    assert audit["d50_boundary_status"] == "k2_equal_component_fallback"
    with pytest.raises(d50.D50ProbeError, match="K2 component-equivalence"):
        _strategy(full, full + 0.01)


def test_fold_mean_and_partition_shape_fail_closed() -> None:
    full = np.full((4, 3), 0.5)
    block = full + 0.02
    with pytest.raises(d50.D50ProbeError, match="fold/per-class CE"):
        d50._centered_median_strategy(
            full_per_class_ce=np.zeros(3),
            block_per_class_ce=block.mean(axis=0),
            full_partition=_partition(full),
            block_partition=_partition(block),
            k_shot=4,
            class_count=3,
        )
    with pytest.raises(d50.D50ProbeError, match="fold CE evidence"):
        d50._centered_median_strategy(
            full_per_class_ce=full.mean(axis=0),
            block_per_class_ce=block.mean(axis=0),
            full_partition=_partition(full[:, :2]),
            block_partition=_partition(block),
            k_shot=4,
            class_count=3,
        )


def test_nonfinite_evidence_fails_closed() -> None:
    full = np.full((4, 3), 0.5)
    block = full.copy()
    block[0, 0] = np.nan
    with pytest.raises(d50.D50ProbeError, match="fold CE evidence"):
        _strategy(full, block)
