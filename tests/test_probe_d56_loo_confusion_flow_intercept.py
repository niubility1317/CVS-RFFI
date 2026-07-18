from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d56_loo_confusion_flow_intercept.py"
SPEC = importlib.util.spec_from_file_location("d56_probe_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d56 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d56)


def _audit() -> dict[str, object]:
    return {
        "d46_full_weight_by_class": [0.5, 0.5, 0.5],
        "d46_block_weight_by_class": [0.5, 0.5, 0.5],
        "d46_full_support_logit_rms": 1.0,
        "d46_block_support_logit_rms": 1.0,
    }


def _evidence() -> tuple[np.ndarray, np.ndarray, list[list[int]], np.ndarray]:
    predictions = np.asarray(
        [[1, 1, 0], [0, 0, 2], [0, 1, 2]], dtype=np.int64
    )
    scores = np.full((3, 3, 3), -1.0, dtype=np.float64)
    for fold in range(3):
        for row in range(3):
            scores[fold, row, predictions[fold, row]] = 1.0
    held = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    targets = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    return scores, scores.copy(), held, targets


def test_active_confusion_flow_is_degree_balanced_and_centered() -> None:
    full, block, held, targets = _evidence()
    result = d56._confusion_flow(full, block, held, targets, _audit(), 3, 3)
    assert result["status"] == "loo_confusion_flow_intercept_active"
    assert np.array_equal(result["out_degree"], [1, 1, 1])
    assert np.array_equal(result["in_degree"], [2, 1, 0])
    assert np.allclose(result["delta_intercept"], [-1 / 9, 0.0, 1 / 9])
    assert abs(float(np.sum(result["delta_intercept"]))) < 1.0e-14


@pytest.mark.parametrize("k", [1, 2])
def test_k1_k2_are_exact_d46_fallbacks_without_evidence(k: int) -> None:
    result = d56._confusion_flow(None, None, None, np.zeros(k * 3), {}, k, 3)
    assert result["exact_fallback"] is True
    assert np.array_equal(result["delta_intercept"], np.zeros(3))


def test_class_permutation_is_equivariant() -> None:
    full, block, held, targets = _evidence()
    baseline = d56._confusion_flow(full, block, held, targets, _audit(), 3, 3)
    order = np.asarray([2, 0, 1])
    inverse = np.argsort(order)
    audit = {
        **_audit(),
        "d46_full_weight_by_class": np.asarray(
            _audit()["d46_full_weight_by_class"]
        )[order].tolist(),
        "d46_block_weight_by_class": np.asarray(
            _audit()["d46_block_weight_by_class"]
        )[order].tolist(),
    }
    actual = d56._confusion_flow(
        full[:, :, order],
        block[:, :, order],
        held,
        inverse[targets],
        audit,
        3,
        3,
    )
    assert np.allclose(actual["delta_intercept"], baseline["delta_intercept"][order])


def test_invalid_scores_and_weight_simplex_fail_closed() -> None:
    full, block, held, targets = _evidence()
    bad = _audit()
    bad["d46_full_weight_by_class"] = [0.9, 0.9, 0.9]
    with pytest.raises(d56.D56ProbeError, match="evidence"):
        d56._confusion_flow(full, block, held, targets, bad, 3, 3)
    with pytest.raises(d56.D56ProbeError, match="evidence"):
        d56._confusion_flow(full[:, :, :2], block, held, targets, _audit(), 3, 3)


def test_resource_account_includes_extra_inner_refits() -> None:
    macs = lambda rows, classes: rows * classes * 288
    fits, lda_macs, numeric, comparisons = d56._extra_resource(
        8, 6, 11, 288, macs
    )
    assert fits == 32
    assert lda_macs == 16 * (42 * 6 * 288) + 16 * (77 * 11 * 288)
    assert numeric == sum(6 * 8 * c * c + 4 * 8 * c for c in (6, 11))
    assert comparisons == sum(8 * c * (c - 1) for c in (6, 11))


def test_formula_has_no_tunable_or_role_specific_surface() -> None:
    assert "out_degree_c-in_degree_c" in d56.FORMULA
    assert "k_shot*class_count" in d56.FORMULA
    assert "CE" not in d56.FORMULA
    assert "alpha" not in d56.FORMULA.lower()
    assert "threshold" not in d56.FORMULA.lower()
