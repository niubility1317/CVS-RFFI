from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "cvsrffi" / "stage2_d69_frozen_d62_append.py"
SPEC = importlib.util.spec_from_file_location("stage2_d69_frozen_append_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
d69 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d69
SPEC.loader.exec_module(d69)


def _support(class_count: int, k: int, seed: int = 69) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for class_index in range(class_count):
        center = np.zeros(7, dtype=np.float64)
        center[class_index % 7] = 2.0 + class_index
        rows.append(center + 0.1 * rng.standard_normal((k, 7)))
        labels.extend([class_index] * k)
    return np.concatenate(rows), np.asarray(labels, dtype=np.int64)


def _base_fit(rows, labels, class_count, k_shot):
    means = np.stack([rows[labels == index].mean(axis=0) for index in range(class_count)])
    coefficient = means * float(class_count)
    intercept = -0.5 * np.sum(coefficient * means, axis=1)
    return coefficient.astype(np.float32), intercept.astype(np.float32), {
        "base_class_count": class_count,
        "base_k": k_shot,
    }


def test_freezes_old_rows_and_appends_exact_joint_new_rows() -> None:
    old_rows, old_labels = _support(3, 4)
    final_rows, final_labels = _support(5, 4)
    lifecycle = d69.FrozenD62AppendLifecycle(_base_fit)
    before_coef, before_bias, before = lifecycle(old_rows, old_labels, 3, 4)
    final_coef, final_bias, final = lifecycle(final_rows, final_labels, 5, 4)
    joint_coef, joint_bias, _ = _base_fit(final_rows, final_labels, 5, 4)
    assert np.array_equal(final_coef[:3], before_coef)
    assert np.array_equal(final_bias[:3], before_bias)
    assert np.array_equal(final_coef[3:], joint_coef[3:])
    assert np.array_equal(final_bias[3:], joint_bias[3:])
    assert before["d69_phase"] == "stage2b_d62_fit_and_freeze"
    assert final["d69_phase"] == "stage2c_append_d62_joint_new_rows"
    assert final["d69_old_row_fp32_bitwise_unchanged"] is True
    assert final["d69_new_row_fp32_matches_joint_d62"] is True
    assert lifecycle.completed_pairs == 1 and lifecycle.pending is False


def test_before_is_exact_base_fit_including_k1() -> None:
    rows, labels = _support(3, 1)
    expected_coef, expected_bias, _ = _base_fit(rows, labels, 3, 1)
    lifecycle = d69.FrozenD62AppendLifecycle(_base_fit)
    actual_coef, actual_bias, audit = lifecycle(rows, labels, 3, 1)
    assert np.array_equal(actual_coef, expected_coef)
    assert np.array_equal(actual_bias, expected_bias)
    assert audit["d69_actual_k"] == 1
    assert audit["d69_actual_row_sha256"] == audit["d69_joint_d62_row_sha256"]


def test_new_support_cannot_modify_frozen_old_rows() -> None:
    old_rows, old_labels = _support(3, 4)
    final_rows, final_labels = _support(5, 4)
    lifecycle = d69.FrozenD62AppendLifecycle(_base_fit)
    before_coef, before_bias, _ = lifecycle(old_rows, old_labels, 3, 4)
    changed = final_rows.copy()
    changed[final_labels >= 3] *= 100.0
    final_coef, final_bias, _ = lifecycle(changed, final_labels, 5, 4)
    assert np.array_equal(final_coef[:3], before_coef)
    assert np.array_equal(final_bias[:3], before_bias)


def test_class_permutation_equivariance_within_registry_phases() -> None:
    old_rows, old_labels = _support(3, 4)
    final_rows, final_labels = _support(5, 4)
    first = d69.FrozenD62AppendLifecycle(_base_fit)
    before1, bias1, _ = first(old_rows, old_labels, 3, 4)
    final1, final_bias1, _ = first(final_rows, final_labels, 5, 4)

    permutation = np.asarray([2, 0, 1, 4, 3], dtype=np.int64)
    old_perm = permutation[:3][old_labels]
    final_perm = permutation[final_labels]
    old_order = np.argsort(old_perm, kind="stable")
    final_order = np.argsort(final_perm, kind="stable")
    second = d69.FrozenD62AppendLifecycle(_base_fit)
    before2, bias2, _ = second(
        old_rows[old_order], old_perm[old_order], 3, 4
    )
    final2, final_bias2, _ = second(
        final_rows[final_order], final_perm[final_order], 5, 4
    )
    assert np.allclose(before2[permutation[:3]], before1)
    assert np.allclose(bias2[permutation[:3]], bias1)
    assert np.allclose(final2[permutation], final1)
    assert np.allclose(final_bias2[permutation], final_bias1)


def test_rejects_asymmetric_or_nonfinite_support() -> None:
    rows, labels = _support(3, 4)
    lifecycle = d69.FrozenD62AppendLifecycle(_base_fit)
    with pytest.raises(d69.D69LifecycleError, match="symmetric"):
        lifecycle(rows[:-1], labels[:-1], 3, 4)
    rows[0, 0] = np.nan
    with pytest.raises(d69.D69LifecycleError, match="symmetric"):
        lifecycle(rows, labels, 3, 4)


def test_rejects_mismatched_stage2c_lifecycle() -> None:
    old_rows, old_labels = _support(3, 4)
    final_rows, final_labels = _support(5, 3)
    lifecycle = d69.FrozenD62AppendLifecycle(_base_fit)
    lifecycle(old_rows, old_labels, 3, 4)
    with pytest.raises(d69.D69LifecycleError, match="lifecycle"):
        lifecycle(final_rows, final_labels, 5, 3)


def test_source_declares_zero_ground_and_no_query_role_branch() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"d69_ground_component_input_count": 0' in source
    assert '"d69_old_new_role_specific_query_branch": False' in source
    assert '"d69_uses_outer_held_or_query": False' in source
    assert '"d69_query_joint_optimization": False' in source
