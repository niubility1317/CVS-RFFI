from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "cvsrffi" / "stage2_d70_atomic_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("stage2_d70_atomic_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
d70 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d70
SPEC.loader.exec_module(d70)


def _support(class_count: int, k: int, seed: int = 70) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for class_index in range(class_count):
        center = np.zeros(8)
        center[class_index % 8] = 2.0 + class_index
        rows.append(center + 0.05 * rng.standard_normal((k, 8)))
        labels.extend([class_index] * k)
    return np.concatenate(rows), np.asarray(labels, dtype=np.int64)


def _fit(rows, labels, class_count, k_shot):
    means = np.stack([rows[labels == index].mean(axis=0) for index in range(class_count)])
    coefficient = means * (1.0 + 0.1 * class_count)
    intercept = -0.5 * np.sum(coefficient * means, axis=1)
    return coefficient.astype(np.float32), intercept.astype(np.float32), {
        "base_class_count": class_count,
        "base_k": k_shot,
    }


def test_twofold_partition_is_exact_once_and_disjoint() -> None:
    _, labels = _support(5, 8)
    folds = d70.twofold_rank_partitions(labels, 5, 8)
    assert len(folds) == 2
    assert set(folds[0]).isdisjoint(set(folds[1]))
    assert sorted(np.concatenate(folds).tolist()) == list(range(40))
    assert all(len(fold) == 20 for fold in folds)


def test_atomic_gate_accepts_only_strict_safe_old_row() -> None:
    truth = np.asarray([0, 0, 1, 1, 2, 2])
    base = np.asarray(
        [[3, 1, 0], [1, 2, 0], [0, 3, 1], [0, 3, 1], [0, 0, 3], [0, 0, 3]],
        dtype=np.float64,
    )
    before = base[:, :2].copy()
    before[:, 0] = np.asarray([3, 3, 0, 0, 0, 0])
    gate = d70.atomic_old_row_gate(base, before, truth)
    assert gate["initial_accept"].tolist() == [True, False]
    assert gate["final_accept"].tolist() == [True, False]
    assert gate["atomic_safe"] is True
    assert gate["status"] == "crossfitted_atomic_lifecycle_rows_active"
    assert np.all(gate["joint_positive"] >= gate["base_positive"])
    assert np.all(gate["joint_false_positive"] <= gate["base_false_positive"])


def test_atomic_gate_is_equivariant_to_old_class_permutation() -> None:
    truth = np.asarray([0, 0, 1, 1, 2, 2])
    base = np.asarray(
        [[3, 1, 0], [1, 2, 0], [0, 3, 1], [0, 3, 1], [0, 0, 3], [0, 0, 3]],
        dtype=np.float64,
    )
    before = base[:, :2].copy()
    before[:, 0] = np.asarray([3, 3, 0, 0, 0, 0])
    first = d70.atomic_old_row_gate(base, before, truth)
    permutation = np.asarray([1, 0, 2])
    inverse = np.argsort(permutation)
    second = d70.atomic_old_row_gate(
        base[:, inverse], before[:, inverse[:2]], permutation[truth]
    )
    assert second["final_accept"].tolist() == first["final_accept"][inverse[:2]].tolist()


def test_lifecycle_before_exact_and_k1_final_exact_fallback() -> None:
    old_rows, old_labels = _support(3, 1)
    final_rows, final_labels = _support(5, 1)
    lifecycle = d70.AtomicLifecycleRowReplacement(_fit)
    before_coef, before_bias, before = lifecycle(old_rows, old_labels, 3, 1)
    expected_before = _fit(old_rows, old_labels, 3, 1)
    assert np.array_equal(before_coef, expected_before[0])
    assert np.array_equal(before_bias, expected_before[1])
    final_coef, final_bias, final = lifecycle(final_rows, final_labels, 5, 1)
    expected_final = _fit(final_rows, final_labels, 5, 1)
    assert np.array_equal(final_coef, expected_final[0])
    assert np.array_equal(final_bias, expected_final[1])
    assert final["d70_gate_status"] == "k1_exact_d62_fallback"
    assert final["d70_exact_d62_fallback"] is True
    assert lifecycle.completed_pairs == 1 and lifecycle.inner_fit_count == 0


def test_lifecycle_applies_only_atomic_selected_old_rows(monkeypatch) -> None:
    old_rows, old_labels = _support(3, 4)
    final_rows, final_labels = _support(5, 4)
    lifecycle = d70.AtomicLifecycleRowReplacement(_fit)
    before_coef, before_bias, _ = lifecycle(old_rows, old_labels, 3, 4)

    def gate(base, before, truth):
        class_count, old_count = base.shape[1], before.shape[1]
        zeros = np.zeros(class_count, dtype=np.int64)
        mask = np.asarray([False, True, False])
        return {
            "base_positive": zeros,
            "base_false_positive": zeros,
            "coordinate_positive": np.zeros(old_count, dtype=np.int64),
            "coordinate_false_positive": np.zeros(old_count, dtype=np.int64),
            "joint_positive": zeros,
            "joint_false_positive": zeros,
            "initial_accept": mask,
            "final_accept": mask,
            "atomic_safe": True,
            "status": "crossfitted_atomic_lifecycle_rows_active",
            "exact_fallback": False,
        }

    monkeypatch.setattr(d70, "atomic_old_row_gate", gate)
    final_coef, final_bias, audit = lifecycle(final_rows, final_labels, 5, 4)
    joint_coef, joint_bias, _ = _fit(final_rows, final_labels, 5, 4)
    assert np.array_equal(final_coef[1], before_coef[1])
    assert np.array_equal(final_bias[1], before_bias[1])
    assert np.array_equal(final_coef[[0, 2, 3, 4]], joint_coef[[0, 2, 3, 4]])
    assert np.array_equal(final_bias[[0, 2, 3, 4]], joint_bias[[0, 2, 3, 4]])
    assert audit["d70_final_accept_mask"] == [False, True, False]
    assert audit["d70_new_rows_match_joint_d62"] is True
    assert len(audit["d70_partition_audit"]) == 2
    assert lifecycle.inner_fit_count == 4


def test_rejects_lifecycle_support_drift() -> None:
    old_rows, old_labels = _support(3, 4)
    final_rows, final_labels = _support(5, 4)
    lifecycle = d70.AtomicLifecycleRowReplacement(_fit)
    lifecycle(old_rows, old_labels, 3, 4)
    final_rows[0, 0] += 1.0
    with pytest.raises(d70.D70LifecycleError, match="lifecycle"):
        lifecycle(final_rows, final_labels, 5, 4)


def test_source_locks_protocol_and_zero_ground() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"d70_ground_component_input_count": 0' in source
    assert '"d70_uses_outer_held_or_query": False' in source
    assert '"d70_old_new_role_specific_query_branch": False' in source
    assert '"d70_hyperparameter_count": 0' in source

