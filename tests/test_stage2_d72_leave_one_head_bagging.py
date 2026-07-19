from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d72_leave_one_head_bagging import (
    D72BaggingError,
    fit_leave_one_bagged_affine,
    physical_rank_leave_one_partitions,
)


def _support(class_count: int = 3, k: int = 4, dimension: int = 5):
    rows = []
    labels = []
    for class_index in range(class_count):
        for rank in range(k):
            row = np.zeros(dimension, dtype=np.float32)
            row[class_index] = 1.0 + 0.05 * rank
            row[-1] = 0.01 * (rank + class_index)
            rows.append(row)
            labels.append(class_index)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _mean_fit(rows, labels, class_count, k_shot):
    assert len(rows) == class_count * k_shot
    coefficient = np.stack(
        [rows[labels == index].mean(axis=0) for index in range(class_count)]
    )
    intercept = np.zeros(class_count, dtype=np.float32)
    return coefficient, intercept, {
        "d62_boundary_status": "synthetic",
        "d62_final_accept_mask": [False] * class_count,
    }


def _fallback(rows, labels, class_count):
    coefficient = np.stack(
        [rows[labels == index].mean(axis=0) for index in range(class_count)]
    )
    coefficient -= coefficient.mean(axis=0, keepdims=True)
    return coefficient.astype(np.float32), np.zeros(class_count, dtype=np.float32)


def test_physical_rank_partitions_are_exact_once_and_balanced() -> None:
    _, labels = _support()
    partitions = physical_rank_leave_one_partitions(labels, 3, 4)
    assert len(partitions) == 4
    held = np.concatenate([item[1] for item in partitions])
    assert sorted(held.tolist()) == list(range(12))
    for train, fold_held in partitions:
        assert len(np.intersect1d(train, fold_held)) == 0
        assert np.bincount(labels[fold_held], minlength=3).tolist() == [1, 1, 1]
        assert np.bincount(labels[train], minlength=3).tolist() == [3, 3, 3]


def test_leave_one_mean_head_matches_full_mean_for_linear_fit() -> None:
    rows, labels = _support()
    fallback_w, fallback_b = _fallback(rows, labels, 3)
    coefficient, intercept, audit = fit_leave_one_bagged_affine(
        rows, labels, 3, 4, _mean_fit, fallback_w, fallback_b
    )
    assert np.allclose(coefficient, fallback_w, atol=1.0e-7)
    assert np.array_equal(intercept, fallback_b)
    assert audit["status"] == "physical_rank_leave_one_head_bagging_active"
    assert audit["leave_one_fit_count"] == 4
    assert audit["inner_k_shot"] == 3
    assert audit["partition_exact_once"] is True
    assert coefficient.flags.writeable is False
    assert intercept.flags.writeable is False


def test_leave_one_fit_never_receives_held_rank() -> None:
    rows, labels = _support(class_count=2)
    observed = []

    def fit(train_rows, train_labels, class_count, k_shot):
        observed.append((train_rows.copy(), train_labels.copy(), k_shot))
        return _mean_fit(train_rows, train_labels, class_count, k_shot)

    fallback_w, fallback_b = _fallback(rows, labels, 2)
    _, _, audit = fit_leave_one_bagged_affine(
        rows, labels, 2, 4, fit, fallback_w, fallback_b
    )
    assert len(observed) == 4
    assert all(k == 3 for _, _, k in observed)
    assert all(len(train) == 6 for train, _, _ in observed)
    assert all(item["train_held_overlap_count"] == 0 for item in audit["partition_audit"])


def test_class_permutation_is_equivariant() -> None:
    rows, labels = _support()
    fallback_w, fallback_b = _fallback(rows, labels, 3)
    base_w, base_b, _ = fit_leave_one_bagged_affine(
        rows, labels, 3, 4, _mean_fit, fallback_w, fallback_b
    )
    permutation = np.asarray([2, 0, 1])
    inverse = np.argsort(permutation)
    row_order = np.concatenate(
        [np.flatnonzero(labels == original) for original in permutation]
    )
    permuted_rows = rows[row_order]
    permuted_labels = np.repeat(np.arange(3), 4)
    permuted_fallback_w = fallback_w[permutation]
    permuted_fallback_b = fallback_b[permutation]
    permuted_w, permuted_b, _ = fit_leave_one_bagged_affine(
        permuted_rows,
        permuted_labels,
        3,
        4,
        _mean_fit,
        permuted_fallback_w,
        permuted_fallback_b,
    )
    assert np.allclose(permuted_w[inverse], base_w)
    assert np.allclose(permuted_b[inverse], base_b)


def test_k1_is_exact_d62_fallback() -> None:
    rows, labels = _support(class_count=2, k=1)
    fallback_w, fallback_b = _fallback(rows, labels, 2)
    coefficient, intercept, audit = fit_leave_one_bagged_affine(
        rows, labels, 2, 1, _mean_fit, fallback_w, fallback_b
    )
    assert np.array_equal(coefficient, fallback_w)
    assert np.array_equal(intercept, fallback_b)
    assert audit["status"] == "k1_k2_exact_d62_fallback"
    assert audit["leave_one_fit_count"] == 0


def test_asymmetric_support_fails_closed() -> None:
    rows, labels = _support(class_count=2)
    fallback_w, fallback_b = _fallback(rows, labels, 2)
    with pytest.raises(D72BaggingError):
        fit_leave_one_bagged_affine(
            rows[:-1], labels[:-1], 2, 4, _mean_fit, fallback_w, fallback_b
        )
