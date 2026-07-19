from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d73_conflict_projected_joint_metric import (
    D73MetricError,
    fit_conflict_projected_log_diag,
)


def _support(k_shot: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(73)
    centers = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.2, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.2],
            [1.2, 1.2, 0.0, 0.4, 0.0, 0.0],
            [0.0, 1.2, 1.2, 0.0, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    rows = np.concatenate(
        [center + 0.12 * rng.normal(size=(k_shot, 6)) for center in centers]
    ).astype(np.float32)
    labels = np.repeat(np.arange(len(centers)), k_shot)
    return rows, labels


def test_active_update_is_deterministic_centered_and_first_order_safe() -> None:
    rows, labels = _support()
    base = np.zeros(rows.shape[1], dtype=np.float32)
    first, audit = fit_conflict_projected_log_diag(
        rows, labels, 3, 5, 4, base
    )
    second, repeated = fit_conflict_projected_log_diag(
        rows, labels, 3, 5, 4, base
    )
    assert np.array_equal(first, second)
    assert audit == repeated
    assert first.flags.writeable is False
    assert audit["status"] == "one_step_conflict_projected_joint_metric_active"
    assert audit["stage2c_step_count"] == 1
    assert audit["old_first_order_change"] <= 1e-10
    assert audit["new_first_order_change"] <= 1e-10
    assert abs(audit["delta_mean"]) < 1e-12
    assert audit["delta_l2"] == pytest.approx(np.sqrt(4.0 / 10.0))
    assert audit["ground_component_input_count"] == 0
    assert audit["query_rows_used"] == 0


def test_within_role_class_permutation_is_equivariant() -> None:
    rows, labels = _support()
    base = np.linspace(-0.1, 0.1, rows.shape[1], dtype=np.float32)
    reference, reference_audit = fit_conflict_projected_log_diag(
        rows, labels, 3, 5, 4, base
    )
    mapping = np.asarray([2, 0, 1, 4, 3])
    permuted_labels = mapping[labels]
    permuted, audit = fit_conflict_projected_log_diag(
        rows, permuted_labels, 3, 5, 4, base
    )
    assert np.allclose(reference, permuted, rtol=0.0, atol=1e-7)
    for field in (
        "old_loss_before",
        "new_loss_before",
        "task_gradient_cosine",
        "delta_l2",
    ):
        assert audit[field] == pytest.approx(reference_audit[field])


def test_k1_is_exact_fallback() -> None:
    rows, labels = _support(k_shot=1)
    base = np.linspace(-0.2, 0.2, rows.shape[1], dtype=np.float32)
    updated, audit = fit_conflict_projected_log_diag(
        rows, labels, 3, 5, 1, base
    )
    assert np.array_equal(updated, base)
    assert audit["status"] == "k1_exact_d62_fallback"
    assert audit["stage2c_step_count"] == 0


def test_invalid_asymmetric_support_fails_closed() -> None:
    rows, labels = _support()
    with pytest.raises(D73MetricError, match="exact symmetric"):
        fit_conflict_projected_log_diag(
            rows[:-1], labels[:-1], 3, 5, 4, np.zeros(rows.shape[1])
        )
