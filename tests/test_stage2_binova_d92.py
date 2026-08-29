from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cvsrffi.stage2_binova_d92 import (
    d92_geometry_conditions,
    d92_geometry_features,
    fit_differentiable_d92,
)


def _balanced_rows() -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.tensor(
        [
            [-2.0, -1.0],
            [-2.0, 1.0],
            [2.0, -1.0],
            [2.0, 1.0],
            [-1.0, -4.0],
            [1.0, -4.0],
            [-1.0, 4.0],
            [1.0, 4.0],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    return rows, labels


def test_task_balanced_covariance_is_equal_old_new_average() -> None:
    rows, labels = _balanced_rows()
    state = fit_differentiable_d92(
        rows,
        labels,
        old_class_count=2,
        shrinkage_override=0.0,
        jitter=1.0e-4,
    )
    expected_old = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    expected_new = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    expected = 0.5 * expected_old + 0.5 * expected_new
    expected = expected + 1.0e-4 * torch.eye(2, dtype=torch.float64)
    torch.testing.assert_close(state.covariance, expected, atol=1.0e-10, rtol=0.0)
    assert state.audit["old_covariance_weight"] == 0.5
    assert state.audit["new_covariance_weight"] == 0.5


def test_cholesky_state_is_positive_definite_and_backpropagates() -> None:
    rows, labels = _balanced_rows()
    rows = rows.clone().requires_grad_(True)
    state = fit_differentiable_d92(rows, labels, old_class_count=2)
    eigenvalues = torch.linalg.eigvalsh(state.covariance.detach())
    assert float(eigenvalues.min()) > 0.0
    loss = F.cross_entropy(state.score(rows), labels)
    loss.backward()
    assert rows.grad is not None
    assert torch.isfinite(rows.grad).all()
    assert float(rows.grad.abs().sum()) > 0.0


def test_geometry_features_are_locked_identity160_plus_fft96() -> None:
    identity = torch.zeros(2, 160)
    fft = torch.zeros(2, 96)
    identity[0, 0] = 3.0
    identity[1, 1] = 4.0
    fft[0, 0] = 5.0
    fft[1, 1] = 6.0
    joined = d92_geometry_features(identity, fft)
    assert joined.shape == (2, 256)
    torch.testing.assert_close(torch.linalg.norm(joined, dim=1), torch.ones(2))
    ratio = joined[:, 160:].norm(dim=1) / joined[:, :160].norm(dim=1)
    torch.testing.assert_close(ratio, torch.full((2,), 4.0))


def test_held_row_does_not_enter_its_crossfit_geometry() -> None:
    rows, labels = _balanced_rows()
    held = rows[0:1]
    full_state = fit_differentiable_d92(rows, labels, old_class_count=2)
    fit_mask = torch.ones(len(rows), dtype=torch.bool)
    fit_mask[0] = False
    fit_state = fit_differentiable_d92(rows[fit_mask], labels[fit_mask], old_class_count=2)
    full_condition = d92_geometry_conditions(full_state, held)
    held_out_condition = d92_geometry_conditions(fit_state, held)
    assert full_condition.shape == (1, 6)
    assert not torch.allclose(full_condition, held_out_condition)


def test_geometry_condition_contains_finite_distances_margin_and_entropy() -> None:
    rows, labels = _balanced_rows()
    state = fit_differentiable_d92(rows, labels, old_class_count=2)
    condition = d92_geometry_conditions(state, rows[:3])
    assert condition.shape == (3, 6)
    assert torch.isfinite(condition).all()
    probabilities = torch.softmax(state.score(rows[:3]), dim=1)
    expected_entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(1)
    torch.testing.assert_close(condition[:, 5], expected_entropy)
