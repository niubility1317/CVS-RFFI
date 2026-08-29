from __future__ import annotations

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from cvsrffi.stage2_bisage_d92 import (
    compare_exact_d92_logits,
    fit_bisage_d92,
)


def _balanced_support() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(713102)
    rows = []
    labels = []
    offsets = torch.tensor(
        [
            [-2.0, 0.0, 0.5, 0.0, 0.0, 0.2],
            [2.0, 0.5, 0.0, 0.0, -0.2, 0.0],
            [0.0, -2.0, 0.0, 0.5, 0.2, 0.0],
            [0.5, 2.0, -0.5, 0.0, 0.0, -0.2],
        ],
        dtype=torch.float64,
    )
    scales = torch.tensor([0.2, 0.4, 0.6, 0.3, 0.5, 0.7], dtype=torch.float64)
    for class_index in range(4):
        noise = torch.randn(8, 6, generator=generator, dtype=torch.float64) * scales
        rows.append(offsets[class_index] + noise)
        labels.extend([class_index] * 8)
    return torch.cat(rows), torch.tensor(labels, dtype=torch.long)


def _formal_d92(
    rows: np.ndarray, labels: np.ndarray, old_class_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_count = len(np.unique(labels))

    def group_covariance(indices: np.ndarray) -> np.ndarray:
        mask = np.isin(labels, indices)
        local = {int(value): index for index, value in enumerate(indices.tolist())}
        local_labels = np.asarray([local[int(value)] for value in labels[mask]])
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=np.full(len(indices), 1.0 / len(indices)),
            store_covariance=True,
        ).fit(rows[mask], local_labels)
        return np.asarray(estimator.covariance_, dtype=np.float64)

    old_covariance = group_covariance(np.arange(old_class_count))
    if class_count > old_class_count:
        new_covariance = group_covariance(np.arange(old_class_count, class_count))
        covariance = 0.5 * old_covariance + 0.5 * new_covariance
    else:
        covariance = old_covariance
    centers = np.stack([rows[labels == index].mean(0) for index in range(class_count)])
    coefficient = np.linalg.solve(covariance, centers.T).T
    intercept = -0.5 * np.diag(centers @ coefficient.T)
    intercept += np.log(np.full(class_count, 1.0 / class_count))
    coefficient -= coefficient.mean(axis=0, keepdims=True)
    intercept -= intercept.mean()
    return covariance, coefficient, intercept


def test_torch_d92_matches_formal_sklearn_logits() -> None:
    rows, labels = _balanced_support()
    state = fit_bisage_d92(rows, labels, old_class_count=2)
    covariance, coefficient, intercept = _formal_d92(
        rows.numpy(), labels.numpy(), old_class_count=2
    )
    probe = rows + 0.03
    expected = probe.numpy() @ coefficient.T + intercept
    actual = state.score(probe).detach().numpy()
    np.testing.assert_allclose(state.covariance.detach().numpy(), covariance, atol=1e-10)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    audit = compare_exact_d92_logits(state, rows.numpy(), labels.numpy(), probe.numpy())
    assert audit["max_logit_abs_error"] < 1.0e-4
    assert audit["argmax_mismatch_count"] == 0


def test_d92_backpropagates_through_ledoit_wolf_and_solve() -> None:
    rows, labels = _balanced_support()
    rows.requires_grad_(True)
    state = fit_bisage_d92(rows, labels, old_class_count=2)
    loss = state.score(rows[:5]).square().mean()
    loss.backward()
    assert rows.grad is not None
    assert torch.isfinite(rows.grad).all()
    assert float(rows.grad.abs().sum()) > 0.0


def test_d92_audit_locks_task_balance_and_positive_definiteness() -> None:
    rows, labels = _balanced_support()
    state = fit_bisage_d92(rows, labels, old_class_count=2)
    assert state.audit["covariance_policy"] == "sklearn_auto_per_class_then_task_balanced"
    assert state.audit["old_covariance_weight"] == 0.5
    assert state.audit["new_covariance_weight"] == 0.5
    assert state.audit["query_rows_used"] == 0
    assert state.audit["covariance_eigenvalue_min"] > 0.0
    assert state.audit["covariance_condition"] >= 1.0


def test_old_only_state_uses_one_task_covariance() -> None:
    rows, labels = _balanced_support()
    mask = labels < 2
    state = fit_bisage_d92(rows[mask], labels[mask], old_class_count=2)
    covariance, _, _ = _formal_d92(rows[mask].numpy(), labels[mask].numpy(), 2)
    np.testing.assert_allclose(state.covariance.numpy(), covariance, atol=1e-10)
    assert state.audit["old_covariance_weight"] == 1.0
    assert state.audit["new_covariance_weight"] == 0.0
