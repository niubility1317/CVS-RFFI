"""Autograd-preserving D92 with formal sklearn numerical parity.

The covariance path mirrors the locked D92 implementation: Ledoit-Wolf is
estimated independently for every class after StandardScaler normalization,
classes are averaged with equal priors inside each registration task, and the
old/new task covariances are combined with fixed equal weights. Query samples
are never accepted by this fitting API.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch


class BiSAGED92Error(ValueError):
    """Raised when the locked BiSAGE D92 support geometry is invalid."""


@dataclass(frozen=True)
class BiSAGED92State:
    class_ids: tuple[int, ...]
    old_class_count: int
    centers: torch.Tensor
    covariance: torch.Tensor
    cholesky: torch.Tensor
    coefficient: torch.Tensor
    intercept: torch.Tensor
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    def score(self, rows: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(
            rows, dtype=self.coefficient.dtype, device=self.coefficient.device
        )
        if values.ndim != 2 or values.shape[1] != self.coefficient.shape[1]:
            raise BiSAGED92Error("D92 score feature geometry drift")
        if not torch.isfinite(values).all():
            raise BiSAGED92Error("D92 score rows must be finite")
        return values @ self.coefficient.T + self.intercept


def _ledoit_wolf_standardized(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match sklearn ``_cov(..., shrinkage='auto')`` for one class."""

    sample_count, dimension = rows.shape
    centered = rows - rows.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    scale = torch.sqrt(variance)
    scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    standardized = centered / scale

    # sklearn's LedoitWolf centers again after StandardScaler. Keeping this
    # second centering is required for float64 numerical parity.
    standardized = standardized - standardized.mean(dim=0, keepdim=True)
    squares = standardized.square()
    empirical_trace = squares.sum(dim=0) / float(sample_count)
    mu = empirical_trace.sum() / float(dimension)
    beta_accumulator = torch.sum(squares.T @ squares)
    gram = standardized.T @ standardized
    delta_accumulator = torch.sum(gram.square()) / float(sample_count**2)
    beta = (
        beta_accumulator / float(sample_count) - delta_accumulator
    ) / float(dimension * sample_count)
    delta = (
        delta_accumulator
        - 2.0 * mu * empirical_trace.sum()
        + float(dimension) * mu.square()
    ) / float(dimension)
    beta = torch.minimum(beta, delta)
    safe_delta = torch.where(delta != 0.0, delta, torch.ones_like(delta))
    shrinkage = torch.where(beta == 0.0, torch.zeros_like(beta), beta / safe_delta)

    empirical = gram / float(sample_count)
    identity = torch.eye(dimension, dtype=rows.dtype, device=rows.device)
    standardized_covariance = (
        (1.0 - shrinkage) * empirical + shrinkage * mu * identity
    )
    covariance = scale[:, None] * standardized_covariance * scale[None, :]
    return 0.5 * (covariance + covariance.T), shrinkage


def _equal_class_group_covariance(
    rows: torch.Tensor, labels: torch.Tensor, class_indices: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    covariances = []
    shrinkages = []
    for class_index in class_indices:
        selected = rows[labels == class_index]
        covariance, shrinkage = _ledoit_wolf_standardized(selected)
        covariances.append(covariance)
        shrinkages.append(shrinkage)
    return torch.stack(covariances).mean(dim=0), tuple(shrinkages)


def fit_bisage_d92(
    rows: torch.Tensor,
    labels: torch.Tensor,
    *,
    old_class_count: int,
) -> BiSAGED92State:
    """Fit the locked support-only D92 while preserving feature gradients."""

    values = torch.as_tensor(rows)
    targets = torch.as_tensor(labels, dtype=torch.long, device=values.device)
    if (
        values.ndim != 2
        or len(values) < 1
        or values.shape[1] < 1
        or not torch.is_floating_point(values)
        or not torch.isfinite(values).all()
        or targets.shape != (len(values),)
    ):
        raise BiSAGED92Error("D92 support rows/labels are invalid")
    classes = torch.unique(targets, sorted=True)
    class_count = int(len(classes))
    if not torch.equal(classes, torch.arange(class_count, device=values.device)):
        raise BiSAGED92Error("D92 labels must be a contiguous zero-based registry")
    old_count = int(old_class_count)
    if old_count < 1 or old_count > class_count:
        raise BiSAGED92Error("old_class_count is outside the registry")
    shots = torch.stack([(targets == index).sum() for index in range(class_count)])
    if int(shots.min()) < 2 or not torch.equal(shots, shots[0].expand_as(shots)):
        raise BiSAGED92Error("D92 requires equal K>=2 support for every class")

    centers = torch.stack(
        [values[targets == index].mean(dim=0) for index in range(class_count)]
    )
    old_indices = torch.arange(old_count, device=values.device)
    old_covariance, old_shrinkages = _equal_class_group_covariance(
        values, targets, old_indices
    )
    if class_count > old_count:
        new_indices = torch.arange(old_count, class_count, device=values.device)
        new_covariance, new_shrinkages = _equal_class_group_covariance(
            values, targets, new_indices
        )
        covariance = 0.5 * old_covariance + 0.5 * new_covariance
        old_weight, new_weight = 0.5, 0.5
    else:
        new_shrinkages = ()
        covariance = old_covariance
        old_weight, new_weight = 1.0, 0.0
    covariance = 0.5 * (covariance + covariance.T)
    cholesky, info = torch.linalg.cholesky_ex(covariance)
    if int(info.max().detach().cpu()) != 0:
        raise BiSAGED92Error("D92 balanced covariance is not positive definite")
    coefficient = torch.cholesky_solve(centers.T, cholesky).T
    prior = values.new_full((class_count,), 1.0 / float(class_count))
    intercept = -0.5 * torch.sum(centers * coefficient, dim=1) + torch.log(prior)
    coefficient = coefficient - coefficient.mean(dim=0, keepdim=True)
    intercept = intercept - intercept.mean()

    eigenvalues = torch.linalg.eigvalsh(covariance.detach())
    eigenvalue_min = float(eigenvalues.min().cpu())
    eigenvalue_max = float(eigenvalues.max().cpu())
    return BiSAGED92State(
        class_ids=tuple(range(class_count)),
        old_class_count=old_count,
        centers=centers,
        covariance=covariance,
        cholesky=cholesky,
        coefficient=coefficient,
        intercept=intercept,
        audit={
            "solver": "torch_float64_cholesky_sklearn_lsqr_equivalent",
            "covariance_policy": "sklearn_auto_per_class_then_task_balanced",
            "prior_policy": "equal_1_over_registered_class_count",
            "class_count": class_count,
            "k_shot": int(shots[0]),
            "old_covariance_weight": old_weight,
            "new_covariance_weight": new_weight,
            "old_class_shrinkages": tuple(
                float(value.detach().cpu()) for value in old_shrinkages
            ),
            "new_class_shrinkages": tuple(
                float(value.detach().cpu()) for value in new_shrinkages
            ),
            "covariance_eigenvalue_min": eigenvalue_min,
            "covariance_eigenvalue_max": eigenvalue_max,
            "covariance_condition": eigenvalue_max / eigenvalue_min,
            "query_rows_used": 0,
            "query_truth_access": False,
            "query_role_oracle_access": False,
            "class_common_affine_omitted": True,
        },
    )


def _formal_sklearn_state(
    rows: np.ndarray, labels: np.ndarray, old_class_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    values = np.asarray(rows, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    class_count = len(np.unique(targets))

    def group_covariance(indices: np.ndarray) -> np.ndarray:
        mask = np.isin(targets, indices)
        local = {int(value): index for index, value in enumerate(indices.tolist())}
        local_targets = np.asarray([local[int(value)] for value in targets[mask]])
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=np.full(len(indices), 1.0 / len(indices), dtype=np.float64),
            store_covariance=True,
        ).fit(values[mask], local_targets)
        return np.asarray(estimator.covariance_, dtype=np.float64)

    covariance = group_covariance(np.arange(old_class_count, dtype=np.int64))
    if class_count > old_class_count:
        new_covariance = group_covariance(
            np.arange(old_class_count, class_count, dtype=np.int64)
        )
        covariance = 0.5 * covariance + 0.5 * new_covariance
    centers = np.stack([values[targets == index].mean(0) for index in range(class_count)])
    coefficient = np.linalg.solve(covariance, centers.T).T
    intercept = -0.5 * np.diag(centers @ coefficient.T)
    intercept += np.log(np.full(class_count, 1.0 / class_count, dtype=np.float64))
    coefficient -= coefficient.mean(axis=0, keepdims=True)
    intercept -= intercept.mean()
    return covariance, coefficient, intercept


def compare_exact_d92_logits(
    state: BiSAGED92State,
    support_rows: Any,
    support_labels: Any,
    probe_rows: Any,
) -> dict[str, Any]:
    """Compare a fitted torch state with the formal sklearn D92 path."""

    covariance, coefficient, intercept = _formal_sklearn_state(
        np.asarray(support_rows, dtype=np.float64),
        np.asarray(support_labels, dtype=np.int64),
        state.old_class_count,
    )
    probe = np.asarray(probe_rows, dtype=np.float64)
    expected = probe @ coefficient.T + intercept
    actual = state.score(
        torch.as_tensor(probe, dtype=state.coefficient.dtype, device=state.coefficient.device)
    ).detach().cpu().numpy()
    return {
        "max_logit_abs_error": float(np.max(np.abs(actual - expected))),
        "max_covariance_abs_error": float(
            np.max(np.abs(state.covariance.detach().cpu().numpy() - covariance))
        ),
        "argmax_mismatch_count": int(
            np.sum(np.argmax(actual, axis=1) != np.argmax(expected, axis=1))
        ),
        "probe_count": int(len(probe)),
    }


__all__ = [
    "BiSAGED92Error",
    "BiSAGED92State",
    "compare_exact_d92_logits",
    "fit_bisage_d92",
]
