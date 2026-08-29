"""Differentiable D92 surrogate and exact D92 bridge for BiNOVA."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class BiNOVAD92Error(ValueError):
    """Raised when differentiable or exact D92 geometry is invalid."""


@dataclass(frozen=True)
class DifferentiableD92State:
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
        values = torch.as_tensor(rows, dtype=self.coefficient.dtype, device=self.coefficient.device)
        if values.ndim != 2 or values.shape[1] != self.coefficient.shape[1]:
            raise BiNOVAD92Error("D92 score feature geometry drift")
        return values @ self.coefficient.T + self.intercept


def d92_geometry_features(identity160: torch.Tensor, fft96: torch.Tensor) -> torch.Tensor:
    identity = torch.as_tensor(identity160)
    fft = torch.as_tensor(fft96, dtype=identity.dtype, device=identity.device)
    if identity.ndim != 2 or identity.shape[1] != 160:
        raise BiNOVAD92Error("identity feature geometry must be [N,160]")
    if fft.shape != (len(identity), 96):
        raise BiNOVAD92Error("FFT feature geometry must be [N,96]")
    if not torch.isfinite(identity).all() or not torch.isfinite(fft).all():
        raise BiNOVAD92Error("D92 input features must be finite")
    joined = torch.cat(
        [F.normalize(identity, dim=1), 4.0 * F.normalize(fft, dim=1)], dim=1
    )
    return F.normalize(joined, dim=1)


def _class_centers(rows: torch.Tensor, labels: torch.Tensor, class_count: int) -> torch.Tensor:
    return torch.stack([rows[labels == index].mean(dim=0) for index in range(class_count)])


def _oas_group_covariance(
    rows: torch.Tensor,
    labels: torch.Tensor,
    class_indices: torch.Tensor,
    *,
    shrinkage_override: float | None,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.isin(labels, class_indices)
    group_rows = rows[mask]
    group_labels = labels[mask]
    residual_parts = []
    for class_index in class_indices:
        selected = group_rows[group_labels == class_index]
        residual_parts.append(selected - selected.mean(dim=0, keepdim=True))
    residuals = torch.cat(residual_parts, dim=0)
    dimension = rows.shape[1]
    covariance = residuals.T @ residuals / float(max(len(residuals), 1))
    covariance = 0.5 * (covariance + covariance.T)
    trace = torch.trace(covariance)
    mean_variance = trace / float(dimension)
    if shrinkage_override is None:
        alpha = torch.sum(covariance.square())
        trace_square = trace.square()
        numerator = (1.0 - 2.0 / float(dimension)) * alpha + trace_square
        denominator = (
            float(len(residuals) + 1) - 2.0 / float(dimension)
        ) * (alpha - trace_square / float(dimension))
        ratio = numerator / denominator.clamp_min(torch.finfo(rows.dtype).eps)
        shrinkage = torch.where(
            denominator > torch.finfo(rows.dtype).eps,
            ratio.clamp(0.0, 1.0),
            torch.ones_like(ratio),
        )
    else:
        value = float(shrinkage_override)
        if value < 0.0 or value > 1.0:
            raise BiNOVAD92Error("shrinkage_override must be in [0,1]")
        shrinkage = rows.new_tensor(value)
    identity = torch.eye(dimension, dtype=rows.dtype, device=rows.device)
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * mean_variance * identity
    shrunk = 0.5 * (shrunk + shrunk.T) + float(jitter) * identity
    return shrunk, shrinkage


def fit_differentiable_d92(
    rows: torch.Tensor,
    labels: torch.Tensor,
    *,
    old_class_count: int,
    shrinkage_override: float | None = None,
    jitter: float = 1.0e-5,
) -> DifferentiableD92State:
    values = torch.as_tensor(rows)
    targets = torch.as_tensor(labels, dtype=torch.long, device=values.device)
    if (
        values.ndim != 2
        or len(values) < 1
        or targets.shape != (len(values),)
        or not torch.is_floating_point(values)
        or not torch.isfinite(values).all()
        or jitter <= 0.0
    ):
        raise BiNOVAD92Error("D92 fit rows/labels are invalid")
    classes = torch.unique(targets, sorted=True)
    class_count = len(classes)
    if not torch.equal(classes, torch.arange(class_count, device=values.device)):
        raise BiNOVAD92Error("D92 labels must be a contiguous zero-based registry")
    old_count = int(old_class_count)
    if old_count < 1 or old_count > class_count:
        raise BiNOVAD92Error("old_class_count is outside the registry")
    if any(int((targets == index).sum()) < 1 for index in range(class_count)):
        raise BiNOVAD92Error("every D92 class needs support")
    centers = _class_centers(values, targets, class_count)
    old_indices = torch.arange(old_count, device=values.device)
    old_covariance, old_shrinkage = _oas_group_covariance(
        values,
        targets,
        old_indices,
        shrinkage_override=shrinkage_override,
        jitter=jitter,
    )
    if class_count > old_count:
        new_indices = torch.arange(old_count, class_count, device=values.device)
        new_covariance, new_shrinkage = _oas_group_covariance(
            values,
            targets,
            new_indices,
            shrinkage_override=shrinkage_override,
            jitter=jitter,
        )
        covariance = 0.5 * old_covariance + 0.5 * new_covariance
        old_weight, new_weight = 0.5, 0.5
    else:
        new_covariance = None
        new_shrinkage = None
        covariance = old_covariance
        old_weight, new_weight = 1.0, 0.0
    covariance = 0.5 * (covariance + covariance.T)
    cholesky = torch.linalg.cholesky(covariance)
    coefficient = torch.cholesky_solve(centers.T, cholesky).T
    prior = values.new_full((class_count,), 1.0 / float(class_count))
    intercept = -0.5 * torch.sum(centers * coefficient, dim=1) + torch.log(prior)
    return DifferentiableD92State(
        class_ids=tuple(range(class_count)),
        old_class_count=old_count,
        centers=centers,
        covariance=covariance,
        cholesky=cholesky,
        coefficient=coefficient,
        intercept=intercept,
        audit={
            "solver": "torch_cholesky",
            "shrinkage": "oas_analytic" if shrinkage_override is None else "fixed_test_override",
            "old_covariance_weight": old_weight,
            "new_covariance_weight": new_weight,
            "old_shrinkage": float(old_shrinkage.detach().cpu()),
            "new_shrinkage": (
                None if new_shrinkage is None else float(new_shrinkage.detach().cpu())
            ),
            "query_rows_used": 0,
        },
    )


def _top_two_smallest(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = torch.sort(values, dim=1).values
    first = ordered[:, 0]
    second = ordered[:, 1] if ordered.shape[1] > 1 else first
    return first, second


def d92_geometry_conditions(
    state: DifferentiableD92State,
    rows: torch.Tensor,
) -> torch.Tensor:
    values = torch.as_tensor(rows, dtype=state.centers.dtype, device=state.centers.device)
    differences = values[:, None, :] - state.centers[None, :, :]
    flat = differences.reshape(-1, differences.shape[-1]).T
    solved = torch.cholesky_solve(flat, state.cholesky).T.reshape_as(differences)
    distances = torch.sum(differences * solved, dim=2)
    old_distances = distances[:, : state.old_class_count]
    new_distances = distances[:, state.old_class_count :]
    if new_distances.shape[1] < 1:
        raise BiNOVAD92Error("geometry conditions require registered new classes")
    old_first, old_second = _top_two_smallest(old_distances)
    new_first, new_second = _top_two_smallest(new_distances)
    logits = state.score(values)
    old_top = logits[:, : state.old_class_count].max(dim=1).values
    new_top = logits[:, state.old_class_count :].max(dim=1).values
    probabilities = torch.softmax(logits, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=1)
    return torch.stack(
        [old_first, old_second, new_first, new_second, old_top - new_top, entropy],
        dim=1,
    )


def exact_d92_fit(
    identity160: Any,
    fft96: Any,
    labels: Any,
    *,
    class_ids: Sequence[int],
    old_class_count: int,
    seed: int,
    device: Any = "cpu",
) -> Any:
    """Fit the existing exact no-RF32 D92 state after BiNOVA adaptation."""

    registry = tuple(int(value) for value in class_ids)
    if len(registry) == int(old_class_count):
        from cvsrffi.stage2_sf_erbt_oldonly import fit_old_only_erbt

        return fit_old_only_erbt(
            identity160,
            fft96,
            labels,
            class_ids=registry,
            seed=int(seed),
            device=device,
        )
    from cvsrffi.stage2_sf_erbt_four_state import fit_registered_erbt

    return fit_registered_erbt(
        identity160,
        fft96,
        labels,
        class_ids=registry,
        old_class_count=int(old_class_count),
        seed=int(seed),
        device=device,
    )


__all__ = [
    "BiNOVAD92Error",
    "DifferentiableD92State",
    "d92_geometry_conditions",
    "d92_geometry_features",
    "exact_d92_fit",
    "fit_differentiable_d92",
]
