"""Finite Margin-REx/CVaR and bounded hard-group weighting for CV2."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _finite_margin_vector(margins: Tensor, name: str = "margins") -> Tensor:
    if not torch.is_tensor(margins):
        raise ValueError(f"{name} must be a tensor")
    if margins.ndim != 1:
        raise ValueError(f"{name} must have shape [sample]")
    if not margins.is_floating_point() or margins.numel() == 0:
        raise ValueError(f"{name} must be a non-empty floating-point tensor")
    if not torch.isfinite(margins).all():
        raise ValueError(f"{name} must contain only finite values")
    return margins


def _tail_fraction(value: float, name: str = "tail_fraction") -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in (0,1]") from exc
    if not math.isfinite(resolved) or not 0.0 < resolved <= 1.0:
        raise ValueError(f"{name} must be a finite number in (0,1]")
    return resolved


def _nonnegative(value: float, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return resolved


def _group_ids(
    groups: Tensor | Sequence[Tensor] | None,
    *,
    count: int,
    device: torch.device,
) -> Tensor:
    if groups is None:
        return torch.zeros(count, dtype=torch.long, device=device)
    if torch.is_tensor(groups):
        value = groups
    else:
        if isinstance(groups, (str, bytes)):
            raise ValueError("groups must contain integer group keys")
        try:
            value = torch.as_tensor(groups)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("groups must contain integer group keys") from exc

    if value.ndim == 1:
        if value.numel() != count:
            raise ValueError("groups must match the margin batch")
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise ValueError("groups must contain integer group keys")
        return value.to(device=device, dtype=torch.long)
    if value.ndim == 2:
        if value.size(0) != count or value.size(1) == 0:
            raise ValueError("groups must have shape [sample, key]")
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise ValueError("groups must contain integer group keys")
        rows = value.to(device=device, dtype=torch.long)
        _, inverse = torch.unique(rows, dim=0, sorted=True, return_inverse=True)
        return inverse
    raise ValueError("groups must have shape [sample] or [sample, key]")


def _group_cvar(risk: Tensor, group_ids: Tensor, tail_fraction: float) -> Tensor:
    group_risks: list[Tensor] = []
    for group_id in torch.unique(group_ids, sorted=True):
        values = risk[group_ids == group_id]
        count = max(1, math.ceil(tail_fraction * values.numel()))
        group_risks.append(values.topk(count, largest=True, sorted=False).values.mean())
    if not group_risks:
        raise ValueError("groups must contain at least one group")
    result = torch.stack(group_risks)
    if not torch.isfinite(result).all():
        raise ValueError("group risks must be finite")
    return result


def margin_group_risks(
    margins: Tensor,
    groups: Tensor | Sequence[Tensor] | None = None,
    *,
    tail_fraction: float = 0.2,
) -> Tensor:
    """Return one differentiable worst-tail risk for each class/domain group.

    ``groups`` may be a single integer key or a matrix of integer keys such as
    ``[class, receiver, view]``.  A sample risk is ``softplus(-margin)``;
    each group's result is the mean of its largest ``ceil(alpha*n)`` risks.
    """

    margins = _finite_margin_vector(margins)
    tail_fraction = _tail_fraction(tail_fraction)
    group_ids = _group_ids(groups, count=margins.numel(), device=margins.device)
    risk = F.softplus(-margins)
    if not torch.isfinite(risk).all():
        raise ValueError("margin risks must be finite")
    return _group_cvar(risk, group_ids, tail_fraction)


def margin_rex_cvar_loss(
    margins: Tensor,
    groups: Tensor | Sequence[Tensor] | None = None,
    *,
    tail_fraction: float = 0.2,
    lambda_rex: float = 0.02,
    lambda_cvar: float = 0.05,
) -> Tensor:
    """Combine finite group-risk variance(REx) with group-tail CVaR."""

    lambda_rex = _nonnegative(lambda_rex, "lambda_rex")
    lambda_cvar = _nonnegative(lambda_cvar, "lambda_cvar")
    risks = margin_group_risks(
        margins, groups, tail_fraction=tail_fraction
    )
    rex = risks.var(unbiased=False)
    tail_count = max(1, math.ceil(tail_fraction * risks.numel()))
    cvar = risks.topk(tail_count, largest=True, sorted=False).values.mean()
    loss = lambda_rex * rex + lambda_cvar * cvar
    if not torch.isfinite(loss).all():
        raise ValueError("Margin-REx/CVaR loss must be finite")
    return loss


def bounded_hard_group_weights(
    group_risks: Tensor,
    *,
    hard_fraction: float = 0.2,
    max_hard_fraction: float = 0.30,
) -> Tensor:
    """Return sampling probabilities whose highest-risk group mass is capped.

    The highest-risk ``ceil(hard_fraction*n)`` groups receive exactly the
    capped mass, while the remaining groups receive the complementary mass.
    This keeps the hard-group sampling probability at or below 30% by
    default, even when their risks are much larger than the rest.
    """

    risks = _finite_margin_vector(group_risks, "group_risks")
    hard_fraction = _tail_fraction(hard_fraction, "hard_fraction")
    max_hard_fraction = _tail_fraction(
        max_hard_fraction, "max_hard_fraction"
    )
    count = risks.numel()
    hard_count = min(count, max(1, math.ceil(hard_fraction * count)))
    if hard_count >= count and max_hard_fraction < 1.0:
        raise ValueError("30% hard-group cap is impossible when every group is hard")

    order = torch.argsort(risks.detach(), descending=True)
    hard_indices = order[:hard_count]
    weights = torch.empty_like(risks)
    if hard_count == count:
        weights.fill_(1.0 / count)
    else:
        hard_mass = min(max_hard_fraction, 1.0)
        weights.fill_((1.0 - hard_mass) / (count - hard_count))
        weights[hard_indices] = hard_mass / hard_count
    if not torch.isfinite(weights).all() or torch.any(weights < 0.0):
        raise ValueError("hard-group weights must be finite and non-negative")
    return weights


__all__ = [
    "bounded_hard_group_weights",
    "margin_group_risks",
    "margin_rex_cvar_loss",
]
