"""Phase1.5 objectives aligned with the frozen-prototype decision rule."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def prototype_logits(
    features: Tensor,
    prototypes: Tensor,
    *,
    logit_scale: float,
) -> Tensor:
    if features.ndim != 2 or prototypes.ndim != 2 or features.shape[1] != prototypes.shape[1]:
        raise ValueError("features and prototypes must be width-aligned matrices")
    number = float(logit_scale)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("logit_scale must be finite and positive")
    return number * (
        F.normalize(features.float(), dim=1, eps=1.0e-8)
        @ F.normalize(prototypes.float(), dim=1, eps=1.0e-8).transpose(0, 1)
    )


def frozen_prototype_losses(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    *,
    scale: float,
) -> Tensor:
    if features.ndim != 2 or prototypes.ndim != 2 or features.shape[1] != prototypes.shape[1]:
        raise ValueError("features and prototypes must be width-aligned matrices")
    if labels.ndim != 1 or labels.numel() != features.shape[0]:
        raise ValueError("labels must align with features")
    if bool((labels < 0).any()) or bool((labels >= prototypes.shape[0]).any()):
        raise ValueError("labels must index frozen prototype rows")
    logits = prototype_logits(features, prototypes, logit_scale=scale)
    return F.cross_entropy(logits, labels.long(), reduction="none")


def frozen_prototype_ce(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    *,
    scale: float,
) -> Tensor:
    return frozen_prototype_losses(features, labels, prototypes, scale=scale).mean()


def smooth_class_floor_loss(
    per_sample_loss: Tensor,
    labels: Tensor,
    *,
    temperature: float,
) -> Tensor:
    if per_sample_loss.ndim != 1 or labels.ndim != 1 or per_sample_loss.numel() != labels.numel():
        raise ValueError("per-sample loss and labels must be aligned vectors")
    tau = float(temperature)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("temperature must be finite and positive")
    class_means = torch.stack(
        [per_sample_loss[labels == class_id].mean() for class_id in torch.unique(labels, sorted=True)]
    )
    return tau * torch.logsumexp(class_means / tau, dim=0)


def trust_region_loss(
    adapted: Tensor,
    base: Tensor,
    *,
    max_relative_move: float,
) -> Tensor:
    if adapted.shape != base.shape or adapted.ndim != 2:
        raise ValueError("adapted and base features must be shape-aligned matrices")
    radius = float(max_relative_move)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("max_relative_move must be finite and positive")
    relative = torch.linalg.vector_norm(adapted - base, dim=1) / torch.linalg.vector_norm(
        base, dim=1
    ).clamp_min(1.0e-8)
    return F.relu(relative - radius).square().mean()


__all__ = [
    "frozen_prototype_ce",
    "frozen_prototype_losses",
    "prototype_logits",
    "smooth_class_floor_loss",
    "trust_region_loss",
]
