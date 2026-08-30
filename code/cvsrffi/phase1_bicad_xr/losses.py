"""Task2 finite losses for ADV3B02-BiCAD-XR."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .heads import _validate_tx_labels


def _validate_matrix(value: Tensor, name: str) -> None:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, feature]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if value.size(1) < 1:
        raise ValueError(f"{name} must have a non-empty feature dimension")


def _validate_pair(left: Tensor, right: Tensor, left_name: str, right_name: str) -> None:
    _validate_matrix(left, left_name)
    _validate_matrix(right, right_name)
    if left.shape != right.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape")


def _validate_finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def conditional_cross_covariance(z_id: Tensor, z_dom: Tensor, tx: Tensor | None) -> Tensor:
    """Penalize normalized cross-covariance within each TX group.

    Each TX group is centered independently and contributes only when it has
    at least two samples.  If no group is valid, the returned scalar remains
    connected to both feature tensors through a differentiable zero.
    """

    _validate_matrix(z_id, "z_id")
    _validate_matrix(z_dom, "z_dom")
    if z_id.size(0) != z_dom.size(0):
        raise ValueError("z_id and z_dom batch sizes must match")
    if z_id.device != z_dom.device:
        raise ValueError("z_id and z_dom must be on the same device")
    _validate_finite(z_id, "z_id")
    _validate_finite(z_dom, "z_dom")
    tx = _validate_tx_labels(tx, batch_size=z_id.size(0), device=z_id.device)

    group_losses: list[Tensor] = []
    for class_id in torch.unique(tx, sorted=True):
        mask = tx == class_id
        count = int(mask.sum().item())
        if count < 2:
            continue
        id_group = z_id[mask]
        dom_group = z_dom[mask]
        id_centered = id_group - id_group.mean(dim=0, keepdim=True)
        dom_centered = dom_group - dom_group.mean(dim=0, keepdim=True)
        covariance = id_centered.transpose(0, 1) @ dom_centered / (count - 1)
        group_losses.append(
            covariance.square().sum() / (z_id.size(1) * z_dom.size(1))
        )

    if not group_losses:
        return z_id.sum() * 0.0 + z_dom.sum() * 0.0
    return torch.stack(group_losses).mean()


def classification_margin(logits: Tensor, tx: Tensor | None) -> Tensor:
    """Return the true-class logit minus the strongest competing logit."""

    _validate_matrix(logits, "logits")
    if logits.size(1) < 2:
        raise ValueError("logits must contain at least two classes")
    _validate_finite(logits, "logits")
    tx = _validate_tx_labels(
        tx, batch_size=logits.size(0), num_classes=logits.size(1), device=logits.device
    )
    true_logits = logits.gather(1, tx.unsqueeze(1)).squeeze(1)
    class_mask = F.one_hot(tx, num_classes=logits.size(1)).to(dtype=torch.bool)
    other_logits = logits.masked_fill(class_mask, torch.finfo(logits.dtype).min)
    return true_logits - other_logits.max(dim=1).values


def _validate_weight(weight: float, name: str) -> float:
    try:
        resolved = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return resolved


def _symmetric_prediction_js(clean_logits: Tensor, satellite_logits: Tensor) -> Tensor:
    clean_log_prob = F.log_softmax(clean_logits, dim=1)
    satellite_log_prob = F.log_softmax(satellite_logits, dim=1)
    clean_prob = clean_log_prob.exp()
    satellite_prob = satellite_log_prob.exp()
    mean_prob = 0.5 * (clean_prob + satellite_prob)
    mean_log_prob = mean_prob.clamp_min(torch.finfo(mean_prob.dtype).tiny).log()
    clean_kl = (clean_prob * (clean_log_prob - mean_log_prob)).sum(dim=1)
    satellite_kl = (satellite_prob * (satellite_log_prob - mean_log_prob)).sum(dim=1)
    return 0.5 * (clean_kl + satellite_kl).mean()


def paired_satellite_loss(
    clean_z_id: Tensor,
    satellite_z_id: Tensor,
    clean_logits: Tensor | None = None,
    satellite_logits: Tensor | None = None,
    tx: Tensor | None = None,
    *,
    identity_weight: float = 1.0,
    prediction_weight: float = 1.0,
    margin_weight: float = 0.0,
) -> Tensor:
    """Match clean/satellite identity features and optional TX predictions.

    The identity term is ``1-cosine_similarity``.  When both logit tensors
    are supplied, a symmetric Jensen-Shannon divergence is added.  ``tx`` and
    ``margin_weight`` optionally preserve the clean classification margin;
    their default weight is zero so the basic pair loss has no hidden label
    dependency.  All terms are differentiable with respect to both views.
    """

    _validate_pair(clean_z_id, satellite_z_id, "clean_z_id", "satellite_z_id")
    if clean_z_id.device != satellite_z_id.device:
        raise ValueError("clean_z_id and satellite_z_id must be on the same device")
    _validate_finite(clean_z_id, "clean_z_id")
    _validate_finite(satellite_z_id, "satellite_z_id")
    identity_weight = _validate_weight(identity_weight, "identity_weight")
    prediction_weight = _validate_weight(prediction_weight, "prediction_weight")
    margin_weight = _validate_weight(margin_weight, "margin_weight")

    identity_term = 1.0 - F.cosine_similarity(
        clean_z_id, satellite_z_id, dim=1, eps=1e-8
    ).mean()
    total = identity_weight * identity_term

    if (clean_logits is None) != (satellite_logits is None):
        raise ValueError("clean_logits and satellite_logits must be provided together")
    if clean_logits is not None and satellite_logits is not None:
        _validate_pair(clean_logits, satellite_logits, "clean_logits", "satellite_logits")
        if clean_logits.size(1) < 2:
            raise ValueError("logits must contain at least two classes")
        if clean_logits.size(0) != clean_z_id.size(0):
            raise ValueError("pair feature and logit batch sizes must match")
        _validate_finite(clean_logits, "clean_logits")
        _validate_finite(satellite_logits, "satellite_logits")
        total = total + prediction_weight * _symmetric_prediction_js(
            clean_logits, satellite_logits
        )

    if margin_weight:
        if clean_logits is None or satellite_logits is None:
            raise ValueError("margin preservation requires clean and satellite logits")
        clean_margin = classification_margin(clean_logits, tx)
        satellite_margin = classification_margin(satellite_logits, tx)
        total = total + margin_weight * (clean_margin - satellite_margin).square().mean()

    return total


__all__ = [
    "classification_margin",
    "conditional_cross_covariance",
    "paired_satellite_loss",
]
