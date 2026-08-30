"""Task2 finite losses for ADV3B02-BiCAD-XR."""

from __future__ import annotations

import math
from collections.abc import Sequence

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


def _validate_margin_vector(value: Tensor, name: str) -> None:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [sample]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    _validate_finite(value, name)


def _validate_group_ids(
    groups: Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> Tensor:
    if groups is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    if not torch.is_tensor(groups):
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    if groups.ndim != 1 or groups.numel() != batch_size:
        raise ValueError(f"{name} must match the margin batch")
    if groups.dtype == torch.bool or groups.is_floating_point() or groups.is_complex():
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    return groups.to(device=device, dtype=torch.long)


def _validate_tail_fraction(tail_fraction: float) -> float:
    try:
        resolved = float(tail_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("tail_fraction must be a finite number in (0,1]") from exc
    if not math.isfinite(resolved) or not 0.0 < resolved <= 1.0:
        raise ValueError("tail_fraction must be a finite number in (0,1]")
    return resolved


def _validate_three_weights(weights: Sequence[float]) -> tuple[float, float, float]:
    if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
        raise ValueError("weights must contain exactly three finite non-negative numbers")
    if len(weights) != 3:
        raise ValueError("weights must contain exactly three finite non-negative numbers")
    return tuple(_validate_weight(value, f"weights[{index}]") for index, value in enumerate(weights))  # type: ignore[return-value]


class DetachedEMA:
    """A detached exponential moving average for scalar or tensor risks."""

    def __init__(self, decay: float = 0.9, value: Tensor | None = None) -> None:
        try:
            resolved_decay = float(decay)
        except (TypeError, ValueError) as exc:
            raise ValueError("decay must be a finite number in [0,1)") from exc
        if not math.isfinite(resolved_decay) or not 0.0 <= resolved_decay < 1.0:
            raise ValueError("decay must be a finite number in [0,1)")
        self.decay = resolved_decay
        self._value: Tensor | None = None
        if value is not None:
            self._value = self._detach_value(value, "value")

    @staticmethod
    def _detach_value(value: Tensor, name: str) -> Tensor:
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")
        _validate_finite(value, name)
        return value.detach().clone()

    @property
    def value(self) -> Tensor | None:
        return None if self._value is None else self._value.detach().clone()

    def update(self, value: Tensor) -> Tensor:
        """Update the running state without retaining an autograd edge."""

        detached = self._detach_value(value, "value")
        if self._value is None:
            self._value = detached
        else:
            if self._value.shape != detached.shape:
                raise ValueError("EMA value shape must remain constant")
            if self._value.device != detached.device:
                raise ValueError("EMA value device must remain constant")
            self._value = (
                self.decay * self._value + (1.0 - self.decay) * detached
            ).detach()
        return self._value.detach().clone()

    def reset(self) -> None:
        self._value = None


def detached_ema(
    value: Tensor,
    previous: Tensor | None = None,
    *,
    decay: float = 0.9,
) -> Tensor:
    """Return one detached EMA update for callers without a state object."""

    return DetachedEMA(decay=decay, value=previous).update(value)


def group_margin_cvar(
    margins: Tensor,
    groups: Tensor | None = None,
    *,
    tail_fraction: float = 0.2,
    ema: DetachedEMA | None = None,
) -> Tensor:
    """Compute the mean within-group CVaR of negative classification margins.

    A margin is converted to the finite risk ``softplus(-margin)``.  Each
    group contributes the mean of its largest ``ceil(alpha*n)`` risks, and
    groups are then weighted equally.  Optional EMA state is updated from a
    detached scalar after the differentiable risk has been computed.
    """

    _validate_margin_vector(margins, "margins")
    tail_fraction = _validate_tail_fraction(tail_fraction)
    group_ids = _validate_group_ids(
        groups,
        batch_size=margins.numel(),
        device=margins.device,
        name="groups",
    )
    risk = F.softplus(-margins)
    group_risks: list[Tensor] = []
    for group_id in torch.unique(group_ids, sorted=True):
        values = risk[group_ids == group_id]
        count = max(1, math.ceil(tail_fraction * values.numel()))
        group_risks.append(values.topk(count, largest=True, sorted=False).values.mean())
    result = torch.stack(group_risks).mean()
    if ema is not None:
        if not isinstance(ema, DetachedEMA):
            raise ValueError("ema must be a DetachedEMA or None")
        ema.update(result.detach())
    return result


def three_layer_group_margin_cvar(
    tx_margins: Tensor,
    xdc_margins: Tensor,
    tangent_margins: Tensor,
    *,
    tx_groups: Tensor | None = None,
    xdc_groups: Tensor | None = None,
    tangent_groups: Tensor | None = None,
    tail_fraction: float = 0.2,
    weights: Sequence[float] = (0.6, 0.3, 0.1),
    emas: Sequence[DetachedEMA | None] | None = None,
    ema: DetachedEMA | None = None,
) -> Tensor:
    """Combine TX, XDC and tangent classification tail risks.

    This function intentionally has no domain, GRL or cross-covariance input;
    the tail mechanism can therefore add weight only to the three allowed
    classification risks.
    """

    _validate_margin_vector(tx_margins, "tx_margins")
    _validate_margin_vector(xdc_margins, "xdc_margins")
    _validate_margin_vector(tangent_margins, "tangent_margins")
    if not (
        tx_margins.device == xdc_margins.device == tangent_margins.device
    ):
        raise ValueError("all margin tensors must use the same device")
    resolved_weights = _validate_three_weights(weights)
    if emas is not None:
        if not isinstance(emas, Sequence) or len(emas) != 3:
            raise ValueError("emas must contain exactly three EMA states")
        if any(state is not None and not isinstance(state, DetachedEMA) for state in emas):
            raise ValueError("emas must contain DetachedEMA states or None")
    else:
        emas = (None, None, None)

    tx_risk = group_margin_cvar(
        tx_margins, tx_groups, tail_fraction=tail_fraction, ema=emas[0]
    )
    xdc_risk = group_margin_cvar(
        xdc_margins, xdc_groups, tail_fraction=tail_fraction, ema=emas[1]
    )
    tangent_risk = group_margin_cvar(
        tangent_margins,
        tangent_groups,
        tail_fraction=tail_fraction,
        ema=emas[2],
    )
    result = (
        resolved_weights[0] * tx_risk
        + resolved_weights[1] * xdc_risk
        + resolved_weights[2] * tangent_risk
    )
    if ema is not None:
        if not isinstance(ema, DetachedEMA):
            raise ValueError("ema must be a DetachedEMA or None")
        ema.update(result.detach())
    return result


def apply_margin_tail(
    base_loss: Tensor,
    tx_risk: Tensor,
    xdc_risk: Tensor,
    tangent_risk: Tensor,
    *,
    weights: Sequence[float] = (0.6, 0.3, 0.1),
) -> Tensor:
    """Add only weighted TX/XDC/tangent classification risks."""

    for value, name in (
        (base_loss, "base_loss"),
        (tx_risk, "tx_risk"),
        (xdc_risk, "xdc_risk"),
        (tangent_risk, "tangent_risk"),
    ):
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")
        _validate_finite(value, name)
    resolved_weights = _validate_three_weights(weights)
    total = base_loss
    total = total + (
        resolved_weights[0] * tx_risk.mean()
        + resolved_weights[1] * xdc_risk.mean()
        + resolved_weights[2] * tangent_risk.mean()
    )
    return total


# Short aliases keep the public names discoverable for the task's three
# mechanism families without introducing a second implementation.
margin_tail_loss = apply_margin_tail
three_layer_margin_cvar = three_layer_group_margin_cvar


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
    "DetachedEMA",
    "apply_margin_tail",
    "classification_margin",
    "conditional_cross_covariance",
    "detached_ema",
    "group_margin_cvar",
    "margin_tail_loss",
    "paired_satellite_loss",
    "three_layer_group_margin_cvar",
    "three_layer_margin_cvar",
]
