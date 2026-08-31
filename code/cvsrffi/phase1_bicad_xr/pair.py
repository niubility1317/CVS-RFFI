"""Pure PairBiCAD pair objectives for the ADV3B02-BiCAD-XR family.

The functions in this module operate only on already-computed tensors.  They
do not own a backbone, create a prediction head, read TX labels, or perform a
model forward.  This keeps the pair losses usable from both the labeled and
unlabeled source paths.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .heads import gradient_reverse


def _validate_matrix(value: Tensor, name: str) -> None:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, feature]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if value.size(1) < 1:
        raise ValueError(f"{name} must have a non-empty feature dimension")


def _validate_finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_pair(left: Tensor, right: Tensor, left_name: str, right_name: str) -> None:
    _validate_matrix(left, left_name)
    _validate_matrix(right, right_name)
    if left.shape != right.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape")
    if left.device != right.device:
        raise ValueError(f"{left_name} and {right_name} must use the same device")
    _validate_finite(left, left_name)
    _validate_finite(right, right_name)


def _validate_non_negative(value: float, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return resolved


def _connected_zero(*values: Tensor) -> Tensor:
    result = values[0].sum() * 0.0
    for value in values[1:]:
        result = result + value.sum() * 0.0
    return result


def _safe_mean(value: Tensor, *references: Tensor) -> Tensor:
    if value.numel() == 0:
        return _connected_zero(*references)
    return value.mean()


def _validate_channel_labels(channel: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
    if not torch.is_tensor(channel):
        raise ValueError("channel must be a one-dimensional integer tensor")
    if channel.ndim != 1 or channel.numel() != batch_size:
        raise ValueError("channel must be a one-dimensional tensor matching the batch")
    if channel.dtype == torch.bool or channel.is_floating_point() or channel.is_complex():
        raise ValueError("channel must be a one-dimensional integer tensor")
    if channel.numel() and int(channel.min().item()) < 0:
        raise ValueError("channel labels must be non-negative")
    return channel.to(device=device, dtype=torch.long)


def _channel_count(channel: Tensor) -> int:
    return max(1, int(channel.max().item()) + 1) if channel.numel() else 1


def _class_logits(value: Tensor, num_classes: int) -> Tensor:
    """Fit a feature tensor to a label-sized logit view without a module."""

    if value.size(1) == num_classes:
        return value
    if value.size(1) > num_classes:
        return value[:, :num_classes]
    padding = value.new_zeros(value.size(0), num_classes - value.size(1))
    return torch.cat((value, padding), dim=1)


def _channel_target(channel: Tensor, feature_dim: int, dtype: torch.dtype) -> Tensor:
    target = F.one_hot(channel, num_classes=_channel_count(channel)).to(dtype=dtype)
    if target.size(1) < feature_dim:
        target = F.pad(target, (0, feature_dim - target.size(1)))
    return target[:, :feature_dim]


def pair_identity_hinge(clean: Tensor, satellite: Tensor, epsilon: float = 0.05) -> Tensor:
    """Penalize normalized same-sample identity distance beyond ``epsilon``."""

    _validate_pair(clean, satellite, "clean", "satellite")
    epsilon = _validate_non_negative(epsilon, "epsilon")
    clean_normalized = F.normalize(clean, p=2.0, dim=1, eps=1e-8)
    satellite_normalized = F.normalize(satellite, p=2.0, dim=1, eps=1e-8)
    distance = torch.linalg.vector_norm(clean_normalized - satellite_normalized, dim=1)
    hinge = F.relu(distance - epsilon)
    return _safe_mean(hinge, clean, satellite)


def _vicreg_variance(value: Tensor, gamma: float) -> Tensor:
    if value.size(0) < 2:
        return _connected_zero(value)
    standard_deviation = torch.sqrt(value.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(gamma - standard_deviation).mean()


def _vicreg_covariance(value: Tensor) -> Tensor:
    if value.size(0) < 2:
        return _connected_zero(value)
    centered = value - value.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered / (value.size(0) - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag(diagonal)
    return off_diagonal.square().mean()


def vicreg_pair_loss(clean: Tensor, satellite: Tensor, gamma: float = 1.0) -> dict[str, Tensor]:
    """Return finite VICReg invariance, variance and covariance components."""

    _validate_pair(clean, satellite, "clean", "satellite")
    gamma = _validate_non_negative(gamma, "gamma")
    if clean.numel() == 0:
        invariance = _connected_zero(clean, satellite)
    else:
        invariance = (clean - satellite).square().mean()
    variance = 0.5 * (
        _vicreg_variance(clean, gamma) + _vicreg_variance(satellite, gamma)
    )
    covariance = 0.5 * (
        _vicreg_covariance(clean) + _vicreg_covariance(satellite)
    )
    return {
        "total": invariance + variance + covariance,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }


def _optional_logits(
    value: Tensor | None,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
) -> Tensor | None:
    if value is None:
        return None
    _validate_matrix(value, name)
    if value.size(0) != batch_size:
        raise ValueError(f"{name} batch size must match feature batch")
    if value.device != device:
        raise ValueError(f"{name} must use the same device as feature tensors")
    _validate_finite(value, name)
    return value


def pair_delta_objectives(
    clean_id: Tensor,
    satellite_id: Tensor,
    clean_c: Tensor,
    satellite_c: Tensor,
    channel: Tensor,
    *,
    epsilon: float = 0.05,
    delta_radius: float = 0.25,
    grl_scale: float = 1.0,
    identity_channel_logits: Tensor | None = None,
    channel_logits: Tensor | None = None,
    delta_channel_logits: Tensor | None = None,
    include_delta_norm_hinge: bool = True,
) -> dict[str, Tensor]:
    """Return pure identity/channel delta objectives.

    ``identity_channel_logits``, ``channel_logits`` and
    ``delta_channel_logits`` are optional precomputed logits.  When omitted,
    the corresponding objective uses a tensor-only class-logit view, so this
    function never invokes a module or performs a backbone forward.  The
    identity fallback applies GRL to the identity delta; callers supplying
    logits are responsible for any GRL used while producing those logits.
    """

    _validate_pair(clean_id, satellite_id, "clean_id", "satellite_id")
    _validate_pair(clean_c, satellite_c, "clean_c", "satellite_c")
    if clean_id.size(0) != clean_c.size(0):
        raise ValueError("identity and channel feature batches must match")
    if clean_id.device != clean_c.device:
        raise ValueError("all feature tensors must use the same device")

    channel = _validate_channel_labels(
        channel, batch_size=clean_id.size(0), device=clean_id.device
    )
    epsilon = _validate_non_negative(epsilon, "epsilon")
    delta_radius = _validate_non_negative(delta_radius, "delta_radius")
    grl_scale = _validate_non_negative(grl_scale, "grl_scale")
    if not isinstance(include_delta_norm_hinge, bool):
        raise ValueError("include_delta_norm_hinge must be a bool")

    batch_size = clean_id.size(0)
    num_channels = _channel_count(channel)
    identity_logits = _optional_logits(
        identity_channel_logits,
        name="identity_channel_logits",
        batch_size=batch_size,
        device=clean_id.device,
    )
    channel_prediction_logits = _optional_logits(
        channel_logits,
        name="channel_logits",
        batch_size=batch_size,
        device=clean_id.device,
    )
    delta_prediction_logits = _optional_logits(
        delta_channel_logits,
        name="delta_channel_logits",
        batch_size=batch_size,
        device=clean_id.device,
    )

    identity_delta = satellite_id - clean_id
    channel_delta = satellite_c - clean_c
    if identity_logits is None:
        identity_logits = _class_logits(
            gradient_reverse(identity_delta, grl_scale), num_channels
        )
    else:
        identity_logits = _class_logits(identity_logits, num_channels)
    identity_channel_adversary = (
        F.cross_entropy(identity_logits, channel)
        if batch_size
        else _connected_zero(clean_id, satellite_id)
    )

    if channel_prediction_logits is None:
        channel_prediction_logits = _class_logits(
            0.5 * (clean_c + satellite_c), num_channels
        )
    else:
        channel_prediction_logits = _class_logits(channel_prediction_logits, num_channels)
    channel_prediction = (
        F.cross_entropy(channel_prediction_logits, channel)
        if batch_size
        else _connected_zero(clean_c, satellite_c)
    )

    pair_stability = pair_identity_hinge(clean_id, satellite_id, epsilon)
    channel_target = _channel_target(channel, channel_delta.size(1), channel_delta.dtype)
    if delta_prediction_logits is None:
        channel_equivariance = _safe_mean(
            (channel_delta - channel_target).square(), clean_c, satellite_c
        )
    else:
        delta_prediction_logits = _class_logits(delta_prediction_logits, num_channels)
        channel_equivariance = (
            F.cross_entropy(delta_prediction_logits, channel)
            if batch_size
            else _connected_zero(clean_c, satellite_c)
        )

    components = {
        "identity_channel_adversary": identity_channel_adversary,
        "channel_prediction": channel_prediction,
        "pair_stability": pair_stability,
        "channel_equivariance": channel_equivariance,
    }
    if include_delta_norm_hinge:
        delta_norm = torch.linalg.vector_norm(identity_delta, dim=1)
        components["delta_norm_hinge"] = _safe_mean(
            F.relu(delta_norm - delta_radius), clean_id, satellite_id
        )
    return components


__all__ = [
    "pair_delta_objectives",
    "pair_identity_hinge",
    "vicreg_pair_loss",
]
