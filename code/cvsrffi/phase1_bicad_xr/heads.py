"""Class-conditional and factorized adversarial heads for BiCAD-XR.

The module is deliberately limited to the Task2 training heads.  It does not
own data sampling, candidate scheduling, or any deployment-time inference
state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _GradientReversal(torch.autograd.Function):
    """Identity in the forward pass and a scaled sign flip in backward."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, value: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, gradient: Tensor
    ) -> tuple[Tensor, None]:
        return -ctx.scale * gradient, None


def gradient_reverse(value: Tensor, scale: float = 1.0) -> Tensor:
    """Return ``value`` while reversing its gradient by ``scale``."""

    scale = _validate_grl_scale(scale)
    return _GradientReversal.apply(value, scale)


def _validate_grl_scale(scale: float) -> float:
    try:
        resolved = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("GRL scale must be a finite non-negative number") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError("GRL scale must be a finite non-negative number")
    return resolved


def _validate_feature_tensor(value: Tensor, name: str) -> None:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, feature]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if value.size(1) < 1:
        raise ValueError(f"{name} must have a non-empty feature dimension")


def _validate_finite_tensor(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_tx_labels(
    tx: Tensor | None,
    *,
    batch_size: int,
    num_classes: int | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Validate hard TX labels and return them on ``device`` when requested."""

    if tx is None:
        raise ValueError("TX labels are required for conditional CDAN")
    if not torch.is_tensor(tx):
        raise ValueError("TX labels must be a one-dimensional integer tensor")
    if tx.ndim != 1:
        raise ValueError("TX labels must be a one-dimensional integer tensor")
    if tx.numel() != batch_size:
        raise ValueError("TX labels batch size must match feature batch size")
    if tx.dtype == torch.bool or tx.is_floating_point() or tx.is_complex():
        raise ValueError("TX labels must use an integer dtype")
    if tx.numel() > 0:
        if int(tx.min().item()) < 0:
            raise ValueError("TX labels are out of range")
        if num_classes is not None and int(tx.max().item()) >= int(num_classes):
            raise ValueError("TX labels are out of range")
    if device is not None and tx.device != device:
        tx = tx.to(device=device)
    return tx


def conditional_outer(z_id: Tensor, tx: Tensor | None, num_classes: int) -> Tensor:
    """Map identity features and hard TX labels to a CDAN outer product.

    For a feature vector ``z`` and class ``y``, the output is the flattened
    ``z outer one_hot(y)`` tensor.  The class labels are intentionally hard
    labels; prediction probabilities are not accepted as a substitute.
    """

    _validate_feature_tensor(z_id, "z_id")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    tx = _validate_tx_labels(
        tx, batch_size=z_id.size(0), num_classes=num_classes, device=z_id.device
    )
    one_hot = F.one_hot(tx, num_classes=num_classes).to(dtype=z_id.dtype)
    return torch.einsum("bd,bc->bdc", z_id, one_hot).reshape(z_id.size(0), -1)


@dataclass(frozen=True)
class DomainFactors:
    """The independent receiver, day, channel and interaction projections."""

    z_r: Tensor
    z_d: Tensor
    z_c: Tensor
    z_int: Tensor


class FactorizedDomainProjector(nn.Module):
    """Project ``z_dom`` into factor-specific domain representations."""

    def __init__(self, feature_dim: int, factor_dim: int, interaction_dim: int) -> None:
        super().__init__()
        dimensions = {
            "feature_dim": feature_dim,
            "factor_dim": factor_dim,
            "interaction_dim": interaction_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.feature_dim = feature_dim
        self.factor_dim = factor_dim
        self.interaction_dim = interaction_dim
        self.z_r = nn.Linear(feature_dim, factor_dim)
        self.z_d = nn.Linear(feature_dim, factor_dim)
        self.z_c = nn.Linear(feature_dim, factor_dim)
        self.z_int = nn.Linear(feature_dim, interaction_dim)

    def forward(self, z_dom: Tensor) -> DomainFactors:
        _validate_feature_tensor(z_dom, "z_dom")
        _validate_finite_tensor(z_dom, "z_dom")
        if z_dom.size(1) != self.feature_dim:
            raise ValueError("z_dom feature_dim must match projector feature_dim")
        return DomainFactors(
            z_r=self.z_r(z_dom),
            z_d=self.z_d(z_dom),
            z_c=self.z_c(z_dom),
            z_int=self.z_int(z_dom),
        )


class FactorizedAdversarialHeads(nn.Module):
    """Factorized BiCAD-XR identity and environment prediction heads.

    The three ``id_*`` heads consume the true-TX conditional representation
    through an identity GRL.  The three ``dom_*`` heads are ordinary positive
    environment classifiers on ``z_dom``.  ``dom_tx`` is a TX classifier on
    ``z_dom`` behind its own GRL.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_receivers: int,
        num_days: int,
        num_channels: int,
    ) -> None:
        super().__init__()
        dimensions = {
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "num_receivers": num_receivers,
            "num_days": num_days,
            "num_channels": num_channels,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_receivers = num_receivers
        self.num_days = num_days
        self.num_channels = num_channels

        conditional_dim = feature_dim * num_classes
        self.id_receiver = nn.Linear(conditional_dim, num_receivers)
        self.id_day = nn.Linear(conditional_dim, num_days)
        self.id_channel = nn.Linear(conditional_dim, num_channels)

        self.dom_receiver = nn.Linear(feature_dim, num_receivers)
        self.dom_day = nn.Linear(feature_dim, num_days)
        self.dom_channel = nn.Linear(feature_dim, num_channels)
        self.dom_tx = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        z_id: Tensor,
        z_dom: Tensor,
        tx: Tensor | None,
        grl_identity: float = 1.0,
        grl_tx: float = 1.0,
    ) -> Dict[str, Tensor]:
        _validate_feature_tensor(z_id, "z_id")
        _validate_feature_tensor(z_dom, "z_dom")
        if z_id.size(0) != z_dom.size(0):
            raise ValueError("z_id and z_dom batch sizes must match")
        if z_id.size(1) != self.feature_dim or z_dom.size(1) != self.feature_dim:
            raise ValueError("z_id and z_dom feature dimensions must match feature_dim")
        identity_scale = _validate_grl_scale(grl_identity)
        tx_scale = _validate_grl_scale(grl_tx)

        conditional = conditional_outer(z_id, tx, self.num_classes)
        conditional = gradient_reverse(conditional, identity_scale)
        reversed_z_dom = gradient_reverse(z_dom, tx_scale)

        return {
            "id_receiver": self.id_receiver(conditional),
            "id_day": self.id_day(conditional),
            "id_channel": self.id_channel(conditional),
            "dom_receiver": self.dom_receiver(z_dom),
            "dom_day": self.dom_day(z_dom),
            "dom_channel": self.dom_channel(z_dom),
            "dom_tx": self.dom_tx(reversed_z_dom),
        }


# Keep the short spelling available to callers that use the existing project
# convention, without adding another behavior or module-level state.
grad_reverse = gradient_reverse


__all__ = [
    "DomainFactors",
    "FactorizedAdversarialHeads",
    "FactorizedDomainProjector",
    "conditional_outer",
    "grad_reverse",
    "gradient_reverse",
]
