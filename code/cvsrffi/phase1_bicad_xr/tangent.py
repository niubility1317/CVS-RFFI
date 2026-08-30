"""Source-only receiver tangent geometry for BiCAD-XR F1/F2."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable

import torch
from torch import Tensor

from .gradients import safe_svd


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        resolved = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(resolved)


def _validate_features(features: Tensor, name: str = "features") -> None:
    if not torch.is_tensor(features):
        raise ValueError(f"{name} must be a tensor")
    if features.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, feature]")
    if not features.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if features.size(0) == 0 or features.size(1) == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(features).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_receiver_ids(
    receiver_ids: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str = "receiver_ids",
) -> Tensor:
    if not torch.is_tensor(receiver_ids):
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    if receiver_ids.ndim != 1 or receiver_ids.numel() != batch_size:
        raise ValueError(f"{name} must match the feature batch")
    if receiver_ids.dtype == torch.bool or receiver_ids.is_floating_point() or receiver_ids.is_complex():
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    return receiver_ids.to(device=device, dtype=torch.long)


def _validate_scalar(name: str, value: object, *, non_negative: bool = False) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(resolved) or (non_negative and resolved < 0.0):
        raise ValueError(f"{name} must be a finite number")
    return resolved


class ReceiverTangentBank:
    """Store one detached low-rank tangent basis per source receiver.

    ``fit`` is the only method that learns geometry.  The optional
    ``source_receivers`` allowlist makes accidental heldout-receiver reads
    fail closed; when omitted, receiver IDs in the fit call define the source
    set and all later calls must use that frozen set.
    """

    def __init__(
        self,
        feature_dim: int,
        rank: int = 4,
        *,
        source_receivers: Iterable[int] | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.feature_dim = _positive_integer("feature_dim", feature_dim)
        self.rank = _positive_integer("rank", rank)
        self.eps = _validate_scalar("eps", eps, non_negative=True)
        if self.eps <= 0.0:
            raise ValueError("eps must be a finite positive number")

        self._source_receivers: tuple[int, ...] | None = None
        if source_receivers is not None:
            try:
                values = tuple(int(value) for value in source_receivers)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("source_receivers must contain integer IDs") from exc
            if not values or len(set(values)) != len(values) or min(values) < 0:
                raise ValueError("source_receivers must be unique non-negative IDs")
            self._source_receivers = tuple(sorted(values))
        self._means: dict[int, Tensor] = {}
        self._bases: dict[int, Tensor] = {}
        self._fitted = False

    @property
    def receiver_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._bases))

    @property
    def basis(self) -> dict[int, Tensor]:
        return {receiver: value.detach().clone() for receiver, value in self._bases.items()}

    @property
    def means(self) -> dict[int, Tensor]:
        return {receiver: value.detach().clone() for receiver, value in self._means.items()}

    @staticmethod
    def _receiver_scalar(receiver_id: int | Tensor) -> int:
        if torch.is_tensor(receiver_id):
            if receiver_id.ndim != 0:
                raise ValueError("receiver_id must be a scalar integer")
            if receiver_id.dtype == torch.bool or receiver_id.is_floating_point() or receiver_id.is_complex():
                raise ValueError("receiver_id must be a scalar integer")
            return int(receiver_id.item())
        if isinstance(receiver_id, bool):
            raise ValueError("receiver_id must be a scalar integer")
        try:
            return int(operator.index(receiver_id))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("receiver_id must be a scalar integer") from exc

    def _check_source_receiver(self, receiver_id: int) -> None:
        if receiver_id not in self._bases:
            raise ValueError(
                f"receiver {receiver_id} is not a fitted source receiver; heldout receivers are forbidden"
            )

    def fit(self, features: Tensor, receiver_ids: Tensor) -> "ReceiverTangentBank":
        """Fit detached receiver tangent bases from source features only."""

        _validate_features(features)
        if features.size(1) != self.feature_dim:
            raise ValueError("features feature dimension must match feature_dim")
        ids = _validate_receiver_ids(
            receiver_ids,
            batch_size=features.size(0),
            device=features.device,
        )
        observed = tuple(int(value) for value in torch.unique(ids, sorted=True).tolist())
        if self._source_receivers is None:
            self._source_receivers = observed
        elif set(observed) - set(self._source_receivers):
            raise ValueError("fit may use source receivers only; heldout receiver was provided")

        means: dict[int, Tensor] = {}
        bases: dict[int, Tensor] = {}
        for receiver in self._source_receivers:
            mask = ids == receiver
            if not bool(mask.any()):
                raise ValueError(f"source receiver {receiver} has no source features")
            receiver_features = features[mask]
            mean = receiver_features.mean(dim=0)
            centered = receiver_features - mean
            available_rank = min(self.rank, centered.size(0), centered.size(1))
            _, _, vh = safe_svd(centered, rank=available_rank, eps=self.eps)
            if available_rank < self.rank:
                padding = torch.zeros(
                    self.rank - available_rank,
                    self.feature_dim,
                    dtype=vh.dtype,
                    device=vh.device,
                )
                vh = torch.cat((vh, padding), dim=0)
            if not torch.isfinite(vh).all():
                raise ValueError("receiver tangent basis must be finite")
            means[receiver] = mean.detach().clone()
            bases[receiver] = vh.detach().clone()

        self._means = means
        self._bases = bases
        self._fitted = True
        return self

    def mean_for(self, receiver_id: int | Tensor) -> Tensor:
        resolved = self._receiver_scalar(receiver_id)
        self._check_source_receiver(resolved)
        return self._means[resolved].detach().clone()

    def basis_for(self, receiver_id: int | Tensor) -> Tensor:
        resolved = self._receiver_scalar(receiver_id)
        self._check_source_receiver(resolved)
        return self._bases[resolved].detach().clone()

    def coefficients(self, features: Tensor, receiver_ids: Tensor) -> Tensor:
        """Project features onto the factual source-receiver tangent basis."""

        _validate_features(features)
        if features.size(1) != self.feature_dim:
            raise ValueError("features feature dimension must match feature_dim")
        ids = _validate_receiver_ids(
            receiver_ids,
            batch_size=features.size(0),
            device=features.device,
        )
        bases = []
        means = []
        for receiver in ids.tolist():
            self._check_source_receiver(int(receiver))
            bases.append(self._bases[int(receiver)].to(device=features.device, dtype=features.dtype))
            means.append(self._means[int(receiver)].to(device=features.device, dtype=features.dtype))
        basis = torch.stack(bases, dim=0)
        mean = torch.stack(means, dim=0)
        return torch.einsum("bd,brd->br", features - mean, basis)

    def factual_tangent(
        self,
        features: Tensor,
        receiver_ids: Tensor,
        coefficients: Tensor | None = None,
        *,
        scale: float = 1.0,
    ) -> Tensor:
        """Apply the fitted factual receiver tangent perturbation."""

        _validate_features(features)
        if features.size(1) != self.feature_dim:
            raise ValueError("features feature dimension must match feature_dim")
        ids = _validate_receiver_ids(
            receiver_ids,
            batch_size=features.size(0),
            device=features.device,
        )
        resolved_scale = _validate_scalar("scale", scale, non_negative=True)
        if coefficients is None:
            coefficients = torch.zeros(
                features.size(0), self.rank, dtype=features.dtype, device=features.device
            )
        if not torch.is_tensor(coefficients) or not coefficients.is_floating_point():
            raise ValueError("coefficients must be a floating-point tensor")
        if coefficients.ndim == 1:
            if coefficients.numel() != self.rank:
                raise ValueError("coefficients must match tangent rank")
            coefficients = coefficients.unsqueeze(0).expand(features.size(0), -1)
        if coefficients.shape != (features.size(0), self.rank):
            raise ValueError("coefficients must have shape [batch, rank]")
        if not torch.isfinite(coefficients).all():
            raise ValueError("coefficients must contain only finite values")
        bases = []
        for receiver in ids.tolist():
            self._check_source_receiver(int(receiver))
            bases.append(self._bases[int(receiver)].to(device=features.device, dtype=features.dtype))
        basis = torch.stack(bases, dim=0)
        delta = torch.einsum("br,brd->bd", coefficients, basis)
        result = features + resolved_scale * delta
        if not torch.isfinite(result).all():
            raise ValueError("factual tangent result must be finite")
        return result


def factual_tangent(
    bank: ReceiverTangentBank,
    features: Tensor,
    receiver_ids: Tensor,
    coefficients: Tensor | None = None,
    *,
    scale: float = 1.0,
) -> Tensor:
    """Functional F1 factual tangent application."""

    if not isinstance(bank, ReceiverTangentBank):
        raise ValueError("bank must be a ReceiverTangentBank")
    return bank.factual_tangent(
        features,
        receiver_ids,
        coefficients,
        scale=scale,
    )


def one_step_tangent_worst_direction(
    loss: Tensor,
    coefficients: Tensor,
    *,
    radius: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Return one detached normalized coefficient-gradient ascent direction.

    F2 deliberately performs one ``autograd.grad`` call and does not update
    the coefficient tensor or any bank state.
    """

    if not torch.is_tensor(loss) or loss.numel() != 1 or not loss.is_floating_point():
        raise ValueError("loss must be a scalar floating-point tensor")
    if not torch.isfinite(loss).all():
        raise ValueError("loss must contain only finite values")
    if not torch.is_tensor(coefficients) or not coefficients.is_floating_point():
        raise ValueError("coefficients must be a floating-point tensor")
    if coefficients.numel() == 0:
        raise ValueError("coefficients must be non-empty")
    if not torch.isfinite(coefficients).all():
        raise ValueError("coefficients must contain only finite values")
    resolved_radius = _validate_scalar("radius", radius, non_negative=True)
    resolved_eps = _validate_scalar("eps", eps, non_negative=True)
    if resolved_eps <= 0.0:
        raise ValueError("eps must be a finite positive number")
    if not loss.requires_grad:
        raise ValueError("loss must require gradients with respect to coefficients")

    gradient = torch.autograd.grad(
        loss,
        coefficients,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if gradient is None:
        gradient = torch.zeros_like(coefficients)
    if not torch.isfinite(gradient).all():
        raise ValueError("coefficient gradient must be finite")
    norm = torch.linalg.vector_norm(gradient.reshape(-1))
    if not torch.isfinite(norm):
        raise ValueError("coefficient gradient norm must be finite")
    if float(norm.detach().item()) <= resolved_eps:
        return torch.zeros_like(coefficients).detach()
    return (resolved_radius * gradient / norm.clamp_min(resolved_eps)).detach()


def tangent_worst_direction(
    coefficients: Tensor,
    loss: Tensor,
    *,
    radius: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Argument-order-compatible alias for the F2 one-step direction."""

    return one_step_tangent_worst_direction(
        loss,
        coefficients,
        radius=radius,
        eps=eps,
    )


f1_factual_tangent = factual_tangent
f2_one_step_tangent_worst_direction = one_step_tangent_worst_direction
f2_worst_direction = one_step_tangent_worst_direction


__all__ = [
    "ReceiverTangentBank",
    "f1_factual_tangent",
    "f2_one_step_tangent_worst_direction",
    "f2_worst_direction",
    "factual_tangent",
    "one_step_tangent_worst_direction",
    "tangent_worst_direction",
]
