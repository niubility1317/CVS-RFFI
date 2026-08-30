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
    """Accumulate source class/receiver centers and derive tangent bases.

    Geometry is built only from the offsets of observed class-conditioned
    centers ``mu[y,r]`` from the same class center across source receivers.
    Raw within-cell feature scatter is never an SVD input.  Count shrinkage
    contracts sparse cells toward their class center before a second
    receiver-level shrinkage and finite-safe SVD.
    """

    def __init__(
        self,
        feature_dim: int,
        rank: int = 4,
        *,
        source_receivers: Iterable[int] | None = None,
        shrinkage: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        self.feature_dim = _positive_integer("feature_dim", feature_dim)
        self.rank = _positive_integer("rank", rank)
        self.shrinkage = _validate_scalar("shrinkage", shrinkage, non_negative=True)
        if self.shrinkage <= 0.0:
            raise ValueError("shrinkage must be a finite positive number")
        self.eps = _validate_scalar("eps", eps, non_negative=True)
        if self.eps <= 0.0:
            raise ValueError("eps must be a finite positive number")

        self._source_receivers: tuple[int, ...] | None = None
        if source_receivers is not None:
            values_list: list[int] = []
            for value in source_receivers:
                if isinstance(value, bool):
                    raise ValueError("source_receivers must contain integer IDs")
                try:
                    values_list.append(int(operator.index(value)))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("source_receivers must contain integer IDs") from exc
            values = tuple(values_list)
            if not values or len(set(values)) != len(values) or min(values) < 0:
                raise ValueError("source_receivers must be unique non-negative IDs")
            self._source_receivers = tuple(sorted(values))
        self._cell_sums: dict[tuple[int, int], Tensor] = {}
        self._cell_counts: dict[tuple[int, int], int] = {}
        self._centers: dict[tuple[int, int], Tensor] = {}
        self._means: dict[int, Tensor] = {}
        self._bases: dict[int, Tensor] = {}

    @property
    def receiver_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._bases))

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(sorted({class_id for class_id, _ in self._centers}))

    @property
    def centers(self) -> dict[tuple[int, int], Tensor]:
        return {key: value.detach().clone() for key, value in self._centers.items()}

    @property
    def counts(self) -> dict[tuple[int, int], int]:
        return dict(self._cell_counts)

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
                f"receiver {receiver_id} is not an observed source receiver; heldout receivers are forbidden"
            )

    def _build_geometry(
        self,
        cell_sums: dict[tuple[int, int], Tensor],
        cell_counts: dict[tuple[int, int], int],
    ) -> tuple[
        dict[tuple[int, int], Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
    ]:
        centers = {
            key: (cell_sums[key] / float(cell_counts[key])).detach()
            for key in cell_sums
        }
        if any(not torch.isfinite(center).all() for center in centers.values()):
            raise ValueError("class-conditioned centers must remain finite")

        observed_receivers = tuple(sorted({receiver for _, receiver in centers}))
        receiver_means: dict[int, Tensor] = {}
        for receiver in observed_receivers:
            keys = [key for key in centers if key[1] == receiver]
            total_count = sum(cell_counts[key] for key in keys)
            receiver_means[receiver] = (
                sum(
                    (centers[key] * float(cell_counts[key]) for key in keys),
                    torch.zeros_like(centers[keys[0]]),
                )
                / float(total_count)
            ).detach()

        offsets_by_receiver: dict[int, list[Tensor]] = {
            receiver: [] for receiver in observed_receivers
        }
        for class_id in sorted({class_id for class_id, _ in centers}):
            class_keys = sorted(key for key in centers if key[0] == class_id)
            if len(class_keys) < 2:
                continue
            class_center = torch.stack([centers[key] for key in class_keys]).mean(dim=0)
            for key in class_keys:
                count = float(cell_counts[key])
                cell_reliability = count / (count + self.shrinkage)
                offset = cell_reliability * (centers[key] - class_center)
                offsets_by_receiver[key[1]].append(offset)

        bases: dict[int, Tensor] = {}
        for receiver in observed_receivers:
            offsets = offsets_by_receiver[receiver]
            if not offsets:
                bases[receiver] = torch.zeros(
                    self.rank,
                    self.feature_dim,
                    dtype=receiver_means[receiver].dtype,
                    device=receiver_means[receiver].device,
                )
                continue
            matrix = torch.stack(offsets)
            receiver_reliability = len(offsets) / (len(offsets) + self.shrinkage)
            matrix = receiver_reliability * matrix
            if not torch.isfinite(matrix).all():
                raise ValueError("hierarchically shrunk tangent offsets must remain finite")
            if float(torch.linalg.vector_norm(matrix).item()) <= self.eps:
                basis = torch.zeros(
                    self.rank,
                    self.feature_dim,
                    dtype=matrix.dtype,
                    device=matrix.device,
                )
            else:
                available_rank = min(self.rank, matrix.size(0), matrix.size(1))
                _, _, vh = safe_svd(matrix, rank=available_rank, eps=self.eps)
                if available_rank < self.rank:
                    vh = torch.cat(
                        (
                            vh,
                            torch.zeros(
                                self.rank - available_rank,
                                self.feature_dim,
                                dtype=vh.dtype,
                                device=vh.device,
                            ),
                        ),
                        dim=0,
                    )
                basis = vh
            if not torch.isfinite(basis).all():
                raise ValueError("receiver tangent basis must remain finite")
            bases[receiver] = basis.detach()
        return centers, receiver_means, bases

    def update(self, z: Tensor, tx: Tensor, rx: Tensor) -> "ReceiverTangentBank":
        """Accumulate detached source ``mu[tx,rx]`` sufficient statistics."""

        _validate_features(z, "z")
        if z.size(1) != self.feature_dim:
            raise ValueError("z feature dimension must match feature_dim")
        tx_ids = _validate_receiver_ids(
            tx,
            batch_size=z.size(0),
            device=z.device,
            name="tx",
        )
        rx_ids = _validate_receiver_ids(
            rx,
            batch_size=z.size(0),
            device=z.device,
            name="rx",
        )
        if int(tx_ids.min().item()) < 0 or int(rx_ids.min().item()) < 0:
            raise ValueError("tx and rx IDs must be non-negative")
        observed = tuple(int(value) for value in torch.unique(rx_ids, sorted=True).tolist())
        if self._source_receivers is None:
            self._source_receivers = observed
        elif set(observed) - set(self._source_receivers):
            raise ValueError("update may use source receivers only; heldout receiver was provided")

        new_sums = {key: value.detach().clone() for key, value in self._cell_sums.items()}
        new_counts = dict(self._cell_counts)
        detached_z = z.detach()
        pairs = torch.stack((tx_ids, rx_ids), dim=1)
        for pair in torch.unique(pairs, dim=0, sorted=True):
            class_id, receiver = (int(pair[0].item()), int(pair[1].item()))
            mask = (tx_ids == class_id) & (rx_ids == receiver)
            increment = detached_z[mask].sum(dim=0)
            if not torch.isfinite(increment).all():
                raise ValueError("class-conditioned sums must remain finite")
            key = (class_id, receiver)
            combined = increment if key not in new_sums else new_sums[key] + increment
            if not torch.isfinite(combined).all():
                raise ValueError("class-conditioned sums must remain finite")
            new_sums[key] = combined.detach()
            new_counts[key] = new_counts.get(key, 0) + int(mask.sum().item())

        centers, means, bases = self._build_geometry(new_sums, new_counts)
        self._cell_sums = new_sums
        self._cell_counts = new_counts
        self._centers = centers
        self._means = means
        self._bases = bases
        return self

    def reset(self) -> None:
        self._cell_sums.clear()
        self._cell_counts.clear()
        self._centers.clear()
        self._means.clear()
        self._bases.clear()

    def fit(self, z: Tensor, tx: Tensor, rx: Tensor) -> "ReceiverTangentBank":
        """Reset the bank and perform one source-only ``update``."""

        self.reset()
        return self.update(z, tx, rx)

    def center_for(self, class_id: int, receiver_id: int) -> Tensor:
        class_value = self._receiver_scalar(class_id)
        receiver_value = self._receiver_scalar(receiver_id)
        key = (class_value, receiver_value)
        if key not in self._centers:
            raise ValueError("class/receiver center was not observed in source data")
        return self._centers[key].detach().clone()

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
            coefficients = self.coefficients(features, ids)
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
