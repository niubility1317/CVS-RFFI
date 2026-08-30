"""Sparse cross-receiver donor exchange for BiCAD-XR.

The implementation deliberately operates on row masks rather than assuming a
rectangular episode.  A sampler's physical IDs are provenance only; feature
rows are always addressed by their local row position.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


def _as_integer_labels(value: Any, name: str, *, device: torch.device) -> Tensor:
    try:
        labels = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must contain integer labels") from exc
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if labels.dtype == torch.bool or labels.is_complex():
        raise ValueError(f"{name} must contain integer labels")
    if labels.is_floating_point():
        if not bool(torch.isfinite(labels).all()):
            raise ValueError(f"{name} must contain finite integer labels")
        converted = labels.to(dtype=torch.long)
        if not torch.equal(labels, converted.to(dtype=labels.dtype)):
            raise ValueError(f"{name} must contain integer labels")
    else:
        try:
            converted = labels.to(dtype=torch.long)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must contain integer labels") from exc
    return converted


def _validate_features(value: Tensor, name: str) -> None:
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, feature]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if value.size(1) < 1:
        raise ValueError(f"{name} must have a non-empty feature dimension")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _connected_zero_from_finite(value: Tensor) -> Tensor:
    """Return a finite zero connected to at most one validated input element."""

    if not bool(torch.isfinite(value).all()):
        raise ValueError("z_id must contain only finite values")
    return value.reshape(-1)[:1].sum() * 0.0


def _validate_scalar_float(value: Any, name: str, *, lower: float, strict: bool) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(resolved) or (resolved <= lower if strict else resolved < lower):
        comparator = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {comparator} {lower}")
    return resolved


def _resolve_num_classes(tx: Tensor, requested: int | None) -> int:
    if requested is None:
        if tx.numel() == 0:
            raise ValueError("num_classes is required for an empty batch")
        value = int(tx.max().item()) + 1
    else:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
            raise ValueError("num_classes must be a positive integer")
        value = requested
    if tx.numel() and (int(tx.min().item()) < 0 or int(tx.max().item()) >= value):
        raise ValueError("TX labels are out of range for num_classes")
    return value


def _resolve_num_receivers(receiver: Tensor, requested: int | None) -> int:
    if receiver.numel() and int(receiver.min().item()) < 0:
        raise ValueError("receiver labels must be non-negative")
    if requested is None:
        return int(receiver.max().item()) + 1 if receiver.numel() else 0
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ValueError("num_receivers must be a non-negative integer")
    if receiver.numel() and int(receiver.max().item()) >= requested:
        raise ValueError("receiver labels are out of range for num_receivers")
    return requested


def _validate_physical_indices(value: Any | None, expected_length: int) -> None:
    """Validate provenance IDs without ever using them to index feature rows."""

    if value is None:
        return
    try:
        indices = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("physical_indices must be a one-dimensional integer sequence") from exc
    if indices.ndim != 1 or indices.numel() != expected_length:
        raise ValueError("physical_indices must match the feature row count")
    if indices.dtype == torch.bool or indices.is_complex():
        raise ValueError("physical_indices must contain integer IDs")
    if indices.is_floating_point():
        if not bool(torch.isfinite(indices).all()):
            raise ValueError("physical_indices must contain finite integer IDs")
        converted = indices.to(dtype=torch.long)
        if not torch.equal(indices, converted.to(dtype=indices.dtype)):
            raise ValueError("physical_indices must contain integer IDs")
        indices = converted
    else:
        indices = indices.to(dtype=torch.long)
    if indices.numel() and int(indices.min().item()) < 0:
        raise ValueError("physical_indices must be non-negative")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("physical_indices must be unique")


def _solve_dtype(value: Tensor) -> torch.dtype:
    if value.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return value.dtype


def _classification_margins(logits: Tensor, labels: Tensor) -> Tensor:
    true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    one_hot = F.one_hot(labels, num_classes=logits.size(1)).to(dtype=torch.bool)
    other_logits = logits.masked_fill(one_hot, torch.finfo(logits.dtype).min)
    return true_logits - other_logits.max(dim=1).values


@dataclass(frozen=True)
class DonorBank:
    """Per-receiver ridge donors and detached support-quality diagnostics."""

    weights: Tensor
    quality: Tensor
    valid_receivers: Tensor
    condition_numbers: Tensor
    support_accuracy: Tensor
    support_margin: Tensor
    support_counts: Tensor
    skip_reasons: tuple[str, ...]
    num_classes: int

    @property
    def condition_number(self) -> Tensor:
        """Singular alias for callers that report one condition per donor."""

        return self.condition_numbers

    @property
    def quality_weights(self) -> Tensor:
        return self.quality

    @property
    def support_accuracies(self) -> Tensor:
        return self.support_accuracy

    @property
    def support_margins(self) -> Tensor:
        return self.support_margin

    @property
    def skip_reason(self) -> tuple[str, ...]:
        return self.skip_reasons

    @property
    def detached_weights(self) -> Tensor:
        return self.weights


@dataclass(frozen=True)
class XDCLossOutput:
    """XDC losses, detached teacher ensemble and audit information."""

    total: Tensor
    xdc_cross_entropy: Tensor
    knowledge_distillation: Tensor
    ensemble_logits: Tensor
    donor_query_matrix: Tensor
    detached_donor_weights: tuple[Tensor, ...]
    donor_bank: DonorBank
    skip_reason: str | None

    @property
    def cross_entropy(self) -> Tensor:
        return self.xdc_cross_entropy

    @property
    def donor_query_loss(self) -> Tensor:
        return self.xdc_cross_entropy

    @property
    def kd_loss(self) -> Tensor:
        return self.knowledge_distillation

    @property
    def kd(self) -> Tensor:
        return self.knowledge_distillation


def fit_receiver_donors(
    z_id: Tensor,
    tx: Tensor,
    receiver: Tensor,
    num_classes: int | None = None,
    ridge: float = 1e-2,
    *,
    min_support_accuracy: float = 0.25,
    num_receivers: int | None = None,
    max_condition_number: float | None = None,
    physical_indices: Sequence[int] | Tensor | None = None,
) -> DonorBank:
    """Fit one detached ridge classifier per receiver using sparse row masks.

    A receiver is a donor only when it has at least two observed classes, a
    finite regularized condition number, finite solved weights and support
    accuracy at or above ``min_support_accuracy``.  The support quality is
    shared by all exchange paths and is
    ``accuracy * clamp(mean_margin, min=0) / log1p(condition_number)``.
    """

    _validate_features(z_id, "z_id")
    tx = _as_integer_labels(tx, "tx", device=z_id.device)
    receiver = _as_integer_labels(receiver, "receiver", device=z_id.device)
    if tx.numel() != z_id.size(0) or receiver.numel() != z_id.size(0):
        raise ValueError("z_id, tx, and receiver must have the same batch size")
    resolved_classes = _resolve_num_classes(tx, num_classes)
    resolved_receivers = _resolve_num_receivers(receiver, num_receivers)
    ridge = _validate_scalar_float(ridge, "ridge", lower=0.0, strict=True)
    min_support_accuracy = _validate_scalar_float(
        min_support_accuracy, "min_support_accuracy", lower=0.0, strict=False
    )
    if min_support_accuracy > 1.0:
        raise ValueError("min_support_accuracy must be in [0,1]")
    if max_condition_number is not None:
        max_condition_number = _validate_scalar_float(
            max_condition_number, "max_condition_number", lower=0.0, strict=True
        )
    _validate_physical_indices(physical_indices, z_id.size(0))

    metric_dtype = torch.float64 if z_id.dtype == torch.float64 else torch.float32
    weights = torch.zeros(
        (resolved_receivers, z_id.size(1), resolved_classes),
        dtype=z_id.dtype,
        device=z_id.device,
    )
    condition_numbers = torch.full(
        (resolved_receivers,), float("nan"), dtype=metric_dtype, device=z_id.device
    )
    support_accuracy = torch.full(
        (resolved_receivers,), float("nan"), dtype=metric_dtype, device=z_id.device
    )
    support_margin = torch.full(
        (resolved_receivers,), float("nan"), dtype=metric_dtype, device=z_id.device
    )
    support_counts = torch.zeros(
        (resolved_receivers,), dtype=torch.long, device=z_id.device
    )
    skip_reasons = ["no_support"] * resolved_receivers
    valid_ids: list[int] = []

    for receiver_id in range(resolved_receivers):
        row_mask = receiver == receiver_id
        row_count = int(row_mask.sum().item())
        support_counts[receiver_id] = row_count
        if row_count == 0:
            continue
        z_support = z_id[row_mask]
        y_support = tx[row_mask]
        if torch.unique(y_support).numel() < 2:
            skip_reasons[receiver_id] = "coverage<2_classes"
            continue
        if not bool(torch.isfinite(z_support).all()):
            skip_reasons[receiver_id] = "nonfinite_condition"
            continue

        solve_dtype = _solve_dtype(z_support)
        solve_features = z_support.to(dtype=solve_dtype)
        gram = solve_features @ solve_features.transpose(0, 1)
        regularized = gram + ridge * torch.eye(
            row_count, dtype=solve_dtype, device=z_id.device
        )
        try:
            condition = torch.linalg.cond(regularized)
        except (RuntimeError, ValueError, torch.linalg.LinAlgError):
            skip_reasons[receiver_id] = "condition_failed"
            continue
        if not bool(torch.isfinite(condition).all()):
            skip_reasons[receiver_id] = "nonfinite_condition"
            continue
        condition_value = float(condition.detach().cpu().item())
        condition_numbers[receiver_id] = condition.to(dtype=metric_dtype)
        if max_condition_number is not None and condition_value > max_condition_number:
            skip_reasons[receiver_id] = "condition_above_threshold"
            continue

        one_hot = F.one_hot(y_support, num_classes=resolved_classes).to(
            dtype=solve_dtype
        )
        try:
            alpha = torch.linalg.solve(regularized, one_hot)
        except (RuntimeError, ValueError, torch.linalg.LinAlgError):
            skip_reasons[receiver_id] = "ridge_solve_failed"
            continue
        donor_weight = (solve_features.transpose(0, 1) @ alpha).to(dtype=z_id.dtype)
        if not bool(torch.isfinite(donor_weight).all()):
            skip_reasons[receiver_id] = "nonfinite_weights"
            continue

        support_logits = solve_features @ donor_weight.to(dtype=solve_dtype)
        margins = _classification_margins(support_logits, y_support)
        accuracy = (support_logits.argmax(dim=1) == y_support).to(metric_dtype).mean()
        margin = margins.to(metric_dtype).mean()
        if not bool(torch.isfinite(accuracy).all() and torch.isfinite(margin).all()):
            skip_reasons[receiver_id] = "nonfinite_support_metrics"
            continue
        support_accuracy[receiver_id] = accuracy
        support_margin[receiver_id] = margin
        if float(accuracy.detach().cpu().item()) < min_support_accuracy:
            skip_reasons[receiver_id] = "support_accuracy<min_support_accuracy"
            continue

        weights[receiver_id] = donor_weight.detach()
        condition_numbers[receiver_id] = condition.to(dtype=metric_dtype).detach()
        valid_ids.append(receiver_id)
        skip_reasons[receiver_id] = ""

    # Quality is initialized separately so the per-receiver loop can fail
    # closed without ever retaining an autograd edge.
    quality = torch.full(
        (resolved_receivers,), float("nan"), dtype=metric_dtype, device=z_id.device
    )
    for receiver_id in valid_ids:
        quality[receiver_id] = (
            support_accuracy[receiver_id]
            * support_margin[receiver_id].clamp_min(0.0)
            / torch.log1p(condition_numbers[receiver_id]).clamp_min(
                torch.finfo(metric_dtype).tiny
            )
        ).detach()

    return DonorBank(
        weights=weights.detach(),
        quality=quality.detach(),
        valid_receivers=torch.tensor(valid_ids, dtype=torch.long, device=z_id.device),
        condition_numbers=condition_numbers.detach(),
        support_accuracy=support_accuracy.detach(),
        support_margin=support_margin.detach(),
        support_counts=support_counts.detach(),
        skip_reasons=tuple(skip_reasons),
        num_classes=resolved_classes,
    )


def _resolve_bank_and_inputs(
    z_id: Tensor,
    tx: Tensor,
    receiver: Tensor,
    bank: DonorBank | None,
    *,
    num_classes: int | None,
    ridge: float,
    min_support_accuracy: float,
    num_receivers: int | None,
    max_condition_number: float | None,
    physical_indices: Sequence[int] | Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, DonorBank]:
    _validate_features(z_id, "z_id")
    tx = _as_integer_labels(tx, "tx", device=z_id.device)
    receiver = _as_integer_labels(receiver, "receiver", device=z_id.device)
    if tx.numel() != z_id.size(0) or receiver.numel() != z_id.size(0):
        raise ValueError("z_id, tx, and receiver must have the same batch size")
    if receiver.numel() and int(receiver.min().item()) < 0:
        raise ValueError("receiver labels must be non-negative")
    resolved_classes = _resolve_num_classes(tx, num_classes)
    _validate_physical_indices(physical_indices, z_id.size(0))
    if bank is None:
        bank = fit_receiver_donors(
            z_id,
            tx,
            receiver,
            num_classes=resolved_classes,
            ridge=ridge,
            min_support_accuracy=min_support_accuracy,
            num_receivers=num_receivers,
            max_condition_number=max_condition_number,
            physical_indices=physical_indices,
        )
    else:
        if not isinstance(bank, DonorBank):
            raise ValueError("bank must be a DonorBank or None")
        if bank.num_classes != resolved_classes:
            raise ValueError("bank and input num_classes must match")
        if bank.weights.ndim != 3 or bank.weights.size(1) != z_id.size(1):
            raise ValueError("bank weights must match the feature dimension")
        if bank.weights.device != z_id.device:
            raise ValueError("bank and features must use the same device")
        if receiver.numel() and int(receiver.max().item()) >= bank.weights.size(0):
            raise ValueError("receiver labels are out of range for bank")
    return z_id, tx, receiver, bank


def _matrix_with_bank(
    z_id: Tensor,
    tx: Tensor,
    receiver: Tensor,
    bank: DonorBank,
    *,
    metric: str,
) -> Tensor:
    metric_name = str(metric).strip().lower()
    if metric_name in {"acc", "accuracy"}:
        metric_name = "accuracy"
    elif metric_name in {"ce", "cross_entropy", "loss"}:
        metric_name = "cross_entropy"
    elif metric_name == "margin":
        metric_name = "margin"
    else:
        raise ValueError("metric must be accuracy, cross_entropy, or margin")

    matrix = torch.full(
        (bank.weights.size(0), bank.weights.size(0)),
        float("nan"),
        dtype=torch.float64 if z_id.dtype == torch.float64 else torch.float32,
        device=z_id.device,
    )
    valid_ids = [int(value) for value in bank.valid_receivers.detach().cpu().tolist()]
    for donor_id in valid_ids:
        donor_weight = bank.weights[donor_id].detach()
        for query_id in range(bank.weights.size(0)):
            if query_id == donor_id:
                continue
            query_mask = receiver == query_id
            if not bool(query_mask.any()):
                continue
            z_query = z_id[query_mask]
            y_query = tx[query_mask]
            if not bool(torch.isfinite(z_query).all()):
                continue
            logits = (z_query @ donor_weight).to(dtype=matrix.dtype)
            if metric_name == "accuracy":
                value = (logits.argmax(dim=1) == y_query).to(matrix.dtype).mean()
            elif metric_name == "cross_entropy":
                value = F.cross_entropy(logits, y_query).to(matrix.dtype)
            else:
                value = _classification_margins(logits, y_query).to(matrix.dtype).mean()
            if bool(torch.isfinite(value).all()):
                matrix[donor_id, query_id] = value.detach()
    return matrix.detach()


def donor_query_matrix(
    z_id: Tensor | DonorBank,
    tx: Tensor,
    receiver: Tensor,
    bank: DonorBank | None = None,
    *,
    num_classes: int | None = None,
    ridge: float = 1e-2,
    min_support_accuracy: float = 0.25,
    num_receivers: int | None = None,
    max_condition_number: float | None = None,
    metric: str = "accuracy",
    physical_indices: Sequence[int] | Tensor | None = None,
) -> Tensor:
    """Return donor-row/query-column metrics with unevaluated cells as NaN.

    The normal call is ``donor_query_matrix(z_id, tx, receiver, ...)``.  For
    callers that already fitted a bank, ``donor_query_matrix(z_id, tx,
    receiver, bank)`` is accepted; the bank-first positional form
    ``donor_query_matrix(bank, z_id, tx, receiver)`` is also supported.
    """

    if isinstance(z_id, DonorBank):
        resolved_bank = z_id
        actual_z_id = tx
        actual_tx = receiver
        actual_receiver = bank
        if actual_receiver is None:
            raise ValueError("bank-first form requires receiver metadata")
        z_id = actual_z_id
        tx = actual_tx
        receiver = actual_receiver
        bank = resolved_bank
    if not torch.is_tensor(z_id):
        raise ValueError("z_id must be a tensor")
    z_id, tx, receiver, bank = _resolve_bank_and_inputs(
        z_id,
        tx,
        receiver,
        bank,
        num_classes=num_classes,
        ridge=ridge,
        min_support_accuracy=min_support_accuracy,
        num_receivers=num_receivers,
        max_condition_number=max_condition_number,
        physical_indices=physical_indices,
    )
    return _matrix_with_bank(z_id, tx, receiver, bank, metric=metric)


def xdc_losses(
    z_id: Tensor,
    tx: Tensor,
    receiver: Tensor,
    public_logits: Tensor,
    num_classes: int | None = None,
    temperature: float = 2.0,
    ridge: float = 1e-2,
    *,
    min_support_accuracy: float = 0.25,
    num_receivers: int | None = None,
    max_condition_number: float | None = None,
    kd_weight: float = 1.0,
    bank: DonorBank | None = None,
    physical_indices: Sequence[int] | Tensor | None = None,
) -> XDCLossOutput:
    """Compute sparse cross-receiver CE and temperature-scaled XDC KD.

    Each query receiver is evaluated only against valid donors from other
    receivers.  The donor ensemble is detached before KD, while query feature
    rows remain connected to the XDC cross-entropy.  If no cross-receiver pair
    is available, every returned loss is a differentiable zero connected to
    ``z_id`` and the reason is recorded.
    """

    _validate_features(z_id, "z_id")
    if not torch.is_tensor(public_logits) or public_logits.ndim != 2:
        raise ValueError("public_logits must have shape [batch, classes]")
    if not public_logits.is_floating_point():
        raise ValueError("public_logits must use a floating-point dtype")
    if public_logits.size(0) != z_id.size(0):
        raise ValueError("public_logits and z_id must have the same batch size")
    if not bool(torch.isfinite(public_logits).all()):
        raise ValueError("public_logits must contain only finite values")
    if num_classes is None:
        num_classes = int(public_logits.size(1))
    if public_logits.size(1) != num_classes:
        raise ValueError("public_logits class dimension must match num_classes")
    temperature = _validate_scalar_float(temperature, "temperature", lower=0.0, strict=True)
    kd_weight = _validate_scalar_float(kd_weight, "kd_weight", lower=0.0, strict=False)
    z_id, tx, receiver, bank = _resolve_bank_and_inputs(
        z_id,
        tx,
        receiver,
        bank,
        num_classes=num_classes,
        ridge=ridge,
        min_support_accuracy=min_support_accuracy,
        num_receivers=num_receivers,
        max_condition_number=max_condition_number,
        physical_indices=physical_indices,
    )

    ensemble_logits = torch.zeros_like(public_logits).detach()
    valid_ids = [int(value) for value in bank.valid_receivers.detach().cpu().tolist()]
    evaluated_donors: set[int] = set()
    ce_terms: list[Tensor] = []
    kd_terms: list[Tensor] = []
    evaluated_samples = 0
    skipped_query_reasons: list[str] = []

    for query_id in range(bank.weights.size(0)):
        query_mask = receiver == query_id
        if not bool(query_mask.any()):
            continue
        donor_ids = [donor_id for donor_id in valid_ids if donor_id != query_id]
        if not donor_ids:
            skipped_query_reasons.append("no_valid_cross_receiver_donor")
            continue
        z_query = z_id[query_mask]
        y_query = tx[query_mask]
        if not bool(torch.isfinite(z_query).all()):
            skipped_query_reasons.append("nonfinite_query_features")
            continue

        donor_quality = bank.quality[donor_ids].detach().clamp_min(0.0)
        if not bool(torch.isfinite(donor_quality).all()):
            skipped_query_reasons.append("nonfinite_donor_quality")
            continue
        quality_sum = donor_quality.sum()
        if float(quality_sum.detach().cpu().item()) <= 0.0:
            skipped_query_reasons.append("no_positive_quality_donor")
            continue
        normalized_quality = donor_quality / quality_sum
        donor_logits = torch.stack(
            [z_query @ bank.weights[donor_id].detach() for donor_id in donor_ids], dim=1
        )
        query_ensemble = (donor_logits * normalized_quality.view(1, -1, 1)).sum(dim=1)
        query_ensemble = query_ensemble.to(dtype=public_logits.dtype)
        if not bool(torch.isfinite(query_ensemble).all()):
            skipped_query_reasons.append("nonfinite_ensemble")
            continue

        query_positions = torch.nonzero(query_mask, as_tuple=False).squeeze(1)
        teacher = query_ensemble.detach()
        ensemble_logits[query_positions] = teacher
        ce_terms.append(F.cross_entropy(query_ensemble, y_query) * y_query.numel())
        public_query = public_logits[query_mask]
        kd = F.kl_div(
            F.log_softmax(public_query / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction="batchmean",
        ) * temperature**2
        kd_terms.append(kd * y_query.numel())
        evaluated_samples += y_query.numel()
        evaluated_donors.update(donor_ids)

    if evaluated_samples == 0:
        zero = _connected_zero_from_finite(z_id)
        if not valid_ids or skipped_query_reasons:
            skip_reason = (
                "no_valid_cross_receiver_donor"
                if any(reason == "no_valid_cross_receiver_donor" for reason in skipped_query_reasons)
                or not valid_ids
                else skipped_query_reasons[0]
            )
        else:
            skip_reason = "no_query_rows"
        return XDCLossOutput(
            total=zero,
            xdc_cross_entropy=zero,
            knowledge_distillation=zero,
            ensemble_logits=ensemble_logits.detach(),
            donor_query_matrix=_matrix_with_bank(
                z_id, tx, receiver, bank, metric="accuracy"
            ),
            detached_donor_weights=tuple(
                bank.weights[donor_id].detach() for donor_id in valid_ids
            ),
            donor_bank=bank,
            skip_reason=skip_reason,
        )

    xdc_cross_entropy = torch.stack(ce_terms).sum() / evaluated_samples
    knowledge_distillation = torch.stack(kd_terms).sum() / evaluated_samples
    total = xdc_cross_entropy + kd_weight * knowledge_distillation
    skip_reason = None
    if skipped_query_reasons:
        skip_reason = "partial:" + ",".join(sorted(set(skipped_query_reasons)))
    return XDCLossOutput(
        total=total,
        xdc_cross_entropy=xdc_cross_entropy,
        knowledge_distillation=knowledge_distillation,
        ensemble_logits=ensemble_logits.detach(),
        donor_query_matrix=_matrix_with_bank(
            z_id, tx, receiver, bank, metric="accuracy"
        ),
        detached_donor_weights=tuple(
            bank.weights[donor_id].detach() for donor_id in sorted(evaluated_donors)
        ),
        donor_bank=bank,
        skip_reason=skip_reason,
    )


__all__ = [
    "DonorBank",
    "XDCLossOutput",
    "donor_query_matrix",
    "fit_receiver_donors",
    "xdc_losses",
]
