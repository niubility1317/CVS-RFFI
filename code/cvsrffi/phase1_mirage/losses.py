"""Source-only MIRAGE training losses with frozen causal-arm composition.

The public functions accept tensors that the future trainer has already
obtained from role-safe source batches.  This module intentionally has no
dataset, target, truth, quota, or threshold-search access.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
import torch.nn.functional as functional
from torch import Tensor

from .config import ArmConfig, arm_config
from .head import OpenHeadOutput
from .proxy import ProxyEpisode


_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


@dataclass(frozen=True)
class BoundaryMixupBatch:
    """Strictly cross-class, registered-only normalized boundary mixtures."""

    mixed_embeddings: Tensor
    left_indices: Tensor
    right_indices: Tensor
    lambdas: Tensor


def _connected_zero(reference: Tensor) -> Tensor:
    """Return a differentiable scalar zero connected to a valid tensor graph."""

    return reference.reshape(-1).sum() * 0.0


def _require_floating_tensor(value: object, *, name: str, device: torch.device | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if device is not None and value.device != device:
        raise ValueError(f"{name} device must match the other inputs")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _require_integer_vector(
    value: object,
    *,
    name: str,
    length: int | None = None,
    device: torch.device | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if length is not None and value.shape[0] != length:
        raise ValueError(f"{name} must have shape [{length}]")
    if device is not None and value.device != device:
        raise ValueError(f"{name} device must match the other inputs")
    return value


def _require_bool_vector(value: object, *, name: str, length: int, device: torch.device) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool")
    if value.ndim != 1 or value.shape[0] != length:
        raise ValueError(f"{name} must have shape [{length}]")
    if value.device != device:
        raise ValueError(f"{name} device must match the other inputs")
    return value


def _require_scalar_finite(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")
    if not bool(torch.isfinite(value)):
        raise FloatingPointError(f"{name} must be finite")
    return value


def _broadcast_device(*values: object) -> torch.device | None:
    devices = {value.device for value in values if isinstance(value, Tensor)}
    if len(devices) > 1:
        raise ValueError("pseudo gate tensors must share one device")
    return next(iter(devices), None)


def _pseudo_metric_tensor(value: object, *, name: str, device: torch.device | None) -> Tensor:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a floating scalar or tensor")
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value, device=device)
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} device must match the other inputs")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _pseudo_bool_tensor(value: object, *, name: str, device: torch.device | None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value, device=device)
    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} device must match the other inputs")
    return tensor


def pseudo_accept_mask(
    top1: Tensor | float,
    margin: Tensor | float,
    views_agree: Tensor | bool,
    inside_radius: Tensor | bool,
) -> Tensor:
    """Accept an EMA pseudo label only when all four frozen gates hold.

    Scalars and broadcast-compatible tensors are supported.  Malformed,
    non-finite, incompatible, and empty inputs fail closed by raising before a
    sample can enter an identity-label loss.
    """

    device = _broadcast_device(top1, margin, views_agree, inside_radius)
    top1_tensor = _pseudo_metric_tensor(top1, name="top1", device=device)
    margin_tensor = _pseudo_metric_tensor(margin, name="margin", device=device)
    views_tensor = _pseudo_bool_tensor(views_agree, name="views_agree", device=device)
    radius_tensor = _pseudo_bool_tensor(inside_radius, name="inside_radius", device=device)
    try:
        top1_tensor, margin_tensor, views_tensor, radius_tensor = torch.broadcast_tensors(
            top1_tensor,
            margin_tensor,
            views_tensor,
            radius_tensor,
        )
    except RuntimeError as error:
        raise ValueError("pseudo gate inputs must be broadcast-compatible") from error
    if top1_tensor.numel() == 0:
        raise ValueError("pseudo gate inputs must be non-empty")
    return (
        (top1_tensor >= 0.95)
        & (margin_tensor >= 0.20)
        & views_tensor
        & radius_tensor
    )


def _validate_logits(logits: object, *, name: str, min_classes: int = 1) -> Tensor:
    tensor = _require_floating_tensor(logits, name=name)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [B, C]")
    if tensor.shape[0] == 0 or tensor.shape[1] < min_classes:
        raise ValueError(f"{name} must have a non-empty batch and at least {min_classes} classes")
    return tensor


def _validate_labels(labels: object, *, batch_size: int, class_count: int, device: torch.device) -> Tensor:
    tensor = _require_integer_vector(labels, name="supervised_labels", length=batch_size, device=device)
    if bool((tensor < 0).any()) or bool((tensor >= class_count).any()):
        raise ValueError("supervised_labels must index a registered logit row")
    return tensor


def registered_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    registered_row_mask: Tensor | None = None,
) -> Tensor:
    """Compute supervised CE exclusively for the currently registered rows."""

    logits = _validate_logits(logits, name="supervised_logits")
    labels = _validate_labels(
        labels,
        batch_size=logits.shape[0],
        class_count=logits.shape[1],
        device=logits.device,
    )
    if registered_row_mask is None:
        registered_row_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    else:
        registered_row_mask = _require_bool_vector(
            registered_row_mask,
            name="registered_row_mask",
            length=logits.shape[0],
            device=logits.device,
        )
    if not bool(registered_row_mask.any()):
        return _connected_zero(logits)
    return _require_scalar_finite(
        functional.cross_entropy(logits[registered_row_mask], labels[registered_row_mask]),
        name="registered_ce",
    )


def _pseudo_state(
    student_logits: Tensor,
    teacher_logits: Tensor,
    inside_radius: Tensor,
) -> tuple[Tensor, Tensor]:
    student_logits = _validate_logits(student_logits, name="pseudo_student_logits", min_classes=2)
    teacher_logits = _validate_logits(teacher_logits, name="pseudo_teacher_logits", min_classes=2)
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("pseudo student and teacher logits must have the same shape")
    if teacher_logits.device != student_logits.device:
        raise ValueError("pseudo student and teacher logits must share one device")
    inside_radius = _require_bool_vector(
        inside_radius,
        name="pseudo_inside_radius",
        length=student_logits.shape[0],
        device=student_logits.device,
    )
    teacher_probabilities = functional.softmax(teacher_logits.detach(), dim=1)
    _require_floating_tensor(teacher_probabilities, name="teacher_probabilities", device=student_logits.device)
    top_two_values, top_two_labels = teacher_probabilities.topk(2, dim=1)
    teacher_labels = top_two_labels[:, 0]
    views_agree = teacher_labels == student_logits.argmax(dim=1)
    accepted = pseudo_accept_mask(
        top_two_values[:, 0],
        top_two_values[:, 0] - top_two_values[:, 1],
        views_agree,
        inside_radius,
    )
    return teacher_labels, accepted


def ema_pseudo_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    inside_radius: Tensor,
) -> Tensor:
    """EMA pseudo-label CE, returning a graph-connected zero when none pass."""

    teacher_labels, accepted = _pseudo_state(student_logits, teacher_logits, inside_radius)
    if not bool(accepted.any()):
        return _connected_zero(student_logits)
    return _require_scalar_finite(
        functional.cross_entropy(student_logits[accepted], teacher_labels[accepted]),
        name="ema_pseudo",
    )


def weak_strong_consistency_loss(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """Role-blind consistency for every unlabeled weak/strong view pair."""

    student_logits = _validate_logits(student_logits, name="pseudo_student_logits")
    teacher_logits = _validate_logits(teacher_logits, name="pseudo_teacher_logits")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("pseudo student and teacher logits must have the same shape")
    if student_logits.device != teacher_logits.device:
        raise ValueError("pseudo student and teacher logits must share one device")
    value = functional.mse_loss(
        functional.softmax(student_logits, dim=1),
        functional.softmax(teacher_logits.detach(), dim=1),
    )
    return _require_scalar_finite(value, name="weak_strong_consistency")


def masked_latent_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Predict a masked latent target without allowing target-side gradients."""

    prediction = _require_floating_tensor(prediction, name="masked_prediction")
    target = _require_floating_tensor(target, name="masked_target", device=prediction.device)
    if prediction.ndim < 1 or prediction.numel() == 0 or prediction.shape != target.shape:
        raise ValueError("masked prediction and target must have the same non-empty shape")
    return _require_scalar_finite(
        functional.mse_loss(prediction, target.detach()),
        name="masked_latent",
    )


def cross_receiver_consistency_loss(features: Tensor, receiver_ids: Tensor) -> Tensor:
    """Align source receiver means without consulting class IDs or quotas."""

    features = _require_floating_tensor(features, name="cross_receiver_features")
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("cross_receiver_features must have non-empty shape [B, D]")
    receiver_ids = _require_integer_vector(
        receiver_ids,
        name="cross_receiver_ids",
        length=features.shape[0],
        device=features.device,
    )
    unique_receivers = receiver_ids.unique(sorted=True)
    if unique_receivers.numel() < 2:
        return _connected_zero(features)
    receiver_means = torch.stack(
        tuple(features[receiver_ids == receiver].mean(dim=0) for receiver in unique_receivers),
        dim=0,
    )
    value = functional.mse_loss(receiver_means, receiver_means.mean(dim=0, keepdim=True).expand_as(receiver_means))
    return _require_scalar_finite(value, name="cross_receiver")


def prototype_aware_pseudo_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    inside_radius: Tensor,
    radius_margins: Tensor,
) -> Tensor:
    """Weight accepted pseudo CE by the matched prototype/radius margin."""

    teacher_labels, accepted = _pseudo_state(student_logits, teacher_logits, inside_radius)
    radius_margins = _require_floating_tensor(
        radius_margins,
        name="pseudo_radius_margins",
        device=student_logits.device,
    )
    if radius_margins.ndim != 1 or radius_margins.shape[0] != student_logits.shape[0]:
        raise ValueError(f"pseudo_radius_margins must have shape [{student_logits.shape[0]}]")
    if not bool(accepted.any()):
        return _connected_zero(student_logits)
    per_row = functional.cross_entropy(
        student_logits[accepted],
        teacher_labels[accepted],
        reduction="none",
    )
    prototype_weights = 1.0 / (1.0 + radius_margins[accepted].abs())
    return _require_scalar_finite((per_row * prototype_weights).mean(), name="prototype_pseudo")


def _validate_open_output(output: object, *, batch_size: int, class_count: int, device: torch.device) -> OpenHeadOutput:
    if not isinstance(output, OpenHeadOutput):
        raise TypeError("open_output must be an OpenHeadOutput")
    fields = {
        "class_scores": output.class_scores,
        "class_distances": output.class_distances,
        "radius_margins": output.radius_margins,
        "energy": output.energy,
        "unknown_risk": output.unknown_risk,
    }
    for name, value in fields.items():
        _require_floating_tensor(value, name=f"open_output.{name}", device=device)
    for name in ("class_scores", "class_distances", "radius_margins"):
        if fields[name].shape != (batch_size, class_count):
            raise ValueError(f"open_output.{name} must have shape [{batch_size}, {class_count}]")
    for name in ("energy", "unknown_risk"):
        if fields[name].shape != (batch_size,):
            raise ValueError(f"open_output.{name} must have shape [{batch_size}]")
    if bool(((output.unknown_risk < 0.0) | (output.unknown_risk > 1.0)).any()):
        raise ValueError("open_output.unknown_risk must lie in [0, 1]")
    return output


def _validated_episode_masks(
    episode: object,
    *,
    labels: Tensor,
    class_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(episode, ProxyEpisode):
        raise TypeError("proxy_episode must be a ProxyEpisode")
    batch_size = labels.shape[0]
    device = labels.device
    class_mask = _require_bool_vector(
        episode.registered_class_mask,
        name="proxy_episode.registered_class_mask",
        length=class_count,
        device=device,
    )
    if not bool(class_mask.any()):
        raise ValueError("proxy_episode must retain at least one registered class")
    if isinstance(episode.proxy_class, bool) or not isinstance(episode.proxy_class, Integral):
        raise TypeError("proxy_episode.proxy_class must be an integer")
    if episode.proxy_class < 0 or episode.proxy_class >= class_count or bool(class_mask[episode.proxy_class]):
        raise ValueError("proxy_episode proxy class must be masked from registration")
    registered_rows = _require_integer_vector(
        episode.registered_rows,
        name="proxy_episode.registered_rows",
        device=device,
    )
    proxy_rows = _require_integer_vector(
        episode.proxy_rows,
        name="proxy_episode.proxy_rows",
        device=device,
    )
    all_rows = torch.cat((registered_rows, proxy_rows), dim=0)
    if bool((all_rows < 0).any()) or bool((all_rows >= batch_size).any()):
        raise ValueError("proxy episode rows must index the supervised batch")
    if all_rows.unique().numel() != batch_size or all_rows.numel() != batch_size:
        raise ValueError("proxy episode rows must be a disjoint full partition of the supervised batch")
    if not torch.equal(all_rows.sort().values, torch.arange(batch_size, device=device)):
        raise ValueError("proxy episode rows must cover the supervised batch exactly once")
    if not bool((labels[proxy_rows] == int(episode.proxy_class)).all()):
        raise ValueError("proxy_episode proxy rows must match its source proxy class")
    registered_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    registered_mask[registered_rows] = True
    proxy_mask = ~registered_mask
    return registered_mask, proxy_mask, class_mask


def _mean_or_zero(values: Tensor, *, reference: Tensor) -> Tensor:
    if values.numel() == 0:
        return _connected_zero(reference)
    return values.mean()


def _proxy_open_losses(
    output: OpenHeadOutput,
    *,
    registered_row_mask: Tensor,
    proxy_row_mask: Tensor,
    registered_class_mask: Tensor,
) -> dict[str, Tensor]:
    proxy_target = proxy_row_mask.to(dtype=output.unknown_risk.dtype)
    proxy_bce = functional.binary_cross_entropy(output.unknown_risk, proxy_target)
    active_margins = output.radius_margins[:, registered_class_mask]
    minimum_margin = active_margins.amin(dim=1)
    proxy_radius = functional.softplus(-minimum_margin[proxy_row_mask]).mean()
    registered_radius = _mean_or_zero(
        functional.softplus(minimum_margin[registered_row_mask]),
        reference=output.radius_margins,
    )
    energy_separation = functional.softplus(
        _mean_or_zero(output.energy[registered_row_mask], reference=output.energy)
        - _mean_or_zero(output.energy[proxy_row_mask], reference=output.energy)
    )
    radius_energy = proxy_radius + registered_radius + energy_separation
    return {
        "proxy_bce": _require_scalar_finite(proxy_bce, name="proxy_bce"),
        "radius_energy": _require_scalar_finite(radius_energy, name="radius_energy"),
    }


def _empty_boundary_mixup(embeddings: Tensor) -> BoundaryMixupBatch:
    empty_indices = torch.empty(0, dtype=torch.long, device=embeddings.device)
    return BoundaryMixupBatch(
        mixed_embeddings=embeddings[:0] * 0.0,
        left_indices=empty_indices,
        right_indices=empty_indices.clone(),
        lambdas=embeddings.new_empty((0,)),
    )


def _validated_lambdas(
    lambdas: Tensor | float,
    *,
    pair_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(lambdas, bool):
        raise TypeError("lambdas must be a floating scalar or tensor")
    value = lambdas if isinstance(lambdas, Tensor) else torch.as_tensor(lambdas, device=device, dtype=dtype)
    if not value.is_floating_point():
        raise TypeError("lambdas must use a floating dtype")
    if value.device != device:
        raise ValueError("lambdas device must match embeddings")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("lambdas must be finite")
    if value.ndim == 0:
        value = value.expand(pair_count)
    elif value.ndim != 1 or value.shape[0] != pair_count:
        raise ValueError(f"lambdas must be scalar or have shape [{pair_count}]")
    if bool(((value < 0.35) | (value > 0.65)).any()):
        raise ValueError("lambdas must lie in [0.35, 0.65]")
    return value.to(dtype=dtype)


def build_boundary_mixup(
    embeddings: Tensor,
    labels: Tensor,
    registered_row_mask: Tensor,
    *,
    lambdas: Tensor | float = 0.5,
) -> BoundaryMixupBatch:
    """Build normalized pairs only across different source registered classes."""

    embeddings = _require_floating_tensor(embeddings, name="boundary_embeddings")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("boundary_embeddings must have non-empty shape [B, D]")
    labels = _require_integer_vector(
        labels,
        name="boundary_labels",
        length=embeddings.shape[0],
        device=embeddings.device,
    )
    if bool((labels < 0).any()):
        raise ValueError("boundary_labels must be non-negative")
    registered_row_mask = _require_bool_vector(
        registered_row_mask,
        name="registered_row_mask",
        length=embeddings.shape[0],
        device=embeddings.device,
    )
    registered_indices = torch.nonzero(registered_row_mask, as_tuple=False).flatten()
    if registered_indices.numel() < 2:
        _validated_lambdas(lambdas, pair_count=0, device=embeddings.device, dtype=embeddings.dtype)
        return _empty_boundary_mixup(embeddings)
    registered_labels = labels.index_select(0, registered_indices)
    different_classes = registered_labels[:, None] != registered_labels[None, :]
    pair_positions = torch.nonzero(torch.triu(different_classes, diagonal=1), as_tuple=False)
    if pair_positions.numel() == 0:
        _validated_lambdas(lambdas, pair_count=0, device=embeddings.device, dtype=embeddings.dtype)
        return _empty_boundary_mixup(embeddings)
    left_indices = registered_indices.index_select(0, pair_positions[:, 0])
    right_indices = registered_indices.index_select(0, pair_positions[:, 1])
    pair_lambdas = _validated_lambdas(
        lambdas,
        pair_count=left_indices.numel(),
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    unnormalized = (
        pair_lambdas[:, None] * embeddings.index_select(0, left_indices)
        + (1.0 - pair_lambdas)[:, None] * embeddings.index_select(0, right_indices)
    )
    norms = torch.linalg.vector_norm(unnormalized, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()):
        raise FloatingPointError("boundary mixup norm must be finite")
    nonzero = norms.squeeze(1) > torch.finfo(embeddings.dtype).eps
    if not bool(nonzero.any()):
        return _empty_boundary_mixup(embeddings)
    left_indices = left_indices[nonzero]
    right_indices = right_indices[nonzero]
    pair_lambdas = pair_lambdas[nonzero]
    mixed_embeddings = unnormalized[nonzero] / norms[nonzero]
    if not bool(torch.isfinite(mixed_embeddings).all()):
        raise FloatingPointError("boundary mixed embeddings must be finite")
    return BoundaryMixupBatch(
        mixed_embeddings=mixed_embeddings,
        left_indices=left_indices,
        right_indices=right_indices,
        lambdas=pair_lambdas,
    )


def boundary_mixup_loss(
    mixup: BoundaryMixupBatch,
    output: OpenHeadOutput | None,
    *,
    registered_class_mask: Tensor,
) -> Tensor:
    """Turn a strict boundary mixup batch into an open-world rejection loss."""

    if not isinstance(mixup, BoundaryMixupBatch):
        raise TypeError("boundary_mixup_batch must be a BoundaryMixupBatch")
    mixed = _require_floating_tensor(mixup.mixed_embeddings, name="boundary_mixup_batch.mixed_embeddings")
    if mixed.ndim != 2:
        raise ValueError("boundary mixup embeddings must have shape [B, D]")
    if not isinstance(registered_class_mask, Tensor):
        raise TypeError("registered_class_mask must be a torch.Tensor")
    if registered_class_mask.ndim != 1 or registered_class_mask.numel() == 0:
        raise ValueError("registered_class_mask must be a non-empty one-dimensional tensor")
    registered_class_mask = _require_bool_vector(
        registered_class_mask,
        name="registered_class_mask",
        length=registered_class_mask.shape[0],
        device=mixed.device,
    )
    if not bool(registered_class_mask.any()):
        raise ValueError("registered_class_mask must retain at least one class")
    if mixed.shape[0] == 0:
        return _connected_zero(mixed)
    if output is None:
        raise ValueError("boundary_mixup_output is required when a legal boundary pair exists")
    output = _validate_open_output(
        output,
        batch_size=mixed.shape[0],
        class_count=registered_class_mask.shape[0],
        device=mixed.device,
    )
    active_margins = output.radius_margins[:, registered_class_mask]
    minimum_margin = active_margins.amin(dim=1)
    unknown_target = torch.ones_like(output.unknown_risk)
    value = (
        functional.binary_cross_entropy(output.unknown_risk, unknown_target)
        + functional.softplus(-minimum_margin).mean()
        + functional.softplus(-output.energy).mean()
    )
    return _require_scalar_finite(value, name="boundary_mixup")


def _validate_supplied_boundary_mixup(
    mixup: object,
    *,
    embeddings: Tensor,
    labels: Tensor,
    registered_row_mask: Tensor,
) -> BoundaryMixupBatch:
    """Reject a prebuilt mixup that bypasses the source-only pair constraints."""

    if not isinstance(mixup, BoundaryMixupBatch):
        raise TypeError("boundary_mixup_batch must be a BoundaryMixupBatch")
    embeddings = _require_floating_tensor(embeddings, name="boundary_embeddings")
    if embeddings.ndim != 2 or embeddings.shape[0] != labels.shape[0] or embeddings.shape[1] == 0:
        raise ValueError("boundary_embeddings must have shape [B, D] for the supervised batch")
    if embeddings.device != labels.device:
        raise ValueError("boundary_embeddings device must match supervised labels")
    mixed = _require_floating_tensor(
        mixup.mixed_embeddings,
        name="boundary_mixup_batch.mixed_embeddings",
        device=embeddings.device,
    )
    if mixed.ndim != 2 or mixed.shape[1] != embeddings.shape[1]:
        raise ValueError("boundary mixup embeddings must preserve the embedding width")
    pair_count = mixed.shape[0]
    for name, indices in (("left_indices", mixup.left_indices), ("right_indices", mixup.right_indices)):
        if not isinstance(indices, Tensor):
            raise TypeError(f"boundary_mixup_batch.{name} must be a torch.Tensor")
        if indices.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"boundary_mixup_batch.{name} must use an integer dtype")
        if indices.ndim != 1 or indices.shape[0] != pair_count:
            raise ValueError(f"boundary mixup {name} must have shape [{pair_count}]")
        if indices.device != embeddings.device:
            raise ValueError(f"boundary mixup {name} device must match embeddings")
    if not isinstance(mixup.lambdas, Tensor):
        raise TypeError("boundary_mixup_batch.lambdas must be a torch.Tensor")
    if not mixup.lambdas.is_floating_point():
        raise TypeError("boundary_mixup_batch.lambdas must use a floating dtype")
    if mixup.lambdas.ndim != 1 or mixup.lambdas.shape[0] != pair_count:
        raise ValueError(f"boundary mixup lambdas must have shape [{pair_count}]")
    if mixup.lambdas.device != embeddings.device:
        raise ValueError("boundary mixup lambdas device must match embeddings")
    if not bool(torch.isfinite(mixup.lambdas).all()):
        raise ValueError("boundary mixup lambdas must be finite")
    if bool(((mixup.lambdas < 0.35) | (mixup.lambdas > 0.65)).any()):
        raise ValueError("boundary mixup lambdas must lie in [0.35, 0.65]")
    if pair_count == 0:
        return mixup
    if bool((mixup.left_indices < 0).any()) or bool((mixup.left_indices >= embeddings.shape[0]).any()):
        raise ValueError("boundary mixup left_indices must index the supervised batch")
    if bool((mixup.right_indices < 0).any()) or bool((mixup.right_indices >= embeddings.shape[0]).any()):
        raise ValueError("boundary mixup right_indices must index the supervised batch")
    if bool((mixup.left_indices == mixup.right_indices).any()):
        raise ValueError("boundary mixup pairs must contain two distinct rows")
    if not bool(registered_row_mask[mixup.left_indices].all()) or not bool(registered_row_mask[mixup.right_indices].all()):
        raise ValueError("boundary mixup pairs must use registered rows only")
    if bool((labels[mixup.left_indices] == labels[mixup.right_indices]).any()):
        raise ValueError("boundary mixup pairs must use different registered classes")
    unnormalized = (
        mixup.lambdas[:, None].to(dtype=embeddings.dtype) * embeddings[mixup.left_indices]
        + (1.0 - mixup.lambdas)[:, None].to(dtype=embeddings.dtype) * embeddings[mixup.right_indices]
    )
    norms = torch.linalg.vector_norm(unnormalized, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= torch.finfo(embeddings.dtype).eps).any()):
        raise ValueError("boundary mixup pairs must have finite non-zero norms")
    expected = unnormalized / norms
    if mixed.dtype != embeddings.dtype or not bool(torch.allclose(mixed, expected, rtol=1e-5, atol=1e-6)):
        raise ValueError("boundary mixup embeddings must be the normalized declared pair mixture")
    return mixup


def resolve_group_ids(
    receiver_ids: Tensor,
    day_ids: Tensor,
    scene_ids: Tensor,
    *,
    min_group_size: int = 16,
) -> Tensor:
    """Resolve the fixed receiver×day×scene fallback hierarchy from counts only."""

    if isinstance(min_group_size, bool) or not isinstance(min_group_size, Integral):
        raise TypeError("min_group_size must be an integer")
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    receiver_ids = _require_integer_vector(receiver_ids, name="receiver_ids")
    day_ids = _require_integer_vector(
        day_ids,
        name="day_ids",
        length=receiver_ids.shape[0],
        device=receiver_ids.device,
    )
    scene_ids = _require_integer_vector(
        scene_ids,
        name="scene_ids",
        length=receiver_ids.shape[0],
        device=receiver_ids.device,
    )
    rows = tuple(
        zip(
            (int(value) for value in receiver_ids.detach().cpu().tolist()),
            (int(value) for value in day_ids.detach().cpu().tolist()),
            (int(value) for value in scene_ids.detach().cpu().tolist()),
        )
    )
    receiver_day_scene_counts = Counter(rows)
    receiver_scene_counts = Counter((receiver, scene) for receiver, _, scene in rows)
    receiver_counts = Counter(receiver for receiver, _, _ in rows)
    key_to_group: dict[tuple[object, ...], int] = {}
    assignments: list[int] = []
    for receiver, day, scene in rows:
        if receiver_day_scene_counts[(receiver, day, scene)] >= min_group_size:
            key: tuple[object, ...] = ("receiver_day_scene", receiver, day, scene)
        elif receiver_scene_counts[(receiver, scene)] >= min_group_size:
            key = ("receiver_scene", receiver, scene)
        elif receiver_counts[receiver] >= min_group_size:
            key = ("receiver", receiver)
        else:
            key = ("global",)
        assignments.append(key_to_group.setdefault(key, len(key_to_group)))
    return torch.tensor(assignments, dtype=torch.int64, device=receiver_ids.device)


def group_cvar(losses: Tensor, groups: Tensor, *, tail_fraction: float = 0.30) -> Tensor:
    """Average the largest ``ceil(fraction × group_count)`` mean group losses."""

    losses = _require_floating_tensor(losses, name="losses")
    if losses.ndim != 1 or losses.numel() == 0:
        raise ValueError("losses must be a non-empty one-dimensional tensor")
    groups = _require_integer_vector(
        groups,
        name="groups",
        length=losses.shape[0],
        device=losses.device,
    )
    if isinstance(tail_fraction, bool) or not isinstance(tail_fraction, Real):
        raise TypeError("tail_fraction must be a real number")
    tail_fraction = float(tail_fraction)
    if not math.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    unique_groups = groups.unique(sorted=True)
    group_means = torch.stack(
        tuple(losses[groups == group].mean() for group in unique_groups),
        dim=0,
    )
    selected_count = max(1, math.ceil(group_means.numel() * tail_fraction))
    value = group_means.topk(selected_count).values.mean()
    return _require_scalar_finite(value, name="group_cvar")


def _resolved_config(arm: str | ArmConfig) -> ArmConfig:
    if isinstance(arm, ArmConfig):
        return arm
    if not isinstance(arm, str):
        raise TypeError("arm must be an arm ID string or ArmConfig")
    return arm_config(arm)


def compute_arm_losses(
    arm: str | ArmConfig,
    *,
    supervised_logits: Tensor,
    supervised_labels: Tensor,
    pseudo_student_logits: Tensor,
    pseudo_teacher_logits: Tensor,
    pseudo_inside_radius: Tensor,
    pseudo_radius_margins: Tensor | None = None,
    masked_prediction: Tensor | None = None,
    masked_target: Tensor | None = None,
    cross_receiver_features: Tensor | None = None,
    cross_receiver_ids: Tensor | None = None,
    proxy_episode: ProxyEpisode | None = None,
    open_output: OpenHeadOutput | None = None,
    boundary_embeddings: Tensor | None = None,
    boundary_mixup_batch: BoundaryMixupBatch | None = None,
    boundary_mixup_output: OpenHeadOutput | None = None,
    boundary_lambdas: Tensor | float = 0.5,
    group_losses: Tensor | None = None,
    group_ids: Tensor | None = None,
    tail_fraction: float = 0.30,
) -> dict[str, Tensor]:
    """Compose exactly the mechanisms declared by a frozen B0/A/B/C arm.

    Inputs for later arms are intentionally ignored by earlier arms, so their
    graphs cannot leak proxy rejection or Group-CVaR gradients into B0/A/B.
    """

    config = _resolved_config(arm)
    supervised_logits = _validate_logits(supervised_logits, name="supervised_logits")
    supervised_labels = _validate_labels(
        supervised_labels,
        batch_size=supervised_logits.shape[0],
        class_count=supervised_logits.shape[1],
        device=supervised_logits.device,
    )
    registered_rows = torch.ones(
        supervised_logits.shape[0], dtype=torch.bool, device=supervised_logits.device
    )
    proxy_rows = torch.zeros_like(registered_rows)
    registered_classes = torch.ones(
        supervised_logits.shape[1], dtype=torch.bool, device=supervised_logits.device
    )
    if config.arm_id in {"B", "C"}:
        if proxy_episode is None:
            raise ValueError("proxy_episode is required for B/C proxy rejection losses")
        registered_rows, proxy_rows, registered_classes = _validated_episode_masks(
            proxy_episode,
            labels=supervised_labels,
            class_count=supervised_logits.shape[1],
        )

    losses: dict[str, Tensor] = {
        "registered_ce": registered_cross_entropy(
            supervised_logits,
            supervised_labels,
            registered_row_mask=registered_rows,
        ),
        "ema_pseudo": ema_pseudo_loss(
            pseudo_student_logits,
            pseudo_teacher_logits,
            pseudo_inside_radius,
        ),
        "weak_strong_consistency": weak_strong_consistency_loss(
            pseudo_student_logits,
            pseudo_teacher_logits,
        ),
    }

    if config.arm_id in {"A", "B", "C"}:
        if masked_prediction is None or masked_target is None:
            raise ValueError("masked_prediction and masked_target are required for A/B/C")
        if cross_receiver_features is None or cross_receiver_ids is None:
            raise ValueError("cross_receiver_features and cross_receiver_ids are required for A/B/C")
        if pseudo_radius_margins is None:
            raise ValueError("pseudo_radius_margins is required for A/B/C")
        losses["masked_latent"] = masked_latent_loss(masked_prediction, masked_target)
        losses["cross_receiver"] = cross_receiver_consistency_loss(
            cross_receiver_features,
            cross_receiver_ids,
        )
        losses["prototype_pseudo"] = prototype_aware_pseudo_loss(
            pseudo_student_logits,
            pseudo_teacher_logits,
            pseudo_inside_radius,
            pseudo_radius_margins,
        )

    if config.arm_id in {"B", "C"}:
        if open_output is None:
            raise ValueError("open_output is required for B/C proxy rejection losses")
        validated_open_output = _validate_open_output(
            open_output,
            batch_size=supervised_logits.shape[0],
            class_count=supervised_logits.shape[1],
            device=supervised_logits.device,
        )
        losses.update(
            _proxy_open_losses(
                validated_open_output,
                registered_row_mask=registered_rows,
                proxy_row_mask=proxy_rows,
                registered_class_mask=registered_classes,
            )
        )
        if boundary_embeddings is None:
            raise ValueError("boundary_embeddings are required for B/C boundary mixup")
        boundary_embeddings = _require_floating_tensor(
            boundary_embeddings,
            name="boundary_embeddings",
            device=supervised_logits.device,
        )
        if boundary_embeddings.ndim != 2 or boundary_embeddings.shape[0] != supervised_logits.shape[0]:
            raise ValueError("boundary_embeddings must have shape [B, D] for the supervised batch")
        if boundary_mixup_batch is None:
            boundary_mixup_batch = build_boundary_mixup(
                boundary_embeddings,
                supervised_labels,
                registered_rows,
                lambdas=boundary_lambdas,
            )
        else:
            boundary_mixup_batch = _validate_supplied_boundary_mixup(
                boundary_mixup_batch,
                embeddings=boundary_embeddings,
                labels=supervised_labels,
                registered_row_mask=registered_rows,
            )
        losses["boundary_mixup"] = boundary_mixup_loss(
            boundary_mixup_batch,
            boundary_mixup_output,
            registered_class_mask=registered_classes,
        )

    if config.arm_id == "C":
        if group_losses is None or group_ids is None:
            raise ValueError("group_losses and group_ids are required for C Group-CVaR")
        losses["group_cvar"] = group_cvar(group_losses, group_ids, tail_fraction=tail_fraction)

    for name, value in tuple(losses.items()):
        _require_scalar_finite(value, name=name)
    total = _connected_zero(supervised_logits)
    for value in losses.values():
        total = total + value
    losses["total"] = _require_scalar_finite(total, name="total")
    return losses


__all__ = [
    "BoundaryMixupBatch",
    "build_boundary_mixup",
    "boundary_mixup_loss",
    "compute_arm_losses",
    "ema_pseudo_loss",
    "group_cvar",
    "pseudo_accept_mask",
    "registered_cross_entropy",
    "resolve_group_ids",
]
