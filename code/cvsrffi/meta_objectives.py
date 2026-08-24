"""Pure support and outer objectives for the V1 meta-adapter route.

The functions in this module only consume model ``return_aux=True`` mappings,
episode labels/masks, frozen prototype tensors, and adapter parameter snapshots.
They do not create a classifier, mutate an adapter, update query state, or read
any dataset files.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


_LOGIT_KEYS = ("logits", "tx_logits")
# ``feat_cls`` is the canonical CVSincNet identity embedding.  The remaining
# names cover the real ADV3B02 dual-model aliases without changing semantics.
_EMBEDDING_KEYS = (
    "feat_cls",
    "z_id",
    "id_feat_cls",
    "feat_joint",
    "id_feat_joint",
    "base",
    "id_base",
)
_FORBIDDEN_ADAPTER_KEY_PARTS = (
    "log_step_size",
    "cls_head",
    "classifier",
    "prototype",
    "lda",
    "cov",
)
_INNER_LEAF_NAMES = frozenset(
    {
        "down.weight",
        "down.bias",
        "up.weight",
        "up.bias",
        "gate",
    }
)


@dataclass(frozen=True)
class MetaObjectiveConfig:
    """Explicit, immutable weights and numerical constants for V1 losses."""

    lambda_adapt: float = 1.0
    lambda_prototype: float = 1.0
    lambda_view_consistency: float = 0.0
    lambda_l2sp: float = 1.0e-3
    lambda_guard: float = 1.0
    lambda_floor: float = 1.0
    lambda_topology: float = 1.0
    lambda_zero_step: float = 1.0
    floor_tau: float = 0.2
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in (
            "lambda_adapt",
            "lambda_prototype",
            "lambda_view_consistency",
            "lambda_l2sp",
            "lambda_guard",
            "lambda_floor",
            "lambda_topology",
            "lambda_zero_step",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(float(self.floor_tau)) or float(self.floor_tau) <= 0.0:
            raise ValueError("floor_tau must be finite and positive")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError("eps must be finite and positive")

    # Equation-friendly aliases keep the mathematical notation available to
    # later trainer code while the public field names remain descriptive.
    @property
    def lambda_p(self) -> float:
        return self.lambda_prototype

    @property
    def lambda_c(self) -> float:
        return self.lambda_view_consistency

    @property
    def lambda_sp(self) -> float:
        return self.lambda_l2sp

    @property
    def lambda_g(self) -> float:
        return self.lambda_guard

    @property
    def lambda_f(self) -> float:
        return self.lambda_floor

    @property
    def lambda_t(self) -> float:
        return self.lambda_topology

    @property
    def lambda_0(self) -> float:
        return self.lambda_zero_step

    @property
    def tau(self) -> float:
        return self.floor_tau


@dataclass(frozen=True)
class LossBreakdown:
    """Individual differentiable terms and episode routing counts."""

    total: Tensor
    adapt: Tensor
    guard: Tensor
    floor: Tensor
    topology: Tensor
    zero_step: Tensor
    prototype: Tensor
    l2sp: Tensor
    adapt_count: int
    guard_count: int


def _require_config(config: MetaObjectiveConfig) -> MetaObjectiveConfig:
    if not isinstance(config, MetaObjectiveConfig):
        raise TypeError("config must be a MetaObjectiveConfig")
    return config


def _zero(reference: Tensor) -> Tensor:
    """Return a finite scalar zero that preserves the reference gradient path."""

    return reference.sum() * 0.0


def _extract_outputs(outputs: Mapping[str, Any], *, role: str) -> tuple[Tensor, Tensor]:
    """Extract fixed-head logits and the explicitly chosen pre-normalization embedding."""

    if not isinstance(outputs, Mapping):
        raise TypeError(f"{role} must be a mapping returned by model(return_aux=True)")

    def find_tensor(keys: tuple[str, ...], field: str) -> Tensor:
        for key in keys:
            if key in outputs:
                value = outputs[key]
                if not torch.is_tensor(value):
                    raise ValueError(f"{role} {field} key {key!r} must be a tensor")
                return value
        joined = ", ".join(keys)
        raise ValueError(f"{role} is missing fixed-head {field}; expected one of {joined}")

    logits = find_tensor(_LOGIT_KEYS, "logits")
    embedding = find_tensor(_EMBEDDING_KEYS, "embedding")
    if logits.ndim != 2:
        raise ValueError(f"{role} logits must have shape [batch, classes]")
    if embedding.ndim != 2:
        raise ValueError(f"{role} embedding must have shape [batch, dimension]")
    if logits.size(0) != embedding.size(0):
        raise ValueError(f"{role} logits and embedding batch dimensions must match")
    if not logits.is_floating_point() or not embedding.is_floating_point():
        raise ValueError(f"{role} logits and embedding must be floating-point tensors")
    if logits.device != embedding.device:
        raise ValueError(f"{role} logits and embedding must be on the same device")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(embedding).all()):
        raise ValueError(f"{role} logits and embedding must be finite")
    return logits, embedding


def _validate_labels(
    labels: Tensor,
    *,
    batch_size: int,
    class_count: int,
    device: torch.device,
) -> Tensor:
    if not torch.is_tensor(labels):
        raise TypeError("labels must be a tensor")
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional tensor")
    if labels.numel() != batch_size:
        raise ValueError("labels length must match output batch size")
    if labels.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("labels must use an integer dtype")
    labels = labels.to(device=device, dtype=torch.long)
    if labels.numel() and (int(labels.min().item()) < 0 or int(labels.max().item()) >= class_count):
        raise ValueError("labels are out of range for the fixed classification head")
    return labels


def _validate_prototypes(
    frozen_prototypes: Tensor,
    *,
    class_count: int,
    embedding_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not torch.is_tensor(frozen_prototypes):
        raise TypeError("frozen_prototypes must be a tensor")
    if frozen_prototypes.ndim != 2:
        raise ValueError("frozen_prototypes must have shape [classes, dimension]")
    if tuple(frozen_prototypes.shape) != (class_count, embedding_dim):
        raise ValueError("frozen_prototypes shape must match fixed classes and embedding dimension")
    if not frozen_prototypes.is_floating_point():
        raise ValueError("frozen_prototypes must be floating-point")
    if not bool(torch.isfinite(frozen_prototypes).all()):
        raise ValueError("frozen_prototypes must be finite")
    # Detach before the device/dtype conversion so prototype anchors can never
    # receive objective gradients, even if a caller passed requires_grad=True.
    return frozen_prototypes.detach().to(device=device, dtype=dtype)


def _validate_mask(mask: Tensor, *, name: str, batch_size: int, device: torch.device) -> Tensor:
    if not torch.is_tensor(mask):
        raise TypeError(f"{name} must be a tensor")
    if mask.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if mask.numel() != batch_size:
        raise ValueError(f"{name} length must match output batch size")
    if mask.dtype is not torch.bool:
        raise ValueError(f"{name} must use boolean dtype")
    return mask.to(device=device)


def _validate_adapter_mapping(
    values: Mapping[str, Tensor],
    *,
    name: str,
    reference: Tensor,
) -> dict[str, Tensor]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping of inner adapter tensors")
    result: dict[str, Tensor] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        if not torch.is_tensor(value):
            raise ValueError(f"{name}[{key!r}] must be a tensor")
        lowered = key.lower()
        if any(part in lowered for part in _FORBIDDEN_ADAPTER_KEY_PARTS):
            if "log_step_size" in lowered:
                raise ValueError("log_step_size is not an inner adapter parameter")
            raise ValueError(f"{name}[{key!r}] is not an allowed adapter-only parameter")
        if lowered.endswith(".gate"):
            leaf = "gate"
        else:
            leaf = ".".join(lowered.split(".")[-2:]) if "." in lowered else lowered
        if leaf not in _INNER_LEAF_NAMES:
            raise ValueError(f"{name}[{key!r}] is not an adapter parameter")
        if not value.is_floating_point():
            raise ValueError(f"{name}[{key!r}] must be floating-point")
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(f"{name}[{key!r}] must match the objective device and dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name}[{key!r}] must be finite")
        result[key] = value
    return result


def _row_cross_entropy(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        return logits.new_empty((0,))
    return F.cross_entropy(logits[mask], labels[mask], reduction="none")


def _mean_cross_entropy(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    rows = _row_cross_entropy(logits, labels, mask)
    return _zero(logits) if rows.numel() == 0 else rows.mean()


def _floor_loss(logits: Tensor, labels: Tensor, mask: Tensor, tau: float) -> Tensor:
    row_ce = _row_cross_entropy(logits, labels, mask)
    if row_ce.numel() == 0:
        return _zero(logits)
    selected_labels = labels[mask]
    class_losses = torch.stack(
        [row_ce[selected_labels == cls].mean() for cls in torch.unique(selected_labels, sorted=True)]
    )
    return float(tau) * torch.logsumexp(class_losses / float(tau), dim=0)


def _prototype_loss(
    embedding: Tensor,
    labels: Tensor,
    frozen_prototypes: Tensor,
    eps: float,
) -> Tensor:
    if labels.numel() == 0:
        return _zero(embedding)
    normalized_embedding = F.normalize(embedding, dim=1, eps=float(eps))
    normalized_prototypes = F.normalize(frozen_prototypes[labels], dim=1, eps=float(eps))
    return (1.0 - (normalized_embedding * normalized_prototypes).sum(dim=1)).mean()


def _l2sp_loss(
    initial_adapter: Mapping[str, Tensor],
    current_adapter: Mapping[str, Tensor],
    *,
    reference: Tensor,
) -> Tensor:
    initial = _validate_adapter_mapping(initial_adapter, name="initial_adapter", reference=reference)
    current = _validate_adapter_mapping(current_adapter, name="current_adapter", reference=reference)
    if set(initial) != set(current):
        raise ValueError("initial_adapter and current_adapter must have the same keys")
    if not initial:
        return _zero(reference)
    terms = []
    for key in sorted(initial):
        if initial[key].shape != current[key].shape:
            raise ValueError(f"adapter tensor shape mismatch for {key!r}")
        # The initial state is an immutable anchor; gradients belong only to
        # the current adapter values.
        terms.append((current[key] - initial[key].detach()).pow(2).sum())
    return torch.stack(terms).sum()


def _topology_loss(
    pre_embedding: Tensor,
    post_embedding: Tensor,
    labels: Tensor,
    mask: Tensor,
    eps: float,
) -> Tensor:
    selected_labels = labels[mask]
    if selected_labels.numel() == 0:
        return _zero(post_embedding)
    classes = torch.unique(selected_labels, sorted=True)
    if classes.numel() < 2:
        return _zero(post_embedding)
    pre_centers = torch.stack(
        [pre_embedding[mask][selected_labels == cls].mean(dim=0) for cls in classes]
    )
    post_centers = torch.stack(
        [post_embedding[mask][selected_labels == cls].mean(dim=0) for cls in classes]
    )
    pre_centers = F.normalize(pre_centers, dim=1, eps=float(eps))
    post_centers = F.normalize(post_centers, dim=1, eps=float(eps))
    pre_pairwise = pre_centers @ pre_centers.transpose(0, 1)
    post_pairwise = post_centers @ post_centers.transpose(0, 1)
    return (pre_pairwise - post_pairwise).pow(2).mean()


def support_objective(
    outputs: Mapping[str, Any],
    labels: Tensor,
    frozen_prototypes: Tensor,
    initial_adapter: Mapping[str, Tensor],
    current_adapter: Mapping[str, Tensor],
    config: MetaObjectiveConfig,
) -> LossBreakdown:
    """Compute the fixed-head support inner-loop objective."""

    config = _require_config(config)
    logits, embedding = _extract_outputs(outputs, role="support outputs")
    labels = _validate_labels(
        labels,
        batch_size=logits.size(0),
        class_count=logits.size(1),
        device=logits.device,
    )
    prototypes = _validate_prototypes(
        frozen_prototypes,
        class_count=logits.size(1),
        embedding_dim=embedding.size(1),
        device=embedding.device,
        dtype=embedding.dtype,
    )
    all_rows = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
    adapt = _mean_cross_entropy(logits, labels, all_rows)
    prototype = _prototype_loss(embedding, labels, prototypes, config.eps)
    l2sp = _l2sp_loss(initial_adapter, current_adapter, reference=logits)
    zero = _zero(logits)
    total = config.lambda_adapt * adapt + config.lambda_prototype * prototype + config.lambda_l2sp * l2sp
    return LossBreakdown(
        total=total,
        adapt=adapt,
        guard=zero,
        floor=zero,
        topology=zero,
        zero_step=zero,
        prototype=prototype,
        l2sp=l2sp,
        adapt_count=0,
        guard_count=0,
    )


def outer_objective(
    pre_outputs: Mapping[str, Any],
    post_outputs: Mapping[str, Any],
    labels: Tensor,
    adapt_mask: Tensor,
    guard_mask: Tensor,
    frozen_prototypes: Tensor,
    config: MetaObjectiveConfig,
) -> LossBreakdown:
    """Compute the independent-query outer objective after adapter updates."""

    config = _require_config(config)
    pre_logits, pre_embedding = _extract_outputs(pre_outputs, role="pre outputs")
    post_logits, post_embedding = _extract_outputs(post_outputs, role="post outputs")
    if pre_logits.shape != post_logits.shape:
        raise ValueError("pre and post logits shapes must match")
    if pre_embedding.shape != post_embedding.shape:
        raise ValueError("pre and post embedding shapes must match")
    if pre_logits.device != post_logits.device or pre_logits.dtype != post_logits.dtype:
        raise ValueError("pre and post logits must share device and dtype")
    if pre_embedding.device != post_embedding.device or pre_embedding.dtype != post_embedding.dtype:
        raise ValueError("pre and post embeddings must share device and dtype")

    labels = _validate_labels(
        labels,
        batch_size=post_logits.size(0),
        class_count=post_logits.size(1),
        device=post_logits.device,
    )
    adapt_mask = _validate_mask(
        adapt_mask,
        name="adapt_mask",
        batch_size=post_logits.size(0),
        device=post_logits.device,
    )
    guard_mask = _validate_mask(
        guard_mask,
        name="guard_mask",
        batch_size=post_logits.size(0),
        device=post_logits.device,
    )
    if bool((adapt_mask & guard_mask).any()):
        raise ValueError("adapt_mask and guard_mask must not overlap")
    prototypes = _validate_prototypes(
        frozen_prototypes,
        class_count=post_logits.size(1),
        embedding_dim=post_embedding.size(1),
        device=post_embedding.device,
        dtype=post_embedding.dtype,
    )
    del prototypes  # The outer topology is defined by pre/post centers only.

    query_mask = adapt_mask | guard_mask
    adapt = _mean_cross_entropy(post_logits, labels, adapt_mask)
    guard = _mean_cross_entropy(post_logits, labels, guard_mask)
    floor = _floor_loss(post_logits, labels, query_mask, config.floor_tau)
    topology = _topology_loss(
        pre_embedding,
        post_embedding,
        labels,
        query_mask,
        config.eps,
    )
    zero_step = _mean_cross_entropy(pre_logits, labels, query_mask)
    zero = _zero(post_logits)
    total = (
        config.lambda_adapt * adapt
        + config.lambda_guard * guard
        + config.lambda_floor * floor
        + config.lambda_topology * topology
        + config.lambda_zero_step * zero_step
    )
    return LossBreakdown(
        total=total,
        adapt=adapt,
        guard=guard,
        floor=floor,
        topology=topology,
        zero_step=zero_step,
        prototype=zero,
        l2sp=zero,
        adapt_count=int(adapt_mask.sum().item()),
        guard_count=int(guard_mask.sum().item()),
    )


__all__ = [
    "LossBreakdown",
    "MetaObjectiveConfig",
    "outer_objective",
    "support_objective",
]
