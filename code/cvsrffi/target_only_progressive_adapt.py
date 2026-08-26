"""Report-parity SF-TAPFT target-only progressive fine-tuning.

The method in the user-supplied report trains a persistent target classifier.
That behavior is intentionally exposed only as ``DIAGNOSTIC_NON_FORMAL``: it
does not satisfy the current formal ``p2_min_v1`` frozen-prototype boundary.
The implementation never accepts a source loader, source samples, query data,
or target-evaluation labels.
"""

from __future__ import annotations

import copy
import inspect
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .meta_adapter import ResidualMetaAdapter


_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_PHASES = ("A", "B", "C")
_EPS = 1.0e-8


@dataclass(frozen=True)
class TargetOnlyAdaptationDataset:
    """Immutable target-train-only carrier; no eval or source role exists."""

    received_iq: Tensor
    labels: Tensor
    physical_ids: tuple[str, ...]
    groups: tuple[str, ...] | None = None
    role: str = "target_train"
    physical_id_origin: str = "provided"

    def __post_init__(self) -> None:
        if self.role != "target_train":
            raise ValueError("SF-TAPFT dataset role must be target_train")
        if self.physical_id_origin not in {"provided", "validated_support_row_index"}:
            raise ValueError("unsupported physical_id_origin")
        if not torch.is_tensor(self.received_iq) or self.received_iq.ndim < 2:
            raise ValueError("received_iq must be a tensor with a non-empty batch")
        if self.received_iq.size(0) <= 0 or not self.received_iq.is_floating_point():
            raise ValueError("received_iq must be non-empty floating-point target train data")
        if not bool(torch.isfinite(self.received_iq).all()):
            raise ValueError("received_iq must contain only finite values")
        if not torch.is_tensor(self.labels) or self.labels.ndim != 1:
            raise ValueError("labels must be a one-dimensional integer tensor")
        if self.labels.dtype not in _INTEGER_DTYPES:
            raise ValueError("labels must use an integer dtype")
        count = int(self.received_iq.size(0))
        if int(self.labels.numel()) != count:
            raise ValueError("labels must align with received_iq")
        physical_ids = tuple(self.physical_ids)
        groups = tuple(self.groups) if self.groups is not None else ()
        if len(physical_ids) != count or (groups and len(groups) != count):
            raise ValueError("physical_ids and groups must align with received_iq")
        if any(not isinstance(value, str) or not value.strip() for value in physical_ids):
            raise ValueError("physical_ids must contain non-empty strings")
        if len(set(physical_ids)) != count:
            raise ValueError("physical_ids must be unique")
        if groups and any(not isinstance(value, str) or not value.strip() for value in groups):
            raise ValueError("groups must contain non-empty strings")
        object.__setattr__(self, "received_iq", self.received_iq.detach().clone())
        object.__setattr__(self, "labels", self.labels.detach().clone().long())
        object.__setattr__(self, "physical_ids", physical_ids)
        object.__setattr__(self, "groups", groups)

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in torch.unique(self.labels, sorted=True).tolist())


@dataclass(frozen=True)
class SFTAPFTConfig:
    """The report's first executable SF-TAPFT configuration."""

    adapter_rank: int = 16
    classifier_source_target_interpolation: float = 0.5
    prototype_scale: float = 8.0
    label_smoothing: float = 0.05
    lambda_proto: float = 0.5
    lambda_l2sp: float = 1.0e-4
    selective_kd_weight: float = 0.0
    selective_kd_temperature: float = 2.0
    selective_kd_gamma: float = 2.0
    phase_steps: tuple[int, int, int] = (500, 1500, 2500)
    lr_head_initial: float = 1.0e-3
    lr_norm: float = 1.0e-4
    lr_head_middle: float = 3.0e-4
    lr_adapter: float = 3.0e-4
    lr_head_late: float = 1.0e-4
    lr_adapter_late: float = 1.0e-4
    lr_last_block: float = 3.0e-5
    weight_decay: float = 1.0e-4
    warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    checkpoint_average_top_k: int = 3
    mixed_precision: bool = True
    seed: int = 392002

    def __post_init__(self) -> None:
        if isinstance(self.adapter_rank, bool) or int(self.adapter_rank) <= 0:
            raise ValueError("adapter_rank must be a positive integer")
        if len(self.phase_steps) != 3 or any(
            isinstance(value, bool) or int(value) < 0 for value in self.phase_steps
        ):
            raise ValueError("phase_steps must contain three non-negative integers")
        if sum(int(value) for value in self.phase_steps) <= 0:
            raise ValueError("phase_steps must contain at least one optimizer step")
        bounded = {
            "classifier_source_target_interpolation": (0.0, 1.0),
            "label_smoothing": (0.0, 1.0),
            "warmup_ratio": (0.0, 1.0),
        }
        for name, (lower, upper) in bounded.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < lower or value > upper:
                raise ValueError(f"{name} must be finite in [{lower}, {upper}]")
        positive = (
            "prototype_scale",
            "selective_kd_temperature",
            "gradient_clip_norm",
            "lr_head_initial",
            "lr_norm",
            "lr_head_middle",
            "lr_adapter",
            "lr_head_late",
            "lr_adapter_late",
            "lr_last_block",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = (
            "lambda_proto",
            "lambda_l2sp",
            "selective_kd_weight",
            "selective_kd_gamma",
            "weight_decay",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if isinstance(self.checkpoint_average_top_k, bool) or int(self.checkpoint_average_top_k) <= 0:
            raise ValueError("checkpoint_average_top_k must be a positive integer")
        if not isinstance(self.mixed_precision, bool):
            raise ValueError("mixed_precision must be a boolean")
        object.__setattr__(self, "phase_steps", tuple(int(value) for value in self.phase_steps))


class TargetPrototypeHead(nn.Module):
    """Trainable normalized target classifier from the report."""

    def __init__(self, weight: Tensor, class_ids: Sequence[int], scale: float = 8.0):
        super().__init__()
        if not torch.is_tensor(weight) or weight.ndim != 2 or weight.size(0) <= 0:
            raise ValueError("weight must have shape [classes, dimension]")
        ids = tuple(int(value) for value in class_ids)
        if len(ids) != int(weight.size(0)) or len(set(ids)) != len(ids):
            raise ValueError("class_ids must uniquely align with weight rows")
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        self.weight = nn.Parameter(F.normalize(weight.detach().clone(), dim=1, eps=_EPS))
        self.class_ids = ids
        self.scale = scale

    @classmethod
    def from_source_and_target(
        cls,
        *,
        source_weights: Tensor,
        target_prototypes: Tensor,
        source_class_ids: Sequence[int],
        target_class_ids: Sequence[int],
        rho: float,
        scale: float,
    ) -> "TargetPrototypeHead":
        if source_weights.ndim != 2 or target_prototypes.ndim != 2:
            raise ValueError("source_weights and target_prototypes must be matrices")
        if source_weights.size(1) != target_prototypes.size(1):
            raise ValueError("source and target classifier dimensions must match")
        source_ids = tuple(int(value) for value in source_class_ids)
        target_ids = tuple(int(value) for value in target_class_ids)
        if len(source_ids) != source_weights.size(0) or len(set(source_ids)) != len(source_ids):
            raise ValueError("source_class_ids must uniquely align with source_weights")
        if len(target_ids) != target_prototypes.size(0) or len(set(target_ids)) != len(target_ids):
            raise ValueError("target_class_ids must uniquely align with target_prototypes")
        rho = float(rho)
        if not math.isfinite(rho) or rho < 0.0 or rho > 1.0:
            raise ValueError("rho must be finite in [0, 1]")
        source = F.normalize(source_weights.detach(), dim=1, eps=_EPS)
        target = F.normalize(target_prototypes.detach(), dim=1, eps=_EPS)
        target_by_id = {class_id: target[index] for index, class_id in enumerate(target_ids)}
        rows: list[Tensor] = []
        output_ids: list[int] = []
        for index, class_id in enumerate(source_ids):
            row = source[index]
            if class_id in target_by_id:
                row = (1.0 - rho) * row + rho * target_by_id[class_id]
            rows.append(F.normalize(row, dim=0, eps=_EPS))
            output_ids.append(class_id)
        for class_id in target_ids:
            if class_id not in set(source_ids):
                rows.append(target_by_id[class_id])
                output_ids.append(class_id)
        return cls(torch.stack(rows), output_ids, scale=scale)

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 2 or embeddings.size(1) != self.weight.size(1):
            raise ValueError("embeddings must align with target classifier dimension")
        return self.scale * F.normalize(embeddings, dim=1, eps=_EPS) @ F.normalize(
            self.weight, dim=1, eps=_EPS
        ).transpose(0, 1)


def leave_one_out_prototype_logits(
    embeddings: Tensor,
    labels: Tensor,
    *,
    class_count: int,
    fallback_weights: Tensor,
    scale: float,
) -> Tensor:
    """Return per-sample prototype logits with the current sample excluded."""

    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.size(0) != labels.numel():
        raise ValueError("embeddings and labels must be row aligned")
    if fallback_weights.shape != (int(class_count), embeddings.size(1)):
        raise ValueError("fallback_weights must align with classes and embedding dimension")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= int(class_count)):
        raise ValueError("labels must be local classifier row indices")
    normalized_embeddings = F.normalize(embeddings, dim=1, eps=_EPS)
    fallback = F.normalize(fallback_weights.detach(), dim=1, eps=_EPS)
    rows: list[Tensor] = []
    indices = torch.arange(embeddings.size(0), device=embeddings.device)
    for sample_index in range(embeddings.size(0)):
        prototypes: list[Tensor] = []
        for class_index in range(int(class_count)):
            mask = labels == class_index
            if int(labels[sample_index]) == class_index:
                mask = mask & (indices != sample_index)
            members = embeddings[mask]
            prototype = members.mean(dim=0) if members.size(0) else fallback[class_index]
            prototypes.append(F.normalize(prototype, dim=0, eps=_EPS))
        rows.append(normalized_embeddings[sample_index] @ torch.stack(prototypes).transpose(0, 1))
    return float(scale) * torch.stack(rows)


class L2SPRegularizer:
    """Normalized parameter-distance anchor to the input checkpoint."""

    def __init__(self, anchors: Mapping[str, Tensor], weights: Mapping[str, float] | None = None):
        self._anchors = MappingProxyType(
            {name: value.detach().clone() for name, value in anchors.items()}
        )
        self._weights = MappingProxyType(dict(weights or {}))

    @classmethod
    def from_named_parameters(
        cls,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> "L2SPRegularizer":
        return cls({name: parameter for name, parameter in named_parameters}, weights=weights)

    def __call__(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> Tensor:
        current = dict(named_parameters)
        if not self._anchors:
            raise ValueError("L2-SP requires at least one anchored parameter")
        first = next(iter(current.values()), None)
        if first is None:
            raise ValueError("current named_parameters is empty")
        total = first.new_zeros(())
        for name, anchor in self._anchors.items():
            if name not in current:
                raise ValueError(f"anchored parameter disappeared: {name!r}")
            parameter = current[name]
            if parameter.shape != anchor.shape:
                raise ValueError(f"anchored parameter shape changed: {name!r}")
            weight = float(self._weights.get(name, 1.0))
            total = total + weight * (parameter - anchor.to(parameter)).pow(2).mean()
        return total


def _identity_backbone(model: nn.Module) -> tuple[nn.Module, str]:
    backbone = getattr(model, "id_backbone", None)
    if isinstance(backbone, nn.Module):
        return backbone, "id_backbone."
    return model, ""


def ensure_time_adapter(model: nn.Module, rank: int = 16) -> ResidualMetaAdapter:
    """Attach the report's exact-identity time adapter after checkpoint load."""

    backbone, _ = _identity_backbone(model)
    if not hasattr(backbone, "meta_adapter_time"):
        raise ValueError("model does not expose meta_adapter_time")
    current = getattr(backbone, "meta_adapter_time")
    if isinstance(current, ResidualMetaAdapter):
        if int(current.down.out_features) != int(rank):
            raise ValueError("existing time adapter rank does not match SF-TAPFT config")
        return current
    if not isinstance(current, nn.Identity):
        raise ValueError("meta_adapter_time must be Identity or ResidualMetaAdapter")
    dimension = int(getattr(backbone, "emb_dim", 0))
    if dimension <= 0:
        raise ValueError("model must expose a positive emb_dim")
    adapter = ResidualMetaAdapter(dimension, rank=int(rank))
    device = next(backbone.parameters()).device
    dtype = next(parameter.dtype for parameter in backbone.parameters() if parameter.is_floating_point())
    adapter.to(device=device, dtype=dtype)
    with torch.no_grad():
        adapter.up.weight.zero_()
        adapter.up.bias.zero_()
    setattr(backbone, "meta_adapter_time", adapter)
    return adapter


class ProgressiveTrainabilityPolicy:
    """A/B/C allowlist for norm, time adapter, and final time block."""

    @staticmethod
    def parameter_names(model: nn.Module, phase: str) -> tuple[str, ...]:
        if phase not in _PHASES:
            raise ValueError("phase must be A, B or C")
        _, prefix = _identity_backbone(model)
        norm_prefixes = (
            f"{prefix}time_fuse.1.",
            f"{prefix}t1.norm.",
            f"{prefix}t2.norm.",
            f"{prefix}t3.norm.",
        )
        adapter_prefix = f"{prefix}meta_adapter_time."
        last_prefix = f"{prefix}t3."
        names = []
        for name, _ in model.named_parameters():
            norm = any(name.startswith(candidate) for candidate in norm_prefixes)
            adapter = name.startswith(adapter_prefix)
            last_block = name.startswith(last_prefix)
            if norm or (phase in {"B", "C"} and adapter) or (phase == "C" and last_block):
                names.append(name)
        if not any(name.startswith(f"{prefix}t3.norm.") for name in names):
            raise ValueError("model must expose t3.norm affine parameters")
        if phase in {"B", "C"} and not any(name.startswith(adapter_prefix) for name in names):
            raise ValueError("model must expose the SF-TAPFT time adapter")
        return tuple(sorted(set(names)))

    def apply(self, model: nn.Module, phase: str) -> tuple[str, ...]:
        allowed = set(self.parameter_names(model, phase))
        model.eval()  # freezes Dropout and every BN running statistic
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in allowed)
        return tuple(sorted(allowed))


@dataclass(frozen=True)
class FoldMetrics:
    balanced_accuracy: float
    nll: float
    true_class_margin: float
    fold_variance: float
    source_distance: float
    non_degrading_fold_fraction: float

    def __post_init__(self) -> None:
        for name in (
            "balanced_accuracy",
            "nll",
            "true_class_margin",
            "fold_variance",
            "source_distance",
            "non_degrading_fold_fraction",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


class GroupedTargetCVSelector:
    """Grouped folds and the report's deterministic hierarchical selector."""

    def __init__(self, folds: int = 4, seed: int = 392002):
        if isinstance(folds, bool) or int(folds) < 2:
            raise ValueError("folds must be at least two")
        self.folds = int(folds)
        self.seed = int(seed)

    def split(
        self, *, labels: Tensor, groups: Sequence[str] | None
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if not groups:
            buckets: list[list[int]] = [[] for _ in range(self.folds)]
            rng = random.Random(self.seed)
            for class_id in torch.unique(labels, sorted=True).tolist():
                indices = torch.nonzero(labels == int(class_id), as_tuple=False).flatten().tolist()
                if len(indices) < self.folds:
                    raise ValueError("every class must have at least folds rows for stratified fallback")
                rng.shuffle(indices)
                offset = rng.randrange(self.folds)
                for position, index in enumerate(indices):
                    buckets[(position + offset) % self.folds].append(int(index))
            rows = set(range(int(labels.numel())))
            result = []
            for bucket in buckets:
                val = tuple(sorted(bucket))
                train = tuple(sorted(rows.difference(val)))
                if not train or not val:
                    raise ValueError("every stratified fold must contain train and validation rows")
                result.append((train, val))
            return tuple(result)
        if labels.numel() != len(groups):
            raise ValueError("labels and groups must be row aligned")
        unique_groups = sorted(set(groups))
        if len(unique_groups) < self.folds:
            raise ValueError("unique group count must be at least folds")
        rng = random.Random(self.seed)
        rng.shuffle(unique_groups)
        buckets = [unique_groups[index :: self.folds] for index in range(self.folds)]
        rows = tuple(range(int(labels.numel())))
        result = []
        for bucket in buckets:
            validation_groups = set(bucket)
            val = tuple(index for index in rows if groups[index] in validation_groups)
            train = tuple(index for index in rows if groups[index] not in validation_groups)
            if not train or not val:
                raise ValueError("every grouped fold must contain train and validation rows")
            if {groups[index] for index in train} & {groups[index] for index in val}:
                raise RuntimeError("group leakage detected")
            result.append((train, val))
        return tuple(result)

    @staticmethod
    def choose(*, frozen: FoldMetrics, adapted: FoldMetrics) -> str:
        if adapted.non_degrading_fold_fraction <= 0.5:
            return "zero_adapt"
        tolerance = 1.0e-12
        nll_improved = frozen.nll - adapted.nll > tolerance
        accuracy_improved = adapted.balanced_accuracy - frozen.balanced_accuracy > tolerance
        margin_improved = adapted.true_class_margin - frozen.true_class_margin > tolerance
        return "adapted" if nll_improved and (accuracy_improved or margin_improved) else "zero_adapt"


class CheckpointAverager:
    def __init__(self, top_k: int = 3):
        if isinstance(top_k, bool) or int(top_k) <= 0:
            raise ValueError("top_k must be a positive integer")
        self.top_k = int(top_k)

    def average(
        self,
        states: Sequence[tuple[Mapping[str, Tensor], float | tuple[float, ...]]],
    ) -> dict[str, Tensor]:
        if not states:
            raise ValueError("at least one checkpoint state is required")
        selected = sorted(states, key=lambda item: item[1], reverse=True)[: self.top_k]
        keys = tuple(selected[0][0].keys())
        if any(tuple(state.keys()) != keys for state, _ in selected[1:]):
            raise ValueError("checkpoint state keys must match")
        averaged: dict[str, Tensor] = {}
        for key in keys:
            tensors = [state[key] for state, _ in selected]
            if any(value.shape != tensors[0].shape or value.dtype != tensors[0].dtype for value in tensors):
                raise ValueError(f"checkpoint tensor mismatch for {key!r}")
            if tensors[0].is_floating_point():
                averaged[key] = torch.stack([value.detach() for value in tensors]).mean(dim=0)
            else:
                if any(not torch.equal(value, tensors[0]) for value in tensors[1:]):
                    raise ValueError(f"non-floating checkpoint tensor changed for {key!r}")
                averaged[key] = tensors[0].detach().clone()
        return averaged


@dataclass(frozen=True)
class SFTAPFTAudit:
    method: str
    permission: str
    total_steps: int
    phase_steps: tuple[int, int, int]
    trainable_names_by_phase: Mapping[str, tuple[str, ...]]
    updated_parameter_names: tuple[str, ...]
    support_losses: tuple[float, ...]
    source_loader_opened: bool
    source_samples_opened: bool
    source_cache_opened: bool
    target_eval_opened: bool
    query_opened: bool
    bn_running_stats_updated: bool
    checkpoint_selection_role: str


@dataclass(frozen=True)
class SFTAPFTResult:
    model: nn.Module
    head: TargetPrototypeHead
    audit: SFTAPFTAudit


@dataclass(frozen=True)
class SFTAPFTFoldRow:
    fold: int
    train_groups: frozenset[str]
    validation_groups: frozenset[str]
    frozen_balanced_accuracy: float
    adapted_balanced_accuracy: float
    frozen_nll: float
    adapted_nll: float
    frozen_margin: float
    adapted_margin: float
    source_distance: float
    query_opened: bool = False


@dataclass(frozen=True)
class SFTAPFTSelectionResult:
    selected: str
    frozen_metrics: FoldMetrics
    adapted_metrics: FoldMetrics
    fold_rows: tuple[SFTAPFTFoldRow, ...]
    adapted_result: SFTAPFTResult | None


def _forward_aux(model: nn.Module, values: Tensor) -> Mapping[str, Any]:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError) as exc:
        raise ValueError("cannot inspect model.forward") from exc
    kwargs: dict[str, Any] = {}
    if "return_aux" in parameters:
        kwargs["return_aux"] = True
    for label_name in ("y", "y_tx"):
        if label_name in parameters:
            kwargs[label_name] = None
            break
    outputs = model(values, **kwargs)
    if not isinstance(outputs, Mapping):
        raise ValueError("SF-TAPFT model must return an auxiliary mapping")
    return outputs


def _extract_joint_embedding(outputs: Mapping[str, Any], batch_size: int) -> Tensor:
    nested = outputs.get("aux_id")
    if isinstance(nested, Mapping):
        value = nested.get("feat_joint")
        if torch.is_tensor(value):
            if value.ndim != 2 or value.size(0) != batch_size:
                raise ValueError("aux_id.feat_joint must have shape [batch, dimension]")
            return value
    for name in ("feat_joint", "z_id", "feat_cls", "embedding"):
        value = outputs.get(name)
        if torch.is_tensor(value):
            if value.ndim != 2 or value.size(0) != batch_size:
                raise ValueError(f"{name} must have shape [batch, dimension]")
            return value
    raise ValueError("model output must expose feat_joint, z_id, feat_cls or embedding")


def _source_classifier_weight(model: nn.Module) -> Tensor:
    backbone, _ = _identity_backbone(model)
    try:
        weight = backbone.cls_head.head.weight
    except AttributeError as exc:
        raise ValueError("model must expose cls_head.head.weight") from exc
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise ValueError("cls_head.head.weight must be a matrix")
    return weight.detach().clone()


def _local_labels(labels: Tensor, class_ids: Sequence[int], device: torch.device) -> Tensor:
    ids = torch.tensor(tuple(class_ids), dtype=torch.long, device=device)
    labels = labels.to(device=device, dtype=torch.long)
    matches = labels[:, None] == ids[None, :]
    if not bool(matches.any(dim=1).all()):
        raise ValueError("target labels do not map to the target classifier")
    return matches.long().argmax(dim=1)


def _target_prototypes(embeddings: Tensor, labels: Tensor, class_ids: Sequence[int]) -> Tensor:
    rows = []
    for class_id in class_ids:
        members = embeddings[labels == int(class_id)]
        if members.size(0) == 0:
            raise ValueError("every target class must have at least one support sample")
        rows.append(F.normalize(members.mean(dim=0), dim=0, eps=_EPS))
    return torch.stack(rows)


def _class_balanced_weights(labels: Tensor, class_count: int) -> Tensor:
    counts = torch.bincount(labels, minlength=class_count).to(dtype=torch.float32)
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = float(labels.numel()) / (float(present.sum()) * counts[present])
    return weights.to(device=labels.device)


def _selective_kd_loss(
    teacher_logits: Tensor,
    student_logits: Tensor,
    local_labels: Tensor,
    *,
    source_class_count: int,
    temperature: float,
    gamma: float,
) -> Tensor:
    valid = local_labels < int(source_class_count)
    if not bool(valid.any()):
        return student_logits.new_zeros(())
    teacher = teacher_logits[valid, :source_class_count].detach()
    student = student_logits[valid, :source_class_count]
    targets = local_labels[valid]
    reliability = teacher.softmax(dim=1).gather(1, targets[:, None]).squeeze(1).pow(float(gamma))
    teacher_soft = (teacher / float(temperature)).softmax(dim=1)
    student_log = (student / float(temperature)).log_softmax(dim=1)
    row_kl = F.kl_div(student_log, teacher_soft, reduction="none").sum(dim=1)
    return float(temperature) ** 2 * (reliability * row_kl).sum() / (reliability.sum() + _EPS)


def _learning_rate_factor(step: int, total_steps: int, warmup_ratio: float) -> float:
    warmup_steps = int(round(total_steps * float(warmup_ratio)))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    # Keep the final scheduled update non-zero. The checkpoint after the last
    # optimizer step is the candidate, so reaching exactly zero one step too
    # early would silently skip a one-step final phase in small diagnostic runs.
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _phase_for_step(step: int, phase_steps: tuple[int, int, int]) -> str:
    if step < phase_steps[0]:
        return "A"
    if step < phase_steps[0] + phase_steps[1]:
        return "B"
    return "C"


def _group_base_lrs(config: SFTAPFTConfig, phase: str) -> dict[str, float]:
    if phase == "A":
        return {"head": config.lr_head_initial, "norm": config.lr_norm, "adapter": 0.0, "last": 0.0}
    if phase == "B":
        return {"head": config.lr_head_middle, "norm": config.lr_norm, "adapter": config.lr_adapter, "last": 0.0}
    return {
        "head": config.lr_head_late,
        "norm": config.lr_norm,
        "adapter": config.lr_adapter_late,
        "last": config.lr_last_block,
    }


def fit_sf_tapft(
    checkpoint_model: nn.Module,
    target_train: TargetOnlyAdaptationDataset,
    config: SFTAPFTConfig | None = None,
    *,
    checkpoint_validation: TargetOnlyAdaptationDataset | None = None,
) -> SFTAPFTResult:
    """Fit report-parity SF-TAPFT using target train only.

    The signature deliberately has no source/eval/query argument. The returned
    classifier is persistent, so the audit always labels this implementation
    diagnostic and non-formal under the current project protocol.
    """

    if not isinstance(checkpoint_model, nn.Module):
        raise TypeError("checkpoint_model must be a torch module")
    if not isinstance(target_train, TargetOnlyAdaptationDataset):
        raise TypeError("target_train must be TargetOnlyAdaptationDataset")
    if checkpoint_validation is not None:
        if not isinstance(checkpoint_validation, TargetOnlyAdaptationDataset):
            raise TypeError("checkpoint_validation must be TargetOnlyAdaptationDataset")
        if set(target_train.physical_ids) & set(checkpoint_validation.physical_ids):
            raise ValueError("target train and checkpoint validation physical IDs must be disjoint")
    config = config or SFTAPFTConfig()
    if not isinstance(config, SFTAPFTConfig):
        raise TypeError("config must be SFTAPFTConfig")
    if config.checkpoint_average_top_k > 1 and checkpoint_validation is None:
        raise ValueError("checkpoint_average_top_k > 1 requires target inner validation")
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))

    teacher = copy.deepcopy(checkpoint_model)
    student = copy.deepcopy(checkpoint_model)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    source_weights = _source_classifier_weight(teacher)
    source_class_ids = tuple(range(int(source_weights.size(0))))
    ensure_time_adapter(student, rank=config.adapter_rank)
    student.eval()
    device = next(student.parameters()).device
    dtype = next(parameter.dtype for parameter in student.parameters() if parameter.is_floating_point())
    target_x = target_train.received_iq.to(device=device, dtype=dtype)
    target_labels = target_train.labels.to(device=device, dtype=torch.long)

    with torch.no_grad():
        initial_embeddings = _extract_joint_embedding(
            _forward_aux(student, target_x), int(target_x.size(0))
        )
        target_class_ids = target_train.class_ids
        prototypes = _target_prototypes(initial_embeddings, target_labels, target_class_ids)
    head = TargetPrototypeHead.from_source_and_target(
        source_weights=source_weights.to(device=device, dtype=dtype),
        target_prototypes=prototypes,
        source_class_ids=source_class_ids,
        target_class_ids=target_class_ids,
        rho=config.classifier_source_target_interpolation,
        scale=config.prototype_scale,
    ).to(device=device, dtype=dtype)
    local_labels = _local_labels(target_labels, head.class_ids, device)
    class_weights = _class_balanced_weights(local_labels, len(head.class_ids)).to(dtype=dtype)

    policy = ProgressiveTrainabilityPolicy()
    phase_names = {phase: policy.parameter_names(student, phase) for phase in _PHASES}
    norm_names = set(phase_names["A"])
    _, identity_prefix = _identity_backbone(student)
    adapter_prefix = f"{identity_prefix}meta_adapter_time."
    adapter_names = {name for name in phase_names["B"] if name.startswith(adapter_prefix)}
    last_names = set(phase_names["C"]) - norm_names - adapter_names
    named = dict(student.named_parameters())
    groups = [
        {"name": "head", "params": list(head.parameters()), "lr": config.lr_head_initial},
        {"name": "norm", "params": [named[name] for name in sorted(norm_names)], "lr": config.lr_norm},
        {"name": "adapter", "params": [named[name] for name in sorted(adapter_names)], "lr": config.lr_adapter},
        {"name": "last", "params": [named[name] for name in sorted(last_names)], "lr": config.lr_last_block},
    ]
    if any(not group["params"] for group in groups):
        missing = [str(group["name"]) for group in groups if not group["params"]]
        raise ValueError(f"SF-TAPFT trainability group is empty: {missing}")
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.weight_decay))
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    l2sp_names = sorted(norm_names | last_names)
    l2sp = L2SPRegularizer.from_named_parameters((name, named[name]) for name in l2sp_names)
    initial_state = {name: value.detach().clone() for name, value in student.state_dict().items()}
    bn_before = {
        name: value.detach().clone()
        for name, value in student.state_dict().items()
        if name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked")
    }
    total_steps = sum(config.phase_steps)
    losses: list[float] = []
    snapshots: list[tuple[dict[str, Tensor], float | tuple[float, ...]]] = []
    current_phase = ""

    for step in range(total_steps):
        phase = _phase_for_step(step, config.phase_steps)
        if phase != current_phase:
            policy.apply(student, phase)
            for parameter in head.parameters():
                parameter.requires_grad_(True)
            current_phase = phase
        lr_factor = _learning_rate_factor(step, total_steps, config.warmup_ratio)
        base_lrs = _group_base_lrs(config, phase)
        for group in optimizer.param_groups:
            group["lr"] = float(base_lrs[str(group["name"])]) * lr_factor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = _forward_aux(student, target_x)
            embeddings = _extract_joint_embedding(outputs, int(target_x.size(0)))
            logits = head(embeddings)
            ce = F.cross_entropy(
                logits,
                local_labels,
                weight=class_weights,
                label_smoothing=float(config.label_smoothing),
            )
            proto_logits = leave_one_out_prototype_logits(
                embeddings,
                local_labels,
                class_count=len(head.class_ids),
                fallback_weights=head.weight,
                scale=config.prototype_scale,
            )
            proto = F.cross_entropy(proto_logits, local_labels, weight=class_weights)
            anchor = l2sp(student.named_parameters())
            kd = logits.new_zeros(())
            if config.selective_kd_weight > 0.0:
                with torch.no_grad():
                    teacher_outputs = _forward_aux(teacher, target_x)
                    teacher_logits = teacher_outputs.get("tx_logits", teacher_outputs.get("logits"))
                    if not torch.is_tensor(teacher_logits):
                        raise ValueError("teacher output must expose logits for selective KD")
                kd = _selective_kd_loss(
                    teacher_logits,
                    logits,
                    local_labels,
                    source_class_count=int(source_weights.size(0)),
                    temperature=config.selective_kd_temperature,
                    gamma=config.selective_kd_gamma,
                )
            loss = (
                ce
                + float(config.lambda_proto) * proto
                + float(config.lambda_l2sp) * anchor
                + float(config.selective_kd_weight) * kd
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("SF-TAPFT loss became non-finite")
        scaler.scale(loss).backward()
        trainable = [
            parameter
            for parameter in list(student.parameters()) + list(head.parameters())
            if parameter.requires_grad
        ]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, float(config.gradient_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach())
        losses.append(loss_value)
        if checkpoint_validation is None:
            score: float | tuple[float, ...] = -loss_value
        else:
            with torch.no_grad():
                validation_x = checkpoint_validation.received_iq.to(device=device, dtype=dtype)
                validation_embeddings = _extract_joint_embedding(
                    _forward_aux(student, validation_x), len(checkpoint_validation.physical_ids)
                )
                validation_logits = head(validation_embeddings)
                validation_labels = _local_labels(
                    checkpoint_validation.labels, head.class_ids, device
                )
                validation_accuracy, validation_nll, validation_margin = _classification_metrics(
                    validation_logits, validation_labels
                )
                score = (
                    validation_accuracy,
                    -validation_nll,
                    validation_margin,
                    -_checkpoint_distance(teacher, student),
                )
        qualifies = (
            len(snapshots) < int(config.checkpoint_average_top_k)
            or score > min(item[1] for item in snapshots)
        )
        if qualifies:
            combined = {
                **{f"model.{name}": value.detach().clone() for name, value in student.state_dict().items()},
                **{f"head.{name}": value.detach().clone() for name, value in head.state_dict().items()},
            }
            snapshots.append((combined, score))
            snapshots.sort(key=lambda item: item[1], reverse=True)
            del snapshots[int(config.checkpoint_average_top_k) :]

    averaged = CheckpointAverager(config.checkpoint_average_top_k).average(snapshots)
    student.load_state_dict(
        {name.removeprefix("model."): value for name, value in averaged.items() if name.startswith("model.")}
    )
    head.load_state_dict(
        {name.removeprefix("head."): value for name, value in averaged.items() if name.startswith("head.")}
    )
    student.eval()
    head.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    updated = tuple(
        sorted(
            name
            for name, value in student.state_dict().items()
            if name not in initial_state or not torch.equal(value, initial_state[name])
        )
    )
    bn_after = student.state_dict()
    bn_updated = any(not torch.equal(value, bn_after[name]) for name, value in bn_before.items())
    audit = SFTAPFTAudit(
        method="sf_tapft_v1",
        permission="DIAGNOSTIC_NON_FORMAL",
        total_steps=total_steps,
        phase_steps=config.phase_steps,
        trainable_names_by_phase=MappingProxyType(phase_names),
        updated_parameter_names=updated,
        support_losses=tuple(losses),
        source_loader_opened=False,
        source_samples_opened=False,
        source_cache_opened=False,
        target_eval_opened=False,
        query_opened=False,
        bn_running_stats_updated=bn_updated,
        checkpoint_selection_role=(
            "target_inner_validation" if checkpoint_validation is not None else "target_train_loss_single"
        ),
    )
    return SFTAPFTResult(model=student, head=head, audit=audit)


def _subset_target_train(
    dataset: TargetOnlyAdaptationDataset, indices: Sequence[int]
) -> TargetOnlyAdaptationDataset:
    index = torch.tensor(tuple(indices), dtype=torch.long)
    return TargetOnlyAdaptationDataset(
        received_iq=dataset.received_iq[index],
        labels=dataset.labels[index],
        physical_ids=tuple(dataset.physical_ids[item] for item in indices),
        groups=(tuple(dataset.groups[item] for item in indices) if dataset.groups else None),
        physical_id_origin=dataset.physical_id_origin,
    )


def _classification_metrics(logits: Tensor, labels: Tensor) -> tuple[float, float, float]:
    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.numel():
        raise ValueError("metric logits and labels must be row aligned")
    predictions = logits.argmax(dim=1)
    per_class = []
    for class_index in torch.unique(labels, sorted=True):
        mask = labels == class_index
        per_class.append((predictions[mask] == labels[mask]).float().mean())
    balanced = float(torch.stack(per_class).mean())
    nll = float(F.cross_entropy(logits, labels))
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    margin = float((true_logits - masked.max(dim=1).values).mean())
    return balanced, nll, margin


def _checkpoint_distance(reference: nn.Module, adapted: nn.Module) -> float:
    left = reference.state_dict()
    right = adapted.state_dict()
    values = []
    for name, baseline in left.items():
        current = right.get(name)
        if current is None or not baseline.is_floating_point():
            continue
        values.append((current.detach().cpu() - baseline.detach().cpu()).pow(2).mean())
    return float(torch.stack(values).sum()) if values else 0.0


@torch.no_grad()
def _frozen_validation_logits(model: nn.Module, dataset: TargetOnlyAdaptationDataset) -> tuple[Tensor, Tensor]:
    model = copy.deepcopy(model).eval()
    device = next(model.parameters()).device
    dtype = next(parameter.dtype for parameter in model.parameters() if parameter.is_floating_point())
    outputs = _forward_aux(model, dataset.received_iq.to(device=device, dtype=dtype))
    logits = outputs.get("tx_logits", outputs.get("logits"))
    if not torch.is_tensor(logits):
        raise ValueError("frozen model output must expose tx_logits or logits")
    labels = dataset.labels.to(device=device, dtype=torch.long)
    if labels.numel() and int(labels.max()) >= logits.size(1):
        raise ValueError("grouped SF-TAPFT selection currently requires old registered target classes")
    return logits, labels


@torch.no_grad()
def _adapted_validation_logits(
    result: SFTAPFTResult, dataset: TargetOnlyAdaptationDataset
) -> tuple[Tensor, Tensor]:
    model = result.model
    head = result.head
    device = next(model.parameters()).device
    dtype = next(parameter.dtype for parameter in model.parameters() if parameter.is_floating_point())
    embeddings = _extract_joint_embedding(
        _forward_aux(model, dataset.received_iq.to(device=device, dtype=dtype)),
        len(dataset.physical_ids),
    )
    logits = head(embeddings)
    labels = _local_labels(dataset.labels, head.class_ids, device)
    return logits, labels


def _aggregate_fold_metrics(
    rows: Sequence[SFTAPFTFoldRow], *, adapted: bool
) -> FoldMetrics:
    prefix = "adapted" if adapted else "frozen"
    accuracy = torch.tensor(
        [getattr(row, f"{prefix}_balanced_accuracy") for row in rows], dtype=torch.float64
    )
    nll = torch.tensor([getattr(row, f"{prefix}_nll") for row in rows], dtype=torch.float64)
    margin = torch.tensor([getattr(row, f"{prefix}_margin") for row in rows], dtype=torch.float64)
    non_degrading = (
        sum(row.adapted_balanced_accuracy >= row.frozen_balanced_accuracy for row in rows)
        / float(len(rows))
        if adapted
        else 1.0
    )
    return FoldMetrics(
        balanced_accuracy=float(accuracy.mean()),
        nll=float(nll.mean()),
        true_class_margin=float(margin.mean()),
        fold_variance=float(accuracy.var(unbiased=False)),
        source_distance=(
            float(sum(row.source_distance for row in rows) / len(rows)) if adapted else 0.0
        ),
        non_degrading_fold_fraction=float(non_degrading),
    )


def select_sf_tapft_by_grouped_cv(
    checkpoint_model: nn.Module,
    target_train: TargetOnlyAdaptationDataset,
    config: SFTAPFTConfig | None = None,
    *,
    folds: int = 4,
) -> SFTAPFTSelectionResult:
    """Use target-train grouped OOF evidence for one domain-level fallback."""

    config = config or SFTAPFTConfig()
    selector = GroupedTargetCVSelector(folds=folds, seed=config.seed)
    splits = selector.split(labels=target_train.labels, groups=target_train.groups)
    rows = []
    fitted_folds = []
    for fold, (train_indices, validation_indices) in enumerate(splits):
        inner_train = _subset_target_train(target_train, train_indices)
        inner_validation = _subset_target_train(target_train, validation_indices)
        fitted = fit_sf_tapft(
            copy.deepcopy(checkpoint_model),
            inner_train,
            config,
            checkpoint_validation=inner_validation,
        )
        fitted_folds.append(fitted)
        frozen_logits, frozen_labels = _frozen_validation_logits(
            checkpoint_model, inner_validation
        )
        adapted_logits, adapted_labels = _adapted_validation_logits(
            fitted, inner_validation
        )
        frozen_accuracy, frozen_nll, frozen_margin = _classification_metrics(
            frozen_logits, frozen_labels
        )
        adapted_accuracy, adapted_nll, adapted_margin = _classification_metrics(
            adapted_logits, adapted_labels
        )
        train_groups = frozenset(inner_train.groups or inner_train.physical_ids)
        validation_groups = frozenset(inner_validation.groups or inner_validation.physical_ids)
        if train_groups & validation_groups:
            raise RuntimeError("grouped target CV leaked a group")
        rows.append(
            SFTAPFTFoldRow(
                fold=fold,
                train_groups=train_groups,
                validation_groups=validation_groups,
                frozen_balanced_accuracy=frozen_accuracy,
                adapted_balanced_accuracy=adapted_accuracy,
                frozen_nll=frozen_nll,
                adapted_nll=adapted_nll,
                frozen_margin=frozen_margin,
                adapted_margin=adapted_margin,
                source_distance=_checkpoint_distance(checkpoint_model, fitted.model),
            )
        )
    fold_rows = tuple(rows)
    frozen_metrics = _aggregate_fold_metrics(fold_rows, adapted=False)
    adapted_metrics = _aggregate_fold_metrics(fold_rows, adapted=True)
    selected = selector.choose(frozen=frozen_metrics, adapted=adapted_metrics)
    # The deployment candidate is fixed to the first seeded fold. Its temporal
    # checkpoint average was selected only on that fold's disjoint target-inner
    # validation rows; OOF evidence from all folds decides adapted vs fallback.
    adapted_result = fitted_folds[0] if selected == "adapted" else None
    return SFTAPFTSelectionResult(
        selected=selected,
        frozen_metrics=frozen_metrics,
        adapted_metrics=adapted_metrics,
        fold_rows=fold_rows,
        adapted_result=adapted_result,
    )


__all__ = [
    "CheckpointAverager",
    "FoldMetrics",
    "GroupedTargetCVSelector",
    "L2SPRegularizer",
    "ProgressiveTrainabilityPolicy",
    "SFTAPFTAudit",
    "SFTAPFTConfig",
    "SFTAPFTResult",
    "SFTAPFTFoldRow",
    "SFTAPFTSelectionResult",
    "TargetOnlyAdaptationDataset",
    "TargetPrototypeHead",
    "ensure_time_adapter",
    "fit_sf_tapft",
    "leave_one_out_prototype_logits",
    "select_sf_tapft_by_grouped_cv",
]
