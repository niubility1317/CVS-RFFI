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
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .meta_adapter import ResidualMetaAdapter


_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_TRAINABILITY_PROFILES = (
    "p0_head_only",
    "p1_head_norm",
    "p2_time_adapter",
    "p3_full_t3",
    "p4_time_fusion",
)
_NORM_SCOPES = {
    "all": ("time_fuse", "t1", "t2", "t3"),
    "t3": ("t3",),
    "t2_t3": ("t2", "t3"),
    "backbone_no_fuse": ("t1", "t2", "t3"),
    "fuse": ("time_fuse",),
    "t1": ("t1",),
    "t2": ("t2",),
    "t3_fuse": ("time_fuse", "t3"),
    "t2_t3_fuse": ("time_fuse", "t2", "t3"),
}
_NORM_AFFINES = ("weight_bias", "weight", "bias")
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
    trainability_profile: str = "p3_full_t3"
    norm_scope: str = "all"
    norm_affine: str = "weight_bias"
    norm_rules: tuple[tuple[str, str], ...] = ()
    classifier_source_target_interpolation: float = 0.5
    prototype_scale: float = 8.0
    inference_temperature: float = 1.0
    label_smoothing: float = 0.05
    lambda_proto: float = 0.5
    lambda_l2sp: float = 1.0e-4
    selective_kd_weight: float = 0.0
    selective_kd_temperature: float = 2.0
    selective_kd_gamma: float = 2.0
    phase_steps: tuple[int, int, int] = (500, 1500, 2500)
    scheduler_reference_steps: int = 0
    lr_head_initial: float = 1.0e-3
    lr_norm: float = 1.0e-4
    lr_head_middle: float = 3.0e-4
    lr_adapter: float = 3.0e-4
    lr_head_late: float = 1.0e-4
    lr_adapter_late: float = 1.0e-4
    lr_last_block: float = 3.0e-5
    lr_fusion: float = 1.0e-5
    weight_decay: float = 1.0e-4
    warmup_ratio: float = 0.05
    head_prefit_steps: int = 0
    validation_steps: tuple[int, ...] = ()
    oof_temperature_calibration: bool = False
    fast_tail_start_step: int = 0
    fast_tail_steps: int = 0
    fast_tail_lr_head_start: float = 2.0e-4
    fast_tail_lr_head_end: float = 2.0e-5
    fast_tail_lr_norm_start: float = 3.0e-5
    fast_tail_lr_norm_end: float = 3.0e-6
    head_polish_steps: int = 0
    head_polish_lr: float = 5.0e-5
    trainable_delta_ema_decay: float = 0.0
    use_class_adaptive_rho: bool = False
    class_adaptive_rho_min: float = 0.25
    class_adaptive_rho_max: float = 0.75
    class_adaptive_rho_temperature: float = 0.10
    head_anchor_weight: float = 0.0
    gradient_clip_norm: float = 1.0
    checkpoint_average_top_k: int = 3
    mixed_precision: bool = True
    seed: int = 392002

    def __post_init__(self) -> None:
        if self.trainability_profile not in _TRAINABILITY_PROFILES:
            raise ValueError(
                f"trainability_profile must be one of {_TRAINABILITY_PROFILES}"
            )
        if self.norm_scope not in _NORM_SCOPES:
            raise ValueError(f"norm_scope must be one of {tuple(_NORM_SCOPES)}")
        if self.norm_affine not in _NORM_AFFINES:
            raise ValueError(f"norm_affine must be one of {_NORM_AFFINES}")
        normalized_rules = tuple((str(scope), str(affine)) for scope, affine in self.norm_rules)
        for scope, affine in normalized_rules:
            if scope not in {name for values in _NORM_SCOPES.values() for name in values}:
                raise ValueError(f"norm_rules contains unknown norm scope: {scope!r}")
            if affine not in _NORM_AFFINES:
                raise ValueError(f"norm_rules contains unknown norm affine: {affine!r}")
        if len({scope for scope, _ in normalized_rules}) != len(normalized_rules):
            raise ValueError("norm_rules must not repeat a norm scope")
        if isinstance(self.adapter_rank, bool) or int(self.adapter_rank) <= 0:
            raise ValueError("adapter_rank must be a positive integer")
        if len(self.phase_steps) != 3 or any(
            isinstance(value, bool) or int(value) < 0 for value in self.phase_steps
        ):
            raise ValueError("phase_steps must contain three non-negative integers")
        if sum(int(value) for value in self.phase_steps) <= 0:
            raise ValueError("phase_steps must contain at least one optimizer step")
        total_steps = sum(int(value) for value in self.phase_steps)
        if (
            isinstance(self.head_prefit_steps, bool)
            or int(self.head_prefit_steps) < 0
            or int(self.head_prefit_steps) > total_steps
        ):
            raise ValueError("head_prefit_steps must be in [0, total_steps]")
        validation_steps = tuple(int(value) for value in self.validation_steps)
        if any(isinstance(value, bool) for value in self.validation_steps):
            raise ValueError("validation_steps must contain integer optimizer steps")
        if validation_steps and (
            tuple(sorted(set(validation_steps))) != validation_steps
            or validation_steps[0] < 1
            or validation_steps[-1] > total_steps
            or validation_steps[-1] != total_steps
        ):
            raise ValueError(
                "validation_steps must be sorted, unique, in range, and include the final step"
            )
        if (
            isinstance(self.scheduler_reference_steps, bool)
            or int(self.scheduler_reference_steps) < 0
            or (
                int(self.scheduler_reference_steps) > 0
                and int(self.scheduler_reference_steps) < total_steps
            )
        ):
            raise ValueError(
                "scheduler_reference_steps must be zero or at least the training step count"
            )
        integer_controls = {
            "fast_tail_start_step": self.fast_tail_start_step,
            "fast_tail_steps": self.fast_tail_steps,
            "head_polish_steps": self.head_polish_steps,
        }
        for name, value in integer_controls.items():
            if isinstance(value, bool) or int(value) < 0 or int(value) > total_steps:
                raise ValueError(f"{name} must be an integer in [0, total_steps]")
        if bool(self.fast_tail_steps) != bool(self.fast_tail_start_step):
            raise ValueError("fast tail start and steps must either both be zero or both be positive")
        if self.fast_tail_steps and self.fast_tail_start_step + self.fast_tail_steps > total_steps:
            raise ValueError("fast tail schedule exceeds total_steps")
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
            "inference_temperature",
            "selective_kd_temperature",
            "gradient_clip_norm",
            "lr_head_initial",
            "lr_norm",
            "lr_head_middle",
            "lr_adapter",
            "lr_head_late",
            "lr_adapter_late",
            "lr_last_block",
            "lr_fusion",
            "fast_tail_lr_head_start",
            "fast_tail_lr_head_end",
            "fast_tail_lr_norm_start",
            "fast_tail_lr_norm_end",
            "head_polish_lr",
            "class_adaptive_rho_temperature",
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
            "trainable_delta_ema_decay",
            "head_anchor_weight",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if isinstance(self.checkpoint_average_top_k, bool) or int(self.checkpoint_average_top_k) <= 0:
            raise ValueError("checkpoint_average_top_k must be a positive integer")
        if not isinstance(self.mixed_precision, bool):
            raise ValueError("mixed_precision must be a boolean")
        if not isinstance(self.oof_temperature_calibration, bool):
            raise ValueError("oof_temperature_calibration must be a boolean")
        if not isinstance(self.use_class_adaptive_rho, bool):
            raise ValueError("use_class_adaptive_rho must be a boolean")
        if not 0.0 <= float(self.trainable_delta_ema_decay) < 1.0:
            raise ValueError("trainable_delta_ema_decay must be in [0, 1)")
        if not 0.0 <= float(self.class_adaptive_rho_min) <= float(self.class_adaptive_rho_max) <= 1.0:
            raise ValueError("class adaptive rho bounds must satisfy 0 <= min <= max <= 1")
        object.__setattr__(self, "phase_steps", tuple(int(value) for value in self.phase_steps))
        object.__setattr__(self, "norm_rules", normalized_rules)
        object.__setattr__(self, "head_prefit_steps", int(self.head_prefit_steps))
        object.__setattr__(self, "validation_steps", validation_steps)
        object.__setattr__(
            self, "scheduler_reference_steps", int(self.scheduler_reference_steps)
        )
        for name in integer_controls:
            object.__setattr__(self, name, int(getattr(self, name)))


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    nll_before: float
    nll_after: float
    argmax_preserved: bool


def fit_positive_temperature(logits: Tensor, labels: Tensor) -> TemperatureCalibration:
    """Fit one positive temperature on OOF logits without changing class order."""

    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.numel():
        raise ValueError("temperature logits and labels must be row aligned")
    if logits.numel() == 0 or not bool(torch.isfinite(logits).all()):
        raise ValueError("temperature logits must be non-empty and finite")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= logits.size(1)):
        raise ValueError("temperature labels must index logit columns")
    work_logits = logits.detach().to(device="cpu", dtype=torch.float64)
    work_labels = labels.detach().to(device="cpu", dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=80, tolerance_grad=1.0e-10, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-6.0, 6.0).exp()
        loss = F.cross_entropy(work_logits / temperature, work_labels)
        loss.backward()
        return loss

    before = float(F.cross_entropy(work_logits, work_labels))
    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-6.0, 6.0).exp())
    calibrated = work_logits / temperature
    after = float(F.cross_entropy(calibrated, work_labels))
    if after > before:
        temperature = 1.0
        calibrated = work_logits
        after = before
    return TemperatureCalibration(
        temperature=temperature,
        nll_before=before,
        nll_after=after,
        argmax_preserved=bool(torch.equal(work_logits.argmax(1), calibrated.argmax(1))),
    )


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
        rho: float | Tensor | Sequence[float],
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
        rho_values = torch.as_tensor(rho, dtype=target_prototypes.dtype, device=target_prototypes.device)
        if rho_values.ndim == 0:
            rho_values = rho_values.expand(len(target_ids))
        if rho_values.shape != (len(target_ids),) or not bool(torch.isfinite(rho_values).all()):
            raise ValueError("rho must be a finite scalar or target-class vector")
        if bool(((rho_values < 0.0) | (rho_values > 1.0)).any()):
            raise ValueError("rho must be finite in [0, 1]")
        source = F.normalize(source_weights.detach(), dim=1, eps=_EPS)
        target = F.normalize(target_prototypes.detach(), dim=1, eps=_EPS)
        target_by_id = {class_id: target[index] for index, class_id in enumerate(target_ids)}
        rho_by_id = {class_id: rho_values[index] for index, class_id in enumerate(target_ids)}
        rows: list[Tensor] = []
        output_ids: list[int] = []
        for index, class_id in enumerate(source_ids):
            row = source[index]
            if class_id in target_by_id:
                class_rho = rho_by_id[class_id]
                row = (1.0 - class_rho) * row + class_rho * target_by_id[class_id]
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
    class_sums = embeddings.new_zeros((int(class_count), embeddings.size(1)))
    class_sums.index_add_(0, labels, embeddings)
    counts = torch.bincount(labels, minlength=int(class_count)).to(
        device=embeddings.device, dtype=embeddings.dtype
    )
    loo_sums = class_sums.unsqueeze(0).expand(embeddings.size(0), -1, -1).clone()
    row_indices = torch.arange(embeddings.size(0), device=embeddings.device)
    loo_sums[row_indices, labels] -= embeddings
    loo_counts = counts.unsqueeze(0).expand(embeddings.size(0), -1).clone()
    loo_counts[row_indices, labels] -= 1.0
    means = loo_sums / loo_counts.clamp_min(1.0).unsqueeze(-1)
    means = torch.where((loo_counts > 0).unsqueeze(-1), means, fallback.unsqueeze(0))
    prototypes = F.normalize(means, dim=2, eps=_EPS)
    return float(scale) * torch.einsum("nd,ncd->nc", normalized_embeddings, prototypes)


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
    """Nested P0-P4 allowlist applied across the A/B/C schedule."""

    def __init__(
        self,
        profile: str = "p3_full_t3",
        *,
        norm_scope: str = "all",
        norm_affine: str = "weight_bias",
        norm_rules: Sequence[tuple[str, str]] = (),
    ) -> None:
        if profile not in _TRAINABILITY_PROFILES:
            raise ValueError(f"trainability_profile must be one of {_TRAINABILITY_PROFILES}")
        if norm_scope not in _NORM_SCOPES:
            raise ValueError(f"norm_scope must be one of {tuple(_NORM_SCOPES)}")
        if norm_affine not in _NORM_AFFINES:
            raise ValueError(f"norm_affine must be one of {_NORM_AFFINES}")
        self.profile = profile
        self.norm_scope = norm_scope
        self.norm_affine = norm_affine
        self.norm_rules = tuple((str(scope), str(affine)) for scope, affine in norm_rules)
        valid_rule_scopes = {name for values in _NORM_SCOPES.values() for name in values}
        if any(scope not in valid_rule_scopes for scope, _ in self.norm_rules):
            raise ValueError("norm_rules contains an unknown norm scope")
        if any(affine not in _NORM_AFFINES for _, affine in self.norm_rules):
            raise ValueError("norm_rules contains an unknown norm affine")
        if len({scope for scope, _ in self.norm_rules}) != len(self.norm_rules):
            raise ValueError("norm_rules must not repeat a norm scope")

    def parameter_names(self, model: nn.Module, phase: str) -> tuple[str, ...]:
        if phase not in _PHASES:
            raise ValueError("phase must be A, B or C")
        _, prefix = _identity_backbone(model)
        norm_prefix_by_scope = {
            "time_fuse": f"{prefix}time_fuse.1.",
            "t1": f"{prefix}t1.norm.",
            "t2": f"{prefix}t2.norm.",
            "t3": f"{prefix}t3.norm.",
        }
        norm_rule_by_prefix = (
            {norm_prefix_by_scope[scope]: affine for scope, affine in self.norm_rules}
            if self.norm_rules
            else {
                norm_prefix_by_scope[name]: self.norm_affine
                for name in _NORM_SCOPES[self.norm_scope]
            }
        )
        norm_prefixes = tuple(norm_rule_by_prefix)
        adapter_prefix = f"{prefix}meta_adapter_time."
        last_prefix = f"{prefix}t3."
        p4_prefixes = (
            f"{prefix}t2.pw.",
            f"{prefix}time_fuse.0.",
            f"{prefix}fuse.",
            f"{prefix}cls_head.id_proj.",
        )
        profile_index = _TRAINABILITY_PROFILES.index(self.profile)
        names = []
        for name, _ in model.named_parameters():
            matched_norm_prefix = next(
                (candidate for candidate in norm_prefixes if name.startswith(candidate)), None
            )
            norm = matched_norm_prefix is not None
            if norm and norm_rule_by_prefix[matched_norm_prefix] != "weight_bias":
                norm = name.endswith(f".{norm_rule_by_prefix[matched_norm_prefix]}")
            adapter = name.startswith(adapter_prefix)
            last_block = name.startswith(last_prefix)
            p4_extra = any(name.startswith(candidate) for candidate in p4_prefixes)
            if (
                (profile_index >= 1 and norm)
                or (profile_index >= 2 and phase in {"B", "C"} and adapter)
                or (profile_index >= 3 and phase == "C" and last_block)
                or (profile_index >= 4 and phase == "C" and p4_extra)
            ):
                names.append(name)
        if profile_index >= 1 and not any(
            any(name.startswith(candidate) for candidate in norm_prefixes)
            for name in names
        ):
            raise ValueError("model must expose the requested norm affine parameters")
        if (
            profile_index >= 2
            and phase in {"B", "C"}
            and not any(name.startswith(adapter_prefix) for name in names)
        ):
            raise ValueError("model must expose the SF-TAPFT time adapter")
        if profile_index >= 4 and phase == "C" and not any(
            name.startswith(f"{prefix}t2.pw.") for name in names
        ):
            raise ValueError("P4 model must expose t2.pw parameters")
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
    class_floor: float = 0.0
    ece: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "balanced_accuracy",
            "nll",
            "true_class_margin",
            "fold_variance",
            "source_distance",
            "non_degrading_fold_fraction",
            "class_floor",
            "ece",
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


class TrainableDeltaEMA:
    """EMA only the explicitly permitted floating-point deltas from an anchor."""

    def __init__(
        self,
        anchor_state: Mapping[str, Tensor],
        *,
        permitted_names: Iterable[str],
        decay: float,
    ) -> None:
        decay = float(decay)
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("decay must be finite in [0, 1)")
        self._anchor = {name: value.detach().clone() for name, value in anchor_state.items()}
        self._permitted = frozenset(str(name) for name in permitted_names)
        if self._permitted.difference(self._anchor):
            raise ValueError("permitted names must exist in anchor_state")
        if any(not self._anchor[name].is_floating_point() for name in self._permitted):
            raise ValueError("permitted EMA state must be floating point")
        self._decay = decay
        self._delta: dict[str, Tensor] | None = None

    def update(self, state: Mapping[str, Tensor]) -> None:
        if set(state) != set(self._anchor):
            raise ValueError("EMA state keys must match anchor_state")
        current = {
            name: state[name].detach().to(dtype=torch.float64) - self._anchor[name].to(dtype=torch.float64)
            for name in self._permitted
        }
        if self._delta is None:
            self._delta = current
        else:
            self._delta = {
                name: self._decay * self._delta[name] + (1.0 - self._decay) * current[name]
                for name in self._permitted
            }

    def state(self) -> dict[str, Tensor]:
        if self._delta is None:
            raise ValueError("EMA requires at least one update")
        result = {name: value.detach().clone() for name, value in self._anchor.items()}
        for name in self._permitted:
            result[name] = (
                self._anchor[name].to(dtype=torch.float64) + self._delta[name]
            ).to(dtype=self._anchor[name].dtype)
        return result


class TrainableDeltaAverager:
    """Average only permitted trainable deltas against an immutable anchor."""

    def __init__(self, top_k: int = 3):
        if isinstance(top_k, bool) or int(top_k) <= 0:
            raise ValueError("top_k must be a positive integer")
        self.top_k = int(top_k)

    def average(
        self,
        states: Sequence[tuple[Mapping[str, Tensor], float | tuple[float, ...]]],
        *,
        anchor_state: Mapping[str, Tensor],
        permitted_names: Iterable[str],
    ) -> dict[str, Tensor]:
        if not states:
            raise ValueError("at least one checkpoint state is required")
        selected = sorted(states, key=lambda item: item[1], reverse=True)[: self.top_k]
        keys = tuple(anchor_state.keys())
        if any(tuple(state.keys()) != keys for state, _ in selected):
            raise ValueError("checkpoint state keys must match the anchor state")
        permitted = frozenset(str(name) for name in permitted_names)
        unknown = permitted.difference(keys)
        if unknown:
            raise ValueError(f"permitted names are absent from anchor state: {sorted(unknown)}")
        averaged: dict[str, Tensor] = {}
        for key in keys:
            anchor = anchor_state[key]
            tensors = [state[key] for state, _ in selected]
            if any(
                value.shape != anchor.shape or value.dtype != anchor.dtype for value in tensors
            ):
                raise ValueError(f"checkpoint tensor mismatch for {key!r}")
            if key not in permitted:
                averaged[key] = anchor.detach().clone()
                continue
            if not anchor.is_floating_point():
                if any(not torch.equal(value, anchor) for value in tensors):
                    raise ValueError(f"non-floating permitted checkpoint tensor changed for {key!r}")
                averaged[key] = anchor.detach().clone()
                continue
            anchor64 = anchor.detach().to(dtype=torch.float64)
            deltas = [value.detach().to(dtype=torch.float64) - anchor64 for value in tensors]
            averaged[key] = (anchor64 + torch.stack(deltas).mean(dim=0)).to(dtype=anchor.dtype)
        return averaged


@dataclass(frozen=True)
class StageValidationMetrics:
    balanced_accuracy: float
    macro_f1: float
    class_floor: float
    nll: float
    per_class_recall: tuple[float, ...]
    per_class_margin: tuple[float, ...]
    positive_flips: int
    negative_flips: int
    permitted_parameter_distance: float


@dataclass(frozen=True)
class StageValidationRow:
    phase: str
    best_step_in_phase: int
    best_global_step: int
    best_metrics: StageValidationMetrics
    end_metrics: StageValidationMetrics


@dataclass(frozen=True)
class SFTAPFTAudit:
    method: str
    permission: str
    total_steps: int
    phase_steps: tuple[int, int, int]
    trainable_names_by_phase: Mapping[str, tuple[str, ...]]
    updated_parameter_names: tuple[str, ...]
    permitted_changed_names: tuple[str, ...]
    nonpermitted_changed_names: tuple[str, ...]
    support_losses: tuple[float, ...]
    source_loader_opened: bool
    source_samples_opened: bool
    source_cache_opened: bool
    target_eval_opened: bool
    query_opened: bool
    bn_running_stats_updated: bool
    checkpoint_selection_role: str
    selected_checkpoint_steps: tuple[int, ...]
    training_sample_count: int
    stage_validation_rows: tuple[StageValidationRow, ...]
    head_prefit_steps: int
    backbone_optimizer_steps: int
    backbone_train_forward_steps: int
    validation_forward_steps: tuple[int, ...]
    snapshot_tensor_bytes: int
    trainable_parameter_elements: int
    actual_changed_elements: int
    head_polish_steps: int
    cached_head_forward_steps: int
    trainable_delta_ema_decay: float
    class_adaptive_rho: tuple[float, ...]
    class_reliability: tuple[float, ...]


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
    stage_validation_rows: tuple[StageValidationRow, ...] = ()
    query_opened: bool = False
    frozen_class_floor: float = 0.0
    adapted_class_floor: float = 0.0
    frozen_ece: float = 0.0
    adapted_ece: float = 0.0
    frozen_per_class_nll: tuple[float, ...] = ()
    adapted_per_class_nll: tuple[float, ...] = ()


@dataclass(frozen=True)
class SFTAPFTSelectionResult:
    selected: str
    frozen_metrics: FoldMetrics
    adapted_metrics: FoldMetrics
    fold_rows: tuple[SFTAPFTFoldRow, ...]
    selected_phase_steps: tuple[int, int, int]
    adapted_result: SFTAPFTResult | None
    full_support_result: SFTAPFTResult | None = None
    final_training_sample_count: int = 0
    fold0_as_final: bool = False
    temperature_calibration: TemperatureCalibration | None = None


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


def class_adaptive_rho(
    embeddings: Tensor,
    source_logits: Tensor,
    labels: Tensor,
    *,
    class_count: int,
    rho_min: float,
    rho_max: float,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Map support-only class compactness to bounded source/target mixing weights."""

    if (
        embeddings.ndim != 2
        or source_logits.ndim != 2
        or labels.ndim != 1
        or embeddings.size(0) != labels.numel()
        or source_logits.shape != (labels.numel(), int(class_count))
    ):
        raise ValueError("embeddings, source_logits and labels must be row aligned")
    if int(class_count) < 2 or labels.numel() == 0:
        raise ValueError("class_adaptive_rho requires at least two non-empty classes")
    if int(labels.min()) < 0 or int(labels.max()) >= int(class_count):
        raise ValueError("labels must index class_count")
    if not 0.0 <= float(rho_min) <= float(rho_max) <= 1.0:
        raise ValueError("rho bounds must satisfy 0 <= min <= max <= 1")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if any(not bool((labels == index).any()) for index in range(int(class_count))):
        raise ValueError("every class must have support rows")
    normalized = F.normalize(embeddings, dim=1, eps=_EPS)
    concentration = torch.stack(
        [normalized[labels == index].sum(0).norm() / float((labels == index).sum()) for index in range(int(class_count))]
    )
    true_logits = source_logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = source_logits.clone()
    other_logits.scatter_(1, labels[:, None], float("-inf"))
    margins = true_logits - other_logits.max(dim=1).values
    class_margins = torch.stack(
        [margins[labels == index].mean() for index in range(int(class_count))]
    )
    reliability = concentration * torch.sigmoid(class_margins / float(temperature))
    rho = float(rho_min) + (float(rho_max) - float(rho_min)) * (1.0 - reliability)
    return rho, reliability


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


def _target_train_snapshot_score(loss_value: float, step: int) -> float:
    """Score a target-train checkpoint without changing the fit contract."""

    del step
    return -float(loss_value)


def _phase_for_step(step: int, phase_steps: tuple[int, int, int]) -> str:
    if step < phase_steps[0]:
        return "A"
    if step < phase_steps[0] + phase_steps[1]:
        return "B"
    return "C"


def _group_base_lrs(config: SFTAPFTConfig, phase: str) -> dict[str, float]:
    if phase == "A":
        return {
            "head": config.lr_head_initial,
            "norm": config.lr_norm,
            "adapter": 0.0,
            "last": 0.0,
            "fusion": 0.0,
        }
    if phase == "B":
        return {
            "head": config.lr_head_middle,
            "norm": config.lr_norm,
            "adapter": config.lr_adapter,
            "last": 0.0,
            "fusion": 0.0,
        }
    return {
        "head": config.lr_head_late,
        "norm": config.lr_norm,
        "adapter": config.lr_adapter_late,
        "last": config.lr_last_block,
        "fusion": config.lr_fusion,
    }


def _fast_strong_group_lrs(
    config: SFTAPFTConfig, step: int, phase: str
) -> dict[str, float]:
    """Return base LRs, replacing only the pre-registered local tail window."""

    values = _group_base_lrs(config, phase)
    start = int(config.fast_tail_start_step)
    count = int(config.fast_tail_steps)
    if not count or step < start or step >= start + count:
        return values
    progress = float(step - start) / float(max(1, count - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    values["head"] = float(config.fast_tail_lr_head_end) + (
        float(config.fast_tail_lr_head_start) - float(config.fast_tail_lr_head_end)
    ) * cosine
    values["norm"] = float(config.fast_tail_lr_norm_end) + (
        float(config.fast_tail_lr_norm_start) - float(config.fast_tail_lr_norm_end)
    ) * cosine
    return values


def _make_grad_scaler(device: torch.device, *, enabled: bool):
    scaler_type = getattr(torch.amp, "GradScaler", None)
    if scaler_type is not None:
        return scaler_type(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def fit_sf_tapft(
    checkpoint_model: nn.Module,
    target_train: TargetOnlyAdaptationDataset,
    config: SFTAPFTConfig | None = None,
    *,
    checkpoint_validation: TargetOnlyAdaptationDataset | None = None,
    checkpoint_selection_mode: str = "best",
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
    if checkpoint_selection_mode not in ("best", "final_step"):
        raise ValueError("checkpoint_selection_mode must be 'best' or 'final_step'")
    if checkpoint_validation is not None:
        if not isinstance(checkpoint_validation, TargetOnlyAdaptationDataset):
            raise TypeError("checkpoint_validation must be TargetOnlyAdaptationDataset")
        if set(target_train.physical_ids) & set(checkpoint_validation.physical_ids):
            raise ValueError("target train and checkpoint validation physical IDs must be disjoint")
    config = config or SFTAPFTConfig()
    if not isinstance(config, SFTAPFTConfig):
        raise TypeError("config must be SFTAPFTConfig")
    if (
        checkpoint_selection_mode == "best"
        and config.checkpoint_average_top_k > 1
        and checkpoint_validation is None
    ):
        raise ValueError("checkpoint_average_top_k > 1 requires target inner validation")
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))

    teacher = (
        copy.deepcopy(checkpoint_model) if config.selective_kd_weight > 0.0 else None
    )
    student = copy.deepcopy(checkpoint_model)
    if teacher is not None:
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    source_weights = _source_classifier_weight(checkpoint_model)
    source_class_ids = tuple(range(int(source_weights.size(0))))
    ensure_time_adapter(student, rank=config.adapter_rank)
    student.eval()
    device = next(student.parameters()).device
    dtype = next(parameter.dtype for parameter in student.parameters() if parameter.is_floating_point())
    target_x = target_train.received_iq.to(device=device, dtype=dtype)
    target_labels = target_train.labels.to(device=device, dtype=torch.long)

    with torch.no_grad():
        initial_outputs = _forward_aux(student, target_x)
        initial_embeddings = _extract_joint_embedding(
            initial_outputs, int(target_x.size(0))
        )
        target_class_ids = target_train.class_ids
        prototypes = _target_prototypes(initial_embeddings, target_labels, target_class_ids)
        frozen_source_logits = initial_outputs.get("tx_logits", initial_outputs.get("logits"))
        if not torch.is_tensor(frozen_source_logits):
            raise ValueError("frozen source model must expose logits for SF-TAPFT")
        if max(target_class_ids) >= frozen_source_logits.size(1):
            raise ValueError("target support class is absent from frozen source logits")
        if config.use_class_adaptive_rho:
            rho_values, reliability_values = class_adaptive_rho(
                initial_embeddings,
                frozen_source_logits[:, list(target_class_ids)],
                _local_labels(target_labels, target_class_ids, device),
                class_count=len(target_class_ids),
                rho_min=config.class_adaptive_rho_min,
                rho_max=config.class_adaptive_rho_max,
                temperature=config.class_adaptive_rho_temperature,
            )
        else:
            rho_values = initial_embeddings.new_full(
                (len(target_class_ids),), config.classifier_source_target_interpolation
            )
            reliability_values = initial_embeddings.new_zeros(len(target_class_ids))
    head = TargetPrototypeHead.from_source_and_target(
        source_weights=source_weights.to(device=device, dtype=dtype),
        target_prototypes=prototypes,
        source_class_ids=source_class_ids,
        target_class_ids=target_class_ids,
        rho=rho_values,
        scale=config.prototype_scale,
    ).to(device=device, dtype=dtype)
    local_labels = _local_labels(target_labels, head.class_ids, device)
    class_weights = _class_balanced_weights(local_labels, len(head.class_ids)).to(dtype=dtype)

    policy = ProgressiveTrainabilityPolicy(
        config.trainability_profile,
        norm_scope=config.norm_scope,
        norm_affine=config.norm_affine,
        norm_rules=config.norm_rules,
    )
    phase_names = {phase: policy.parameter_names(student, phase) for phase in _PHASES}
    norm_names = set(phase_names["A"])
    _, identity_prefix = _identity_backbone(student)
    adapter_prefix = f"{identity_prefix}meta_adapter_time."
    adapter_names = {name for name in phase_names["B"] if name.startswith(adapter_prefix)}
    fusion_prefixes = (
        f"{identity_prefix}t2.pw.",
        f"{identity_prefix}time_fuse.0.",
        f"{identity_prefix}fuse.",
        f"{identity_prefix}cls_head.id_proj.",
    )
    fusion_names = {
        name
        for name in phase_names["C"]
        if any(name.startswith(prefix) for prefix in fusion_prefixes)
    }
    last_names = set(phase_names["C"]) - norm_names - adapter_names - fusion_names
    named = dict(student.named_parameters())
    initial_model_state = {
        name: value.detach().clone() for name, value in student.state_dict().items()
    }
    initial_head_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
    initial_head_weight = head.weight.detach().clone()
    head_anchor_reliability = head.weight.new_zeros(len(head.class_ids))
    for index, class_id in enumerate(target_class_ids):
        head_anchor_reliability[head.class_ids.index(class_id)] = reliability_values[index]
    validation_x: Tensor | None = None
    validation_labels: Tensor | None = None
    frozen_validation_logits: Tensor | None = None
    if checkpoint_validation is not None:
        validation_x = checkpoint_validation.received_iq.to(device=device, dtype=dtype)
        validation_labels = _local_labels(checkpoint_validation.labels, head.class_ids, device)
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        with torch.no_grad():
            frozen_validation_embeddings = _extract_joint_embedding(
                _forward_aux(student, validation_x), len(checkpoint_validation.physical_ids)
            )
            frozen_validation_logits = head(frozen_validation_embeddings).detach().clone()
        student.load_state_dict(initial_model_state)
        head.load_state_dict(initial_head_state)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)
    groups = [
        {"name": "head", "params": list(head.parameters()), "lr": config.lr_head_initial},
        {"name": "norm", "params": [named[name] for name in sorted(norm_names)], "lr": config.lr_norm},
        {"name": "adapter", "params": [named[name] for name in sorted(adapter_names)], "lr": config.lr_adapter},
        {"name": "last", "params": [named[name] for name in sorted(last_names)], "lr": config.lr_last_block},
        {"name": "fusion", "params": [named[name] for name in sorted(fusion_names)], "lr": config.lr_fusion},
    ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.weight_decay))
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = _make_grad_scaler(device, enabled=use_amp)
    l2sp_names = sorted(norm_names | last_names | fusion_names)
    l2sp = (
        L2SPRegularizer.from_named_parameters((name, named[name]) for name in l2sp_names)
        if l2sp_names
        else None
    )
    bn_before = {
        name: value.detach().clone() for name, value in initial_model_state.items()
        if name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked")
    }
    total_steps = sum(config.phase_steps)
    scheduler_steps = int(config.scheduler_reference_steps) or total_steps
    validation_step_set = set(config.validation_steps)
    losses: list[float] = []
    snapshots: list[
        tuple[dict[str, Tensor], float | tuple[float, ...], int]
    ] = []
    stage_best: dict[str, tuple[int, int, StageValidationMetrics]] = {}
    stage_end: dict[str, StageValidationMetrics] = {}
    current_phase = ""
    validation_forward_steps: list[int] = []
    permitted_model_names = norm_names | adapter_names | last_names | fusion_names
    full_anchor_state = {
        **{f"model.{name}": value for name, value in initial_model_state.items()},
        **{f"head.{name}": value for name, value in initial_head_state.items()},
    }
    compact_anchor_state = {
        **{
            f"model.{name}": initial_model_state[name]
            for name in sorted(permitted_model_names)
        },
        **{f"head.{name}": value for name, value in initial_head_state.items()},
    }
    ema = (
        TrainableDeltaEMA(
            compact_anchor_state,
            permitted_names=compact_anchor_state.keys(),
            decay=config.trainable_delta_ema_decay,
        )
        if config.trainable_delta_ema_decay > 0.0
        else None
    )
    cached_head_embeddings: Tensor | None = None
    cached_head_forward_steps = 0

    for step in range(total_steps):
        phase = _phase_for_step(step, config.phase_steps)
        polishing = config.head_polish_steps > 0 and step >= total_steps - config.head_polish_steps
        optimization_phase = "HEAD" if step < config.head_prefit_steps or polishing else phase
        if optimization_phase != current_phase:
            if optimization_phase == "HEAD":
                student.eval()
                for parameter in student.parameters():
                    parameter.requires_grad_(False)
                if polishing and cached_head_embeddings is None:
                    with torch.no_grad():
                        cached_head_embeddings = _extract_joint_embedding(
                            _forward_aux(student, target_x), int(target_x.size(0))
                        ).detach()
                    cached_head_forward_steps += 1
            else:
                policy.apply(student, phase)
            for parameter in head.parameters():
                parameter.requires_grad_(True)
            current_phase = optimization_phase
        lr_factor = _learning_rate_factor(step, scheduler_steps, config.warmup_ratio)
        base_lrs = _fast_strong_group_lrs(config, step, phase)
        if config.fast_tail_steps and config.fast_tail_start_step <= step < config.fast_tail_start_step + config.fast_tail_steps:
            lr_factor = 1.0
        if polishing:
            base_lrs = {name: 0.0 for name in base_lrs}
            base_lrs["head"] = config.head_polish_lr
            lr_factor = 1.0
        for group in optimizer.param_groups:
            group["lr"] = float(base_lrs[str(group["name"])]) * lr_factor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if optimization_phase == "HEAD":
                embeddings = (
                    cached_head_embeddings
                    if polishing and cached_head_embeddings is not None
                    else initial_embeddings.detach()
                )
            else:
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
            anchor = (
                l2sp(student.named_parameters()) if l2sp is not None else logits.new_zeros(())
            )
            head_anchor_rows = 1.0 - F.cosine_similarity(
                head.weight, initial_head_weight, dim=1, eps=_EPS
            )
            head_anchor = (
                (head_anchor_reliability * head_anchor_rows).sum()
                / head_anchor_reliability.sum().clamp_min(_EPS)
                if config.use_class_adaptive_rho
                else head_anchor_rows.mean()
            )
            kd = logits.new_zeros(())
            if config.selective_kd_weight > 0.0:
                assert teacher is not None
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
                + float(config.head_anchor_weight) * head_anchor
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
        if ema is not None:
            ema.update(
                {
                    **{
                        f"model.{name}": dict(student.state_dict())[name]
                        for name in sorted(permitted_model_names)
                    },
                    **{f"head.{name}": value for name, value in head.state_dict().items()},
                }
            )
        loss_value = float(loss.detach())
        losses.append(loss_value)
        evaluate_checkpoint = (
            checkpoint_validation is None
            or not validation_step_set
            or step + 1 in validation_step_set
        )
        score: float | tuple[float, ...] | None = None
        if checkpoint_validation is None:
            score: float | tuple[float, ...] = _target_train_snapshot_score(
                loss_value, step + 1
            )
        elif evaluate_checkpoint:
            assert validation_x is not None
            assert validation_labels is not None
            assert frozen_validation_logits is not None
            with torch.no_grad():
                validation_embeddings = _extract_joint_embedding(
                    _forward_aux(student, validation_x), len(checkpoint_validation.physical_ids)
                )
                validation_logits = head(validation_embeddings)
                stage_metrics = _stage_validation_metrics(
                    validation_logits,
                    frozen_validation_logits,
                    validation_labels,
                    registered_class_indices=range(len(head.class_ids)),
                    permitted_parameter_distance=_permitted_parameter_distance(
                        initial_model_state, student, phase_names["C"]
                    ),
                )
                validation_accuracy, validation_nll, validation_margin = _classification_metrics(
                    validation_logits, validation_labels
                )
                score = (
                    validation_accuracy,
                    -validation_nll,
                    validation_margin,
                    -stage_metrics.permitted_parameter_distance,
                )
            validation_forward_steps.append(step + 1)
            phase_offset = sum(config.phase_steps[: _PHASES.index(phase)])
            step_in_phase = step - phase_offset + 1
            current_best = stage_best.get(phase)
            if current_best is None or _stage_metric_order_key(stage_metrics) > _stage_metric_order_key(
                current_best[2]
            ):
                stage_best[phase] = (step_in_phase, step + 1, stage_metrics)
            stage_end[phase] = stage_metrics
        qualifies = (
            score is not None
            and checkpoint_selection_mode == "final_step"
            and step + 1 == total_steps
        )
        if checkpoint_selection_mode == "best" and score is not None:
            qualifies = (
                len(snapshots) < int(config.checkpoint_average_top_k)
                or score > min(item[1] for item in snapshots)
            )
        if qualifies:
            combined = {
                **{
                    f"model.{name}": dict(student.state_dict())[name].detach().clone()
                    for name in sorted(permitted_model_names)
                },
                **{f"head.{name}": value.detach().clone() for name, value in head.state_dict().items()},
            }
            if ema is not None:
                combined = ema.state()
            if checkpoint_selection_mode == "final_step":
                snapshots = [(combined, score, step + 1)]
            else:
                snapshots.append((combined, score, step + 1))
                snapshots.sort(key=lambda item: item[1], reverse=True)
                del snapshots[int(config.checkpoint_average_top_k) :]

    anchor_state = compact_anchor_state
    permitted_snapshot_names = {
        *(f"model.{name}" for name in norm_names | adapter_names | last_names | fusion_names),
        *(f"head.{name}" for name in initial_head_state),
    }
    checkpoint_average_top_k = (
        1 if checkpoint_selection_mode == "final_step" else config.checkpoint_average_top_k
    )
    averaged = TrainableDeltaAverager(checkpoint_average_top_k).average(
        [(state, score) for state, score, _ in snapshots],
        anchor_state=anchor_state,
        permitted_names=permitted_snapshot_names,
    )
    averaged_model = {
        name.removeprefix("model."): value
        for name, value in averaged.items()
        if name.startswith("model.")
    }
    restored_model_state = {
        name: value.detach().clone() for name, value in initial_model_state.items()
    }
    restored_model_state.update(averaged_model)
    student.load_state_dict(restored_model_state)
    head.load_state_dict(
        {name.removeprefix("head."): value for name, value in averaged.items() if name.startswith("head.")}
    )
    student.eval()
    head.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    final_state = {
        **{f"model.{name}": value for name, value in student.state_dict().items()},
        **{f"head.{name}": value for name, value in head.state_dict().items()},
    }
    permitted_changed = tuple(
        sorted(
            name
            for name, value in final_state.items()
            if name in permitted_snapshot_names and not torch.equal(value, full_anchor_state[name])
        )
    )
    nonpermitted_changed = tuple(
        sorted(
            name
            for name, value in final_state.items()
            if name not in permitted_snapshot_names and not torch.equal(value, full_anchor_state[name])
        )
    )
    if nonpermitted_changed:
        raise RuntimeError(
            f"SF-TAPFT checkpoint averaging changed non-permitted state: {nonpermitted_changed}"
        )
    updated = tuple(
        name.removeprefix("model.")
        for name in permitted_changed
        if name.startswith("model.")
    )
    bn_after = student.state_dict()
    bn_updated = any(not torch.equal(value, bn_after[name]) for name, value in bn_before.items())
    stage_validation_rows = tuple(
        StageValidationRow(
            phase=phase,
            best_step_in_phase=stage_best[phase][0],
            best_global_step=stage_best[phase][1],
            best_metrics=stage_best[phase][2],
            end_metrics=stage_end[phase],
        )
        for phase in _PHASES
        if phase in stage_best
    )
    audit = SFTAPFTAudit(
        method="sf_tapft_v1",
        permission="DIAGNOSTIC_NON_FORMAL",
        total_steps=total_steps,
        phase_steps=config.phase_steps,
        trainable_names_by_phase=MappingProxyType(phase_names),
        updated_parameter_names=updated,
        permitted_changed_names=permitted_changed,
        nonpermitted_changed_names=nonpermitted_changed,
        support_losses=tuple(losses),
        source_loader_opened=False,
        source_samples_opened=False,
        source_cache_opened=False,
        target_eval_opened=False,
        query_opened=False,
        bn_running_stats_updated=bn_updated,
        checkpoint_selection_role=(
            "fixed_final_step"
            if checkpoint_selection_mode == "final_step"
            else (
                "target_inner_validation"
                if checkpoint_validation is not None
                else "target_train_loss_single"
            )
        ),
        selected_checkpoint_steps=tuple(step for _, _, step in snapshots),
        training_sample_count=len(target_train.physical_ids),
        stage_validation_rows=stage_validation_rows,
        head_prefit_steps=config.head_prefit_steps,
        backbone_optimizer_steps=total_steps - config.head_prefit_steps - config.head_polish_steps,
        backbone_train_forward_steps=total_steps - config.head_prefit_steps - config.head_polish_steps,
        validation_forward_steps=tuple(validation_forward_steps),
        snapshot_tensor_bytes=sum(
            int(value.numel() * value.element_size()) for value in compact_anchor_state.values()
        ),
        trainable_parameter_elements=(
            sum(int(named[name].numel()) for name in permitted_model_names)
            + sum(int(parameter.numel()) for parameter in head.parameters())
        ),
        actual_changed_elements=sum(
            int(final_state[name].numel()) for name in permitted_changed
        ),
        head_polish_steps=config.head_polish_steps,
        cached_head_forward_steps=cached_head_forward_steps,
        trainable_delta_ema_decay=float(config.trainable_delta_ema_decay),
        class_adaptive_rho=tuple(float(value) for value in rho_values.detach().cpu()),
        class_reliability=tuple(
            float(value) for value in reliability_values.detach().cpu()
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


def _calibration_metrics(
    logits: Tensor, labels: Tensor, *, bins: int = 10
) -> tuple[float, float, tuple[float, ...]]:
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = (predictions == labels).to(dtype=probabilities.dtype)
    ece = probabilities.new_zeros(())
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    for index in range(bins):
        mask = (confidence > boundaries[index]) & (confidence <= boundaries[index + 1])
        if bool(mask.any()):
            ece = ece + mask.float().mean() * (
                confidence[mask].mean() - correct[mask].mean()
            ).abs()
    recalls = []
    per_class_nll = []
    row_nll = F.cross_entropy(logits, labels, reduction="none")
    for class_index in torch.unique(labels, sorted=True):
        mask = labels == class_index
        recalls.append(float((predictions[mask] == labels[mask]).float().mean()))
        per_class_nll.append(float(row_nll[mask].mean()))
    return float(min(recalls)), float(ece), tuple(per_class_nll)


def _stage_validation_metrics(
    adapted_logits: Tensor,
    frozen_logits: Tensor,
    labels: Tensor,
    *,
    registered_class_indices: Sequence[int],
    permitted_parameter_distance: float,
) -> StageValidationMetrics:
    if (
        adapted_logits.ndim != 2
        or adapted_logits.size(1) < 2
        or frozen_logits.shape != adapted_logits.shape
        or labels.ndim != 1
        or labels.numel() != adapted_logits.size(0)
        or labels.numel() == 0
    ):
        raise ValueError("stage metric logits and labels must be non-empty and row aligned")
    distance = float(permitted_parameter_distance)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("permitted_parameter_distance must be finite and non-negative")
    class_indices = tuple(int(value) for value in registered_class_indices)
    if (
        not class_indices
        or len(set(class_indices)) != len(class_indices)
        or min(class_indices) < 0
        or max(class_indices) >= adapted_logits.size(1)
    ):
        raise ValueError("registered_class_indices must uniquely identify valid logit columns")
    if not set(labels.detach().cpu().tolist()).issubset(class_indices):
        raise ValueError("validation labels must belong to the registered class universe")
    predictions = adapted_logits.argmax(dim=1)
    frozen_predictions = frozen_logits.argmax(dim=1)
    recalls: list[float] = []
    f1_values: list[float] = []
    margins: list[float] = []
    true_logits = adapted_logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = adapted_logits.clone()
    other_logits.scatter_(1, labels[:, None], float("-inf"))
    row_margins = true_logits - other_logits.max(dim=1).values
    for class_index in class_indices:
        true_mask = labels == class_index
        predicted_mask = predictions == class_index
        true_positive = (true_mask & predicted_mask).sum().float()
        false_positive = ((~true_mask) & predicted_mask).sum().float()
        false_negative = (true_mask & (~predicted_mask)).sum().float()
        true_count = int(true_mask.sum())
        recalls.append(float(true_positive / true_count) if true_count else 0.0)
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1_values.append(float((2.0 * true_positive / denominator) if denominator > 0 else 0.0))
        margins.append(float(row_margins[true_mask].mean()) if true_count else 0.0)
    adapted_correct = predictions == labels
    frozen_correct = frozen_predictions == labels
    return StageValidationMetrics(
        balanced_accuracy=float(sum(recalls) / len(recalls)),
        macro_f1=float(sum(f1_values) / len(f1_values)),
        class_floor=float(min(recalls)),
        nll=float(F.cross_entropy(adapted_logits, labels)),
        per_class_recall=tuple(recalls),
        per_class_margin=tuple(margins),
        positive_flips=int(((~frozen_correct) & adapted_correct).sum()),
        negative_flips=int((frozen_correct & (~adapted_correct)).sum()),
        permitted_parameter_distance=distance,
    )


def _stage_metric_order_key(metrics: StageValidationMetrics) -> tuple[float, ...]:
    mean_margin = sum(metrics.per_class_margin) / len(metrics.per_class_margin)
    return (
        metrics.balanced_accuracy,
        metrics.class_floor,
        -metrics.nll,
        metrics.macro_f1,
        mean_margin,
        -metrics.permitted_parameter_distance,
    )


def _permitted_parameter_distance(
    reference_state: Mapping[str, Tensor],
    adapted: nn.Module,
    permitted_names: Iterable[str],
) -> float:
    current = dict(adapted.named_parameters())
    values = []
    for name in sorted(set(permitted_names)):
        if name not in current or name not in reference_state:
            raise ValueError(f"permitted parameter is absent from distance state: {name!r}")
        values.append((current[name].detach() - reference_state[name]).pow(2).mean())
    return float(torch.stack(values).sum()) if values else 0.0


def _lower_median(values: Sequence[int]) -> int:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("lower median requires at least one value")
    return ordered[(len(ordered) - 1) // 2]


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
        class_floor=float(
            min(getattr(row, f"{prefix}_class_floor") for row in rows)
        ),
        ece=float(sum(getattr(row, f"{prefix}_ece") for row in rows) / len(rows)),
    )


def _full_support_refit_config(
    config: SFTAPFTConfig, selected_phase_steps: tuple[int, int, int]
) -> SFTAPFTConfig:
    selected = tuple(int(value) for value in selected_phase_steps)
    tail_steps = selected[1] if config.fast_tail_steps else 0
    return replace(
        config,
        phase_steps=selected,
        head_prefit_steps=min(config.head_prefit_steps, sum(selected)),
        fast_tail_start_step=selected[0] if tail_steps else 0,
        fast_tail_steps=tail_steps,
        head_polish_steps=min(config.head_polish_steps, selected[2]),
        validation_steps=(),
        checkpoint_average_top_k=1,
    )


def select_sf_tapft_by_grouped_cv(
    checkpoint_model: nn.Module,
    target_train: TargetOnlyAdaptationDataset,
    config: SFTAPFTConfig | None = None,
    *,
    folds: int = 4,
    full_support_refit: bool = False,
) -> SFTAPFTSelectionResult:
    """Use target-train grouped OOF evidence for one domain-level fallback."""

    config = config or SFTAPFTConfig()
    if not isinstance(full_support_refit, bool):
        raise ValueError("full_support_refit must be a boolean")
    selector = GroupedTargetCVSelector(folds=folds, seed=config.seed)
    splits = selector.split(labels=target_train.labels, groups=target_train.groups)
    rows = []
    fitted_folds = []
    adapted_oof: list[tuple[Tensor, Tensor]] = []
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
        adapted_oof.append(
            (adapted_logits.detach().cpu(), adapted_labels.detach().cpu())
        )
        frozen_accuracy, frozen_nll, frozen_margin = _classification_metrics(
            frozen_logits, frozen_labels
        )
        adapted_accuracy, adapted_nll, adapted_margin = _classification_metrics(
            adapted_logits, adapted_labels
        )
        frozen_floor, frozen_ece, frozen_per_class_nll = _calibration_metrics(
            frozen_logits, frozen_labels
        )
        adapted_floor, adapted_ece, adapted_per_class_nll = _calibration_metrics(
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
                stage_validation_rows=fitted.audit.stage_validation_rows,
                frozen_class_floor=frozen_floor,
                adapted_class_floor=adapted_floor,
                frozen_ece=frozen_ece,
                adapted_ece=adapted_ece,
                frozen_per_class_nll=frozen_per_class_nll,
                adapted_per_class_nll=adapted_per_class_nll,
            )
        )
    temperature_calibration = None
    if config.oof_temperature_calibration:
        temperature_calibration = fit_positive_temperature(
            torch.cat([logits for logits, _ in adapted_oof]),
            torch.cat([labels for _, labels in adapted_oof]),
        )
        rows = [
            replace(
                row,
                adapted_nll=_classification_metrics(
                    logits / temperature_calibration.temperature, labels
                )[1],
                adapted_margin=_classification_metrics(
                    logits / temperature_calibration.temperature, labels
                )[2],
                adapted_class_floor=_calibration_metrics(
                    logits / temperature_calibration.temperature, labels
                )[0],
                adapted_ece=_calibration_metrics(
                    logits / temperature_calibration.temperature, labels
                )[1],
                adapted_per_class_nll=_calibration_metrics(
                    logits / temperature_calibration.temperature, labels
                )[2],
            )
            for row, (logits, labels) in zip(rows, adapted_oof)
        ]
    fold_rows = tuple(rows)
    frozen_metrics = _aggregate_fold_metrics(fold_rows, adapted=False)
    adapted_metrics = _aggregate_fold_metrics(fold_rows, adapted=True)
    selected = selector.choose(frozen=frozen_metrics, adapted=adapted_metrics)
    selected_phase_steps = tuple(
        _lower_median(
            [
                stage.best_step_in_phase
                for row in fold_rows
                for stage in row.stage_validation_rows
                if stage.phase == phase
            ]
        )
        if any(stage.phase == phase for row in fold_rows for stage in row.stage_validation_rows)
        else 0
        for phase in _PHASES
    )
    full_support_result = None
    final_training_sample_count = 0
    fold0_as_final = False
    adapted_result = None
    if selected == "adapted":
        if full_support_refit:
            refit_config = _full_support_refit_config(config, selected_phase_steps)
            full_support_result = fit_sf_tapft(
                copy.deepcopy(checkpoint_model),
                target_train,
                refit_config,
                checkpoint_selection_mode="final_step",
            )
            if temperature_calibration is not None:
                full_support_result.head.scale /= temperature_calibration.temperature
            adapted_result = full_support_result
            final_training_sample_count = len(target_train.physical_ids)
        else:
            # Preserve the V1 return behavior until the R0 runner opts in.
            adapted_result = fitted_folds[0]
            if temperature_calibration is not None:
                adapted_result.head.scale /= temperature_calibration.temperature
            final_training_sample_count = adapted_result.audit.training_sample_count
            fold0_as_final = True
    return SFTAPFTSelectionResult(
        selected=selected,
        frozen_metrics=frozen_metrics,
        adapted_metrics=adapted_metrics,
        fold_rows=fold_rows,
        selected_phase_steps=selected_phase_steps,
        adapted_result=adapted_result,
        full_support_result=full_support_result,
        final_training_sample_count=final_training_sample_count,
        fold0_as_final=fold0_as_final,
        temperature_calibration=temperature_calibration,
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
    "StageValidationMetrics",
    "StageValidationRow",
    "TemperatureCalibration",
    "TargetOnlyAdaptationDataset",
    "TargetPrototypeHead",
    "TrainableDeltaAverager",
    "TrainableDeltaEMA",
    "class_adaptive_rho",
    "ensure_time_adapter",
    "fit_sf_tapft",
    "fit_positive_temperature",
    "leave_one_out_prototype_logits",
    "select_sf_tapft_by_grouped_cv",
]
