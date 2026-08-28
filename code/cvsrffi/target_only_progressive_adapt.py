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
from dataclasses import dataclass, field, replace
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
    head_cvar_weight: float = 0.0
    head_cvar_top_k: int = 2
    head_cvar_steps: int = 0
    trainable_delta_ema_decay: float = 0.0
    use_class_adaptive_rho: bool = False
    class_adaptive_rho_min: float = 0.25
    class_adaptive_rho_max: float = 0.75
    class_adaptive_rho_temperature: float = 0.10
    head_anchor_weight: float = 0.0
    hard_pair_weight: float = 0.0
    hard_pair_margin: float = 0.2
    pace_expand_start_step: int = 0
    pace_norm_rules: tuple[tuple[str, str], ...] = ()
    pace_tail_weight: float = 0.0
    pace_preserve_weight: float = 0.0
    pace_preserve_temperature: float = 2.0
    pace_bias_steps: int = 0
    pace_bias_lr: float = 0.05
    pace_bias_l2: float = 0.01
    rse_view_weight: float = 0.0
    rse_view_phase_radians: float = 0.0
    rse_snapshot_steps: tuple[int, ...] = ()
    prefix_cache_dtype: str = "off"
    cache_storage_dtype: str = ""
    suffix_compute_dtype: str = ""
    cache_device: str = "model"
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
        rse_snapshot_steps = tuple(int(value) for value in self.rse_snapshot_steps)
        if any(isinstance(value, bool) for value in self.rse_snapshot_steps) or (
            rse_snapshot_steps
            and (
                tuple(sorted(set(rse_snapshot_steps))) != rse_snapshot_steps
                or rse_snapshot_steps[0] < 1
                or rse_snapshot_steps[-1] > total_steps
            )
        ):
            raise ValueError("rse_snapshot_steps must be sorted, unique, and in [1, total_steps]")
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
            "head_cvar_steps": self.head_cvar_steps,
            "pace_expand_start_step": self.pace_expand_start_step,
        }
        for name, value in integer_controls.items():
            if isinstance(value, bool) or int(value) < 0 or int(value) > total_steps:
                raise ValueError(f"{name} must be an integer in [0, total_steps]")
        if bool(self.fast_tail_steps) != bool(self.fast_tail_start_step):
            raise ValueError("fast tail start and steps must either both be zero or both be positive")
        if self.fast_tail_steps and self.fast_tail_start_step + self.fast_tail_steps > total_steps:
            raise ValueError("fast tail schedule exceeds total_steps")
        if self.head_cvar_steps > self.head_polish_steps:
            raise ValueError("head_cvar_steps must not exceed head_polish_steps")
        pace_rules = tuple((str(scope), str(affine)) for scope, affine in self.pace_norm_rules)
        for scope, affine in pace_rules:
            if scope not in {name for values in _NORM_SCOPES.values() for name in values}:
                raise ValueError(f"pace_norm_rules contains unknown norm scope: {scope!r}")
            if affine not in _NORM_AFFINES:
                raise ValueError(f"pace_norm_rules contains unknown norm affine: {affine!r}")
        if len({scope for scope, _ in pace_rules}) != len(pace_rules):
            raise ValueError("pace_norm_rules must not repeat a norm scope")
        if pace_rules and int(self.pace_expand_start_step) <= 0:
            raise ValueError("pace_expand_start_step must be positive when pace_norm_rules are set")
        if int(self.pace_bias_steps) > 0 and int(self.pace_expand_start_step) <= 0:
            raise ValueError("pace_bias_steps require a positive pace_expand_start_step")
        if isinstance(self.pace_bias_steps, bool) or int(self.pace_bias_steps) < 0:
            raise ValueError("pace_bias_steps must be a non-negative integer")
        if isinstance(self.head_cvar_top_k, bool) or int(self.head_cvar_top_k) <= 0:
            raise ValueError("head_cvar_top_k must be a positive integer")
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
            "pace_preserve_temperature",
            "pace_bias_lr",
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
            "hard_pair_weight",
            "hard_pair_margin",
            "head_cvar_weight",
            "pace_tail_weight",
            "pace_preserve_weight",
            "pace_bias_l2",
            "rse_view_weight",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(float(self.rse_view_phase_radians)):
            raise ValueError("rse_view_phase_radians must be finite")
        if bool(self.rse_view_weight) != bool(self.rse_view_phase_radians):
            raise ValueError("rse_view_weight and rse_view_phase_radians must be enabled together")
        if (self.rse_snapshot_steps or self.rse_view_weight > 0.0) and normalized_rules != (
            ("t3", "weight_bias"),
        ):
            raise ValueError("RSE requires exactly t3 norm weight_bias")
        if (self.rse_snapshot_steps or self.rse_view_weight > 0.0) and (
            self.trainability_profile != "p1_head_norm"
            or self.pace_norm_rules
            or self.pace_bias_steps
            or self.hard_pair_weight > 0.0
            or self.trainable_delta_ema_decay > 0.0
        ):
            raise ValueError("RSE requires the compact E0 permission boundary")
        if isinstance(self.checkpoint_average_top_k, bool) or int(self.checkpoint_average_top_k) <= 0:
            raise ValueError("checkpoint_average_top_k must be a positive integer")
        if not isinstance(self.mixed_precision, bool):
            raise ValueError("mixed_precision must be a boolean")
        if not isinstance(self.oof_temperature_calibration, bool):
            raise ValueError("oof_temperature_calibration must be a boolean")
        if not isinstance(self.use_class_adaptive_rho, bool):
            raise ValueError("use_class_adaptive_rho must be a boolean")
        if self.prefix_cache_dtype not in {"off", "float32", "float16"}:
            raise ValueError("prefix_cache_dtype must be off, float32 or float16")
        cache_dtypes = {"off", "float16", "bfloat16", "float32"}
        storage_dtype = self.cache_storage_dtype or self.prefix_cache_dtype
        compute_dtype = self.suffix_compute_dtype or (
            "off" if storage_dtype == "off" else "float32"
        )
        if storage_dtype not in cache_dtypes:
            raise ValueError("cache_storage_dtype must be off, float16, bfloat16 or float32")
        if compute_dtype not in {"off", "float32"}:
            raise ValueError(
                "suffix_compute_dtype currently supports only off or float32; "
                "low-precision suffix compute requires a separate equivalence-qualified path"
            )
        if (storage_dtype == "off") != (compute_dtype == "off"):
            raise ValueError("cache storage and suffix compute must either both be off or enabled")
        if self.cache_device not in {"model", "cpu", "cuda"}:
            raise ValueError("cache_device must be model, cpu or cuda")
        object.__setattr__(self, "cache_storage_dtype", storage_dtype)
        object.__setattr__(self, "suffix_compute_dtype", compute_dtype)
        if not 0.0 <= float(self.trainable_delta_ema_decay) < 1.0:
            raise ValueError("trainable_delta_ema_decay must be in [0, 1)")
        if not 0.0 <= float(self.class_adaptive_rho_min) <= float(self.class_adaptive_rho_max) <= 1.0:
            raise ValueError("class adaptive rho bounds must satisfy 0 <= min <= max <= 1")
        object.__setattr__(self, "phase_steps", tuple(int(value) for value in self.phase_steps))
        object.__setattr__(self, "norm_rules", normalized_rules)
        object.__setattr__(self, "pace_norm_rules", pace_rules)
        object.__setattr__(self, "head_prefit_steps", int(self.head_prefit_steps))
        object.__setattr__(self, "validation_steps", validation_steps)
        object.__setattr__(self, "rse_snapshot_steps", rse_snapshot_steps)
        object.__setattr__(
            self, "scheduler_reference_steps", int(self.scheduler_reference_steps)
        )
        for name in integer_controls:
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "head_cvar_top_k", int(self.head_cvar_top_k))
        object.__setattr__(self, "pace_bias_steps", int(self.pace_bias_steps))


def class_cvar_from_class_losses(class_losses: Tensor, *, top_k: int) -> Tensor:
    """Return the mean of the largest class-mean losses without class-ID branches."""

    if class_losses.ndim != 1 or class_losses.numel() == 0:
        raise ValueError("class losses must be a non-empty vector")
    if isinstance(top_k, bool) or int(top_k) <= 0 or int(top_k) > class_losses.numel():
        raise ValueError("top_k must be in [1, class_count]")
    if not bool(torch.isfinite(class_losses).all()):
        raise ValueError("class losses must be finite")
    return torch.topk(class_losses, k=int(top_k)).values.mean()


def phase_rotate_iq(values: Tensor, *, radians: float) -> Tensor:
    """Apply one identity-preserving phase rotation to paired IQ channels."""

    if values.ndim != 3 or values.size(1) < 2 or values.size(1) % 2:
        raise ValueError("IQ phase rotation requires paired real/imag channels")
    if not math.isfinite(float(radians)):
        raise ValueError("phase rotation radians must be finite")
    angle = values.new_tensor(float(radians))
    cosine, sine = torch.cos(angle), torch.sin(angle)
    paired = values.reshape(values.size(0), values.size(1) // 2, 2, values.size(2))
    real, imag = paired[:, :, 0], paired[:, :, 1]
    rotated = torch.stack(
        (cosine * real - sine * imag, sine * real + cosine * imag), dim=2
    )
    return rotated.reshape_as(values)


@dataclass(frozen=True)
class RobustSupportRisk:
    total: float
    macro_ce: float
    class_cvar: float
    view_js: float
    margin_regression: float


def interpolate_trainable_state(
    anchor: Mapping[str, Tensor], snapshot: Mapping[str, Tensor], *, alpha: float
) -> Mapping[str, Tensor]:
    """Interpolate one registered trainable snapshot from its common anchor."""

    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite in [0, 1]")
    if set(anchor) != set(snapshot):
        raise ValueError("anchor and snapshot must have aligned keys")
    output: dict[str, Tensor] = {}
    for name in anchor:
        left, right = anchor[name], snapshot[name]
        if left.shape != right.shape:
            raise ValueError(f"anchor and snapshot shape mismatch: {name}")
        output[name] = left.detach().cpu() + alpha * (
            right.detach().cpu() - left.detach().cpu()
        )
    return MappingProxyType(output)


def average_trainable_states(
    anchor: Mapping[str, Tensor], snapshots: Sequence[Mapping[str, Tensor]]
) -> Mapping[str, Tensor]:
    """Average aligned trainable deltas without touching unregistered state."""

    if not snapshots:
        raise ValueError("at least one trainable snapshot is required")
    if any(set(snapshot) != set(anchor) for snapshot in snapshots):
        raise ValueError("all trainable states must have aligned keys")
    output: dict[str, Tensor] = {}
    for name, base in anchor.items():
        deltas = []
        for snapshot in snapshots:
            value = snapshot[name]
            if value.shape != base.shape:
                raise ValueError(f"trainable state shape mismatch: {name}")
            deltas.append(value.detach().cpu() - base.detach().cpu())
        output[name] = base.detach().cpu() + torch.stack(deltas).mean(dim=0)
    return MappingProxyType(output)


def balanced_rse_subsets(
    labels: Tensor, *, per_class: int, count: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Build deterministic class-balanced support subsamples for delta averaging."""

    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels must be a non-empty vector")
    if isinstance(per_class, bool) or int(per_class) <= 0:
        raise ValueError("per_class must be positive")
    if isinstance(count, bool) or int(count) <= 0:
        raise ValueError("count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    class_ids = torch.unique(labels.detach().cpu(), sorted=True)
    pools: dict[int, Tensor] = {}
    for class_id_tensor in class_ids:
        class_id = int(class_id_tensor)
        indices = torch.nonzero(labels.detach().cpu() == class_id, as_tuple=False).flatten()
        if indices.numel() < int(per_class):
            raise ValueError("per_class exceeds the smallest class support count")
        pools[class_id] = indices
    rows = []
    attempts = 0
    while len(rows) < int(count) and attempts < int(count) * 32:
        attempts += 1
        selected = []
        for class_id in sorted(pools):
            pool = pools[class_id]
            order = torch.randperm(pool.numel(), generator=generator)[: int(per_class)]
            selected.extend(int(value) for value in pool[order].tolist())
        candidate = tuple(sorted(selected))
        if candidate not in rows:
            rows.append(candidate)
    if len(rows) != int(count):
        raise ValueError("RSE subset sampling could not produce unique subsets")
    return tuple(rows)


def select_rse_strength(
    fold_risks: Mapping[tuple[int, float], Sequence[float]],
) -> tuple[int, float]:
    """Select the lowest mean cross-fit risk with conservative deterministic ties."""

    if not fold_risks:
        raise ValueError("RSE strength selection requires candidate risks")
    normalized: list[tuple[float, float, int, float]] = []
    for (step, alpha), values in fold_risks.items():
        if isinstance(step, bool) or int(step) <= 0:
            raise ValueError("RSE candidate step must be positive")
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("RSE candidate alpha must be in [0, 1]")
        risks = tuple(float(value) for value in values)
        if not risks or any(not math.isfinite(value) for value in risks):
            raise ValueError("RSE fold risks must be finite and non-empty")
        normalized.append((sum(risks) / len(risks), alpha, int(step), alpha))
    _, _, step, alpha = min(normalized)
    return step, alpha


def robust_support_risk(
    logits: Tensor,
    labels: Tensor,
    *,
    frozen_logits: Tensor,
    second_view_logits: Tensor | None = None,
    cvar_top_k: int = 2,
) -> RobustSupportRisk:
    """Compute the report-defined support-only RSE risk for one held-out fold."""

    if logits.ndim != 2 or frozen_logits.shape != logits.shape:
        raise ValueError("adapted and frozen logits must be aligned matrices")
    if labels.ndim != 1 or labels.numel() != logits.size(0):
        raise ValueError("labels must align with logits")
    if second_view_logits is not None and second_view_logits.shape != logits.shape:
        raise ValueError("second-view logits must align with original-view logits")
    row_ce = F.cross_entropy(logits.float(), labels, reduction="none")
    class_ids = tuple(int(value) for value in torch.unique(labels, sorted=True).tolist())
    class_losses = torch.stack([row_ce[labels == class_id].mean() for class_id in class_ids])
    macro_ce = class_losses.mean()
    class_cvar = class_cvar_from_class_losses(
        class_losses, top_k=min(int(cvar_top_k), len(class_ids))
    )
    true = labels[:, None]
    frozen_true = frozen_logits.gather(1, true).squeeze(1)
    adapted_true = logits.gather(1, true).squeeze(1)
    frozen_other = frozen_logits.masked_fill(
        F.one_hot(labels, num_classes=logits.size(1)).bool(), float("-inf")
    ).max(dim=1).values
    adapted_other = logits.masked_fill(
        F.one_hot(labels, num_classes=logits.size(1)).bool(), float("-inf")
    ).max(dim=1).values
    margin_regression = F.relu(
        (frozen_true - frozen_other) - (adapted_true - adapted_other)
    ).mean()
    view_js = logits.new_zeros((), dtype=torch.float32)
    if second_view_logits is not None:
        p = F.softmax(logits.float(), dim=1)
        q = F.softmax(second_view_logits.float(), dim=1)
        mixture = 0.5 * (p + q)
        view_js = 0.5 * (
            F.kl_div(mixture.log(), p, reduction="batchmean")
            + F.kl_div(mixture.log(), q, reduction="batchmean")
        )
    total = macro_ce + 0.30 * class_cvar + 0.10 * view_js + 0.05 * margin_regression
    return RobustSupportRisk(
        total=float(total.detach()),
        macro_ce=float(macro_ce.detach()),
        class_cvar=float(class_cvar.detach()),
        view_js=float(view_js.detach()),
        margin_regression=float(margin_regression.detach()),
    )


def stable_support_weights(teacher_logits: Tensor, labels: Tensor) -> Tensor:
    """Return detached D0 support stability weights without class-specific rules."""

    if (
        teacher_logits.ndim != 2
        or labels.ndim != 1
        or teacher_logits.size(0) != labels.numel()
        or teacher_logits.size(1) < 2
    ):
        raise ValueError("teacher logits and labels must be row aligned")
    if not bool(torch.isfinite(teacher_logits).all()):
        raise ValueError("teacher logits must be finite")
    labels = labels.to(device=teacher_logits.device, dtype=torch.long)
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= teacher_logits.size(1)):
        raise ValueError("labels must index teacher logit columns")
    detached = teacher_logits.detach().float()
    probability = detached.softmax(dim=1).gather(1, labels[:, None]).squeeze(1)
    true_logits = detached.gather(1, labels[:, None]).squeeze(1)
    other_logits = detached.masked_fill(
        F.one_hot(labels, num_classes=detached.size(1)).bool(), float("-inf")
    ).max(dim=1).values
    weights = probability * torch.sigmoid(true_logits - other_logits)
    return (weights / weights.max().clamp_min(_EPS)).detach()


def stable_preservation_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    stable_weights: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Weighted support-only KL from the frozen D0 teacher to the PACE student."""

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must have the same matrix shape")
    if stable_weights.shape != (student_logits.size(0),):
        raise ValueError("stable_weights must align with support rows")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    teacher_probability = (teacher_logits.detach() / float(temperature)).softmax(dim=1)
    student_log_probability = (student_logits / float(temperature)).log_softmax(dim=1)
    row_kl = F.kl_div(
        student_log_probability, teacher_probability, reduction="none"
    ).sum(dim=1)
    weights = stable_weights.detach().to(device=row_kl.device, dtype=row_kl.dtype)
    return float(temperature) ** 2 * (weights * row_kl).sum() / weights.sum().clamp_min(_EPS)


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    nll_before: float
    nll_after: float
    argmax_preserved: bool


@dataclass(frozen=True)
class BiasCalibration:
    bias: Tensor
    nll_before: float
    nll_after: float
    steps: int


def fit_zero_sum_class_bias(
    logits: Tensor,
    labels: Tensor,
    *,
    steps: int = 40,
    lr: float = 0.05,
    l2: float = 0.01,
) -> BiasCalibration:
    """Fit a cross-fitted class bias with an exact zero-sum parameterization."""

    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.numel():
        raise ValueError("bias calibration logits and labels must be row aligned")
    if logits.numel() == 0 or not bool(torch.isfinite(logits).all()):
        raise ValueError("bias calibration logits must be non-empty and finite")
    if isinstance(steps, bool) or int(steps) <= 0:
        raise ValueError("bias calibration steps must be positive")
    if not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise ValueError("bias calibration lr must be finite and positive")
    if not math.isfinite(float(l2)) or float(l2) < 0.0:
        raise ValueError("bias calibration l2 must be finite and non-negative")
    work_logits = logits.detach().float()
    work_labels = labels.detach().to(device=work_logits.device, dtype=torch.long)
    raw = torch.zeros(work_logits.size(1), device=work_logits.device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=float(lr))
    before = float(F.cross_entropy(work_logits, work_labels))
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        bias = raw - raw.mean()
        loss = F.cross_entropy(work_logits + bias, work_labels) + float(l2) * bias.square().mean()
        loss.backward()
        optimizer.step()
    bias = (raw.detach() - raw.detach().mean()).to(dtype=logits.dtype)
    after = float(F.cross_entropy(work_logits + bias.float(), work_labels))
    if after > before:
        bias = torch.zeros_like(bias)
        after = before
    return BiasCalibration(bias=bias, nll_before=before, nll_after=after, steps=int(steps))


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

    def __init__(
        self,
        weight: Tensor,
        class_ids: Sequence[int],
        scale: float = 8.0,
        bias: Tensor | None = None,
    ):
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
        bias_value = torch.zeros(len(ids), dtype=weight.dtype, device=weight.device)
        if bias is not None:
            if not torch.is_tensor(bias) or bias.shape != (len(ids),):
                raise ValueError("bias must align with class_ids")
            if not bool(torch.isfinite(bias).all()):
                raise ValueError("bias must be finite")
            bias_value = bias.detach().to(device=weight.device, dtype=weight.dtype).clone()
        self.register_buffer("bias", bias_value, persistent=False)

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
        ).transpose(0, 1) + self.bias


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
    head_cvar_steps: int
    head_cvar_weight: float
    head_cvar_top_k: int
    head_cvar_losses: tuple[float, ...]
    trainable_delta_ema_decay: float
    class_adaptive_rho: tuple[float, ...]
    class_reliability: tuple[float, ...]
    prefix_cache_dtype: str
    prefix_cache_build_forward_steps: int
    cached_suffix_forward_steps: int
    hard_pair_weight: float
    hard_pair_margin: float
    prefix_cache_tensor_bytes: int
    support_safety_checked: bool
    support_safety_passed: bool
    support_safety_prediction_mismatches: int
    support_safety_per_class_recall_mismatches: int
    support_safety_max_abs_logit_delta: float
    support_safety_fallback_to_float32: bool
    pace_expand_start_step: int = 0
    pace_teacher_snapshot_count: int = 0
    pace_expanded_optimizer_steps: int = 0
    pace_tail_losses: tuple[float, ...] = ()
    pace_preserve_losses: tuple[float, ...] = ()
    effective_view_count: int = 1
    view_consistency_losses: tuple[float, ...] = ()
    suffix_backward_steps: int = 0
    head_optimizer_steps: int = 0


@dataclass(frozen=True)
class SFTAPFTResult:
    model: nn.Module
    head: TargetPrototypeHead
    audit: SFTAPFTAudit
    base_parameter_anchors: Mapping[str, Tensor]
    retained_trainable_snapshots: Mapping[int, Mapping[str, Tensor]] = field(
        default_factory=lambda: MappingProxyType({})
    )


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


@dataclass(frozen=True)
class RSEStrengthFoldRow:
    repeat: int
    fold: int
    step: int
    alpha: float
    risk: RobustSupportRisk


@dataclass(frozen=True)
class RSEStrengthSelectionResult:
    result: SFTAPFTResult
    selected_step: int
    selected_alpha: float
    fold_rows: tuple[RSEStrengthFoldRow, ...]
    crossfit_fit_count: int
    crossfit_validation_forward_steps: int
    crossfit_validation_suffix_forward_steps: int


@dataclass(frozen=True)
class RSEDeltaEnsembleResult:
    result: SFTAPFTResult
    subset_indices: tuple[tuple[int, ...], ...]
    subset_fit_count: int
    polish_steps: int
    common_anchor: Mapping[str, Tensor]


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


@dataclass(frozen=True)
class H6PrefixCache:
    """Frozen inputs needed to replay only the trainable H6 suffix."""

    pre_t3_norm: Tensor
    frozen_fuse_tail: Tensor
    cls_head_kwargs: Mapping[str, Tensor]
    storage_dtype: torch.dtype
    batch_size: int
    source_tensor_bytes: int = 0

    @property
    def tensor_bytes(self) -> int:
        tensors = (
            self.pre_t3_norm,
            self.frozen_fuse_tail,
            *self.cls_head_kwargs.values(),
        )
        return sum(int(value.numel() * value.element_size()) for value in tensors)

    @property
    def storage_tensor_bytes(self) -> int:
        return self.source_tensor_bytes or self.tensor_bytes

    def materialize_once(self, *, device: torch.device, dtype: torch.dtype) -> "H6PrefixCache":
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("H6 compute cache dtype must be float16, bfloat16 or float32")
        return H6PrefixCache(
            pre_t3_norm=self.pre_t3_norm.to(device=device, dtype=dtype),
            frozen_fuse_tail=self.frozen_fuse_tail.to(device=device, dtype=dtype),
            cls_head_kwargs=MappingProxyType(
                {
                    name: value.to(device=device, dtype=dtype)
                    for name, value in self.cls_head_kwargs.items()
                }
            ),
            storage_dtype=dtype,
            batch_size=self.batch_size,
            source_tensor_bytes=self.storage_tensor_bytes,
        )


@dataclass(frozen=True)
class H6SupportSafetyAudit:
    passed: bool
    prediction_mismatches: int
    per_class_recall_mismatches: int
    positive_margin_regressions: int
    max_abs_logit_delta: float
    minimum_full_path_margin: float


@dataclass(frozen=True)
class TimeNormPrefixCache:
    """Frozen graph inputs for a suffix beginning at one time-path Norm."""

    boundary: str
    pre_norm: Tensor
    frozen_fuse_tail: Tensor
    cls_head_kwargs: Mapping[str, Tensor]
    storage_dtype: torch.dtype
    batch_size: int
    source_tensor_bytes: int = 0

    @property
    def tensor_bytes(self) -> int:
        return sum(
            int(value.numel() * value.element_size())
            for value in (self.pre_norm, self.frozen_fuse_tail, *self.cls_head_kwargs.values())
        )

    @property
    def storage_tensor_bytes(self) -> int:
        return self.source_tensor_bytes or self.tensor_bytes

    def materialize_once(self, *, device: torch.device, dtype: torch.dtype) -> "TimeNormPrefixCache":
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("time suffix compute cache dtype must be floating point")
        return TimeNormPrefixCache(
            boundary=self.boundary,
            pre_norm=self.pre_norm.to(device=device, dtype=dtype),
            frozen_fuse_tail=self.frozen_fuse_tail.to(device=device, dtype=dtype),
            cls_head_kwargs=MappingProxyType(
                {name: value.to(device=device, dtype=dtype) for name, value in self.cls_head_kwargs.items()}
            ),
            storage_dtype=dtype,
            batch_size=self.batch_size,
            source_tensor_bytes=self.storage_tensor_bytes,
        )


class CompactTimeNormSuffix(nn.Module):
    """Independent time suffix selected by the earliest trainable Norm."""

    _ORDER = ("time_fuse.1", "t1.norm", "t2.norm", "t3.norm")

    def __init__(
        self,
        *,
        time_fuse: nn.Module,
        time_down: nn.Module,
        t1: nn.Module,
        t2: nn.Module,
        t3: nn.Module,
        t_pool: nn.Module,
        t_proj: nn.Module,
        meta_adapter_time: nn.Module,
        fuse: nn.Module,
        meta_adapter_fusion: nn.Module,
        cls_head: nn.Module,
        target_head: TargetPrototypeHead,
        cache: TimeNormPrefixCache,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.time_fuse = time_fuse
        self.time_down = time_down
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t_pool = t_pool
        self.t_proj = t_proj
        self.meta_adapter_time = meta_adapter_time
        self.fuse = fuse
        self.meta_adapter_fusion = meta_adapter_fusion
        self.cls_head = cls_head
        self.target_head = target_head
        self.cache = cache
        self.emb_dim = int(emb_dim)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        start = self._ORDER.index(cache.boundary)
        norm_modules = (self.time_fuse[1], self.t1.norm, self.t2.norm, self.t3.norm)
        for module in norm_modules[start:]:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        for parameter in self.target_head.parameters():
            parameter.requires_grad_(True)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        head: TargetPrototypeHead,
        cache: TimeNormPrefixCache,
    ) -> "CompactTimeNormSuffix":
        backbone, _ = _identity_backbone(model)
        module_by_boundary = {
            "time_fuse.1": backbone.time_fuse[1],
            "t1.norm": backbone.t1.norm,
            "t2.norm": backbone.t2.norm,
            "t3.norm": backbone.t3.norm,
        }
        if cache.boundary not in module_by_boundary:
            raise ValueError("unsupported time suffix cache boundary")
        reference = next(module_by_boundary[cache.boundary].parameters())
        compute_cache = cache.materialize_once(device=reference.device, dtype=reference.dtype)
        return cls(
            time_fuse=copy.deepcopy(backbone.time_fuse),
            time_down=copy.deepcopy(backbone.time_down),
            t1=copy.deepcopy(backbone.t1),
            t2=copy.deepcopy(backbone.t2),
            t3=copy.deepcopy(backbone.t3),
            t_pool=copy.deepcopy(backbone.t_pool),
            t_proj=copy.deepcopy(backbone.t_proj),
            meta_adapter_time=copy.deepcopy(backbone.meta_adapter_time),
            fuse=copy.deepcopy(backbone.fuse),
            meta_adapter_fusion=copy.deepcopy(backbone.meta_adapter_fusion),
            cls_head=copy.deepcopy(backbone.cls_head),
            target_head=copy.deepcopy(head),
            cache=compute_cache,
            emb_dim=int(backbone.emb_dim),
        )

    def embedding(self) -> Tensor:
        value = self.cache.pre_norm
        boundary = self.cache.boundary
        if boundary == "time_fuse.1":
            value = self.time_fuse[1](value)
            value = self.time_fuse[2](value)
            value = self.time_down(value)
            value = self.t1(value)
            value = self.t2(value)
            value = self.t3(value)
        elif boundary == "t1.norm":
            value = self.t1.drop(self.t1.pool(self.t1.act(self.t1.norm(value))))
            value = self.t2(value)
            value = self.t3(value)
        elif boundary == "t2.norm":
            value = self.t2.drop(self.t2.pool(self.t2.act(self.t2.norm(value))))
            value = self.t3(value)
        elif boundary == "t3.norm":
            value = self.t3.drop(self.t3.pool(self.t3.act(self.t3.norm(value))))
        else:
            raise ValueError("unsupported time suffix cache boundary")
        time_embedding = self.meta_adapter_time(self.t_proj(self.t_pool(value).squeeze(-1)))
        tail = self.cache.frozen_fuse_tail
        base_input = torch.cat([time_embedding, tail], dim=1) if tail.size(1) else time_embedding
        base = self.meta_adapter_fusion(self.fuse(base_input))
        output = self.cls_head(base, labels=None, return_emb=True, **dict(self.cache.cls_head_kwargs))
        embedding = output.get("feat_joint") if isinstance(output, Mapping) else output[-1]
        if not torch.is_tensor(embedding) or embedding.ndim != 2:
            raise ValueError("compact time suffix did not expose feat_joint")
        return embedding

    def logits(self) -> Tensor:
        return self.target_head(self.embedding())


class CompactH6Suffix(nn.Module):
    """Independent H6 suffix that never retains the complete checkpoint model."""

    def __init__(
        self,
        *,
        t3: nn.Module,
        t_pool: nn.Module,
        t_proj: nn.Module,
        meta_adapter_time: nn.Module,
        fuse: nn.Module,
        meta_adapter_fusion: nn.Module,
        cls_head: nn.Module,
        target_head: TargetPrototypeHead,
        cache: H6PrefixCache,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.t3 = t3
        self.t_pool = t_pool
        self.t_proj = t_proj
        self.meta_adapter_time = meta_adapter_time
        self.fuse = fuse
        self.meta_adapter_fusion = meta_adapter_fusion
        self.cls_head = cls_head
        self.target_head = target_head
        self.cache = cache
        self.emb_dim = int(emb_dim)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.t3.norm.parameters():
            parameter.requires_grad_(True)
        for parameter in self.target_head.parameters():
            parameter.requires_grad_(True)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        head: TargetPrototypeHead,
        cache: H6PrefixCache,
    ) -> "CompactH6Suffix":
        backbone, _ = _identity_backbone(model)
        reference = next(backbone.t3.norm.parameters())
        compute_cache = cache.materialize_once(
            device=reference.device,
            dtype=reference.dtype,
        )
        return cls(
            t3=copy.deepcopy(backbone.t3),
            t_pool=copy.deepcopy(backbone.t_pool),
            t_proj=copy.deepcopy(backbone.t_proj),
            meta_adapter_time=copy.deepcopy(backbone.meta_adapter_time),
            fuse=copy.deepcopy(backbone.fuse),
            meta_adapter_fusion=copy.deepcopy(backbone.meta_adapter_fusion),
            cls_head=copy.deepcopy(backbone.cls_head),
            target_head=copy.deepcopy(head),
            cache=compute_cache,
            emb_dim=int(backbone.emb_dim),
        )

    def embedding(self) -> Tensor:
        t = self.t3.norm(self.cache.pre_t3_norm)
        t = self.t3.act(t)
        t = self.t3.pool(t)
        t = self.t3.drop(t)
        t_emb = self.meta_adapter_time(self.t_proj(self.t_pool(t).squeeze(-1)))
        tail = self.cache.frozen_fuse_tail
        base_input = torch.cat([t_emb, tail], dim=1) if tail.size(1) else t_emb
        base = self.meta_adapter_fusion(self.fuse(base_input))
        output = self.cls_head(
            base,
            labels=None,
            return_emb=True,
            **dict(self.cache.cls_head_kwargs),
        )
        if isinstance(output, Mapping):
            embedding = output.get("feat_joint")
        elif isinstance(output, (tuple, list)) and output:
            embedding = output[-1]
        else:
            embedding = None
        if not torch.is_tensor(embedding) or embedding.ndim != 2:
            raise ValueError("compact H6 suffix did not expose feat_joint")
        if int(embedding.size(0)) != self.cache.batch_size:
            raise ValueError("compact H6 suffix batch size drifted")
        return embedding

    def logits(self) -> Tensor:
        return self.target_head(self.embedding())

    def export_permitted_state(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {
                "model.t3.norm.weight": self.t3.norm.weight.detach().clone(),
                "model.t3.norm.bias": self.t3.norm.bias.detach().clone(),
                "head.weight": self.target_head.weight.detach().clone(),
            }
        )


@dataclass(frozen=True)
class H6SuffixTrainer:
    """Reference-only deployment view over one model, one head and one cache."""

    model: nn.Module
    head: TargetPrototypeHead
    cache: H6PrefixCache

    @property
    def cache_tensor_bytes(self) -> int:
        return self.cache.tensor_bytes

    def embedding(self) -> Tensor:
        return forward_h6_suffix(self.model, self.cache)

    def logits(self) -> Tensor:
        return self.head(self.embedding())


def _cache_dtype_from_name(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"unsupported H6 cache dtype: {value}") from exc

def _capture_h6_prefix_cache(
    model: nn.Module,
    values: Tensor,
    *,
    storage_dtype: torch.dtype,
) -> tuple[H6PrefixCache, Mapping[str, Any]]:
    if storage_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("H6 prefix cache storage dtype must be float16, bfloat16 or float32")
    backbone, _ = _identity_backbone(model)
    required = (
        "t3",
        "t_pool",
        "t_proj",
        "meta_adapter_time",
        "fuse",
        "meta_adapter_fusion",
        "cls_head",
        "emb_dim",
    )
    if any(not hasattr(backbone, name) for name in required):
        raise ValueError("model does not expose the H6 cached-suffix interface")
    t3 = getattr(backbone, "t3")
    if not all(hasattr(t3, name) for name in ("norm", "act", "pool", "drop")):
        raise ValueError("model t3 does not expose the H6 cached-suffix interface")

    captured: dict[str, Any] = {}

    def capture_norm(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not args or not torch.is_tensor(args[0]):
            raise ValueError("t3.norm did not receive a tensor")
        captured["pre_t3_norm"] = args[0].detach().to(dtype=storage_dtype).clone()

    def capture_fuse(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not args or not torch.is_tensor(args[0]):
            raise ValueError("identity fuse did not receive a tensor")
        captured["fuse_input"] = args[0].detach().to(dtype=storage_dtype).clone()

    def capture_head(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        del args
        names = ("dac_local", "pa_local", "dac_delta", "pa_delta")
        if any(not torch.is_tensor(kwargs.get(name)) for name in names):
            raise ValueError("identity head did not expose frozen auxiliary tensors")
        captured["cls_head_kwargs"] = {
            name: kwargs[name].detach().to(dtype=storage_dtype).clone() for name in names
        }

    handles = [
        t3.norm.register_forward_pre_hook(capture_norm),
        backbone.fuse.register_forward_pre_hook(capture_fuse),
        backbone.cls_head.register_forward_pre_hook(capture_head, with_kwargs=True),
    ]
    try:
        with torch.no_grad():
            outputs = _forward_aux(model, values)
    finally:
        for handle in handles:
            handle.remove()
    required_captures = {"pre_t3_norm", "fuse_input", "cls_head_kwargs"}
    if set(captured) != required_captures:
        raise ValueError("model forward did not close the H6 prefix cache")
    fuse_input = captured["fuse_input"]
    emb_dim = int(backbone.emb_dim)
    if fuse_input.ndim != 2 or emb_dim <= 0 or fuse_input.size(1) < emb_dim:
        raise ValueError("identity fuse input is incompatible with H6 prefix caching")
    cache = H6PrefixCache(
        pre_t3_norm=captured["pre_t3_norm"],
        frozen_fuse_tail=fuse_input[:, emb_dim:].clone(),
        cls_head_kwargs=MappingProxyType(captured["cls_head_kwargs"]),
        storage_dtype=storage_dtype,
        batch_size=int(values.size(0)),
    )
    return cache, outputs


def _capture_trainable_suffix_prefix(
    model: nn.Module,
    values: Tensor,
    earliest_trainable_node: str,
    *,
    storage_dtype: torch.dtype = torch.float32,
) -> tuple[TimeNormPrefixCache, Mapping[str, Any]]:
    """Capture the frozen graph before the earliest trainable time-path Norm."""

    if earliest_trainable_node not in CompactTimeNormSuffix._ORDER:
        raise ValueError("earliest_trainable_node must name a supported time Norm")
    if storage_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("time suffix cache storage dtype must be floating point")
    backbone, _ = _identity_backbone(model)
    required = (
        "time_fuse", "time_down", "t1", "t2", "t3", "t_pool", "t_proj",
        "meta_adapter_time", "fuse", "meta_adapter_fusion", "cls_head", "emb_dim",
    )
    if any(not hasattr(backbone, name) for name in required):
        raise ValueError("model does not expose the generalized time suffix interface")
    boundary_modules = {
        "time_fuse.1": backbone.time_fuse[1],
        "t1.norm": backbone.t1.norm,
        "t2.norm": backbone.t2.norm,
        "t3.norm": backbone.t3.norm,
    }
    captured: dict[str, Any] = {}

    def capture_norm(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not args or not torch.is_tensor(args[0]):
            raise ValueError("time suffix boundary did not receive a tensor")
        captured["pre_norm"] = args[0].detach().to(dtype=storage_dtype).clone()

    def capture_fuse(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not args or not torch.is_tensor(args[0]):
            raise ValueError("identity fuse did not receive a tensor")
        captured["fuse_input"] = args[0].detach().to(dtype=storage_dtype).clone()

    def capture_head(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        del args
        names = ("dac_local", "pa_local", "dac_delta", "pa_delta")
        if any(not torch.is_tensor(kwargs.get(name)) for name in names):
            raise ValueError("identity head did not expose frozen auxiliary tensors")
        captured["cls_head_kwargs"] = {
            name: kwargs[name].detach().to(dtype=storage_dtype).clone() for name in names
        }

    handles = [
        boundary_modules[earliest_trainable_node].register_forward_pre_hook(capture_norm),
        backbone.fuse.register_forward_pre_hook(capture_fuse),
        backbone.cls_head.register_forward_pre_hook(capture_head, with_kwargs=True),
    ]
    try:
        with torch.no_grad():
            outputs = _forward_aux(model, values)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"pre_norm", "fuse_input", "cls_head_kwargs"}:
        raise ValueError("model forward did not close the generalized time suffix cache")
    fuse_input = captured["fuse_input"]
    emb_dim = int(backbone.emb_dim)
    if fuse_input.ndim != 2 or fuse_input.size(1) < emb_dim:
        raise ValueError("identity fuse input is incompatible with time suffix caching")
    cache = TimeNormPrefixCache(
        boundary=earliest_trainable_node,
        pre_norm=captured["pre_norm"],
        frozen_fuse_tail=fuse_input[:, emb_dim:].clone(),
        cls_head_kwargs=MappingProxyType(captured["cls_head_kwargs"]),
        storage_dtype=storage_dtype,
        batch_size=int(values.size(0)),
    )
    return cache, outputs


def encode_trainable_suffix_prefix(
    model: nn.Module,
    values: Tensor,
    earliest_trainable_node: str,
    *,
    storage_dtype: torch.dtype = torch.float32,
) -> TimeNormPrefixCache:
    cache, _ = _capture_trainable_suffix_prefix(
        model,
        values,
        earliest_trainable_node,
        storage_dtype=storage_dtype,
    )
    return cache


def build_h6_prefix_cache(
    model: nn.Module,
    values: Tensor,
    *,
    storage_dtype: torch.dtype = torch.float32,
) -> H6PrefixCache:
    """Run the frozen graph once and retain only the H6 suffix inputs."""

    cache, _ = _capture_h6_prefix_cache(model, values, storage_dtype=storage_dtype)
    return cache


def encode_h6_prefix(
    model: nn.Module,
    values: Tensor,
    *,
    storage_dtype: torch.dtype = torch.float32,
) -> H6PrefixCache:
    """Stable deployment API for encoding the frozen H6 prefix."""

    return build_h6_prefix_cache(model, values, storage_dtype=storage_dtype)


def forward_h6_prefix_cache(model: nn.Module, cache: H6PrefixCache) -> Tensor:
    """Replay t3.norm and the frozen identity suffix with gradients intact."""

    if not isinstance(cache, H6PrefixCache):
        raise TypeError("cache must be H6PrefixCache")
    backbone, _ = _identity_backbone(model)
    t3 = backbone.t3
    reference = next(t3.norm.parameters())

    def materialize(value: Tensor) -> Tensor:
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError("H6 compute cache must be materialized before suffix replay")
        return value

    t = t3.norm(materialize(cache.pre_t3_norm))
    t = t3.act(t)
    t = t3.pool(t)
    t = t3.drop(t)
    t_emb = backbone.meta_adapter_time(backbone.t_proj(backbone.t_pool(t).squeeze(-1)))
    tail = materialize(cache.frozen_fuse_tail)
    base_input = torch.cat([t_emb, tail], dim=1) if tail.size(1) else t_emb
    base = backbone.meta_adapter_fusion(backbone.fuse(base_input))
    kwargs = {name: materialize(value) for name, value in cache.cls_head_kwargs.items()}
    output = backbone.cls_head(base, labels=None, return_emb=True, **kwargs)
    if isinstance(output, Mapping):
        embedding = output.get("feat_joint")
    elif isinstance(output, (tuple, list)) and output:
        embedding = output[-1]
    else:
        embedding = None
    if not torch.is_tensor(embedding) or embedding.ndim != 2:
        raise ValueError("identity head cached suffix did not expose feat_joint")
    if int(embedding.size(0)) != cache.batch_size:
        raise ValueError("H6 cached suffix batch size drifted")
    return embedding


def forward_h6_suffix(model: nn.Module, cache: H6PrefixCache) -> Tensor:
    """Stable deployment API for replaying the trainable H6 suffix."""

    return forward_h6_prefix_cache(model, cache)


def audit_h6_support_safety(
    model: nn.Module,
    head: TargetPrototypeHead,
    cache: H6PrefixCache,
    support_values: Tensor,
    support_labels: Tensor,
) -> H6SupportSafetyAudit:
    """Compare cached suffix inference with one FP32 full-path support forward."""

    model.eval()
    head.eval()
    with torch.no_grad():
        full_embedding = _extract_joint_embedding(
            _forward_aux(model, support_values), int(support_values.size(0))
        )
        head_dtype = head.weight.dtype
        full_logits = head(full_embedding.to(dtype=head_dtype)).float()
        cached_logits = head(forward_h6_suffix(model, cache)).float()
    if full_logits.shape != cached_logits.shape:
        raise ValueError("H6 support safety logits shape mismatch")
    labels = _local_labels(support_labels, head.class_ids, full_logits.device)
    finite = bool(torch.isfinite(full_logits).all() and torch.isfinite(cached_logits).all())
    full_prediction = full_logits.argmax(dim=1)
    cached_prediction = cached_logits.argmax(dim=1)
    prediction_mismatches = int((full_prediction != cached_prediction).sum().item())
    recall_mismatches = 0
    for class_index in range(len(head.class_ids)):
        mask = labels == class_index
        if not bool(mask.any()):
            raise ValueError("H6 support safety requires every registered class")
        full_correct = int((full_prediction[mask] == class_index).sum().item())
        cached_correct = int((cached_prediction[mask] == class_index).sum().item())
        recall_mismatches += int(full_correct != cached_correct)
    true_logits = full_logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = full_logits.masked_fill(
        F.one_hot(labels, num_classes=full_logits.size(1)).bool(), float("-inf")
    ).max(dim=1).values
    full_margin = true_logits - other_logits
    cached_true = cached_logits.gather(1, labels[:, None]).squeeze(1)
    cached_other = cached_logits.masked_fill(
        F.one_hot(labels, num_classes=cached_logits.size(1)).bool(), float("-inf")
    ).max(dim=1).values
    cached_margin = cached_true - cached_other
    positive_margin_regressions = int(
        ((full_margin >= 0.0) & (cached_margin < 0.0)).sum().item()
    )
    max_abs_logit_delta = float(torch.max(torch.abs(full_logits - cached_logits)).item())
    passed = bool(
        finite
        and prediction_mismatches == 0
        and recall_mismatches == 0
        and positive_margin_regressions == 0
    )
    return H6SupportSafetyAudit(
        passed=passed,
        prediction_mismatches=prediction_mismatches,
        per_class_recall_mismatches=recall_mismatches,
        positive_margin_regressions=positive_margin_regressions,
        max_abs_logit_delta=max_abs_logit_delta,
        minimum_full_path_margin=float(full_margin.min().item()),
    )


def support_hard_pair_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    class_count: int,
    margin: float,
) -> Tensor:
    """Class-permutation-invariant hardest-pair hinge derived from support only."""

    if (
        logits.ndim != 2
        or labels.ndim != 1
        or logits.size(0) != labels.numel()
        or int(class_count) != int(logits.size(1))
        or int(class_count) < 2
    ):
        raise ValueError("hard-pair logits, labels and class_count are incompatible")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("hard-pair margin must be finite and non-negative")
    rows = []
    for class_index in range(int(class_count)):
        mask = labels == class_index
        if not bool(mask.any()):
            raise ValueError("hard-pair loss requires every registered class in support")
        class_logits = logits[mask]
        competing_mean = class_logits.mean(dim=0).clone()
        competing_mean[class_index] = -torch.inf
        hardest = int(competing_mean.argmax().item())
        rows.append(
            F.relu(
                float(margin)
                + class_logits[:, hardest]
                - class_logits[:, class_index]
            ).mean()
        )
    return torch.stack(rows).mean()


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
    """Fit SF-TAPFT while preserving the caller-owned checkpoint model."""

    return _fit_sf_tapft(
        checkpoint_model,
        target_train,
        config,
        checkpoint_validation=checkpoint_validation,
        checkpoint_selection_mode=checkpoint_selection_mode,
        copy_checkpoint_model=True,
    )


def fit_sf_tapft_inplace(
    checkpoint_model: nn.Module,
    target_train: TargetOnlyAdaptationDataset,
    config: SFTAPFTConfig | None = None,
    *,
    checkpoint_validation: TargetOnlyAdaptationDataset | None = None,
    checkpoint_selection_mode: str = "final_step",
) -> SFTAPFTResult:
    """Fit the deployment path on a model instance exclusively owned by the caller."""

    effective_config = config or SFTAPFTConfig()
    result = _fit_sf_tapft(
        checkpoint_model,
        target_train,
        effective_config,
        checkpoint_validation=checkpoint_validation,
        checkpoint_selection_mode=checkpoint_selection_mode,
        copy_checkpoint_model=False,
    )
    if (
        effective_config.cache_storage_dtype in {"float16", "bfloat16"}
        and not result.audit.support_safety_passed
    ):
        named = dict(checkpoint_model.named_parameters())
        with torch.no_grad():
            for anchor_name, anchor in result.base_parameter_anchors.items():
                if not anchor_name.startswith("model."):
                    continue
                name = anchor_name.removeprefix("model.")
                named[name].copy_(anchor.to(device=named[name].device, dtype=named[name].dtype))
        fallback = _fit_sf_tapft(
            checkpoint_model,
            target_train,
            replace(
                effective_config,
                prefix_cache_dtype="float32",
                cache_storage_dtype="float32",
                suffix_compute_dtype="float32",
            ),
            checkpoint_validation=checkpoint_validation,
            checkpoint_selection_mode=checkpoint_selection_mode,
            copy_checkpoint_model=False,
        )
        return replace(
            fallback,
            audit=replace(
                fallback.audit,
                support_safety_fallback_to_float32=True,
            ),
        )
    return result


def _fit_sf_tapft(
    checkpoint_model: nn.Module,
    target_train: TargetOnlyAdaptationDataset,
    config: SFTAPFTConfig | None = None,
    *,
    checkpoint_validation: TargetOnlyAdaptationDataset | None = None,
    checkpoint_selection_mode: str = "best",
    copy_checkpoint_model: bool,
    prototype_reference: TargetOnlyAdaptationDataset | None = None,
    initial_trainable_state: Mapping[str, Tensor] | None = None,
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
    if prototype_reference is not None and not isinstance(
        prototype_reference, TargetOnlyAdaptationDataset
    ):
        raise TypeError("prototype_reference must be TargetOnlyAdaptationDataset")
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
    if not copy_checkpoint_model:
        if checkpoint_validation is not None or checkpoint_selection_mode != "final_step":
            raise ValueError(
                "in-place SF-TAPFT requires final_step selection without validation"
            )
        if config.selective_kd_weight > 0.0:
            raise ValueError("in-place SF-TAPFT does not retain a full teacher model")
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))

    teacher = (
        copy.deepcopy(checkpoint_model) if config.selective_kd_weight > 0.0 else None
    )
    student = copy.deepcopy(checkpoint_model) if copy_checkpoint_model else checkpoint_model
    if teacher is not None:
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    source_weights = _source_classifier_weight(checkpoint_model).detach().clone()
    source_class_ids = tuple(range(int(source_weights.size(0))))
    ensure_time_adapter(student, rank=config.adapter_rank)
    student.eval()
    device = next(student.parameters()).device
    dtype = next(parameter.dtype for parameter in student.parameters() if parameter.is_floating_point())
    target_x = target_train.received_iq.to(device=device, dtype=dtype)
    target_labels = target_train.labels.to(device=device, dtype=torch.long)
    physical_support_count = int(target_x.size(0))
    effective_view_count = 1
    if config.rse_view_weight > 0.0:
        target_x = torch.cat(
            (
                target_x,
                phase_rotate_iq(target_x, radians=config.rse_view_phase_radians),
            ),
            dim=0,
        )
        target_labels = torch.cat((target_labels, target_labels), dim=0)
        effective_view_count = 2

    prefix_cache: H6PrefixCache | TimeNormPrefixCache | None = None
    prefix_cache_build_forward_steps = 0
    if config.cache_storage_dtype != "off":
        normalized_rules = (
            (config.pace_norm_rules or config.norm_rules)
            if (config.pace_norm_rules or config.norm_rules)
            else tuple(
                (scope, config.norm_affine) for scope in _NORM_SCOPES[config.norm_scope]
            )
        )
        if config.trainability_profile != "p1_head_norm":
            raise ValueError("time prefix caching requires p1_head_norm")
        storage_dtype = _cache_dtype_from_name(config.cache_storage_dtype)
        if normalized_rules == (("t3", "weight_bias"),):
            prefix_cache, initial_outputs = _capture_h6_prefix_cache(
                student, target_x, storage_dtype=storage_dtype
            )
        else:
            scope_order = ("time_fuse", "t1", "t2", "t3")
            selected_scopes = {scope for scope, _ in normalized_rules}
            earliest_scope = next(scope for scope in scope_order if scope in selected_scopes)
            boundary = "time_fuse.1" if earliest_scope == "time_fuse" else f"{earliest_scope}.norm"
            prefix_cache, initial_outputs = _capture_trainable_suffix_prefix(
                student, target_x, boundary, storage_dtype=storage_dtype
            )
        storage_device = (
            device
            if config.cache_device == "model"
            else torch.device(config.cache_device)
        )
        cache_tensor = (
            prefix_cache.pre_t3_norm
            if isinstance(prefix_cache, H6PrefixCache)
            else prefix_cache.pre_norm
        )
        if storage_device != cache_tensor.device:
            prefix_cache = prefix_cache.materialize_once(
                device=storage_device,
                dtype=storage_dtype,
            )
        prefix_cache = prefix_cache.materialize_once(
            device=device,
            dtype=_cache_dtype_from_name(config.suffix_compute_dtype),
        )
        prefix_cache_build_forward_steps = effective_view_count
    else:
        with torch.no_grad():
            initial_outputs = _forward_aux(student, target_x)
    with torch.no_grad():
        initial_embeddings = _extract_joint_embedding(
            initial_outputs, int(target_x.size(0))
        )
        target_class_ids = target_train.class_ids
        prototype_embeddings = initial_embeddings[:physical_support_count]
        prototype_labels = target_labels[:physical_support_count]
        frozen_source_logits = initial_outputs.get("tx_logits", initial_outputs.get("logits"))
        if not torch.is_tensor(frozen_source_logits):
            raise ValueError("frozen source model must expose logits for SF-TAPFT")
        if max(target_class_ids) >= frozen_source_logits.size(1):
            raise ValueError("target support class is absent from frozen source logits")
        if prototype_reference is not None:
            reference_x = prototype_reference.received_iq.to(device=device, dtype=dtype)
            reference_labels = prototype_reference.labels.to(device=device, dtype=torch.long)
            reference_outputs = _forward_aux(student, reference_x)
            prototype_embeddings = _extract_joint_embedding(
                reference_outputs, len(prototype_reference.physical_ids)
            )
            prototype_labels = reference_labels
            frozen_source_logits = reference_outputs.get(
                "tx_logits", reference_outputs.get("logits")
            )
            if not torch.is_tensor(frozen_source_logits):
                raise ValueError("prototype reference must expose frozen source logits")
        prototypes = _target_prototypes(
            prototype_embeddings, prototype_labels, target_class_ids
        )
        if config.use_class_adaptive_rho:
            rho_values, reliability_values = class_adaptive_rho(
                prototype_embeddings,
                frozen_source_logits[:, list(target_class_ids)],
                _local_labels(
                    prototype_labels, target_class_ids, device
                ),
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
    if initial_trainable_state is not None:
        student_named_initial = dict(student.named_parameters())
        with torch.no_grad():
            for name, value in initial_trainable_state.items():
                if name.startswith("model."):
                    parameter_name = name.removeprefix("model.")
                    if parameter_name not in student_named_initial:
                        raise ValueError(
                            f"initial RSE state has unknown model parameter: {parameter_name}"
                        )
                    student_named_initial[parameter_name].copy_(
                        value.to(student_named_initial[parameter_name])
                    )
                elif name == "head.weight":
                    head.weight.copy_(value.to(head.weight))
                else:
                    raise ValueError(f"initial RSE state has unregistered key: {name}")
    local_labels = _local_labels(target_labels, head.class_ids, device)
    class_weights = _class_balanced_weights(local_labels, len(head.class_ids)).to(dtype=dtype)

    policy = ProgressiveTrainabilityPolicy(
        config.trainability_profile,
        norm_scope=config.norm_scope,
        norm_affine=config.norm_affine,
        norm_rules=(config.pace_norm_rules or config.norm_rules),
    )
    expanded_phase_names = {
        phase: policy.parameter_names(student, phase) for phase in _PHASES
    }
    phase_names = dict(expanded_phase_names)
    base_phase_names = dict(expanded_phase_names)
    if config.pace_norm_rules:
        base_policy = ProgressiveTrainabilityPolicy(
            config.trainability_profile,
            norm_scope=config.norm_scope,
            norm_affine=config.norm_affine,
            norm_rules=config.norm_rules,
        )
        base_phase_names = {
            phase: base_policy.parameter_names(student, phase) for phase in _PHASES
        }
        phase_names["A"] = base_phase_names["A"]
    norm_names = {
        name
        for phase in _PHASES
        for name in phase_names[phase]
        if ".norm." in name or ".time_fuse.1." in name
    }
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
    initial_model_state = (
        {name: value.detach().clone() for name, value in student.state_dict().items()}
        if copy_checkpoint_model
        else {
            name: named[name].detach().clone()
            for name in sorted(norm_names | adapter_names | last_names | fusion_names)
        }
    )
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
    compact_suffix: CompactH6Suffix | CompactTimeNormSuffix | None = None
    training_named = dict(named)
    if prefix_cache is not None and not copy_checkpoint_model:
        compact_suffix = (
            CompactH6Suffix.from_model(student, head, prefix_cache)
            if isinstance(prefix_cache, H6PrefixCache)
            else CompactTimeNormSuffix.from_model(student, head, prefix_cache)
        )
        head = compact_suffix.target_head
        compact_named = dict(compact_suffix.named_parameters())
        for name in norm_names:
            compact_name = name.removeprefix(identity_prefix)
            if compact_name not in compact_named:
                raise ValueError(f"compact suffix does not expose permitted parameter: {name}")
            training_named[name] = compact_named[compact_name]
    groups = [
        {"name": "head", "params": list(head.parameters()), "lr": config.lr_head_initial},
        {"name": "norm", "params": [training_named[name] for name in sorted(norm_names)], "lr": config.lr_norm},
        {"name": "adapter", "params": [training_named[name] for name in sorted(adapter_names)], "lr": config.lr_adapter},
        {"name": "last", "params": [training_named[name] for name in sorted(last_names)], "lr": config.lr_last_block},
        {"name": "fusion", "params": [training_named[name] for name in sorted(fusion_names)], "lr": config.lr_fusion},
    ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.weight_decay))
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    permitted_optimizer_ids = {
        *(
            id(training_named[name])
            for name in (norm_names | adapter_names | last_names | fusion_names)
        ),
        *(id(parameter) for parameter in head.parameters()),
    }
    if optimizer_parameter_ids != permitted_optimizer_ids:
        raise RuntimeError("SF-TAPFT optimizer can reach a non-permitted parameter")
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = _make_grad_scaler(device, enabled=use_amp)
    l2sp_names = sorted(norm_names | last_names | fusion_names)
    l2sp = (
        L2SPRegularizer.from_named_parameters((name, training_named[name]) for name in l2sp_names)
        if l2sp_names
        else None
    )
    bn_before = {
        name: value.detach().clone() for name, value in student.state_dict().items()
        if name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked")
    }
    total_steps = sum(config.phase_steps)
    scheduler_steps = int(config.scheduler_reference_steps) or total_steps
    validation_step_set = set(config.validation_steps)
    losses: list[float] = []
    head_cvar_losses: list[float] = []
    pace_tail_losses: list[float] = []
    pace_preserve_losses: list[float] = []
    view_consistency_losses: list[float] = []
    retained_trainable_snapshots: dict[int, Mapping[str, Tensor]] = {}
    snapshots: list[
        tuple[dict[str, Tensor], float | tuple[float, ...], int]
    ] = []
    stage_best: dict[str, tuple[int, int, StageValidationMetrics]] = {}
    stage_end: dict[str, StageValidationMetrics] = {}
    current_phase = ""
    current_trainability_key: tuple[str, bool] | None = None
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
    cached_suffix_forward_steps = 0
    pace_teacher_logits: Tensor | None = None
    pace_stable_weights: Tensor | None = None
    pace_expanded_optimizer_steps = 0

    for step in range(total_steps):
        phase = _phase_for_step(step, config.phase_steps)
        pace_active = bool(
            config.pace_expand_start_step and step >= config.pace_expand_start_step
        )
        if config.pace_expand_start_step:
            polishing = bool(
                config.head_polish_steps > 0
                and config.pace_expand_start_step - config.head_polish_steps <= step
                < config.pace_expand_start_step
            )
        else:
            polishing = config.head_polish_steps > 0 and step >= total_steps - config.head_polish_steps
        optimization_phase = "HEAD" if step < config.head_prefit_steps or polishing else phase
        if config.pace_expand_start_step and step == config.pace_expand_start_step:
            with torch.no_grad():
                teacher_embeddings = (
                    compact_suffix.embedding()
                    if compact_suffix is not None
                    else (
                        forward_h6_prefix_cache(student, prefix_cache)
                        if isinstance(prefix_cache, H6PrefixCache)
                        else _extract_joint_embedding(
                            _forward_aux(student, target_x), int(target_x.size(0))
                        )
                    )
                )
                pace_teacher_logits = head(teacher_embeddings).detach()
                pace_stable_weights = stable_support_weights(
                    pace_teacher_logits, local_labels
                )
        trainability_key = (optimization_phase, pace_active)
        if trainability_key != current_trainability_key:
            if optimization_phase == "HEAD":
                student.eval()
                for parameter in student.parameters():
                    parameter.requires_grad_(False)
                if polishing and cached_head_embeddings is None:
                    with torch.no_grad():
                        cached_head_embeddings = (
                            (
                                compact_suffix.embedding()
                                if compact_suffix is not None
                                else forward_h6_prefix_cache(student, prefix_cache)
                            )
                            if prefix_cache is not None
                            else _extract_joint_embedding(
                                _forward_aux(student, target_x), int(target_x.size(0))
                            )
                        ).detach()
                    cached_head_forward_steps += 1
            else:
                allowed_names = set(
                    expanded_phase_names[phase]
                    if pace_active
                    else base_phase_names[phase]
                )
                if compact_suffix is None:
                    student.eval()
                    for name, parameter in student.named_parameters():
                        parameter.requires_grad_(name in allowed_names)
                else:
                    for parameter in compact_suffix.parameters():
                        parameter.requires_grad_(False)
                    compact_named = dict(compact_suffix.named_parameters())
                    for name in allowed_names:
                        compact_named[name.removeprefix(identity_prefix)].requires_grad_(True)
            for parameter in head.parameters():
                parameter.requires_grad_(True)
            current_phase = optimization_phase
            current_trainability_key = trainability_key
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
                if prefix_cache is not None:
                    embeddings = (
                        compact_suffix.embedding()
                        if compact_suffix is not None
                        else forward_h6_prefix_cache(student, prefix_cache)
                    )
                    cached_suffix_forward_steps += 1
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
            cvar = logits.new_zeros(())
            cvar_active = (
                config.head_cvar_steps > 0
                and step >= total_steps - config.head_cvar_steps
            ) or (pace_active and config.pace_tail_weight > 0.0)
            if cvar_active:
                row_losses = F.cross_entropy(
                    logits,
                    local_labels,
                    reduction="none",
                    label_smoothing=float(config.label_smoothing),
                )
                class_losses = torch.stack(
                    [
                        row_losses[local_labels == class_index].mean()
                        for class_index in range(len(head.class_ids))
                    ]
                )
                cvar = class_cvar_from_class_losses(
                    class_losses,
                    top_k=config.head_cvar_top_k,
                )
            proto_embeddings = (
                embeddings[:physical_support_count]
                if effective_view_count == 2
                else embeddings
            )
            proto_labels = (
                local_labels[:physical_support_count]
                if effective_view_count == 2
                else local_labels
            )
            proto_logits = leave_one_out_prototype_logits(
                proto_embeddings,
                proto_labels,
                class_count=len(head.class_ids),
                fallback_weights=head.weight,
                scale=config.prototype_scale,
            )
            proto = F.cross_entropy(proto_logits, proto_labels, weight=class_weights)
            hard_pair = (
                support_hard_pair_loss(
                    logits,
                    local_labels,
                    class_count=len(head.class_ids),
                    margin=config.hard_pair_margin,
                )
                if config.hard_pair_weight > 0.0
                else logits.new_zeros(())
            )
            anchor = (
                l2sp((name, training_named[name]) for name in l2sp_names)
                if l2sp is not None
                else logits.new_zeros(())
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
            preservation = logits.new_zeros(())
            if pace_active and config.pace_preserve_weight > 0.0:
                if pace_teacher_logits is None or pace_stable_weights is None:
                    raise RuntimeError("PACE expansion started without a D0 teacher snapshot")
                preservation = stable_preservation_kl(
                    logits,
                    pace_teacher_logits,
                    pace_stable_weights,
                    temperature=config.pace_preserve_temperature,
                )
            view_consistency = logits.new_zeros(())
            if effective_view_count == 2:
                original_logits = logits[:physical_support_count]
                augmented_logits = logits[physical_support_count:]
                original_prob = F.softmax(original_logits.float(), dim=1)
                augmented_prob = F.softmax(augmented_logits.float(), dim=1)
                mixture = 0.5 * (original_prob + augmented_prob)
                view_consistency = 0.5 * (
                    F.kl_div(mixture.log(), original_prob, reduction="batchmean")
                    + F.kl_div(mixture.log(), augmented_prob, reduction="batchmean")
                )
            loss = (
                ce
                + float(config.lambda_proto) * proto
                + float(config.lambda_l2sp) * anchor
                + float(config.selective_kd_weight) * kd
                + float(config.head_anchor_weight) * head_anchor
                + float(config.hard_pair_weight) * hard_pair
                + float(config.head_cvar_weight) * cvar
                + float(config.pace_tail_weight) * cvar
                + float(config.pace_preserve_weight) * preservation
                + float(config.rse_view_weight) * view_consistency
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("SF-TAPFT loss became non-finite")
        scaler.scale(loss).backward()
        trainable = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, float(config.gradient_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        if pace_active:
            pace_expanded_optimizer_steps += 1
        if ema is not None:
            ema.update(
                {
                    **{
                        f"model.{name}": training_named[name]
                        for name in sorted(permitted_model_names)
                    },
                    **{f"head.{name}": value for name, value in head.state_dict().items()},
                }
            )
        loss_value = float(loss.detach())
        losses.append(loss_value)
        if cvar_active:
            head_cvar_losses.append(float(cvar.detach()))
        if pace_active:
            pace_tail_losses.append(float(cvar.detach()))
            pace_preserve_losses.append(float(preservation.detach()))
        if effective_view_count == 2:
            view_consistency_losses.append(float(view_consistency.detach()))
        if step + 1 in set(config.rse_snapshot_steps):
            retained_trainable_snapshots[step + 1] = MappingProxyType(
                {
                    **{
                        f"model.{name}": training_named[name].detach().cpu().clone()
                        for name in sorted(permitted_model_names)
                    },
                    **{
                        f"head.{name}": value.detach().cpu().clone()
                        for name, value in head.state_dict().items()
                    },
                }
            )
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
                    f"model.{name}": training_named[name].detach().clone()
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
    if copy_checkpoint_model:
        restored_model_state = {
            name: value.detach().clone() for name, value in initial_model_state.items()
        }
        restored_model_state.update(averaged_model)
        student.load_state_dict(restored_model_state)
    else:
        with torch.no_grad():
            for name, value in averaged_model.items():
                named[name].copy_(value)
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
        **{
            f"model.{name}": dict(student.state_dict())[name]
            for name in (
                student.state_dict().keys()
                if copy_checkpoint_model
                else sorted(permitted_model_names)
            )
        },
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
    support_safety = (
        audit_h6_support_safety(student, head, prefix_cache, target_x, target_labels)
        if isinstance(prefix_cache, H6PrefixCache)
        else None
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
        backbone_train_forward_steps=(
            0
            if prefix_cache is not None
            else total_steps - config.head_prefit_steps - config.head_polish_steps
        ),
        validation_forward_steps=tuple(validation_forward_steps),
        snapshot_tensor_bytes=sum(
            int(value.numel() * value.element_size()) for value in compact_anchor_state.values()
        ),
        trainable_parameter_elements=(
            sum(int(training_named[name].numel()) for name in permitted_model_names)
            + sum(int(parameter.numel()) for parameter in head.parameters())
        ),
        actual_changed_elements=sum(
            int(final_state[name].numel()) for name in permitted_changed
        ),
        head_polish_steps=config.head_polish_steps,
        cached_head_forward_steps=cached_head_forward_steps,
        head_cvar_steps=config.head_cvar_steps,
        head_cvar_weight=float(config.head_cvar_weight),
        head_cvar_top_k=config.head_cvar_top_k,
        head_cvar_losses=tuple(head_cvar_losses),
        trainable_delta_ema_decay=float(config.trainable_delta_ema_decay),
        class_adaptive_rho=tuple(float(value) for value in rho_values.detach().cpu()),
        class_reliability=tuple(
            float(value) for value in reliability_values.detach().cpu()
        ),
        prefix_cache_dtype=config.cache_storage_dtype,
        prefix_cache_build_forward_steps=prefix_cache_build_forward_steps,
        cached_suffix_forward_steps=cached_suffix_forward_steps,
        hard_pair_weight=float(config.hard_pair_weight),
        hard_pair_margin=float(config.hard_pair_margin),
        prefix_cache_tensor_bytes=(
            prefix_cache.tensor_bytes if prefix_cache is not None else 0
        ),
        support_safety_checked=support_safety is not None,
        support_safety_passed=(
            support_safety.passed if support_safety is not None else True
        ),
        support_safety_prediction_mismatches=(
            support_safety.prediction_mismatches if support_safety is not None else 0
        ),
        support_safety_per_class_recall_mismatches=(
            support_safety.per_class_recall_mismatches
            if support_safety is not None
            else 0
        ),
        support_safety_max_abs_logit_delta=(
            support_safety.max_abs_logit_delta if support_safety is not None else 0.0
        ),
        support_safety_fallback_to_float32=False,
        pace_expand_start_step=int(config.pace_expand_start_step),
        pace_teacher_snapshot_count=int(pace_teacher_logits is not None),
        pace_expanded_optimizer_steps=pace_expanded_optimizer_steps,
        pace_tail_losses=tuple(pace_tail_losses),
        pace_preserve_losses=tuple(pace_preserve_losses),
        effective_view_count=effective_view_count,
        view_consistency_losses=tuple(view_consistency_losses),
        suffix_backward_steps=(
            total_steps - config.head_prefit_steps - config.head_polish_steps
        ),
        head_optimizer_steps=total_steps,
    )
    return SFTAPFTResult(
        model=student,
        head=head,
        audit=audit,
        base_parameter_anchors=MappingProxyType(
            {
                name: value.detach().cpu().clone()
                for name, value in compact_anchor_state.items()
            }
        ),
        retained_trainable_snapshots=MappingProxyType(retained_trainable_snapshots),
    )


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


def _apply_trainable_state(result: SFTAPFTResult, state: Mapping[str, Tensor]) -> SFTAPFTResult:
    model = copy.deepcopy(result.model)
    head = copy.deepcopy(result.head)
    model_named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if name.startswith("model."):
                parameter_name = name.removeprefix("model.")
                if parameter_name not in model_named:
                    raise ValueError(f"RSE state has unknown model parameter: {parameter_name}")
                model_named[parameter_name].copy_(value.to(model_named[parameter_name]))
            elif name == "head.weight":
                head.weight.copy_(value.to(head.weight))
            else:
                raise ValueError(f"RSE state has an unregistered key: {name}")
    model.eval()
    head.eval()
    for parameter in (*model.parameters(), *head.parameters()):
        parameter.requires_grad_(False)
    changed = tuple(
        sorted(
            name
            for name, value in state.items()
            if not torch.equal(value.detach().cpu(), result.base_parameter_anchors[name].detach().cpu())
        )
    )
    audit = replace(
        result.audit,
        updated_parameter_names=tuple(
            name.removeprefix("model.") for name in changed if name.startswith("model.")
        ),
        permitted_changed_names=changed,
        nonpermitted_changed_names=(),
        actual_changed_elements=sum(int(state[name].numel()) for name in changed),
    )
    return replace(result, model=model, head=head, audit=audit)


def fit_sf_tapft_rse_strength_selection(
    checkpoint_model: nn.Module,
    target_train: TargetOnlyAdaptationDataset,
    config: SFTAPFTConfig,
    *,
    steps: Sequence[int] = (250, 350, 450, 520),
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    repeats: int = 2,
    folds: int = 2,
) -> RSEStrengthSelectionResult:
    """Select E0 stopping point and delta strength using support-only repeated cross-fit."""

    steps = tuple(int(value) for value in steps)
    alphas = tuple(float(value) for value in alphas)
    if tuple(sorted(set(steps))) != steps or not steps or steps[-1] > sum(config.phase_steps):
        raise ValueError("RSE steps must be sorted, unique, and within the training trajectory")
    if tuple(sorted(set(alphas))) != alphas or not alphas or alphas[0] != 0.0:
        raise ValueError("RSE alphas must be sorted, unique, and start at zero")
    fold_risks: dict[tuple[int, float], list[float]] = {
        (step, alpha): [] for step in steps for alpha in alphas
    }
    rows: list[RSEStrengthFoldRow] = []
    fit_count = 0
    validation_forward_steps = 0
    validation_suffix_forward_steps = 0
    fold_config = replace(
        config,
        rse_snapshot_steps=steps,
        validation_steps=(),
        checkpoint_average_top_k=1,
    )
    for repeat_index in range(int(repeats)):
        selector = GroupedTargetCVSelector(
            folds=int(folds), seed=int(config.seed) + repeat_index
        )
        for fold_index, (train_indices, validation_indices) in enumerate(
            selector.split(labels=target_train.labels, groups=target_train.groups)
        ):
            inner_train = _subset_target_train(target_train, train_indices)
            inner_validation = _subset_target_train(target_train, validation_indices)
            fitted = fit_sf_tapft(
                copy.deepcopy(checkpoint_model),
                inner_train,
                fold_config,
                checkpoint_selection_mode="final_step",
            )
            fit_count += 1
            validation_device = next(fitted.model.parameters()).device
            validation_dtype = next(
                parameter.dtype
                for parameter in fitted.model.parameters()
                if parameter.is_floating_point()
            )
            validation_x = inner_validation.received_iq.to(
                device=validation_device, dtype=validation_dtype
            )
            validation_cache, _ = _capture_h6_prefix_cache(
                fitted.model, validation_x, storage_dtype=torch.float32
            )
            validation_forward_steps += 1
            augmented_validation_cache = None
            if float(config.rse_view_weight) > 0.0:
                augmented_validation_cache, _ = _capture_h6_prefix_cache(
                    fitted.model,
                    phase_rotate_iq(
                        validation_x, radians=float(config.rse_view_phase_radians)
                    ),
                    storage_dtype=torch.float32,
                )
                validation_forward_steps += 1
            anchor_candidate = _apply_trainable_state(
                fitted, fitted.base_parameter_anchors
            )
            with torch.no_grad():
                anchor_logits = anchor_candidate.head(
                    forward_h6_prefix_cache(anchor_candidate.model, validation_cache)
                )
            labels = _local_labels(
                inner_validation.labels,
                anchor_candidate.head.class_ids,
                validation_device,
            )
            validation_suffix_forward_steps += 1
            for step in steps:
                snapshot = fitted.retained_trainable_snapshots[step]
                for alpha in alphas:
                    state = interpolate_trainable_state(
                        fitted.base_parameter_anchors, snapshot, alpha=alpha
                    )
                    candidate = _apply_trainable_state(fitted, state)
                    with torch.no_grad():
                        logits = candidate.head(
                            forward_h6_prefix_cache(candidate.model, validation_cache)
                        )
                    validation_suffix_forward_steps += 1
                    second_view_logits = None
                    if augmented_validation_cache is not None:
                        with torch.no_grad():
                            second_view_logits = candidate.head(
                                forward_h6_prefix_cache(
                                    candidate.model, augmented_validation_cache
                                )
                            )
                        validation_suffix_forward_steps += 1
                    risk = robust_support_risk(
                        logits,
                        labels,
                        frozen_logits=anchor_logits,
                        second_view_logits=second_view_logits,
                    )
                    fold_risks[(step, alpha)].append(risk.total)
                    rows.append(
                        RSEStrengthFoldRow(
                            repeat=repeat_index,
                            fold=fold_index,
                            step=step,
                            alpha=alpha,
                            risk=risk,
                        )
                    )
    selected_step, selected_alpha = select_rse_strength(fold_risks)
    final_config = replace(
        config,
        validation_steps=(),
        rse_snapshot_steps=(selected_step,),
        checkpoint_average_top_k=1,
    )
    final_fit = fit_sf_tapft(
        copy.deepcopy(checkpoint_model),
        target_train,
        final_config,
        checkpoint_selection_mode="final_step",
    )
    committed_state = interpolate_trainable_state(
        final_fit.base_parameter_anchors,
        final_fit.retained_trainable_snapshots[selected_step],
        alpha=selected_alpha,
    )
    committed = _apply_trainable_state(final_fit, committed_state)
    committed = replace(
        committed,
        audit=replace(
            committed.audit,
            selected_checkpoint_steps=(selected_step,),
        ),
    )
    return RSEStrengthSelectionResult(
        result=committed,
        selected_step=selected_step,
        selected_alpha=selected_alpha,
        fold_rows=tuple(rows),
        crossfit_fit_count=fit_count,
        crossfit_validation_forward_steps=validation_forward_steps,
        crossfit_validation_suffix_forward_steps=validation_suffix_forward_steps,
    )


def _trainable_state_from_result(result: SFTAPFTResult) -> Mapping[str, Tensor]:
    model_named = dict(result.model.named_parameters())
    return MappingProxyType(
        {
            name: (
                model_named[name.removeprefix("model.")].detach().cpu().clone()
                if name.startswith("model.")
                else result.head.weight.detach().cpu().clone()
            )
            for name in result.base_parameter_anchors
        }
    )


def fit_sf_tapft_rse_delta_ensemble(
    checkpoint_model: nn.Module,
    target_train: TargetOnlyAdaptationDataset,
    config: SFTAPFTConfig,
    *,
    ensemble_count: int = 2,
    per_class: int = 8,
    polish_steps: int = 30,
) -> RSEDeltaEnsembleResult:
    """Average two aligned E0 support-subset deltas, then polish on full support."""

    subset_indices = balanced_rse_subsets(
        target_train.labels,
        per_class=int(per_class),
        count=int(ensemble_count),
        seed=int(config.seed),
    )
    subset_results = []
    for indices in subset_indices:
        subset_results.append(
            _fit_sf_tapft(
                copy.deepcopy(checkpoint_model),
                _subset_target_train(target_train, indices),
                config,
                checkpoint_selection_mode="final_step",
                copy_checkpoint_model=True,
                prototype_reference=target_train,
            )
        )
    common_anchor = subset_results[0].base_parameter_anchors
    for result in subset_results[1:]:
        if set(result.base_parameter_anchors) != set(common_anchor) or any(
            not torch.equal(
                result.base_parameter_anchors[name].detach().cpu(),
                common_anchor[name].detach().cpu(),
            )
            for name in common_anchor
        ):
            raise RuntimeError("RSE subset fits do not share one trainable anchor")
    averaged_state = average_trainable_states(
        common_anchor,
        tuple(_trainable_state_from_result(result) for result in subset_results),
    )
    if isinstance(polish_steps, bool) or int(polish_steps) <= 0:
        raise ValueError("RSE polish_steps must be positive")
    polish_config = replace(
        config,
        phase_steps=(0, int(polish_steps), 0),
        scheduler_reference_steps=max(int(config.scheduler_reference_steps), int(polish_steps)),
        lr_head_middle=min(float(config.lr_head_middle), 5.0e-5),
        lr_norm=min(float(config.lr_norm), 1.0e-5),
        fast_tail_start_step=0,
        fast_tail_steps=0,
        head_polish_steps=0,
        validation_steps=(),
        rse_snapshot_steps=(),
        checkpoint_average_top_k=1,
        warmup_ratio=0.0,
    )
    polished = _fit_sf_tapft(
        copy.deepcopy(checkpoint_model),
        target_train,
        polish_config,
        checkpoint_selection_mode="final_step",
        copy_checkpoint_model=True,
        prototype_reference=target_train,
        initial_trainable_state=averaged_state,
    )
    polished = replace(
        polished,
        base_parameter_anchors=MappingProxyType(
            {name: value.detach().cpu().clone() for name, value in common_anchor.items()}
        ),
    )
    polished = _apply_trainable_state(polished, _trainable_state_from_result(polished))
    return RSEDeltaEnsembleResult(
        result=polished,
        subset_indices=subset_indices,
        subset_fit_count=len(subset_results),
        polish_steps=int(polish_steps),
        common_anchor=common_anchor,
    )


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


@dataclass(frozen=True)
class HeadBiasOOFCalibration:
    calibration: BiasCalibration
    folds: int
    head_only_steps: int
    embedding_forward_steps: int


def fit_sf_tapft_support_oof_head_bias(
    result: SFTAPFTResult,
    target_train: TargetOnlyAdaptationDataset,
    *,
    folds: int = 4,
    head_only_steps: int = 40,
    lr: float = 0.05,
    l2: float = 0.01,
    seed: int = 392002,
) -> HeadBiasOOFCalibration:
    """Cross-fit only the head on one frozen final-support embedding matrix."""

    if not isinstance(result, SFTAPFTResult):
        raise TypeError("result must be SFTAPFTResult")
    selector = GroupedTargetCVSelector(folds=int(folds), seed=int(seed))
    splits = selector.split(labels=target_train.labels, groups=target_train.groups)
    model = result.model
    device = next(model.parameters()).device
    dtype = next(parameter.dtype for parameter in model.parameters() if parameter.is_floating_point())
    support_x = target_train.received_iq.to(device=device, dtype=dtype)
    local_labels = _local_labels(target_train.labels, result.head.class_ids, device)
    model.eval()
    with torch.no_grad():
        embeddings = _extract_joint_embedding(
            _forward_aux(model, support_x), int(support_x.size(0))
        ).detach()
    oof_logits = torch.empty(
        (embeddings.size(0), len(result.head.class_ids)),
        device=device,
        dtype=torch.float32,
    )
    for train_indices, validation_indices in splits:
        train_index = torch.tensor(train_indices, device=device, dtype=torch.long)
        validation_index = torch.tensor(validation_indices, device=device, dtype=torch.long)
        fold_head = copy.deepcopy(result.head).to(device=device, dtype=dtype)
        fold_head.bias.zero_()
        for parameter in fold_head.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            fold_head.parameters(), lr=float(lr), weight_decay=float(l2)
        )
        for _ in range(int(head_only_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                fold_head(embeddings[train_index]), local_labels[train_index]
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            oof_logits[validation_index] = fold_head(embeddings[validation_index]).float()
    calibration = fit_zero_sum_class_bias(
        oof_logits,
        local_labels,
        steps=int(head_only_steps),
        lr=float(lr),
        l2=float(l2),
    )
    with torch.no_grad():
        result.head.bias.copy_(
            calibration.bias.to(device=result.head.bias.device, dtype=result.head.bias.dtype)
        )
    return HeadBiasOOFCalibration(
        calibration=calibration,
        folds=int(folds),
        head_only_steps=int(head_only_steps) * int(folds) + int(head_only_steps),
        embedding_forward_steps=1,
    )


def fit_sf_tapft_support_oof_temperature(
    checkpoint_model: nn.Module,
    target_train: TargetOnlyAdaptationDataset,
    config: SFTAPFTConfig,
    *,
    folds: int = 4,
) -> TemperatureCalibration:
    """Fit one temperature from fixed-step support OOF logits without model selection."""

    selector = GroupedTargetCVSelector(folds=int(folds), seed=config.seed)
    splits = selector.split(labels=target_train.labels, groups=target_train.groups)
    fold_config = replace(
        config,
        oof_temperature_calibration=False,
        validation_steps=(),
        checkpoint_average_top_k=1,
    )
    adapted_oof: list[tuple[Tensor, Tensor]] = []
    for train_indices, validation_indices in splits:
        inner_train = _subset_target_train(target_train, train_indices)
        inner_validation = _subset_target_train(target_train, validation_indices)
        fitted = fit_sf_tapft(
            copy.deepcopy(checkpoint_model),
            inner_train,
            fold_config,
            checkpoint_selection_mode="final_step",
        )
        logits, labels = _adapted_validation_logits(fitted, inner_validation)
        adapted_oof.append(
            (
                (logits / float(config.inference_temperature)).detach().cpu(),
                labels.detach().cpu(),
            )
        )
    return fit_positive_temperature(
        torch.cat([logits for logits, _ in adapted_oof]),
        torch.cat([labels for _, labels in adapted_oof]),
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
    "BiasCalibration",
    "CheckpointAverager",
    "CompactTimeNormSuffix",
    "FoldMetrics",
    "GroupedTargetCVSelector",
    "HeadBiasOOFCalibration",
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
    "TimeNormPrefixCache",
    "TrainableDeltaAverager",
    "TrainableDeltaEMA",
    "class_adaptive_rho",
    "ensure_time_adapter",
    "encode_trainable_suffix_prefix",
    "fit_sf_tapft",
    "fit_positive_temperature",
    "fit_zero_sum_class_bias",
    "fit_sf_tapft_support_oof_temperature",
    "fit_sf_tapft_support_oof_head_bias",
    "leave_one_out_prototype_logits",
    "stable_preservation_kl",
    "stable_support_weights",
    "select_sf_tapft_by_grouped_cv",
]
