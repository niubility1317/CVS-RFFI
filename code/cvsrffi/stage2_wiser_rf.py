"""Strict WB-FT core for WISER-RF Stage2-B adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as F


_STAGE1_PREFIXES = (
    "id_backbone.t3.",
    "id_backbone.f3.",
    "id_backbone.t_proj.",
    "id_backbone.f_proj.",
    "id_backbone.fuse.",
    "id_backbone.cls_head.id_proj.",
    "id_backbone.cls_head.id_gate.",
    "id_backbone.cls_head.joint_proj.",
)
_STAGE2_PREFIXES = _STAGE1_PREFIXES + (
    "id_backbone.t2.",
    "id_backbone.f2.",
)
_STAGE3_PREFIXES = _STAGE2_PREFIXES + (
    "id_backbone.t1.",
    "id_backbone.f1.",
    "id_backbone.time_fuse.",
    "id_backbone.freq_gate.",
    "id_backbone.freq_stats_proj.",
)


_P3_STAGE1_TIME_PREFIXES = (
    "id_backbone.t3.",
    "id_backbone.t_proj.",
    "id_backbone.fuse.",
    "id_backbone.cls_head.id_proj.",
    "id_backbone.cls_head.id_gate.",
    "id_backbone.cls_head.joint_proj.",
)
_P3_STAGE2_PREFIXES = {
    "stage2_time": _P3_STAGE1_TIME_PREFIXES + ("id_backbone.t2.",),
    "stage2_frequency": _P3_STAGE1_TIME_PREFIXES
    + ("id_backbone.f3.", "id_backbone.f2."),
    "stage2_joint": _P3_STAGE1_TIME_PREFIXES
    + ("id_backbone.t2.", "id_backbone.f3.", "id_backbone.f2."),
}


@dataclass(frozen=True)
class ProgressiveUpdateAudit:
    stage: int
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    source_head_frozen: bool
    domain_branch_frozen: bool
    sinc_frozen: bool


@dataclass(frozen=True)
class P3TimeFirstUpdateAudit:
    """Frozen whitelist receipt for one P3 time-first branch."""

    branch: str
    parent_branch: str | None
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    source_head_frozen: bool
    domain_branch_frozen: bool
    sinc_frozen: bool


@dataclass(frozen=True)
class WISERDualLosses:
    total: torch.Tensor
    source_head: torch.Tensor
    target_proto: torch.Tensor


def configure_progressive_identity_update(
    model: nn.Module,
    *,
    stage: int,
) -> ProgressiveUpdateAudit:
    """Freeze the model, then open only the preregistered primary identity path."""

    stage_value = int(stage)
    if stage_value not in (0, 1, 2, 3):
        raise ValueError("WISER progressive stage must be one of 0,1,2,3")
    prefixes = {
        0: (),
        1: _STAGE1_PREFIXES,
        2: _STAGE2_PREFIXES,
        3: _STAGE3_PREFIXES,
    }[stage_value]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_names = []
    trainable_count = 0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(True)
            trainable_names.append(name)
            trainable_count += int(parameter.numel())
    if stage_value > 0 and not trainable_names:
        raise ValueError("WISER stage did not match any identity parameters")

    source_head_frozen = not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("id_backbone.cls_head.head.")
    )
    domain_branch_frozen = not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(("dom_backbone.", "dom_head.", "adv_head.", "tx_adv_head."))
    )
    sinc_frozen = not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("id_backbone.sinc.")
    )
    if not (source_head_frozen and domain_branch_frozen and sinc_frozen):
        raise RuntimeError("WISER freeze invariant failed")
    return ProgressiveUpdateAudit(
        stage=stage_value,
        trainable_parameter_names=tuple(trainable_names),
        trainable_parameter_count=trainable_count,
        source_head_frozen=source_head_frozen,
        domain_branch_frozen=domain_branch_frozen,
        sinc_frozen=sinc_frozen,
    )


def configure_p3_time_first_update(
    model: nn.Module,
    *,
    branch: str,
    parent_branch: str | None = None,
) -> P3TimeFirstUpdateAudit:
    """Open exactly one P3-primary identity whitelist and freeze all else.

    ``stage3`` deliberately requires the selected Stage2 parent.  This avoids
    silently rebuilding a different branch from a display-name string.
    """

    branch_value = str(branch)
    if branch_value == "stage1_time":
        if parent_branch is not None:
            raise ValueError("stage1_time must not have a parent branch")
        prefixes = _P3_STAGE1_TIME_PREFIXES
    elif branch_value in _P3_STAGE2_PREFIXES:
        if parent_branch not in (None, "stage1_time"):
            raise ValueError("Stage2 P3 branches must inherit stage1_time")
        prefixes = _P3_STAGE2_PREFIXES[branch_value]
    elif branch_value == "stage3":
        if parent_branch not in _P3_STAGE2_PREFIXES:
            raise ValueError("stage3 requires an explicit selected Stage2 parent")
        prefixes = _P3_STAGE2_PREFIXES[parent_branch] + (
            "id_backbone.t1.",
            "id_backbone.time_fuse.",
        )
    else:
        raise ValueError("unknown P3 time-first branch")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names: list[str] = []
    count = 0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(True)
            names.append(name)
            count += int(parameter.numel())
    if not names:
        raise ValueError("P3 branch did not match any identity parameters")

    source_head_frozen = not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("id_backbone.cls_head.head.")
    )
    domain_branch_frozen = not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(("dom_", "adv_", "tx_adv_", "meta_adapter_", "sat_anchor_"))
    )
    sinc_frozen = not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("id_backbone.sinc.")
    )
    if not (source_head_frozen and domain_branch_frozen and sinc_frozen):
        raise RuntimeError("P3 time-first freeze invariant failed")
    return P3TimeFirstUpdateAudit(
        branch=branch_value,
        parent_branch=parent_branch,
        trainable_parameter_names=tuple(names),
        trainable_parameter_count=count,
        source_head_frozen=source_head_frozen,
        domain_branch_frozen=domain_branch_frozen,
        sinc_frozen=sinc_frozen,
    )


def leave_one_out_prototype_logits(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    scale: float = 10.0,
) -> torch.Tensor:
    """Cosine logits whose true-class prototype excludes the current sample."""

    if features.ndim != 2 or features.shape[0] != labels.numel():
        raise ValueError("features and labels must align")
    unique, inverse = torch.unique(labels.view(-1).long(), sorted=True, return_inverse=True)
    class_count = int(unique.numel())
    sums = features.new_zeros((class_count, features.shape[1]))
    sums = sums.index_add(0, inverse, features)
    counts = torch.bincount(inverse, minlength=class_count).to(
        device=features.device, dtype=features.dtype
    )
    if bool((counts < 2).any()):
        raise ValueError("leave-one-out prototype supervision requires K>=2 per class")
    one_hot = F.one_hot(inverse, num_classes=class_count).to(features.dtype)
    prototypes = sums[None, :, :] - one_hot[:, :, None] * features[:, None, :]
    denominators = counts[None, :] - one_hot
    prototypes = prototypes / denominators[:, :, None]
    prototypes = F.normalize(prototypes, dim=-1)
    normalized = F.normalize(features, dim=-1)
    return float(scale) * torch.einsum("nd,ncd->nc", normalized, prototypes)


def wiser_dual_supervision_loss(
    source_logits: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_proto: float = 0.5,
    prototype_scale: float = 10.0,
) -> WISERDualLosses:
    """Frozen source-head CE plus current-feature LOO prototype CE."""

    labels_long = labels.view(-1).long()
    if source_logits.ndim != 2 or source_logits.shape[0] != labels_long.numel():
        raise ValueError("source logits and labels must align")
    if float(lambda_proto) < 0.0:
        raise ValueError("lambda_proto must be nonnegative")
    source_head = F.cross_entropy(source_logits, labels_long)
    proto_logits = leave_one_out_prototype_logits(
        features, labels_long, scale=float(prototype_scale)
    )
    _, inverse = torch.unique(labels_long, sorted=True, return_inverse=True)
    target_proto = F.cross_entropy(proto_logits, inverse)
    total = source_head + float(lambda_proto) * target_proto
    return WISERDualLosses(
        total=total,
        source_head=source_head,
        target_proto=target_proto,
    )


def normalized_l2sp_penalty(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    anchors: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Parameter-count-normalized squared distance to the frozen Phase1 state."""

    total = None
    element_count = 0
    for name, parameter in named_parameters:
        if name not in anchors:
            raise ValueError(f"missing L2-SP anchor for {name}")
        anchor = anchors[name].detach().to(parameter.device, parameter.dtype)
        if anchor.shape != parameter.shape:
            raise ValueError(f"L2-SP anchor shape mismatch for {name}")
        value = (parameter - anchor).square().sum()
        total = value if total is None else total + value
        element_count += int(parameter.numel())
    if total is None or element_count < 1:
        raise ValueError("L2-SP requires at least one named parameter")
    return total / float(element_count)


__all__ = [
    "P3TimeFirstUpdateAudit",
    "ProgressiveUpdateAudit",
    "WISERDualLosses",
    "configure_p3_time_first_update",
    "configure_progressive_identity_update",
    "leave_one_out_prototype_logits",
    "normalized_l2sp_penalty",
    "wiser_dual_supervision_loss",
]
