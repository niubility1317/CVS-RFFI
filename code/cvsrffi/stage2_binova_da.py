"""Support-only NOVA-DA (Stage A) for BiNOVA-D92.

The adapter is a zero-initialized, low-rank nonlinear residual over frozen
identity and late-time features.  Only old-class target support enters fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvsrffi.stage2_binova_d92 import d92_geometry_features, fit_differentiable_d92
from cvsrffi.stage2_binova_features import (
    BiNOVAFeatures,
    BiNOVASupport,
    class_balanced_domain_context,
)


class NOVADAError(ValueError):
    """Raised when the Stage A support-only contract is violated."""


@dataclass(frozen=True)
class NOVA_DA_Config:
    steps: int = 600
    learning_rate: float = 1.0e-3
    late_rank: int = 16
    identity_rank: int = 32
    weight_forgetting: float = 0.30
    weight_contrastive: float = 0.05
    weight_affine_leak: float = 0.10
    maximum_affine_fraction: float = 0.80
    pseudo_registration: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps < 1 or self.learning_rate <= 0:
            raise NOVADAError("Stage A steps and learning rate must be positive")
        if self.late_rank < 1 or self.identity_rank < 1:
            raise NOVADAError("Stage A residual ranks must be positive")
        if not 0.0 <= self.maximum_affine_fraction <= 1.0:
            raise NOVADAError("maximum_affine_fraction must be in [0,1]")


class NOVA_DA_Module(nn.Module):
    """Rank-16 late-time plus rank-32 identity nonlinear residual."""

    def __init__(self, late_rank: int = 16, identity_rank: int = 32) -> None:
        super().__init__()
        self.late_down = nn.Linear(160, int(late_rank), bias=True)
        self.late_up = nn.Linear(int(late_rank), 160, bias=False)
        self.identity_down = nn.Linear(160, int(identity_rank), bias=True)
        self.identity_up = nn.Linear(int(identity_rank), 160, bias=False)
        self.context_gate = nn.Linear(166, 2, bias=True)
        nn.init.zeros_(self.late_up.weight)
        nn.init.zeros_(self.identity_up.weight)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)

    def forward(
        self,
        identity160: torch.Tensor,
        late_time160: torch.Tensor,
        domain160: torch.Tensor,
        physical6: torch.Tensor,
        domain_context166: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.cat([domain160, physical6], dim=1) - domain_context166[None, :]
        gates = 2.0 * torch.sigmoid(self.context_gate(context))
        late_delta = self.late_up(torch.tanh(self.late_down(late_time160)))
        identity_delta = self.identity_up(torch.tanh(self.identity_down(identity160)))
        return identity160 + gates[:, :1] * late_delta + gates[:, 1:] * identity_delta


@dataclass(frozen=True)
class NOVA_DA_State:
    module: NOVA_DA_Module
    config: NOVA_DA_Config
    domain_context166: np.ndarray
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        context = np.asarray(self.domain_context166, dtype=np.float32)
        if context.shape != (166,) or not np.isfinite(context).all():
            raise NOVADAError("Stage A domain context must be finite [166]")
        object.__setattr__(self, "domain_context166", context.copy())
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def build_pseudo_role_rotations(
    class_ids: Sequence[int],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    registry = tuple(int(value) for value in class_ids)
    if len(registry) < 6 or len(set(registry)) != len(registry):
        raise NOVADAError("Stage A role rotation requires at least six unique old classes")
    rotations = []
    for offset in range(0, len(registry), 2):
        pseudo_new = (registry[offset % len(registry)], registry[(offset + 1) % len(registry)])
        base = tuple(value for value in registry if value not in pseudo_new)[:4]
        if len(base) == 4:
            rotations.append((base, pseudo_new))
    return tuple(rotations)


def support_crossfit_masks(labels: Any, ranks: Any) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(labels, dtype=np.int64)
    order = np.asarray(ranks, dtype=np.int64)
    if targets.ndim != 1 or order.shape != targets.shape:
        raise NOVADAError("support labels/ranks are invalid")
    train = np.zeros(len(targets), dtype=bool)
    held = np.zeros(len(targets), dtype=bool)
    for class_id in np.unique(targets):
        indices = np.flatnonzero(targets == class_id)
        if len(indices) < 10:
            raise NOVADAError("Stage A minimum cross-fit requires ten shots per class")
        ranked = indices[np.argsort(order[indices], kind="stable")]
        train[ranked[:8]] = True
        held[ranked[8:10]] = True
    return train, held


def affine_explained_ratio(inputs: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(inputs)
    y = torch.as_tensor(residuals, dtype=x.dtype, device=x.device)
    if x.ndim != 2 or y.shape != x.shape:
        raise NOVADAError("affine norm leak inputs must be aligned matrices")
    if len(x) < 5:
        raise NOVADAError("affine leak cross-fit requires at least five rows")
    validation = torch.arange(len(x), device=x.device) % 5 == 0
    train = ~validation
    train_x_mean = x[train].mean(dim=0, keepdim=True)
    train_y_mean = y[train].mean(dim=0, keepdim=True)
    fit_x = x[train] - train_x_mean
    fit_y = y[train] - train_y_mean
    validation_x = x[validation] - train_x_mean
    validation_y = y[validation] - train_y_mean
    total = validation_y.square().sum()
    if float(total.detach()) <= torch.finfo(x.dtype).eps:
        return total * 0.0
    slope = torch.sum(fit_x * fit_y, dim=0) / torch.sum(fit_x.square(), dim=0).clamp_min(
        torch.finfo(x.dtype).eps
    )
    error = (validation_y - validation_x * slope).square().sum()
    return (1.0 - error / total).clamp(0.0, 1.0)


def _domain_context(support: BiNOVASupport) -> np.ndarray:
    rows = np.concatenate(
        [support.features.domain160, support.features.physical6], axis=1
    )
    return class_balanced_domain_context(rows, support.labels)


def _feature_tensors(features: BiNOVAFeatures, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.as_tensor(value.copy(), dtype=torch.float32, device=device)
        for value in (
            features.identity160,
            features.late_time160,
            features.domain160,
            features.physical6,
            features.fft96,
        )
    )


def _compactness_loss(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(rows, dim=1)
    centers = torch.stack(
        [normalized[labels == class_id].mean(dim=0) for class_id in torch.unique(labels)]
    )
    centers = F.normalize(centers, dim=1)
    mapped = torch.searchsorted(torch.unique(labels), labels)
    return (1.0 - torch.sum(normalized * centers[mapped], dim=1)).mean()


def fit_nova_da(
    support: BiNOVASupport,
    config: NOVA_DA_Config | None = None,
    *,
    device: str | torch.device,
) -> NOVA_DA_State:
    if not isinstance(support, BiNOVASupport):
        raise TypeError("Stage A requires a support-only BiNOVASupport object")
    settings = NOVA_DA_Config() if config is None else config
    target_device = torch.device(device)
    torch.manual_seed(int(settings.seed))
    module = NOVA_DA_Module(settings.late_rank, settings.identity_rank).to(target_device)
    identity, late, domain, physical, fft = _feature_tensors(support.features, target_device)
    labels = torch.as_tensor(support.labels.copy(), dtype=torch.long, device=target_device)
    train_mask_np, held_mask_np = support_crossfit_masks(support.labels, support.ranks)
    train_mask = torch.as_tensor(train_mask_np, device=target_device)
    ranks = torch.as_tensor(support.ranks.copy(), dtype=torch.long, device=target_device)
    optimizer_fit_mask = train_mask & (ranks < 6)
    optimizer_score_mask = train_mask & (ranks >= 6)
    context_np = _domain_context(support)
    context = torch.as_tensor(context_np, dtype=torch.float32, device=target_device)
    class_registry = tuple(int(v) for v in np.unique(support.labels))
    rotations = (
        build_pseudo_role_rotations(class_registry)
        if settings.pseudo_registration
        else ((class_registry, tuple()),)
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=settings.learning_rate, weight_decay=1.0e-4)
    history: list[float] = []
    for step in range(settings.steps):
        base, pseudo_new = rotations[step % len(rotations)]
        registry = base + pseudo_new
        pseudo_old_count = 4 if pseudo_new else len(registry)
        selected = torch.zeros_like(labels, dtype=torch.bool)
        remapped = torch.full_like(labels, -1)
        for new_index, class_id in enumerate(registry):
            class_mask = labels == int(class_id)
            selected |= class_mask
            remapped[class_mask] = new_index
        adapted = module(identity, late, domain, physical, context)
        geometry = d92_geometry_features(adapted, fft)
        fit_mask = selected & optimizer_fit_mask
        score_mask = selected & optimizer_score_mask
        d92 = fit_differentiable_d92(
            geometry[fit_mask],
            remapped[fit_mask],
            old_class_count=pseudo_old_count,
            shrinkage_override=0.25,
            jitter=1.0e-3,
        )
        classification = F.cross_entropy(d92.score(geometry[score_mask]), remapped[score_mask])
        old_score_mask = score_mask & (remapped < pseudo_old_count)
        forgetting = F.cross_entropy(
            d92.score(geometry[old_score_mask])[:, :pseudo_old_count],
            remapped[old_score_mask],
        )
        compactness = _compactness_loss(adapted[fit_mask], remapped[fit_mask])
        residual = adapted[train_mask] - identity[train_mask]
        affine_ratio = affine_explained_ratio(identity[train_mask], residual)
        affine_leak = F.relu(affine_ratio - settings.maximum_affine_fraction)
        loss = (
            classification
            + settings.weight_forgetting * forgetting
            + settings.weight_contrastive * compactness
            + settings.weight_affine_leak * affine_leak
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        adapted = module(identity, late, domain, physical, context)
        affine_fraction = float(affine_explained_ratio(identity, adapted - identity).cpu())
    return NOVA_DA_State(
        module=module,
        config=settings,
        domain_context166=context_np,
        audit={
            "query_rows_used": 0,
            "role_rotation_count": len(rotations),
            "pseudo_registration": settings.pseudo_registration,
            "crossfit_train_per_class": 8,
            "crossfit_held_per_class": 2,
            "optimizer_d92_fit_per_class": 6,
            "optimizer_score_per_class": 2,
            "gate_held_labels_used_for_gradient": False,
            "affine_fraction": affine_fraction,
            "non_affine_fraction": 1.0 - affine_fraction,
            "final_loss": history[-1],
        },
    )


def apply_nova_da(
    state_or_module: NOVA_DA_State | NOVA_DA_Module,
    features: BiNOVAFeatures,
) -> np.ndarray:
    if isinstance(state_or_module, NOVA_DA_State):
        module = state_or_module.module
        context_np = state_or_module.domain_context166
    elif isinstance(state_or_module, NOVA_DA_Module):
        module = state_or_module
        context_np = np.zeros(166, dtype=np.float32)
    else:
        raise TypeError("apply_nova_da requires a Stage A state or module")
    device = next(module.parameters()).device
    identity, late, domain, physical, _ = _feature_tensors(features, device)
    context = torch.as_tensor(context_np, dtype=torch.float32, device=device)
    with torch.inference_mode():
        result = module(identity, late, domain, physical, context)
    return result.detach().cpu().numpy().astype(np.float32, copy=False)


def evaluate_nova_da_crossfit(
    state: NOVA_DA_State,
    support: BiNOVASupport,
    *,
    device: str | torch.device = "cpu",
) -> Mapping[str, float]:
    """Evaluate Stage A only on the deterministic held support rows."""

    if not isinstance(state, NOVA_DA_State) or not isinstance(support, BiNOVASupport):
        raise TypeError("Stage A cross-fit evaluation requires state and support")
    target_device = torch.device(device)
    train_np, held_np = support_crossfit_masks(support.labels, support.ranks)
    labels = torch.as_tensor(support.labels.copy(), dtype=torch.long, device=target_device)
    fft = torch.as_tensor(support.features.fft96.copy(), dtype=torch.float32, device=target_device)
    baseline_identity = torch.as_tensor(
        support.features.identity160.copy(), dtype=torch.float32, device=target_device
    )
    adapted_identity = torch.as_tensor(
        apply_nova_da(state, support.features), dtype=torch.float32, device=target_device
    )
    train = torch.as_tensor(train_np, device=target_device)
    held = torch.as_tensor(held_np, device=target_device)
    rotations = build_pseudo_role_rotations(tuple(int(v) for v in np.unique(support.labels)))
    old_correct = new_correct = old_total = new_total = 0
    baseline_old_correct = 0
    class_accuracies: list[float] = []
    for base, pseudo_new in rotations:
        registry = base + pseudo_new
        selected = torch.zeros_like(labels, dtype=torch.bool)
        remapped = torch.full_like(labels, -1)
        for index, class_id in enumerate(registry):
            mask = labels == class_id
            selected |= mask
            remapped[mask] = index
        fit_mask, score_mask = selected & train, selected & held
        baseline_geometry = d92_geometry_features(baseline_identity, fft)
        adapted_geometry = d92_geometry_features(adapted_identity, fft)
        baseline_d92 = fit_differentiable_d92(
            baseline_geometry[fit_mask], remapped[fit_mask], old_class_count=4,
            shrinkage_override=0.25, jitter=1.0e-3,
        )
        adapted_d92 = fit_differentiable_d92(
            adapted_geometry[fit_mask], remapped[fit_mask], old_class_count=4,
            shrinkage_override=0.25, jitter=1.0e-3,
        )
        truth = remapped[score_mask]
        prediction = adapted_d92.score(adapted_geometry[score_mask]).argmax(dim=1)
        baseline_prediction = baseline_d92.score(baseline_geometry[score_mask]).argmax(dim=1)
        old = truth < 4
        new = ~old
        old_correct += int((prediction[old] == truth[old]).sum())
        new_correct += int((prediction[new] == truth[new]).sum())
        baseline_old_correct += int((baseline_prediction[old] == truth[old]).sum())
        old_total += int(old.sum())
        new_total += int(new.sum())
        for class_index in range(4):
            class_rows = truth == class_index
            class_accuracies.append(float((prediction[class_rows] == truth[class_rows]).float().mean()))
    old_accuracy = old_correct / max(old_total, 1)
    new_accuracy = new_correct / max(new_total, 1)
    harmonic = 2.0 * old_accuracy * new_accuracy / max(old_accuracy + new_accuracy, 1.0e-12)
    baseline_old_accuracy = baseline_old_correct / max(old_total, 1)
    return MappingProxyType(
        {
            "pseudo_old_accuracy": old_accuracy,
            "pseudo_new_accuracy": new_accuracy,
            "pseudo_h": harmonic,
            "pseudo_old_floor": min(class_accuracies),
            "pseudo_forgetting": baseline_old_accuracy - old_accuracy,
            "query_rows_used": 0.0,
        }
    )


__all__ = [
    "NOVADAError",
    "NOVA_DA_Config",
    "NOVA_DA_Module",
    "NOVA_DA_State",
    "affine_explained_ratio",
    "apply_nova_da",
    "build_pseudo_role_rotations",
    "evaluate_nova_da_crossfit",
    "fit_nova_da",
    "support_crossfit_masks",
]
