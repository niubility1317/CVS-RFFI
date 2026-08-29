"""Support-only NOVA-REG (Stage B) for BiNOVA-D92."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvsrffi.stage2_binova_d92 import (
    DifferentiableD92State,
    d92_geometry_conditions,
    d92_geometry_features,
    fit_differentiable_d92,
)
from cvsrffi.stage2_binova_da import (
    NOVA_DA_State,
    apply_nova_da,
    support_crossfit_masks,
)
from cvsrffi.stage2_binova_features import BiNOVAFeatures, BiNOVASupport


class NOVAREGError(ValueError):
    """Raised when the Stage B registration contract is invalid."""


@dataclass(frozen=True)
class NOVA_REG_Config:
    steps: int = 400
    learning_rate: float = 8.0e-4
    rank: int = 16
    margin: float = 0.20
    weight_margin: float = 0.50
    weight_forgetting: float = 0.40
    weight_topology: float = 0.10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps < 1 or self.learning_rate <= 0 or self.rank < 1:
            raise NOVAREGError("Stage B steps, learning rate, and rank must be positive")
        if self.margin < 0:
            raise NOVAREGError("Stage B margin must be nonnegative")


class NOVA_REG_Module(nn.Module):
    """Zero-initialized rank-16 joint identity/FFT residual."""

    def __init__(self, rank: int = 16) -> None:
        super().__init__()
        self.down = nn.Linear(262, int(rank), bias=True)
        self.identity_up = nn.Linear(int(rank), 160, bias=False)
        self.fft_up = nn.Linear(int(rank), 96, bias=False)
        nn.init.zeros_(self.identity_up.weight)
        nn.init.zeros_(self.fft_up.weight)

    def forward(
        self,
        identity160: torch.Tensor,
        fft96: torch.Tensor,
        geometry6: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.down(torch.cat([identity160, fft96, geometry6], dim=1)))
        return identity160 + self.identity_up(hidden), fft96 + self.fft_up(hidden)


@dataclass(frozen=True)
class NOVA_REG_State:
    module: NOVA_REG_Module
    stage_a: NOVA_DA_State
    conditioning_d92: DifferentiableD92State
    condition_mean6: torch.Tensor
    condition_scale6: torch.Tensor
    old_class_count: int
    config: NOVA_REG_Config
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.condition_mean6.shape != (6,) or self.condition_scale6.shape != (6,):
            raise NOVAREGError("Stage B condition normalization must be [6]")
        if not torch.isfinite(self.condition_mean6).all() or not torch.isfinite(self.condition_scale6).all():
            raise NOVAREGError("Stage B condition normalization must be finite")
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def project_conflicting_gradient(
    gradient: torch.Tensor, reference: torch.Tensor, *, epsilon: float = 1.0e-12
) -> torch.Tensor:
    """Project ``gradient`` off ``reference`` only when their dot is negative."""

    dot = torch.sum(gradient * reference)
    if float(dot.detach()) >= 0.0:
        return gradient
    return gradient - dot / torch.sum(reference.square()).clamp_min(epsilon) * reference


def old_new_margin_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    old_class_count: int,
    margin: float,
) -> tuple[torch.Tensor, Mapping[str, int]]:
    values = torch.as_tensor(scores)
    targets = torch.as_tensor(labels, dtype=torch.long, device=values.device)
    old_count = int(old_class_count)
    if values.ndim != 2 or values.shape[0] != len(targets) or not 0 < old_count < values.shape[1]:
        raise NOVAREGError("old/new margin geometry is invalid")
    row = torch.arange(len(targets), device=values.device)
    true_score = values[row, targets]
    old_rows = targets < old_count
    new_rows = ~old_rows
    old_intruder = values[:, :old_count].max(dim=1).values
    new_intruder = values[:, old_count:].max(dim=1).values
    old_terms = F.relu(new_intruder[old_rows] - true_score[old_rows] + float(margin))
    new_terms = F.relu(old_intruder[new_rows] - true_score[new_rows] + float(margin))
    parts = [part for part in (old_terms, new_terms) if part.numel()]
    loss = torch.cat(parts).mean() if parts else values.sum() * 0.0
    return loss, {
        "old_intrusion_rows": int((old_terms > 0).sum().detach().cpu()),
        "new_intrusion_rows": int((new_terms > 0).sum().detach().cpu()),
    }


def _pairwise_topology(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    classes = torch.unique(labels, sorted=True)
    centers = torch.stack([rows[labels == value].mean(dim=0) for value in classes])
    return torch.pdist(F.normalize(centers, dim=1), p=2)


def _set_projected_gradients(
    parameters: list[torch.nn.Parameter],
    primary: torch.Tensor,
    retention: torch.Tensor,
) -> None:
    primary_grads = torch.autograd.grad(primary, parameters, retain_graph=True, allow_unused=True)
    retention_grads = torch.autograd.grad(retention, parameters, allow_unused=True)
    for parameter, main, keep in zip(parameters, primary_grads, retention_grads):
        if main is None and keep is None:
            parameter.grad = None
        elif main is None:
            parameter.grad = keep
        elif keep is None:
            parameter.grad = main
        else:
            parameter.grad = main + project_conflicting_gradient(keep, main)


def _tensors(features: BiNOVAFeatures, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(features.fft96.copy(), dtype=torch.float32, device=device),
        torch.as_tensor(features.identity160.copy(), dtype=torch.float32, device=device),
    )


def fit_nova_reg(
    stage_a: NOVA_DA_State,
    support: BiNOVASupport,
    *,
    old_class_count: int,
    config: NOVA_REG_Config | None = None,
    device: str | torch.device,
) -> NOVA_REG_State:
    if not isinstance(stage_a, NOVA_DA_State) or not isinstance(support, BiNOVASupport):
        raise TypeError("Stage B requires frozen Stage A state and support")
    settings = NOVA_REG_Config() if config is None else config
    target_device = torch.device(device)
    torch.manual_seed(int(settings.seed))
    old_count = int(old_class_count)
    classes = np.unique(support.labels)
    if not np.array_equal(classes, np.arange(len(classes))) or not 0 < old_count < len(classes):
        raise NOVAREGError("Stage B requires contiguous old plus registered-new labels")
    for parameter in stage_a.module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    stage_a.module.eval()
    base_identity_np = apply_nova_da(stage_a, support.features)
    base_identity = torch.as_tensor(base_identity_np, dtype=torch.float32, device=target_device)
    fft, _ = _tensors(support.features, target_device)
    labels = torch.as_tensor(support.labels.copy(), dtype=torch.long, device=target_device)
    train_np, held_np = support_crossfit_masks(support.labels, support.ranks)
    train = torch.as_tensor(train_np, dtype=torch.bool, device=target_device)
    held = torch.as_tensor(held_np, dtype=torch.bool, device=target_device)
    base_geometry = d92_geometry_features(base_identity, fft)
    conditioning = fit_differentiable_d92(
        base_geometry[train], labels[train], old_class_count=old_count,
        shrinkage_override=0.25, jitter=1.0e-3,
    )
    with torch.no_grad():
        raw_conditions = d92_geometry_conditions(conditioning, base_geometry)
        condition_mean = raw_conditions[train].mean(dim=0)
        condition_scale = raw_conditions[train].std(dim=0, unbiased=False).clamp_min(1.0e-4)
        conditions = (raw_conditions - condition_mean) / condition_scale
        baseline_scores = conditioning.score(base_geometry[held])
        baseline_topology = _pairwise_topology(base_geometry[train], labels[train])
    module = NOVA_REG_Module(settings.rank).to(target_device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=settings.learning_rate, weight_decay=1.0e-4)
    final_intrusion: Mapping[str, int] = {}
    final_loss = 0.0
    parameters = list(module.parameters())
    for _ in range(settings.steps):
        adapted_identity, adapted_fft = module(base_identity, fft, conditions)
        geometry = d92_geometry_features(adapted_identity, adapted_fft)
        d92 = fit_differentiable_d92(
            geometry[train], labels[train], old_class_count=old_count,
            shrinkage_override=0.25, jitter=1.0e-3,
        )
        scores = d92.score(geometry[held])
        classification = F.cross_entropy(scores, labels[held])
        margin_loss, final_intrusion = old_new_margin_loss(
            scores, labels[held], old_class_count=old_count, margin=settings.margin
        )
        topology = F.mse_loss(
            _pairwise_topology(geometry[train], labels[train]), baseline_topology
        )
        old_held = labels[held] < old_count
        retention = F.kl_div(
            F.log_softmax(scores[old_held, :old_count], dim=1),
            F.softmax(baseline_scores[old_held, :old_count], dim=1),
            reduction="batchmean",
        )
        primary = classification + settings.weight_margin * margin_loss + settings.weight_topology * topology
        optimizer.zero_grad(set_to_none=True)
        _set_projected_gradients(parameters, primary, settings.weight_forgetting * retention)
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        final_loss = float((primary + settings.weight_forgetting * retention).detach().cpu())
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return NOVA_REG_State(
        module=module,
        stage_a=stage_a,
        conditioning_d92=conditioning,
        condition_mean6=condition_mean,
        condition_scale6=condition_scale,
        old_class_count=old_count,
        config=settings,
        audit={
            "query_rows_used": 0,
            "stage_a_frozen": True,
            "gradient_merge": "conflict_projection",
            "final_loss": final_loss,
            **dict(final_intrusion),
        },
    )


def apply_nova_reg(
    state: NOVA_REG_State,
    features: BiNOVAFeatures,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(state, NOVA_REG_State):
        raise TypeError("apply_nova_reg requires a Stage B state")
    device = next(state.module.parameters()).device
    base_identity = torch.as_tensor(
        apply_nova_da(state.stage_a, features), dtype=torch.float32, device=device
    )
    fft = torch.as_tensor(features.fft96.copy(), dtype=torch.float32, device=device)
    with torch.inference_mode():
        geometry = d92_geometry_features(base_identity, fft)
        raw_conditions = d92_geometry_conditions(state.conditioning_d92, geometry)
        conditions = (raw_conditions - state.condition_mean6) / state.condition_scale6
        identity, adapted_fft = state.module(base_identity, fft, conditions)
    return (
        identity.detach().cpu().numpy().astype(np.float32, copy=False),
        adapted_fft.detach().cpu().numpy().astype(np.float32, copy=False),
    )


__all__ = [
    "NOVAREGError",
    "NOVA_REG_Config",
    "NOVA_REG_Module",
    "NOVA_REG_State",
    "apply_nova_reg",
    "fit_nova_reg",
    "old_new_margin_loss",
    "project_conflicting_gradient",
]
