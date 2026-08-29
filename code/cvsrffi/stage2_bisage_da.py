"""Stage A SAGE-D: old-class-only class-extrapolative domain adaptation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvsrffi.stage2_binova_d92 import d92_geometry_features
from cvsrffi.stage2_binova_features import (
    BiNOVAFeatures,
    BiNOVASupport,
    class_balanced_domain_context,
)
from cvsrffi.stage2_bisage_d92 import BiSAGED92State, fit_bisage_d92


class SAGEDError(ValueError):
    """Raised when Stage A violates its old-support-only contract."""


@dataclass(frozen=True)
class RoleRotation:
    pseudo_base: tuple[int, ...]
    pseudo_new: tuple[int, ...]


@dataclass(frozen=True)
class SAGEDConfig:
    steps: int = 3000
    learning_rate: float = 3.0e-4
    late_rank: int = 32
    identity_rank: int = 32
    context_dim: int = 32
    covariance_rank: int = 16
    weight_forgetting: float = 0.20
    weight_covariance: float = 0.05
    weight_topology: float = 0.05
    weight_nonaffine: float = 0.05
    weight_movement: float = 1.0e-3
    minimum_nonaffine: float = 0.20
    maximum_nonaffine: float = 0.70
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps < 1 or self.learning_rate <= 0.0:
            raise SAGEDError("Stage A steps and learning rate must be positive")
        if min(self.late_rank, self.identity_rank, self.context_dim) < 1:
            raise SAGEDError("Stage A ranks must be positive")
        if self.covariance_rank < 1:
            raise SAGEDError("Stage A covariance rank must be positive")
        if not 0.0 <= self.minimum_nonaffine <= self.maximum_nonaffine <= 1.0:
            raise SAGEDError("Stage A non-affine interval is invalid")


class SAGEDModule(nn.Module):
    """Zero-initialized, sample-conditioned late/identity tangent residual."""

    def __init__(
        self,
        late_rank: int = 32,
        identity_rank: int = 32,
        context_dim: int = 32,
    ) -> None:
        super().__init__()
        context_dim = int(context_dim)
        self.context = nn.Sequential(
            nn.Linear(166, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim)
        )
        self.late_down = nn.Linear(160, int(late_rank), bias=True)
        self.late_fft = nn.Linear(96, int(late_rank), bias=False)
        self.late_context = nn.Linear(context_dim, int(late_rank), bias=False)
        self.late_gate = nn.Linear(160 + 96 + context_dim, int(late_rank))
        self.late_up = nn.Linear(int(late_rank), 160, bias=False)
        self.identity_down = nn.Linear(160, int(identity_rank), bias=True)
        self.identity_fft = nn.Linear(96, int(identity_rank), bias=False)
        self.identity_context = nn.Linear(context_dim, int(identity_rank), bias=False)
        self.identity_gate = nn.Linear(160 + 96 + context_dim, int(identity_rank))
        self.identity_up = nn.Linear(int(identity_rank), 160, bias=False)
        nn.init.zeros_(self.late_up.weight)
        nn.init.zeros_(self.identity_up.weight)

    def forward(
        self,
        identity160: torch.Tensor,
        late_time160: torch.Tensor,
        fft96: torch.Tensor,
        domain_context166: torch.Tensor,
    ) -> torch.Tensor:
        if (
            identity160.ndim != 2
            or late_time160.shape != identity160.shape
            or identity160.shape[1] != 160
            or fft96.shape != (len(identity160), 96)
            or domain_context166.shape != (166,)
        ):
            raise SAGEDError("Stage A forward feature geometry drift")
        context = self.context(domain_context166)[None, :].expand(len(identity160), -1)
        conditions = torch.cat([identity160, fft96, context], dim=1)
        late_hidden = torch.tanh(
            self.late_down(late_time160)
            + self.late_fft(fft96)
            + self.late_context(context)
        )
        late_hidden = torch.sigmoid(self.late_gate(conditions)) * late_hidden
        late_delta = self.late_up(late_hidden)
        identity_hidden = torch.tanh(
            self.identity_down(identity160)
            + self.identity_fft(fft96)
            + self.identity_context(context)
        )
        identity_hidden = torch.sigmoid(self.identity_gate(conditions)) * identity_hidden
        raw_delta = late_delta + self.identity_up(identity_hidden)
        unit = F.normalize(identity160, dim=1)
        tangent_delta = raw_delta - torch.sum(raw_delta * unit, dim=1, keepdim=True) * unit
        return identity160 + tangent_delta


@dataclass(frozen=True)
class SAGEDState:
    module: SAGEDModule
    config: SAGEDConfig
    domain_context166: np.ndarray
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        context = np.asarray(self.domain_context166, dtype=np.float64)
        if context.shape != (166,) or not np.isfinite(context).all():
            raise SAGEDError("Stage A domain context must be finite [166]")
        object.__setattr__(self, "domain_context166", context.copy())
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def build_role_rotations(class_ids: Sequence[int]) -> tuple[RoleRotation, ...]:
    registry = tuple(int(value) for value in class_ids)
    if len(registry) != 6 or len(set(registry)) != 6:
        raise SAGEDError("Stage A requires exactly six unique old classes")
    return tuple(
        RoleRotation(
            pseudo_base=tuple(value for value in registry if value not in registry[offset : offset + 2]),
            pseudo_new=registry[offset : offset + 2],
        )
        for offset in (0, 2, 4)
    )


def support_crossfit_masks(labels: Any, ranks: Any) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(labels, dtype=np.int64)
    order = np.asarray(ranks, dtype=np.int64)
    if targets.ndim != 1 or order.shape != targets.shape:
        raise SAGEDError("Stage A support labels/ranks are invalid")
    fit = np.zeros(len(targets), dtype=bool)
    held = np.zeros(len(targets), dtype=bool)
    counts = []
    for class_id in np.unique(targets):
        indices = np.flatnonzero(targets == class_id)
        counts.append(len(indices))
        if len(indices) < 2:
            raise SAGEDError("Stage A cross-fit requires K>=2")
        ranked = indices[np.argsort(order[indices], kind="stable")]
        held_count = max(1, int(round(0.2 * len(ranked))))
        fit[ranked[:-held_count]] = True
        held[ranked[-held_count:]] = True
    if len(set(counts)) != 1:
        raise SAGEDError("Stage A requires equal K for all old classes")
    return fit, held


def coordinate_median_consensus(
    gradients: Sequence[torch.Tensor], *, normalize: bool = True
) -> torch.Tensor:
    if len(gradients) != 3 or any(item.shape != gradients[0].shape for item in gradients):
        raise SAGEDError("gradient consensus requires three aligned tensors")
    stack = torch.stack(tuple(gradients))
    if normalize:
        flat_norms = stack.reshape(3, -1).norm(dim=1).clamp_min(1.0e-12)
        normalized = stack / flat_norms.reshape((3,) + (1,) * gradients[0].ndim)
        median = normalized.median(dim=0).values
        return median * flat_norms.median()
    return stack.median(dim=0).values


def _feature_tensors(
    features: BiNOVAFeatures, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.as_tensor(value.copy(), dtype=dtype, device=device)
        for value in (
            features.identity160,
            features.late_time160,
            features.domain160,
            features.physical6,
            features.fft96,
        )
    )


def _domain_context(support: BiNOVASupport, fit_mask: np.ndarray | None = None) -> np.ndarray:
    mask = np.ones(len(support.labels), dtype=bool) if fit_mask is None else fit_mask
    rows = np.concatenate(
        [support.features.domain160[mask], support.features.physical6[mask]], axis=1
    )
    return np.asarray(
        class_balanced_domain_context(rows, support.labels[mask]), dtype=np.float64
    )


def _remap(labels: torch.Tensor, rotation: RoleRotation) -> torch.Tensor:
    registry = rotation.pseudo_base + rotation.pseudo_new
    result = torch.full_like(labels, -1)
    for new_index, class_id in enumerate(registry):
        result[labels == int(class_id)] = new_index
    return result


def _true_margin(logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    true_score = logits.gather(1, truth[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, truth[:, None], -torch.inf)
    return true_score - masked.max(dim=1).values


def _topology(state: BiSAGED92State) -> torch.Tensor:
    differences = state.centers[:, None, :] - state.centers[None, :, :]
    solved = torch.cholesky_solve(
        differences.reshape(-1, differences.shape[-1]).T, state.cholesky
    ).T.reshape_as(differences)
    return torch.sum(differences * solved, dim=-1)


def _covariance_homogeneity(
    rows: torch.Tensor, labels: torch.Tensor, projection: torch.Tensor
) -> torch.Tensor:
    projected = rows @ projection
    covariances = []
    epsilon = torch.finfo(rows.dtype).eps * 100.0
    identity = torch.eye(projection.shape[1], dtype=rows.dtype, device=rows.device)
    for class_id in torch.unique(labels, sorted=True):
        selected = projected[labels == class_id]
        centered = selected - selected.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / float(len(selected)) + epsilon * identity
        covariances.append(0.5 * (covariance + covariance.T))
    mean_covariance = torch.stack(covariances).mean(dim=0)

    def matrix_log(value: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(value)
        return (eigenvectors * eigenvalues.clamp_min(epsilon).log()[None, :]) @ eigenvectors.T

    mean_log = matrix_log(mean_covariance)
    return torch.stack([(matrix_log(value) - mean_log).square().mean() for value in covariances]).mean()


def _nonaffine_energy(
    baseline: torch.Tensor, adapted: torch.Tensor, projection: torch.Tensor
) -> torch.Tensor:
    residual = adapted - baseline
    design = torch.cat(
        [torch.ones(len(baseline), 1, dtype=baseline.dtype, device=baseline.device), baseline @ projection],
        dim=1,
    )
    ridge = 1.0e-4 * torch.eye(design.shape[1], dtype=baseline.dtype, device=baseline.device)
    coefficients = torch.linalg.solve(design.T @ design + ridge, design.T @ residual)
    explained = (design @ coefficients).square().sum()
    total = residual.square().sum()
    return (total - explained).clamp_min(0.0) / (total + 1.0e-12)


def _rotation_loss(
    geometry: torch.Tensor,
    baseline_geometry: torch.Tensor,
    labels: torch.Tensor,
    fit_mask: torch.Tensor,
    held_mask: torch.Tensor,
    rotation: RoleRotation,
    projection: torch.Tensor,
    baseline_topology: torch.Tensor,
    identity: torch.Tensor,
    adapted_identity: torch.Tensor,
    settings: SAGEDConfig,
) -> torch.Tensor:
    remapped = _remap(labels, rotation)
    registered = fit_bisage_d92(
        geometry[fit_mask], remapped[fit_mask], old_class_count=4
    )
    held_truth = remapped[held_mask]
    held_logits = registered.score(geometry[held_mask])
    pseudo_registration = F.cross_entropy(held_logits, held_truth)

    base_fit = fit_mask & (remapped < 4)
    base_held = held_mask & (remapped < 4)
    before = fit_bisage_d92(geometry[base_fit], remapped[base_fit], old_class_count=4)
    pre_margin = _true_margin(before.score(geometry[base_held]), remapped[base_held])
    post_margin = _true_margin(registered.score(geometry[base_held]), remapped[base_held])
    forgetting = F.relu(pre_margin - post_margin).mean()
    covariance = _covariance_homogeneity(geometry[fit_mask], remapped[fit_mask], projection)
    topology = (_topology(registered) - baseline_topology).square().mean()
    nonaffine = _nonaffine_energy(identity[fit_mask], adapted_identity[fit_mask], projection[:160])
    nonaffine_penalty = F.relu(settings.minimum_nonaffine - nonaffine).square()
    nonaffine_penalty += F.relu(nonaffine - settings.maximum_nonaffine).square()
    movement = (adapted_identity[fit_mask] - identity[fit_mask]).square().mean()
    return (
        pseudo_registration
        + settings.weight_forgetting * forgetting
        + settings.weight_covariance * covariance
        + settings.weight_topology * topology
        + settings.weight_nonaffine * nonaffine_penalty
        + settings.weight_movement * movement
    )


def fit_sage_d(
    support: BiNOVASupport,
    config: SAGEDConfig | None = None,
    *,
    device: str | torch.device,
) -> SAGEDState:
    if not isinstance(support, BiNOVASupport):
        raise TypeError("Stage A requires an old-class support-only object")
    settings = SAGEDConfig() if config is None else config
    classes, counts = np.unique(support.labels, return_counts=True)
    if len(classes) != 6 or not np.array_equal(classes, np.arange(6)):
        raise SAGEDError("Stage A requires the six contiguous historical old classes")
    if len(set(counts.tolist())) != 1:
        raise SAGEDError("Stage A requires equal K old support")
    target_device = torch.device(device)
    torch.manual_seed(int(settings.seed))
    module = SAGEDModule(
        settings.late_rank, settings.identity_rank, settings.context_dim
    ).to(device=target_device, dtype=torch.float64)

    if int(counts[0]) < 5:
        context_np = _domain_context(support)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return SAGEDState(
            module=module,
            config=settings,
            domain_context166=context_np,
            audit={
                "selected_mode": "S0",
                "k1_fallback": int(counts[0]) == 1,
                "low_k_fallback": True,
                "k_shot": int(counts[0]),
                "query_rows_used": 0,
                "query_truth_access": False,
                "role_rotation_count": 0,
            },
        )

    fit_np, held_np = support_crossfit_masks(support.labels, support.ranks)
    context_np = _domain_context(support, fit_np)
    identity, late, _, _, fft = _feature_tensors(support.features, target_device, torch.float64)
    labels = torch.as_tensor(support.labels.copy(), dtype=torch.long, device=target_device)
    fit_mask = torch.as_tensor(fit_np, dtype=torch.bool, device=target_device)
    held_mask = torch.as_tensor(held_np, dtype=torch.bool, device=target_device)
    context = torch.as_tensor(context_np, dtype=torch.float64, device=target_device)
    baseline_geometry = d92_geometry_features(identity, fft)
    rank = min(settings.covariance_rank, baseline_geometry.shape[1], int(fit_mask.sum()))
    _, _, projection = torch.pca_lowrank(baseline_geometry[fit_mask], q=rank, center=True)
    projection = projection.detach()
    identity_projection = projection[:160]
    rotations = build_role_rotations(classes.tolist())
    baseline_topologies = []
    for rotation in rotations:
        remapped = _remap(labels, rotation)
        state = fit_bisage_d92(
            baseline_geometry[fit_mask], remapped[fit_mask], old_class_count=4
        )
        baseline_topologies.append(_topology(state).detach())

    optimizer = torch.optim.AdamW(
        module.parameters(), lr=settings.learning_rate, weight_decay=1.0e-4
    )
    parameters = tuple(module.parameters())
    history = []
    for _ in range(settings.steps):
        adapted_identity = module(identity, late, fft, context)
        geometry = d92_geometry_features(adapted_identity, fft)
        losses = [
            _rotation_loss(
                geometry,
                baseline_geometry,
                labels,
                fit_mask,
                held_mask,
                rotation,
                projection,
                baseline_topology,
                identity,
                adapted_identity,
                settings,
            )
            for rotation, baseline_topology in zip(rotations, baseline_topologies)
        ]
        gradients = [
            torch.autograd.grad(loss, parameters, retain_graph=index < len(losses) - 1, allow_unused=True)
            for index, loss in enumerate(losses)
        ]
        optimizer.zero_grad(set_to_none=True)
        for parameter_index, parameter in enumerate(parameters):
            aligned = [
                torch.zeros_like(parameter) if row[parameter_index] is None else row[parameter_index]
                for row in gradients
            ]
            parameter.grad = coordinate_median_consensus(aligned)
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        history.append(float(torch.stack(losses).mean().detach().cpu()))

    module.eval()
    with torch.inference_mode():
        adapted = module(identity, late, fft, context)
        nonaffine = float(
            _nonaffine_energy(identity[fit_mask], adapted[fit_mask], identity_projection).cpu()
        )
        movement = float((adapted - identity).square().mean().sqrt().cpu())
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return SAGEDState(
        module=module,
        config=settings,
        domain_context166=context_np,
        audit={
            "selected_mode": "S1_CANDIDATE",
            "k1_fallback": False,
            "low_k_fallback": False,
            "k_shot": int(counts[0]),
            "role_rotation_count": 3,
            "gradient_consensus": "normalized_coordinate_median",
            "crossfit_fit_per_class": int(np.sum(fit_np & (support.labels == 0))),
            "crossfit_held_per_class": int(np.sum(held_np & (support.labels == 0))),
            "nonaffine_energy": nonaffine,
            "movement_rms": movement,
            "final_loss": history[-1],
            "query_rows_used": 0,
            "query_truth_access": False,
            "forward_class_id_input": False,
        },
    )


def apply_sage_d(
    state_or_module: SAGEDState | SAGEDModule,
    features: BiNOVAFeatures,
    domain_context166: Any | None = None,
) -> np.ndarray:
    if isinstance(state_or_module, SAGEDState):
        module = state_or_module.module
        context_np = state_or_module.domain_context166
        if state_or_module.audit.get("selected_mode") == "S0":
            return np.asarray(features.identity160, dtype=np.float32).copy()
    elif isinstance(state_or_module, SAGEDModule):
        module = state_or_module
        if domain_context166 is None:
            raise SAGEDError("direct Stage A module inference requires frozen context")
        context_np = np.asarray(domain_context166, dtype=np.float64)
    else:
        raise TypeError("apply_sage_d requires a Stage A state or module")
    parameter = next(module.parameters())
    identity, late, _, _, fft = _feature_tensors(features, parameter.device, parameter.dtype)
    context = torch.as_tensor(context_np, dtype=parameter.dtype, device=parameter.device)
    with torch.inference_mode():
        result = module(identity, late, fft, context)
    return np.asarray(result.detach().cpu().tolist(), dtype=np.float32)


def _harmonic(old: float, new: float) -> float:
    return 2.0 * old * new / max(old + new, 1.0e-12)


def _lcb(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean() - 1.96 * array.std(ddof=0) / math.sqrt(len(array)))


def evaluate_sage_d_crossfit(
    state: SAGEDState,
    support: BiNOVASupport,
    *,
    device: str | torch.device = "cpu",
) -> Mapping[str, float]:
    if not isinstance(state, SAGEDState) or not isinstance(support, BiNOVASupport):
        raise TypeError("Stage A evaluation requires state and old support")
    if state.audit.get("selected_mode") == "S0":
        raise SAGEDError("Stage A low-K fallback has no valid cross-fit gate")
    fit_np, held_np = support_crossfit_masks(support.labels, support.ranks)
    target_device = torch.device(device)
    labels = torch.as_tensor(support.labels.copy(), dtype=torch.long, device=target_device)
    fit = torch.as_tensor(fit_np, dtype=torch.bool, device=target_device)
    held = torch.as_tensor(held_np, dtype=torch.bool, device=target_device)
    baseline_identity = torch.as_tensor(
        support.features.identity160.copy(), dtype=torch.float64, device=target_device
    )
    adapted_identity = torch.as_tensor(
        apply_sage_d(state, support.features), dtype=torch.float64, device=target_device
    )
    fft = torch.as_tensor(support.features.fft96.copy(), dtype=torch.float64, device=target_device)
    baseline_geometry = d92_geometry_features(baseline_identity, fft)
    adapted_geometry = d92_geometry_features(adapted_identity, fft)

    totals = {"old": 0, "new": 0}
    correct = {"old": 0, "new": 0, "base_old": 0}
    baseline_correct = {"old": 0, "new": 0, "base_old": 0}
    class_correct: dict[tuple[int, int], list[int]] = {}
    baseline_class_correct: dict[tuple[int, int], list[int]] = {}
    adapted_h = []
    baseline_h = []
    changes = 0
    for rotation_index, rotation in enumerate(build_role_rotations(range(6))):
        remapped = _remap(labels, rotation)
        base_fit = fit & (remapped < 4)
        base_held = held & (remapped < 4)
        baseline_pre = fit_bisage_d92(
            baseline_geometry[base_fit], remapped[base_fit], old_class_count=4
        )
        adapted_pre = fit_bisage_d92(
            adapted_geometry[base_fit], remapped[base_fit], old_class_count=4
        )
        baseline_full = fit_bisage_d92(
            baseline_geometry[fit], remapped[fit], old_class_count=4
        )
        adapted_full = fit_bisage_d92(
            adapted_geometry[fit], remapped[fit], old_class_count=4
        )
        truth = remapped[held]
        baseline_prediction = baseline_full.score(baseline_geometry[held]).argmax(1)
        prediction = adapted_full.score(adapted_geometry[held]).argmax(1)
        changes += int((baseline_prediction != prediction).sum())
        old = truth < 4
        new = ~old
        old_acc = float((prediction[old] == truth[old]).double().mean())
        new_acc = float((prediction[new] == truth[new]).double().mean())
        baseline_old_acc = float(
            (baseline_prediction[old] == truth[old]).double().mean()
        )
        baseline_new_acc = float(
            (baseline_prediction[new] == truth[new]).double().mean()
        )
        adapted_h.append(_harmonic(old_acc, new_acc))
        baseline_h.append(_harmonic(baseline_old_acc, baseline_new_acc))
        for key, mask in (("old", old), ("new", new)):
            totals[key] += int(mask.sum())
            correct[key] += int((prediction[mask] == truth[mask]).sum())
            baseline_correct[key] += int((baseline_prediction[mask] == truth[mask]).sum())
        correct["base_old"] += int(
            (adapted_pre.score(adapted_geometry[base_held]).argmax(1) == remapped[base_held]).sum()
        )
        baseline_correct["base_old"] += int(
            (baseline_pre.score(baseline_geometry[base_held]).argmax(1) == remapped[base_held]).sum()
        )
        for class_index in range(4):
            mask = old & (truth == class_index)
            class_correct[(rotation_index, class_index)] = [
                int((prediction[mask] == truth[mask]).sum()), int(mask.sum())
            ]
            baseline_class_correct[(rotation_index, class_index)] = [
                int((baseline_prediction[mask] == truth[mask]).sum()), int(mask.sum())
            ]
    old_accuracy = correct["old"] / max(totals["old"], 1)
    baseline_old_accuracy = baseline_correct["old"] / max(totals["old"], 1)
    old_pre_total = totals["old"]
    old_pre = correct["base_old"] / max(old_pre_total, 1)
    baseline_old_pre = baseline_correct["base_old"] / max(old_pre_total, 1)
    floor = min(value[0] / max(value[1], 1) for value in class_correct.values())
    baseline_floor = min(
        value[0] / max(value[1], 1) for value in baseline_class_correct.values()
    )
    result = {
        "pseudo_old_accuracy": old_accuracy,
        "baseline_pseudo_old_accuracy": baseline_old_accuracy,
        "pseudo_new_accuracy": correct["new"] / max(totals["new"], 1),
        "baseline_pseudo_new_accuracy": baseline_correct["new"] / max(totals["new"], 1),
        "pseudo_old_floor": floor,
        "baseline_pseudo_old_floor": baseline_floor,
        "pseudo_forgetting": old_pre - old_accuracy,
        "baseline_pseudo_forgetting": baseline_old_pre - baseline_old_accuracy,
        "lcb_h_pseudo": _lcb(adapted_h),
        "baseline_lcb_h_pseudo": _lcb(baseline_h),
        "delta_lcb_h_pseudo": _lcb(adapted_h) - _lcb(baseline_h),
        "prediction_change_count": float(changes),
        "nonaffine_energy": float(state.audit["nonaffine_energy"]),
        "query_rows_used": 0.0,
    }
    return MappingProxyType(result)


def stage_a_gate(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    required = (
        "prediction_change_count",
        "delta_lcb_h_pseudo",
        "nonaffine_energy",
        "pseudo_old_accuracy",
        "baseline_pseudo_old_accuracy",
        "pseudo_old_floor",
        "baseline_pseudo_old_floor",
        "pseudo_forgetting",
        "baseline_pseudo_forgetting",
    )
    if any(key not in metrics or not math.isfinite(float(metrics[key])) for key in required):
        raise SAGEDError("Stage A gate metrics are incomplete or non-finite")
    checks = {
        "prediction_changed": float(metrics["prediction_change_count"]) > 0.0,
        "delta_lcb_h_pseudo": float(metrics["delta_lcb_h_pseudo"]) > 0.001,
        "nonaffine_energy": float(metrics["nonaffine_energy"]) >= 0.10,
        "old_post_non_decreasing": float(metrics["pseudo_old_accuracy"])
        >= float(metrics["baseline_pseudo_old_accuracy"]),
        "old_floor_non_decreasing": float(metrics["pseudo_old_floor"])
        >= float(metrics["baseline_pseudo_old_floor"]),
        "forgetting_non_increasing": float(metrics["pseudo_forgetting"])
        <= float(metrics["baseline_pseudo_forgetting"]),
    }
    passed = all(checks.values())
    return MappingProxyType(
        {
            "stage_a_gate_passed": passed,
            "status": "STAGE_A_GATE_PASSED" if passed else "STOPPED_SCIENTIFIC_GATE",
            "checks": MappingProxyType(checks),
        }
    )


__all__ = [
    "RoleRotation",
    "SAGEDConfig",
    "SAGEDError",
    "SAGEDModule",
    "SAGEDState",
    "apply_sage_d",
    "build_role_rotations",
    "coordinate_median_consensus",
    "evaluate_sage_d_crossfit",
    "fit_sage_d",
    "stage_a_gate",
    "support_crossfit_masks",
]
