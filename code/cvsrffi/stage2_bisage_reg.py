"""Stage B SAGE-R: boundary-local registration adaptation with old-risk control."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvsrffi.stage2_binova_d92 import d92_geometry_features
from cvsrffi.stage2_binova_features import BiNOVAFeatures, BiNOVASupport
from cvsrffi.stage2_bisage_d92 import BiSAGED92State, fit_bisage_d92
from cvsrffi.stage2_bisage_da import (
    SAGEDState,
    apply_sage_d,
    support_crossfit_masks,
)


class SAGERError(ValueError):
    """Raised when Stage B registration geometry or protocol is invalid."""


@dataclass(frozen=True)
class SAGERConfig:
    steps: int = 2000
    learning_rate: float = 3.0e-4
    rank: int = 16
    boundary_tau: float = 1.0
    boundary_gamma: float = 5.0
    intrusion_margin: float = 0.20
    epsilon_old: float = 0.0
    rho: float = 2.0
    weight_old_to_new: float = 0.20
    weight_new_to_old: float = 0.10
    weight_forgetting: float = 0.20
    weight_topology: float = 0.05
    weight_condition: float = 0.01
    condition_log_threshold: float = 18.42
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps < 1 or self.learning_rate <= 0.0 or self.rank < 1:
            raise SAGERError("Stage B steps, learning rate, and rank must be positive")
        if self.boundary_tau < 0.0 or self.boundary_gamma <= 0.0:
            raise SAGERError("Stage B boundary gate configuration is invalid")
        if self.intrusion_margin < 0.0 or self.epsilon_old < 0.0 or self.rho <= 0.0:
            raise SAGERError("Stage B risk configuration is invalid")


def boundary_gate(margin: torch.Tensor, *, tau: float, gamma: float) -> torch.Tensor:
    values = torch.as_tensor(margin)
    return torch.sigmoid(float(gamma) * (float(tau) - values.abs()))


def update_eta(*, eta: float, violation: float, rho: float) -> float:
    return max(0.0, float(eta) + float(rho) * float(violation))


class SAGERModule(nn.Module):
    """Zero-initialized joint identity/FFT residual with D92 boundary gating."""

    def __init__(
        self,
        rank: int = 16,
        boundary_tau: float = 1.0,
        boundary_gamma: float = 5.0,
    ) -> None:
        super().__init__()
        width = int(rank)
        self.feature_down = nn.Linear(256, width, bias=True)
        self.condition_down = nn.Linear(6, width, bias=False)
        self.context_down = nn.Linear(8, width, bias=False)
        self.identity_up = nn.Linear(width, 160, bias=False)
        self.fft_up = nn.Linear(width, 96, bias=False)
        self.boundary_tau = float(boundary_tau)
        self.boundary_gamma = float(boundary_gamma)
        nn.init.zeros_(self.identity_up.weight)
        nn.init.zeros_(self.fft_up.weight)

    def forward(
        self,
        identity160: torch.Tensor,
        fft96: torch.Tensor,
        geometry6: torch.Tensor,
        registration_context8: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            identity160.ndim != 2
            or identity160.shape[1] != 160
            or fft96.shape != (len(identity160), 96)
            or geometry6.shape != (len(identity160), 6)
            or registration_context8.shape != (8,)
        ):
            raise SAGERError("Stage B forward feature geometry drift")
        context = registration_context8[None, :].expand(len(identity160), -1)
        hidden = torch.tanh(
            self.feature_down(torch.cat([identity160, fft96], dim=1))
            + self.condition_down(geometry6)
            + self.context_down(context)
        )
        gate = boundary_gate(
            geometry6[:, 0], tau=self.boundary_tau, gamma=self.boundary_gamma
        )[:, None]
        return (
            identity160 + gate * self.identity_up(hidden),
            fft96 + gate * self.fft_up(hidden),
        )


@dataclass(frozen=True)
class SAGERState:
    module: SAGERModule
    stage_a: SAGEDState
    conditioning_d92: BiSAGED92State
    condition_mean6: torch.Tensor
    condition_scale6: torch.Tensor
    registration_context8: torch.Tensor
    old_class_count: int
    config: SAGERConfig
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        for value, shape, name in (
            (self.condition_mean6, (6,), "condition_mean6"),
            (self.condition_scale6, (6,), "condition_scale6"),
            (self.registration_context8, (8,), "registration_context8"),
        ):
            if value.shape != shape or not torch.isfinite(value).all():
                raise SAGERError(f"Stage B {name} is invalid")
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def d92_boundary_conditions(state: BiSAGED92State, rows: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(rows, dtype=state.centers.dtype, device=state.centers.device)
    differences = values[:, None, :] - state.centers[None, :, :]
    solved = torch.cholesky_solve(
        differences.reshape(-1, differences.shape[-1]).T, state.cholesky
    ).T.reshape_as(differences)
    distances = torch.sum(differences * solved, dim=2)
    old_distances = torch.sort(distances[:, : state.old_class_count], dim=1).values
    new_distances = torch.sort(distances[:, state.old_class_count :], dim=1).values
    if new_distances.shape[1] < 1:
        raise SAGERError("Stage B boundary conditions require registered new classes")
    old_second = old_distances[:, 1] if old_distances.shape[1] > 1 else old_distances[:, 0]
    new_second = new_distances[:, 1] if new_distances.shape[1] > 1 else new_distances[:, 0]
    logits = state.score(values)
    margin = logits[:, : state.old_class_count].max(1).values
    margin -= logits[:, state.old_class_count :].max(1).values
    probabilities = torch.softmax(logits, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(1)
    return torch.stack(
        [margin, entropy, old_distances[:, 0], old_second, new_distances[:, 0], new_second],
        dim=1,
    )


def _registration_context(state: BiSAGED92State) -> torch.Tensor:
    eigenvalues = torch.linalg.eigvalsh(state.covariance).clamp_min(1.0e-18)
    differences = state.centers[:, None, :] - state.centers[None, :, :]
    solved = torch.cholesky_solve(
        differences.reshape(-1, differences.shape[-1]).T, state.cholesky
    ).T.reshape_as(differences)
    distances = torch.sum(differences * solved, dim=-1)
    old_count = state.old_class_count

    def off_diagonal_min(value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] < 2:
            return value.new_tensor(0.0)
        mask = ~torch.eye(value.shape[0], dtype=torch.bool, device=value.device)
        return value[mask].min()

    old_old = off_diagonal_min(distances[:old_count, :old_count])
    new_new = off_diagonal_min(distances[old_count:, old_count:])
    old_new = distances[:old_count, old_count:].min()
    middle = eigenvalues[len(eigenvalues) // 2]
    condition = eigenvalues[-1] / eigenvalues[0]
    new_fraction = (len(state.class_ids) - old_count) / float(len(state.class_ids))
    return torch.stack(
        [
            eigenvalues[0].log(),
            middle.log(),
            eigenvalues[-1].log(),
            condition.log(),
            old_new,
            old_old,
            new_new,
            eigenvalues.new_tensor(new_fraction),
        ]
    ).detach()


def _true_margin(logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    true_score = logits.gather(1, truth[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, truth[:, None], -torch.inf)
    return true_score - masked.max(1).values


def _intrusion_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    old_class_count: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    old = labels < old_class_count
    new = ~old
    true_score = logits.gather(1, labels[:, None]).squeeze(1)
    old_to_new = F.relu(
        float(margin)
        + logits[old, old_class_count:].max(1).values
        - true_score[old]
    ).mean()
    new_to_old = F.relu(
        float(margin)
        + logits[new, :old_class_count].max(1).values
        - true_score[new]
    ).mean()
    return old_to_new, new_to_old


def _old_topology(state: BiSAGED92State) -> torch.Tensor:
    centers = state.centers[: state.old_class_count]
    differences = centers[:, None, :] - centers[None, :, :]
    solved = torch.cholesky_solve(
        differences.reshape(-1, differences.shape[-1]).T, state.cholesky
    ).T.reshape_as(differences)
    return torch.sum(differences * solved, dim=-1)


def fit_sage_r(
    stage_a: SAGEDState,
    support: BiNOVASupport,
    *,
    old_class_count: int,
    config: SAGERConfig | None = None,
    device: str | torch.device,
) -> SAGERState:
    if not isinstance(stage_a, SAGEDState) or not isinstance(support, BiNOVASupport):
        raise TypeError("Stage B requires frozen Stage A state and registered support")
    if stage_a.audit.get("selected_mode") == "S0":
        raise SAGERError("Stage B cannot run after Stage A fallback")
    settings = SAGERConfig() if config is None else config
    classes, counts = np.unique(support.labels, return_counts=True)
    old_count = int(old_class_count)
    if (
        not np.array_equal(classes, np.arange(len(classes)))
        or not 0 < old_count < len(classes)
        or len(set(counts.tolist())) != 1
        or int(counts[0]) < 5
    ):
        raise SAGERError("Stage B requires contiguous equal K>=5 old/new support")
    for parameter in stage_a.module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    stage_a.module.eval()

    target_device = torch.device(device)
    torch.manual_seed(int(settings.seed))
    base_identity = torch.as_tensor(
        apply_sage_d(stage_a, support.features), dtype=torch.float64, device=target_device
    )
    fft = torch.as_tensor(
        support.features.fft96.copy(), dtype=torch.float64, device=target_device
    )
    labels = torch.as_tensor(support.labels.copy(), dtype=torch.long, device=target_device)
    fit_np, held_np = support_crossfit_masks(support.labels, support.ranks)
    fit = torch.as_tensor(fit_np, dtype=torch.bool, device=target_device)
    held = torch.as_tensor(held_np, dtype=torch.bool, device=target_device)
    base_geometry = d92_geometry_features(base_identity, fft)
    conditioning = fit_bisage_d92(
        base_geometry[fit], labels[fit], old_class_count=old_count
    )
    with torch.no_grad():
        raw_conditions = d92_boundary_conditions(conditioning, base_geometry)
        condition_mean = raw_conditions[fit].mean(0)
        condition_scale = raw_conditions[fit].std(0, unbiased=False).clamp_min(1.0e-6)
        conditions = (raw_conditions - condition_mean) / condition_scale
        registration_context = _registration_context(conditioning)
        baseline_scores = conditioning.score(base_geometry[held])
        old_held = labels[held] < old_count
        baseline_old_risk = F.cross_entropy(
            baseline_scores[old_held], labels[held][old_held]
        ).detach()
        baseline_old_margin = _true_margin(
            baseline_scores[old_held], labels[held][old_held]
        )
        baseline_topology = _old_topology(conditioning).detach()

    module = SAGERModule(
        settings.rank, settings.boundary_tau, settings.boundary_gamma
    ).to(device=target_device, dtype=torch.float64)
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=settings.learning_rate, weight_decay=1.0e-4
    )
    eta = 0.0
    final = {}
    for _ in range(settings.steps):
        adapted_identity, adapted_fft = module(
            base_identity, fft, conditions, registration_context
        )
        geometry = d92_geometry_features(adapted_identity, adapted_fft)
        d92 = fit_bisage_d92(geometry[fit], labels[fit], old_class_count=old_count)
        scores = d92.score(geometry[held])
        held_labels = labels[held]
        old_rows = held_labels < old_count
        new_rows = ~old_rows
        old_risk = F.cross_entropy(scores[old_rows], held_labels[old_rows])
        new_risk = F.cross_entropy(scores[new_rows], held_labels[new_rows])
        old_to_new, new_to_old = _intrusion_losses(
            scores, held_labels, old_count, settings.intrusion_margin
        )
        post_old_margin = _true_margin(scores[old_rows], held_labels[old_rows])
        forgetting = F.relu(baseline_old_margin - post_old_margin).mean()
        topology = (_old_topology(d92) - baseline_topology).square().mean()
        eigenvalues = torch.linalg.eigvalsh(d92.covariance)
        log_condition = torch.log(eigenvalues[-1] / eigenvalues[0])
        condition_penalty = F.relu(log_condition - settings.condition_log_threshold).square()
        violation = old_risk - baseline_old_risk - settings.epsilon_old
        loss = (
            new_risk
            + settings.weight_old_to_new * old_to_new
            + settings.weight_new_to_old * new_to_old
            + settings.weight_forgetting * forgetting
            + settings.weight_topology * topology
            + settings.weight_condition * condition_penalty
            + float(eta) * violation
            + 0.5 * settings.rho * F.relu(violation).square()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
        optimizer.step()
        eta = update_eta(
            eta=eta, violation=float(violation.detach().cpu()), rho=settings.rho
        )
        final = {
            "final_loss": float(loss.detach().cpu()),
            "final_old_risk": float(old_risk.detach().cpu()),
            "baseline_old_risk": float(baseline_old_risk.cpu()),
            "final_new_risk": float(new_risk.detach().cpu()),
            "final_old_to_new": float(old_to_new.detach().cpu()),
            "final_new_to_old": float(new_to_old.detach().cpu()),
            "final_forgetting_margin_loss": float(forgetting.detach().cpu()),
            "final_eta": eta,
            "final_old_risk_violation": float(violation.detach().cpu()),
        }

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return SAGERState(
        module=module,
        stage_a=stage_a,
        conditioning_d92=conditioning,
        condition_mean6=condition_mean.detach(),
        condition_scale6=condition_scale.detach(),
        registration_context8=registration_context.detach(),
        old_class_count=old_count,
        config=settings,
        audit={
            "stage_a_frozen": True,
            "old_risk_constraint": "augmented_lagrangian",
            "boundary_gate": "sigmoid_gamma_tau_abs_old_new_margin",
            "crossfit_fit_per_class": int(np.sum(fit_np & (support.labels == 0))),
            "crossfit_held_per_class": int(np.sum(held_np & (support.labels == 0))),
            "query_rows_used": 0,
            "query_truth_access": False,
            **final,
        },
    )


def apply_sage_r(
    state: SAGERState, features: BiNOVAFeatures
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(state, SAGERState):
        raise TypeError("apply_sage_r requires a Stage B state")
    parameter = next(state.module.parameters())
    base_identity = torch.as_tensor(
        apply_sage_d(state.stage_a, features),
        dtype=parameter.dtype,
        device=parameter.device,
    )
    fft = torch.as_tensor(
        features.fft96.copy(), dtype=parameter.dtype, device=parameter.device
    )
    with torch.inference_mode():
        geometry = d92_geometry_features(base_identity, fft)
        raw = d92_boundary_conditions(state.conditioning_d92, geometry)
        conditions = (raw - state.condition_mean6) / state.condition_scale6
        identity, adapted_fft = state.module(
            base_identity, fft, conditions, state.registration_context8
        )
    return (
        np.asarray(identity.detach().cpu().tolist(), dtype=np.float32),
        np.asarray(adapted_fft.detach().cpu().tolist(), dtype=np.float32),
    )


__all__ = [
    "SAGERConfig",
    "SAGERError",
    "SAGERModule",
    "SAGERState",
    "apply_sage_r",
    "boundary_gate",
    "d92_boundary_conditions",
    "fit_sage_r",
    "update_eta",
]
