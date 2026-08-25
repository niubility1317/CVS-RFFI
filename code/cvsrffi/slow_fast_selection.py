"""Support-only selection for cached slow/fast adapters."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from .slow_fast_adapter import (
    SlowFastAdapterState,
    SlowFastCandidate,
    apply_slow_fast,
)


def _validate_support(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    *,
    k_shot: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if (
        features.ndim != 2
        or prototypes.ndim != 2
        or features.shape[1] != prototypes.shape[1]
        or features.shape[0] != labels.numel()
        or labels.ndim != 1
        or not features.is_floating_point()
        or not prototypes.is_floating_point()
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(prototypes).all())
    ):
        raise ValueError("support features, labels and prototypes must be finite and aligned")
    if int(k_shot) < 1:
        raise ValueError("k_shot must be positive")
    labels = labels.long()
    if bool((labels < 0).any()) or bool((labels >= prototypes.shape[0]).any()):
        raise ValueError("support labels must be prototype row indices")
    counts = torch.bincount(labels, minlength=prototypes.shape[0])
    if not bool((counts == int(k_shot)).all()):
        raise ValueError("support must contain exactly k_shot independent rows per class")
    return features.float(), labels, F.normalize(prototypes.float(), dim=1)


def _disabled(state: SlowFastAdapterState) -> SlowFastAdapterState:
    if state.candidate is SlowFastCandidate.COMMON_SHIFT_R4:
        return replace(state, rho=0.0, common_coeff=torch.zeros_like(state.common_coeff))
    return replace(state, rho=0.0)


def _fit_common(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    state: SlowFastAdapterState,
) -> SlowFastAdapterState:
    residual = (features - prototypes[labels]).mean(dim=0)
    basis = state.slow_u.to(features)
    ridge = 1.0e-4 * torch.eye(basis.shape[1], device=features.device)
    coeff = torch.linalg.solve(basis.T @ basis + ridge, basis.T @ residual)
    return replace(state, common_coeff=coeff.detach().cpu())


def _fast_forward(
    features: Tensor,
    state: SlowFastAdapterState,
    gamma: Tensor,
    beta: Tensor,
    direction_gate: Tensor | None,
) -> Tensor:
    u = state.slow_u.to(features)
    v = state.slow_v.to(features)
    hidden = F.layer_norm(features, (features.shape[1],)) @ v
    latent = (1.0 + gamma) * hidden + beta
    if direction_gate is not None:
        latent = torch.sigmoid(direction_gate) * latent
    return F.normalize(
        features + float(state.rho) * (latent @ u.T), dim=1, eps=1.0e-8
    )


def _fit_fast(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    state: SlowFastAdapterState,
    *,
    steps: int,
    step_size: float,
) -> SlowFastAdapterState:
    gamma = state.gamma.to(features).clone().requires_grad_(True)
    beta = state.beta.to(features).clone().requires_grad_(True)
    gate = None
    parameters = [gamma, beta]
    if state.direction_gate is not None:
        gate = state.direction_gate.to(features).clone().requires_grad_(True)
        parameters.append(gate)
    for _ in range(int(steps)):
        adapted = _fast_forward(features, state, gamma, beta, gate)
        loss = F.cross_entropy(8.0 * (adapted @ prototypes.T), labels)
        gradients = torch.autograd.grad(loss, parameters)
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter.add_(gradient, alpha=-float(step_size))
        parameters = [parameter.detach().requires_grad_(True) for parameter in parameters]
        gamma, beta = parameters[:2]
        gate = parameters[2] if len(parameters) == 3 else None
    return replace(
        state,
        gamma=gamma.detach().cpu(),
        beta=beta.detach().cpu(),
        direction_gate=None if gate is None else gate.detach().cpu(),
    )


def _fit(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    state: SlowFastAdapterState,
    *,
    steps: int,
    step_size: float,
) -> SlowFastAdapterState:
    if state.candidate is SlowFastCandidate.COMMON_SHIFT_R4:
        return _fit_common(features, labels, prototypes, state)
    return _fit_fast(
        features,
        labels,
        prototypes,
        state,
        steps=steps,
        step_size=step_size,
    )


def _scaled(state: SlowFastAdapterState, strength: float) -> SlowFastAdapterState:
    strength = float(strength)
    if state.candidate is SlowFastCandidate.COMMON_SHIFT_R4:
        return replace(state, rho=strength, common_coeff=state.common_coeff * strength)
    return replace(state, rho=state.rho * strength)


def _metrics(features: Tensor, labels: Tensor, prototypes: Tensor) -> dict[str, float]:
    scores = features @ prototypes.T
    predictions = scores.argmax(dim=1)
    per_class = []
    for class_index in range(int(prototypes.shape[0])):
        mask = labels == class_index
        per_class.append(float((predictions[mask] == labels[mask]).float().mean()))
    true_scores = scores.gather(1, labels[:, None]).squeeze(1)
    masked = scores.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    margin = float((true_scores - masked.max(dim=1).values).mean())
    return {
        "macro_accuracy": float(sum(per_class) / len(per_class)),
        "floor_accuracy": float(min(per_class)),
        "margin": margin,
    }


def _per_class_losses(features: Tensor, labels: Tensor, prototypes: Tensor) -> tuple[float, ...]:
    losses = F.cross_entropy(8.0 * (features @ prototypes.T), labels, reduction="none")
    return tuple(
        float(losses[labels == class_index].mean())
        for class_index in range(int(prototypes.shape[0]))
    )


def _loo_predictions(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    state: SlowFastAdapterState,
    strengths: Iterable[float],
    *,
    steps: int,
    step_size: float,
) -> dict[float, Tensor]:
    collected = {float(value): [] for value in strengths}
    for held_out in range(int(features.shape[0])):
        keep = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        keep[held_out] = False
        fitted = _fit(
            features[keep], labels[keep], prototypes, state,
            steps=steps, step_size=step_size,
        )
        row = features[held_out : held_out + 1]
        for strength in collected:
            candidate = _disabled(state) if strength == 0.0 else _scaled(fitted, strength)
            collected[strength].append(
                F.normalize(row, dim=1) if strength == 0.0 else apply_slow_fast(row, candidate)
            )
    return {key: torch.cat(value, dim=0) for key, value in collected.items()}


def select_support_only_state(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    initial_state: SlowFastAdapterState,
    *,
    k_shot: int,
    steps: int = 3,
    step_size: float = 0.02,
    trust_radius: float = 2.0,
    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[SlowFastAdapterState, dict[str, Any]]:
    """Select adapter strength using support rows only; query is never an input."""

    features, labels, prototypes = _validate_support(
        features, labels, prototypes, k_shot=k_shot
    )
    if initial_state.feature_dim != int(features.shape[1]):
        raise ValueError("adapter feature width does not match support")
    if int(steps) != 3:
        raise ValueError("formal slow-fast selection requires exactly three updates")
    baseline_state = _disabled(initial_state)
    baseline_features = F.normalize(features, dim=1)
    baseline = _metrics(baseline_features, labels, prototypes)
    if int(k_shot) == 1:
        return baseline_state, {
            "selected_lambda": 0.0,
            "reason": "K1_NO_INDEPENDENT_LOO_FALLBACK_DA0",
            "gradient_updates": 0,
            **{f"baseline_{key}": value for key, value in baseline.items()},
            **{f"selected_{key}": value for key, value in baseline.items()},
            "max_relative_move": 0.0,
        }

    strengths = tuple(sorted(set(float(value) for value in lambda_grid)))
    if not strengths or strengths[0] != 0.0 or strengths[-1] > 1.0:
        raise ValueError("lambda_grid must include 0 and stay within [0, 1]")
    loo = _loo_predictions(
        features, labels, prototypes, initial_state, strengths,
        steps=steps, step_size=step_size,
    )
    baseline = _metrics(loo[0.0], labels, prototypes)
    baseline_class_losses = _per_class_losses(loo[0.0], labels, prototypes)
    selected_strength = 0.0
    selected_metrics = baseline
    selected_move = 0.0
    selected_class_losses = baseline_class_losses
    eps = 1.0e-8
    for strength in strengths[1:]:
        metrics = _metrics(loo[strength], labels, prototypes)
        class_losses = _per_class_losses(loo[strength], labels, prototypes)
        move = float(torch.linalg.vector_norm(loo[strength] - loo[0.0], dim=1).max())
        not_worse = (
            metrics["macro_accuracy"] + eps >= baseline["macro_accuracy"]
            and metrics["floor_accuracy"] + eps >= baseline["floor_accuracy"]
            and metrics["margin"] + eps >= baseline["margin"]
            and move <= float(trust_radius)
        )
        improved = any(metrics[key] > baseline[key] + eps for key in metrics) or any(
            candidate_loss < baseline_loss - eps
            for candidate_loss, baseline_loss in zip(class_losses, baseline_class_losses)
        )
        if not_worse and improved:
            selected_strength = strength
            selected_metrics = metrics
            selected_move = move
            selected_class_losses = class_losses

    if selected_strength == 0.0:
        selected = baseline_state
        reason = "SUPPORT_GATE_FALLBACK_DA0"
        updates = 0
    else:
        fitted = _fit(
            features, labels, prototypes, initial_state,
            steps=steps, step_size=step_size,
        )
        selected = _scaled(fitted, selected_strength)
        reason = "SUPPORT_LOO_NONDEGRADATION_PASS"
        updates = 0 if initial_state.candidate is SlowFastCandidate.COMMON_SHIFT_R4 else int(steps)
    return selected, {
        "selected_lambda": selected_strength,
        "reason": reason,
        "gradient_updates": updates,
        **{f"baseline_{key}": value for key, value in baseline.items()},
        **{f"selected_{key}": value for key, value in selected_metrics.items()},
        "max_relative_move": selected_move,
        "max_per_class_loss_increase": max(
            selected_loss - baseline_loss
            for selected_loss, baseline_loss in zip(
                selected_class_losses, baseline_class_losses
            )
        ),
    }


__all__ = ["select_support_only_state"]
