"""Support-only cross-fit selection for cached slow/fast adapters."""

from __future__ import annotations

from dataclasses import replace
import math
import random
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate, apply_slow_fast
from .slow_fast_objectives import prototype_logits


def _validate_support(features: Tensor, labels: Tensor, prototypes: Tensor, *, k_shot: int) -> tuple[Tensor, Tensor, Tensor]:
    if (features.ndim != 2 or prototypes.ndim != 2 or features.shape[1] != prototypes.shape[1]
            or features.shape[0] != labels.numel() or labels.ndim != 1
            or not features.is_floating_point() or not prototypes.is_floating_point()
            or not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(prototypes).all())):
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
    return replace(state, rho=0.0)


def _fit_common(features: Tensor, labels: Tensor, prototypes: Tensor, state: SlowFastAdapterState) -> SlowFastAdapterState:
    residual = (features - prototypes[labels]).mean(dim=0)
    basis = state.slow_u.to(features)
    ridge = 1.0e-4 * torch.eye(basis.shape[1], device=features.device, dtype=features.dtype)
    coeff = torch.linalg.solve(basis.T @ basis + ridge, basis.T @ residual)
    return replace(state, common_coeff=coeff.detach().cpu())


def _fast_forward(features: Tensor, state: SlowFastAdapterState, gamma: Tensor, beta: Tensor, direction_gate: Tensor | None) -> Tensor:
    u = state.slow_u.to(features)
    v = state.slow_v.to(features)
    hidden = F.layer_norm(features, (features.shape[1],)) @ v
    latent = (1.0 + gamma) * hidden + beta
    if direction_gate is not None:
        latent = torch.tanh(direction_gate) * latent
    return F.normalize(features + float(state.rho) * (latent @ u.T), dim=1, eps=1.0e-8)


def _fit_fast(features: Tensor, labels: Tensor, prototypes: Tensor, state: SlowFastAdapterState, *, steps: int, step_size: float, logit_scale: float) -> SlowFastAdapterState:
    gamma = state.gamma.to(features).clone().requires_grad_(True)
    beta = state.beta.to(features).clone().requires_grad_(True)
    gate = None
    parameters = [gamma, beta]
    if state.direction_gate is not None:
        gate = state.direction_gate.to(features).clone().requires_grad_(True)
        parameters.append(gate)
    for _ in range(int(steps)):
        adapted = _fast_forward(features, state, gamma, beta, gate)
        loss = F.cross_entropy(prototype_logits(adapted, prototypes, logit_scale=logit_scale), labels)
        gradients = torch.autograd.grad(loss, parameters)
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter.add_(gradient, alpha=-float(step_size))
        parameters = [parameter.detach().requires_grad_(True) for parameter in parameters]
        gamma, beta = parameters[:2]
        gate = parameters[2] if len(parameters) == 3 else None
    return replace(state, gamma=gamma.detach().cpu(), beta=beta.detach().cpu(), direction_gate=None if gate is None else gate.detach().cpu())


def _fit(features: Tensor, labels: Tensor, prototypes: Tensor, state: SlowFastAdapterState, *, steps: int, step_size: float, logit_scale: float) -> SlowFastAdapterState:
    if state.candidate is SlowFastCandidate.COMMON_SHIFT_R4:
        return _fit_common(features, labels, prototypes, state)
    return _fit_fast(features, labels, prototypes, state, steps=steps, step_size=step_size, logit_scale=logit_scale)


def _scaled(state: SlowFastAdapterState, strength: float) -> SlowFastAdapterState:
    return replace(state, rho=float(state.rho) * float(strength))


def _stratified_crossfit_splits(labels: Tensor, *, k_shot: int, seed: int, repeats: int) -> tuple[tuple[Tensor, Tensor], ...]:
    if labels.ndim != 1 or int(k_shot) < 2 or int(repeats) < 1:
        raise ValueError("cross-fit requires vector labels, k_shot>=2 and repeats>=1")
    classes = sorted(int(value) for value in torch.unique(labels).tolist())
    rng = random.Random(int(seed))
    splits: list[tuple[Tensor, Tensor]] = []
    for _repeat in range(int(repeats)):
        first: list[int] = []
        second: list[int] = []
        for class_id in classes:
            indices = torch.nonzero(labels == class_id, as_tuple=False).flatten().tolist()
            if len(indices) != int(k_shot):
                raise ValueError("cross-fit labels are not k-shot balanced")
            rng.shuffle(indices)
            first_size = (int(k_shot) + 1) // 2
            first.extend(indices[:first_size])
            second.extend(indices[first_size:])
        device = labels.device
        first_tensor = torch.tensor(first, dtype=torch.long, device=device)
        second_tensor = torch.tensor(second, dtype=torch.long, device=device)
        splits.append((first_tensor, second_tensor))
        splits.append((second_tensor, first_tensor))
    return tuple(splits)


def _trace_metrics(features: Tensor, baseline_features: Tensor, labels: Tensor, prototypes: Tensor, *, logit_scale: float) -> dict[str, Any]:
    normalized = F.normalize(features, dim=1)
    baseline_normalized = F.normalize(baseline_features, dim=1)
    raw_scores = normalized @ prototypes.T
    baseline_scores = baseline_normalized @ prototypes.T
    predictions = raw_scores.argmax(dim=1)
    baseline_predictions = baseline_scores.argmax(dim=1)
    losses = F.cross_entropy(prototype_logits(features, prototypes, logit_scale=logit_scale), labels, reduction="none")
    true_scores = raw_scores.gather(1, labels[:, None]).squeeze(1)
    masked = raw_scores.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    margins = true_scores - masked.max(dim=1).values
    per_class_accuracy: list[float] = []
    per_class_ce: list[float] = []
    per_class_margin: list[float] = []
    for class_index in range(int(prototypes.shape[0])):
        mask = labels == class_index
        per_class_accuracy.append(float((predictions[mask] == labels[mask]).float().mean()))
        per_class_ce.append(float(losses[mask].mean()))
        per_class_margin.append(float(margins[mask].mean()))
    worst_count = max(1, int(math.ceil(0.3 * len(per_class_ce))))
    class_cvar = sum(sorted(per_class_ce, reverse=True)[:worst_count]) / worst_count
    moves = torch.linalg.vector_norm(normalized - baseline_normalized, dim=1)
    baseline_correct = baseline_predictions == labels
    adapted_correct = predictions == labels
    return {
        "macro_accuracy": float(sum(per_class_accuracy) / len(per_class_accuracy)),
        "floor_accuracy": float(min(per_class_accuracy)),
        "macro_ce": float(sum(per_class_ce) / len(per_class_ce)),
        "class_cvar_ce": float(class_cvar),
        "mean_margin": float(margins.mean()),
        "min_class_margin": float(min(per_class_margin)),
        "mean_feature_move": float(moves.mean()),
        "max_feature_move": float(moves.max()),
        "prediction_flip_count": int((predictions != baseline_predictions).sum()),
        "positive_flip_count": int((~baseline_correct & adapted_correct).sum()),
        "negative_flip_count": int((baseline_correct & ~adapted_correct).sum()),
        "per_class_ce": per_class_ce,
    }


def choose_crossfit_lambda(trace: Sequence[Mapping[str, Any]]) -> float:
    eligible = [row for row in trace if float(row["lambda"]) > 0.0 and bool(row.get("eligible", False))]
    if not eligible:
        return 0.0
    return float(min(eligible, key=lambda row: (float(row["risk"]), float(row["lambda"])))["lambda"])


def _crossfit_features(features: Tensor, labels: Tensor, prototypes: Tensor, initial_state: SlowFastAdapterState, strengths: Iterable[float], *, k_shot: int, steps: int, step_size: float, logit_scale: float, seed: int, repeats: int) -> tuple[dict[float, Tensor], Tensor, int]:
    collected = {float(value): [] for value in strengths}
    validation_labels: list[Tensor] = []
    splits = _stratified_crossfit_splits(labels, k_shot=k_shot, seed=seed, repeats=repeats)
    for train, validation in splits:
        fitted = _fit(features.index_select(0, train), labels.index_select(0, train), prototypes, initial_state, steps=steps, step_size=step_size, logit_scale=logit_scale)
        rows = features.index_select(0, validation)
        for strength in collected:
            candidate = _disabled(initial_state) if strength == 0.0 else _scaled(fitted, strength)
            collected[strength].append(F.normalize(rows, dim=1) if strength == 0.0 else apply_slow_fast(rows, candidate))
        validation_labels.append(labels.index_select(0, validation))
    return {key: torch.cat(value, dim=0) for key, value in collected.items()}, torch.cat(validation_labels, dim=0), len(splits)


def fit_support_candidate_states(features: Tensor, labels: Tensor, prototypes: Tensor, initial_state: SlowFastAdapterState, *, steps: int, step_size: float, logit_scale: float, lambda_grid: Iterable[float]) -> dict[float, SlowFastAdapterState]:
    fitted = _fit(features, labels, prototypes, initial_state, steps=steps, step_size=step_size, logit_scale=logit_scale)
    return {float(strength): (_disabled(initial_state) if float(strength) == 0.0 else _scaled(fitted, float(strength))) for strength in lambda_grid}


def select_support_only_state_legacy(features: Tensor, labels: Tensor, prototypes: Tensor, initial_state: SlowFastAdapterState, *, k_shot: int, logit_scale: float, trust_radius: float, steps: int = 3, step_size: float = 0.02, lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> tuple[SlowFastAdapterState, dict[str, Any]]:
    """Reproduce the V1 one-row LOO gate for a preregistered diagnostic shadow."""
    features, labels, prototypes = _validate_support(features, labels, prototypes, k_shot=k_shot)
    strengths = tuple(sorted(set(float(value) for value in lambda_grid)))
    baseline_state = _disabled(initial_state)
    if int(k_shot) == 1:
        return baseline_state, {"selected_lambda": 0.0, "reason": "K1_NO_INDEPENDENT_LOO_FALLBACK_DA0", "loo_fit_count": 0, "attempted_gradient_updates": 0, "committed_gradient_updates": 0}
    collected = {strength: [] for strength in strengths}
    for held_out in range(int(features.shape[0])):
        keep = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        keep[held_out] = False
        fitted = _fit(features[keep], labels[keep], prototypes, initial_state, steps=steps, step_size=step_size, logit_scale=logit_scale)
        row = features[held_out : held_out + 1]
        for strength in strengths:
            state = baseline_state if strength == 0.0 else _scaled(fitted, strength)
            collected[strength].append(F.normalize(row, dim=1) if strength == 0.0 else apply_slow_fast(row, state))
    predictions = {strength: torch.cat(rows, dim=0) for strength, rows in collected.items()}
    baseline = _trace_metrics(predictions[0.0], predictions[0.0], labels, prototypes, logit_scale=logit_scale)
    selected_strength = 0.0
    for strength in strengths[1:]:
        metrics = _trace_metrics(predictions[strength], predictions[0.0], labels, prototypes, logit_scale=logit_scale)
        class_improved = any(candidate < base - 1.0e-8 for candidate, base in zip(metrics["per_class_ce"], baseline["per_class_ce"]))
        not_worse = (metrics["macro_accuracy"] + 1.0e-8 >= baseline["macro_accuracy"] and metrics["floor_accuracy"] + 1.0e-8 >= baseline["floor_accuracy"] and metrics["mean_margin"] + 1.0e-8 >= baseline["mean_margin"] and metrics["max_feature_move"] <= float(trust_radius))
        improved = (metrics["macro_accuracy"] > baseline["macro_accuracy"] + 1.0e-8 or metrics["floor_accuracy"] > baseline["floor_accuracy"] + 1.0e-8 or metrics["mean_margin"] > baseline["mean_margin"] + 1.0e-8 or class_improved)
        if not_worse and improved:
            selected_strength = strength
    full_states = fit_support_candidate_states(features, labels, prototypes, initial_state, steps=steps, step_size=step_size, logit_scale=logit_scale, lambda_grid=strengths)
    committed = int(steps) if selected_strength > 0.0 and initial_state.candidate is not SlowFastCandidate.COMMON_SHIFT_R4 else 0
    attempted = (int(features.shape[0]) + 1) * int(steps) if initial_state.candidate is not SlowFastCandidate.COMMON_SHIFT_R4 else 0
    return full_states[selected_strength], {"selected_lambda": selected_strength, "reason": "SUPPORT_LOO_NONDEGRADATION_PASS" if selected_strength > 0.0 else "SUPPORT_GATE_FALLBACK_DA0", "loo_fit_count": int(features.shape[0]), "attempted_gradient_updates": attempted, "committed_gradient_updates": committed}


def select_support_only_state(features: Tensor, labels: Tensor, prototypes: Tensor, initial_state: SlowFastAdapterState, *, k_shot: int, logit_scale: float, trust_radius: float, steps: int = 3, step_size: float = 0.02, lambda_grid: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0), crossfit_seed: int = 392002, repeats: int = 3, macro_tolerance: float = 0.02, floor_tolerance: float = 0.10, minimum_risk_gain: float = 1.0e-4) -> tuple[SlowFastAdapterState, dict[str, Any]]:
    """Select one adapter state from target support without opening query."""
    features, labels, prototypes = _validate_support(features, labels, prototypes, k_shot=k_shot)
    if initial_state.feature_dim != int(features.shape[1]):
        raise ValueError("adapter feature width does not match support")
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    scale = float(logit_scale)
    radius = float(trust_radius)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("logit_scale must be finite and positive")
    if not math.isfinite(radius) or radius <= 0.0 or radius >= 2.0:
        raise ValueError("trust_radius must be explicit and lie in (0, 2)")
    strengths = tuple(sorted(set(float(value) for value in lambda_grid)))
    if not strengths or strengths[0] != 0.0 or strengths[-1] > 1.0:
        raise ValueError("lambda_grid must include 0 and stay within [0, 1]")
    baseline_state = _disabled(initial_state)
    baseline_features = F.normalize(features, dim=1)
    if int(k_shot) == 1:
        baseline = _trace_metrics(baseline_features, baseline_features, labels, prototypes, logit_scale=scale)
        risk = baseline["macro_ce"] + 0.3 * baseline["class_cvar_ce"]
        trace = [{"lambda": 0.0, **baseline, "risk": risk, "risk_gain": 0.0, "eligible": True, "rejection_reasons": [], "per_class_ce_delta": [0.0 for _ in baseline["per_class_ce"]]}]
        return baseline_state, {"selected_lambda": 0.0, "reason": "K1_NO_INDEPENDENT_LOO_FALLBACK_DA0", "gradient_updates": 0, "attempted_gradient_updates": 0, "committed_gradient_updates": 0, "crossfit_fit_count": 0, "loo_fit_count": 0, "support_logit_scale": scale, "trust_radius": radius, "lambda_trace": trace, **{f"baseline_{key}": value for key, value in baseline.items() if key != "per_class_ce"}, **{f"selected_{key}": value for key, value in baseline.items() if key != "per_class_ce"}}
    crossfit, crossfit_labels, fit_count = _crossfit_features(features, labels, prototypes, initial_state, strengths, k_shot=k_shot, steps=steps, step_size=step_size, logit_scale=scale, seed=crossfit_seed, repeats=repeats)
    baseline = _trace_metrics(crossfit[0.0], crossfit[0.0], crossfit_labels, prototypes, logit_scale=scale)
    baseline_risk = baseline["macro_ce"] + 0.3 * baseline["class_cvar_ce"]
    trace: list[dict[str, Any]] = []
    for strength in strengths:
        metrics = _trace_metrics(crossfit[strength], crossfit[0.0], crossfit_labels, prototypes, logit_scale=scale)
        risk = metrics["macro_ce"] + 0.3 * metrics["class_cvar_ce"] + 0.1 * metrics["mean_feature_move"]
        reasons: list[str] = []
        if strength > 0.0:
            if metrics["macro_accuracy"] < baseline["macro_accuracy"] - float(macro_tolerance): reasons.append("MACRO_TOLERANCE")
            if metrics["floor_accuracy"] < baseline["floor_accuracy"] - float(floor_tolerance): reasons.append("FLOOR_TOLERANCE")
            if metrics["max_feature_move"] > radius: reasons.append("TRUST_RADIUS")
            if baseline_risk - risk < float(minimum_risk_gain): reasons.append("INSUFFICIENT_RISK_GAIN")
        trace.append({"lambda": strength, **metrics, "risk": float(risk), "risk_gain": float(baseline_risk - risk), "eligible": strength == 0.0 or not reasons, "rejection_reasons": reasons, "per_class_ce_delta": [float(candidate - base) for candidate, base in zip(metrics["per_class_ce"], baseline["per_class_ce"])]})
    selected_strength = choose_crossfit_lambda(trace)
    full_states = fit_support_candidate_states(features, labels, prototypes, initial_state, steps=steps, step_size=step_size, logit_scale=scale, lambda_grid=strengths)
    selected = full_states[selected_strength]
    selected_trace = next(row for row in trace if row["lambda"] == selected_strength)
    committed = int(steps) if selected_strength > 0.0 and initial_state.candidate is not SlowFastCandidate.COMMON_SHIFT_R4 else 0
    attempted = (fit_count + 1) * int(steps) if initial_state.candidate is not SlowFastCandidate.COMMON_SHIFT_R4 else 0
    return selected, {"selected_lambda": selected_strength, "reason": "SUPPORT_CROSSFIT_CONTINUOUS_RISK_PASS" if selected_strength > 0.0 else "SUPPORT_CROSSFIT_FALLBACK_DA0", "gradient_updates": committed, "attempted_gradient_updates": attempted, "committed_gradient_updates": committed, "crossfit_fit_count": fit_count, "loo_fit_count": fit_count, "support_logit_scale": scale, "trust_radius": radius, "lambda_trace": trace, **{f"baseline_{key}": value for key, value in baseline.items() if key != "per_class_ce"}, **{f"selected_{key}": value for key, value in selected_trace.items() if key not in {"lambda", "per_class_ce", "per_class_ce_delta", "eligible", "rejection_reasons"}}, "max_per_class_loss_increase": max(selected_trace["per_class_ce_delta"])}


__all__ = ["choose_crossfit_lambda", "fit_support_candidate_states", "select_support_only_state", "select_support_only_state_legacy"]
