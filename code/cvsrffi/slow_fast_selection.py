"""Support-only cross-fit selection for cached slow/fast adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate, apply_slow_fast
from .slow_fast_objectives import prototype_logits


@dataclass(frozen=True)
class SupportTrustPolicy:
    """Frozen support-only geometry and repeated-fold stability limits."""

    q90_move: float
    hard_move: float
    q90_relative_move: float
    minimum_positive_folds: int = 5
    lcb_z: float = 1.2815515655446004
    require_fold_lcb: bool = True

    def __post_init__(self) -> None:
        values = (self.q90_move, self.hard_move, self.q90_relative_move, self.lcb_z)
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("support trust policy limits must be finite and positive")
        if not 0.0 < float(self.q90_move) <= float(self.hard_move) < 2.0:
            raise ValueError("support trust policy requires 0 < q90_move <= hard_move < 2")
        if int(self.minimum_positive_folds) < 0:
            raise ValueError("minimum_positive_folds must be nonnegative")
        if not isinstance(self.require_fold_lcb, bool):
            raise ValueError("require_fold_lcb must be boolean")


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


def _stratified_crossfit_splits(
    labels: Tensor,
    *,
    k_shot: int,
    seed: int,
    repeats: int,
    physical_ids: Sequence[Any] | None = None,
) -> tuple[tuple[Tensor, Tensor], ...]:
    if labels.ndim != 1 or int(k_shot) < 2 or int(repeats) < 1:
        raise ValueError("cross-fit requires vector labels, k_shot>=2 and repeats>=1")
    if physical_ids is not None:
        if len(physical_ids) != int(labels.numel()):
            raise ValueError("physical IDs must align with support rows")
        try:
            unique_physical_ids = set(physical_ids)
        except TypeError as error:
            raise ValueError("physical IDs must be hashable") from error
        if len(unique_physical_ids) != len(physical_ids):
            raise ValueError("physical IDs must be unique across support rows")
    classes = sorted(int(value) for value in torch.unique(labels).tolist())
    rng = random.Random(int(seed))
    splits: list[tuple[Tensor, Tensor]] = []
    canonical_partitions: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    attempts = 0
    maximum_attempts = max(1000, int(repeats) * 1000)
    while len(canonical_partitions) < int(repeats) and attempts < maximum_attempts:
        attempts += 1
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
        ordered_halves = sorted((tuple(sorted(first)), tuple(sorted(second))))
        canonical = (ordered_halves[0], ordered_halves[1])
        if canonical in canonical_partitions:
            continue
        canonical_partitions.add(canonical)
        device = labels.device
        first_tensor = torch.tensor(first, dtype=torch.long, device=device)
        second_tensor = torch.tensor(second, dtype=torch.long, device=device)
        if physical_ids is not None:
            train_ids = {physical_ids[index] for index in first}
            validation_ids = {physical_ids[index] for index in second}
            if not train_ids.isdisjoint(validation_ids):
                raise ValueError("cross-fit train and validation physical IDs overlap")
        splits.append((first_tensor, second_tensor))
        splits.append((second_tensor, first_tensor))
    if not canonical_partitions:
        raise ValueError("cross-fit could not construct a unique stratified partition")
    return tuple(splits)


def _movement_statistics(
    adapted_features: Tensor,
    baseline_features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
) -> dict[str, float]:
    adapted = F.normalize(adapted_features, dim=1)
    baseline = F.normalize(baseline_features, dim=1)
    moves = torch.linalg.vector_norm(adapted - baseline, dim=1)
    scores = baseline @ prototypes.T
    true_scores = scores.gather(1, labels[:, None]).squeeze(1)
    competitors = scores.clone()
    competitors.scatter_(1, labels[:, None], float("-inf"))
    competitor_indices = competitors.argmax(dim=1)
    normals = prototypes.index_select(0, labels) - prototypes.index_select(0, competitor_indices)
    boundary_distance = (true_scores - competitors.max(dim=1).values).abs()
    boundary_distance = boundary_distance / torch.linalg.vector_norm(normals, dim=1).clamp_min(1.0e-8)
    relative_moves = moves / boundary_distance.clamp_min(1.0e-8)
    return {
        "q50_feature_move": float(torch.quantile(moves, 0.50)),
        "q90_feature_move": float(torch.quantile(moves, 0.90)),
        "max_feature_move": float(moves.max()),
        "q90_relative_move": float(torch.quantile(relative_moves, 0.90)),
    }


def _fold_gain_statistics(fold_risk_gains: Sequence[float], *, lcb_z: float) -> dict[str, Any]:
    gains = torch.tensor(tuple(float(value) for value in fold_risk_gains), dtype=torch.float64)
    if gains.numel() == 0:
        return {
            "fold_risk_gains": [],
            "positive_fold_count": 0,
            "fold_gain_mean": 0.0,
            "fold_gain_std": 0.0,
            "fold_gain_lcb90": float("-inf"),
        }
    if not bool(torch.isfinite(gains).all()):
        raise ValueError("fold risk gains must be finite")
    mean = float(gains.mean())
    std = float(gains.std(unbiased=gains.numel() > 1))
    lcb = mean - float(lcb_z) * std / math.sqrt(int(gains.numel()))
    return {
        "fold_risk_gains": [float(value) for value in gains.tolist()],
        "positive_fold_count": int((gains > 0.0).sum()),
        "fold_gain_mean": mean,
        "fold_gain_std": std,
        "fold_gain_lcb90": float(lcb),
    }


def support_state_diagnostics(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    fitted_state: SlowFastAdapterState,
    *,
    nominal_lambda: float,
    policy: SupportTrustPolicy,
    fold_risk_gains: Sequence[float] = (),
) -> dict[str, Any]:
    """Measure one fitted state using support only and normalize its intensity."""

    counts = torch.bincount(labels.long()) if labels.ndim == 1 and labels.numel() else torch.tensor([])
    if counts.numel() == 0 or not bool((counts == counts[0]).all()):
        raise ValueError("support diagnostics require balanced nonempty labels")
    features, labels, prototypes = _validate_support(
        features, labels, prototypes, k_shot=int(counts[0])
    )
    nominal = float(nominal_lambda)
    if not math.isfinite(nominal) or nominal < 0.0 or nominal > 1.0:
        raise ValueError("nominal_lambda must lie in [0, 1]")
    baseline = F.normalize(features, dim=1)
    full_strength = apply_slow_fast(features, _scaled(fitted_state, 1.0))
    full_q90 = _movement_statistics(
        full_strength, baseline, labels, prototypes
    )["q90_feature_move"]
    normalization = min(1.0, float(policy.q90_move) / (full_q90 + 1.0e-8))
    effective = nominal * normalization
    adapted = baseline if effective == 0.0 else apply_slow_fast(
        features, _scaled(fitted_state, effective)
    )
    movement = _movement_statistics(adapted, baseline, labels, prototypes)
    fold_stats = _fold_gain_statistics(fold_risk_gains, lcb_z=policy.lcb_z)
    trust_pass = (
        movement["q90_feature_move"] <= float(policy.q90_move) + 1.0e-8
        and movement["max_feature_move"] <= float(policy.hard_move) + 1.0e-8
        and movement["q90_relative_move"] <= float(policy.q90_relative_move) + 1.0e-8
    )
    stability_pass = (
        fold_stats["positive_fold_count"] >= int(policy.minimum_positive_folds)
        and (not policy.require_fold_lcb or fold_stats["fold_gain_lcb90"] > 0.0)
    )
    return {
        "nominal_lambda": nominal,
        "support_strength_normalizer": float(normalization),
        "effective_lambda": float(effective),
        "full_strength_q90_feature_move": float(full_q90),
        **movement,
        **fold_stats,
        "trust_pass": bool(trust_pass),
        "fold_stability_pass": bool(stability_pass),
    }


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


def evaluate_frozen_support_state(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    state: SlowFastAdapterState,
    *,
    logit_scale: float,
) -> dict[str, Any]:
    """Describe a frozen state on full support; this is diagnostic, not a selector."""

    counts = torch.bincount(labels.long()) if labels.ndim == 1 and labels.numel() else torch.tensor([])
    if counts.numel() == 0 or not bool((counts == counts[0]).all()):
        raise ValueError("frozen support diagnostics require balanced nonempty labels")
    features, labels, prototypes = _validate_support(
        features, labels, prototypes, k_shot=int(counts[0])
    )
    baseline_features = F.normalize(features, dim=1)
    adapted_features = baseline_features if float(state.rho) == 0.0 else apply_slow_fast(features, state)
    baseline = _trace_metrics(
        baseline_features, baseline_features, labels, prototypes, logit_scale=float(logit_scale)
    )
    adapted = _trace_metrics(
        adapted_features, baseline_features, labels, prototypes, logit_scale=float(logit_scale)
    )
    baseline_risk = baseline["macro_ce"] + 0.3 * baseline["class_cvar_ce"]
    adapted_risk = (
        adapted["macro_ce"]
        + 0.3 * adapted["class_cvar_ce"]
        + 0.1 * adapted["mean_feature_move"]
    )
    return {
        "diagnostic_role": "full_support_diagnostic_only",
        "query_opened": False,
        "risk": float(adapted_risk),
        "risk_gain": float(baseline_risk - adapted_risk),
        **_movement_statistics(adapted_features, baseline_features, labels, prototypes),
        "macro_accuracy": adapted["macro_accuracy"],
        "floor_accuracy": adapted["floor_accuracy"],
        "mean_margin": adapted["mean_margin"],
    }


def choose_crossfit_lambda(trace: Sequence[Mapping[str, Any]]) -> float:
    eligible = [row for row in trace if float(row["lambda"]) > 0.0 and bool(row.get("eligible", False))]
    if not eligible:
        return 0.0
    return float(min(eligible, key=lambda row: (float(row["risk"]), float(row["lambda"])))["lambda"])


def _crossfit_features(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    initial_state: SlowFastAdapterState,
    strengths: Mapping[float, float] | Iterable[float],
    *,
    k_shot: int,
    steps: int,
    step_size: float,
    logit_scale: float,
    seed: int,
    repeats: int,
    physical_ids: Sequence[Any] | None = None,
) -> tuple[dict[float, Tensor], Tensor, int, tuple[tuple[dict[float, Tensor], Tensor], ...]]:
    strength_map = (
        {float(key): float(value) for key, value in strengths.items()}
        if isinstance(strengths, Mapping)
        else {float(value): float(value) for value in strengths}
    )
    collected = {nominal: [] for nominal in strength_map}
    validation_labels: list[Tensor] = []
    fold_records: list[tuple[dict[float, Tensor], Tensor]] = []
    splits = _stratified_crossfit_splits(
        labels,
        k_shot=k_shot,
        seed=seed,
        repeats=repeats,
        physical_ids=physical_ids,
    )
    for train, validation in splits:
        fitted = _fit(features.index_select(0, train), labels.index_select(0, train), prototypes, initial_state, steps=steps, step_size=step_size, logit_scale=logit_scale)
        rows = features.index_select(0, validation)
        fold_features: dict[float, Tensor] = {}
        for nominal, effective in strength_map.items():
            candidate = _disabled(initial_state) if effective == 0.0 else _scaled(fitted, effective)
            adapted = F.normalize(rows, dim=1) if effective == 0.0 else apply_slow_fast(rows, candidate)
            collected[nominal].append(adapted)
            fold_features[nominal] = adapted
        fold_labels = labels.index_select(0, validation)
        validation_labels.append(fold_labels)
        fold_records.append((fold_features, fold_labels))
    return (
        {key: torch.cat(value, dim=0) for key, value in collected.items()},
        torch.cat(validation_labels, dim=0),
        len(splits),
        tuple(fold_records),
    )


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


def select_support_only_state(features: Tensor, labels: Tensor, prototypes: Tensor, initial_state: SlowFastAdapterState, *, k_shot: int, logit_scale: float, trust_radius: float, steps: int = 3, step_size: float = 0.02, lambda_grid: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0), crossfit_seed: int = 392002, repeats: int = 3, macro_tolerance: float = 0.02, floor_tolerance: float = 0.10, minimum_risk_gain: float = 1.0e-4, physical_ids: Sequence[Any] | None = None, trust_policy: SupportTrustPolicy | None = None) -> tuple[SlowFastAdapterState, dict[str, Any]]:
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
        return baseline_state, {"selected_lambda": 0.0, "reason": "K1_NO_INDEPENDENT_LOO_FALLBACK_DA0", "selection_protocol": "repeated_stratified_2fold", "gradient_updates": 0, "attempted_gradient_updates": 0, "committed_gradient_updates": 0, "crossfit_fit_count": 0, "loo_fit_count": 0, "crossfit_updates": 0, "deployment_candidate_updates": 0, "support_logit_scale": scale, "trust_radius": radius, "lambda_trace": trace, **{f"baseline_{key}": value for key, value in baseline.items() if key != "per_class_ce"}, **{f"selected_{key}": value for key, value in baseline.items() if key != "per_class_ce"}}
    full_fitted = _fit(
        features,
        labels,
        prototypes,
        initial_state,
        steps=steps,
        step_size=step_size,
        logit_scale=scale,
    )
    strength_normalizer = 1.0
    if trust_policy is not None:
        strength_normalizer = support_state_diagnostics(
            features,
            labels,
            prototypes,
            full_fitted,
            nominal_lambda=1.0,
            policy=trust_policy,
        )["support_strength_normalizer"]
    effective_strengths = {
        strength: float(strength) * float(strength_normalizer) for strength in strengths
    }
    crossfit, crossfit_labels, fit_count, fold_records = _crossfit_features(
        features,
        labels,
        prototypes,
        initial_state,
        effective_strengths,
        k_shot=k_shot,
        steps=steps,
        step_size=step_size,
        logit_scale=scale,
        seed=crossfit_seed,
        repeats=repeats,
        physical_ids=physical_ids,
    )
    baseline = _trace_metrics(crossfit[0.0], crossfit[0.0], crossfit_labels, prototypes, logit_scale=scale)
    baseline_risk = baseline["macro_ce"] + 0.3 * baseline["class_cvar_ce"]
    trace: list[dict[str, Any]] = []
    for strength in strengths:
        metrics = _trace_metrics(crossfit[strength], crossfit[0.0], crossfit_labels, prototypes, logit_scale=scale)
        movement = _movement_statistics(
            crossfit[strength], crossfit[0.0], crossfit_labels, prototypes
        )
        risk = metrics["macro_ce"] + 0.3 * metrics["class_cvar_ce"] + 0.1 * metrics["mean_feature_move"]
        fold_risk_gains: list[float] = []
        for fold_features, fold_labels in fold_records:
            fold_baseline = _trace_metrics(
                fold_features[0.0],
                fold_features[0.0],
                fold_labels,
                prototypes,
                logit_scale=scale,
            )
            fold_metrics = _trace_metrics(
                fold_features[strength],
                fold_features[0.0],
                fold_labels,
                prototypes,
                logit_scale=scale,
            )
            fold_baseline_risk = fold_baseline["macro_ce"] + 0.3 * fold_baseline["class_cvar_ce"]
            fold_risk = (
                fold_metrics["macro_ce"]
                + 0.3 * fold_metrics["class_cvar_ce"]
                + 0.1 * fold_metrics["mean_feature_move"]
            )
            fold_risk_gains.append(float(fold_baseline_risk - fold_risk))
        fold_stats = _fold_gain_statistics(
            fold_risk_gains,
            lcb_z=trust_policy.lcb_z if trust_policy is not None else 1.2815515655446004,
        )
        reasons: list[str] = []
        if strength > 0.0:
            if metrics["macro_accuracy"] < baseline["macro_accuracy"] - float(macro_tolerance): reasons.append("MACRO_TOLERANCE")
            if metrics["floor_accuracy"] < baseline["floor_accuracy"] - float(floor_tolerance): reasons.append("FLOOR_TOLERANCE")
            if trust_policy is None:
                if metrics["max_feature_move"] > radius: reasons.append("TRUST_RADIUS")
            else:
                if movement["q90_feature_move"] > float(trust_policy.q90_move) + 1.0e-8: reasons.append("Q90_MOVE")
                if movement["max_feature_move"] > float(trust_policy.hard_move) + 1.0e-8: reasons.append("HARD_MOVE")
                if movement["q90_relative_move"] > float(trust_policy.q90_relative_move) + 1.0e-8: reasons.append("RELATIVE_MOVE")
                if fold_stats["positive_fold_count"] < int(trust_policy.minimum_positive_folds): reasons.append("POSITIVE_FOLD_COUNT")
                if trust_policy.require_fold_lcb and fold_stats["fold_gain_lcb90"] <= 0.0: reasons.append("FOLD_GAIN_LCB90")
            if baseline_risk - risk < float(minimum_risk_gain): reasons.append("INSUFFICIENT_RISK_GAIN")
        trace.append({"lambda": strength, "effective_lambda": effective_strengths[strength], **metrics, **movement, **fold_stats, "risk": float(risk), "risk_gain": float(baseline_risk - risk), "eligible": strength == 0.0 or not reasons, "rejection_reasons": reasons, "per_class_ce_delta": [float(candidate - base) for candidate, base in zip(metrics["per_class_ce"], baseline["per_class_ce"])]})
    selected_strength = choose_crossfit_lambda(trace)
    selected_effective_strength = effective_strengths[selected_strength]
    selected = baseline_state if selected_effective_strength == 0.0 else _scaled(
        full_fitted, selected_effective_strength
    )
    selected_trace = next(row for row in trace if row["lambda"] == selected_strength)
    committed = int(steps) if selected_strength > 0.0 and initial_state.candidate is not SlowFastCandidate.COMMON_SHIFT_R4 else 0
    attempted = (fit_count + 1) * int(steps) if initial_state.candidate is not SlowFastCandidate.COMMON_SHIFT_R4 else 0
    crossfit_updates = fit_count * int(steps) if initial_state.candidate is not SlowFastCandidate.COMMON_SHIFT_R4 else 0
    deployment_updates = int(steps) if initial_state.candidate is not SlowFastCandidate.COMMON_SHIFT_R4 else 0
    return selected, {"selected_lambda": selected_strength, "selected_effective_lambda": selected_effective_strength, "support_strength_normalizer": float(strength_normalizer), "reason": "SUPPORT_CROSSFIT_CONTINUOUS_RISK_PASS" if selected_strength > 0.0 else "SUPPORT_CROSSFIT_FALLBACK_DA0", "selection_protocol": "repeated_stratified_2fold", "gradient_updates": committed, "attempted_gradient_updates": attempted, "committed_gradient_updates": committed, "crossfit_fit_count": fit_count, "loo_fit_count": 0, "crossfit_updates": crossfit_updates, "deployment_candidate_updates": deployment_updates, "support_logit_scale": scale, "trust_radius": radius, "lambda_trace": trace, **{f"baseline_{key}": value for key, value in baseline.items() if key != "per_class_ce"}, **{f"selected_{key}": value for key, value in selected_trace.items() if key not in {"lambda", "per_class_ce", "per_class_ce_delta", "eligible", "rejection_reasons"}}, "max_per_class_loss_increase": max(selected_trace["per_class_ce_delta"])}


__all__ = ["SupportTrustPolicy", "choose_crossfit_lambda", "evaluate_frozen_support_state", "fit_support_candidate_states", "select_support_only_state", "select_support_only_state_legacy", "support_state_diagnostics"]
