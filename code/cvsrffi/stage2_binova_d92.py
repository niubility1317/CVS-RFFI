"""Differentiable D92 surrogate and exact D92 bridge for BiNOVA."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from cvsrffi.stage2_d38_strong_b3_quantized import (
    FEATURE_NOISE_STD,
    FFT_LOG_SCALE_LIMIT,
    GRAD_CLIP,
    LOG_SCALE_LIMIT,
    PROTOTYPE_ANCHOR_WEIGHT,
    STAGE2B_LR,
    TEMPERATURE,
    WEIGHT_DECAY,
)


_D92_METRIC_EPOCHS = 20
_D92_OLD_CLASS_COUNT = 6
_D92_SEED = 713102


class BiNOVAD92Error(ValueError):
    """Raised when differentiable or exact D92 geometry is invalid."""


@dataclass(frozen=True)
class DifferentiableD92State:
    class_ids: tuple[int, ...]
    old_class_count: int
    centers: torch.Tensor
    covariance: torch.Tensor
    cholesky: torch.Tensor
    coefficient: torch.Tensor
    intercept: torch.Tensor
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    def score(self, rows: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(rows, dtype=self.coefficient.dtype, device=self.coefficient.device)
        if values.ndim != 2 or values.shape[1] != self.coefficient.shape[1]:
            raise BiNOVAD92Error("D92 score feature geometry drift")
        return values @ self.coefficient.T + self.intercept


def d92_geometry_features(identity160: torch.Tensor, fft96: torch.Tensor) -> torch.Tensor:
    identity = torch.as_tensor(identity160)
    fft = torch.as_tensor(fft96, dtype=identity.dtype, device=identity.device)
    if identity.ndim != 2 or identity.shape[1] != 160:
        raise BiNOVAD92Error("identity feature geometry must be [N,160]")
    if fft.shape != (len(identity), 96):
        raise BiNOVAD92Error("FFT feature geometry must be [N,96]")
    if not torch.isfinite(identity).all() or not torch.isfinite(fft).all():
        raise BiNOVAD92Error("D92 input features must be finite")
    both_modalities_zero = (identity.square().sum(dim=1) == 0.0) & (
        fft.square().sum(dim=1) == 0.0
    )
    if bool(both_modalities_zero.any()):
        raise BiNOVAD92Error("both modalities are zero for a D92 feature row")
    joined = torch.cat(
        [F.normalize(identity, dim=1), 4.0 * F.normalize(fft, dim=1)], dim=1
    )
    return F.normalize(joined, dim=1)


def _class_centers(rows: torch.Tensor, labels: torch.Tensor, class_count: int) -> torch.Tensor:
    return torch.stack([rows[labels == index].mean(dim=0) for index in range(class_count)])


def _oas_group_covariance(
    rows: torch.Tensor,
    labels: torch.Tensor,
    class_indices: torch.Tensor,
    *,
    shrinkage_override: float | None,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.isin(labels, class_indices)
    group_rows = rows[mask]
    group_labels = labels[mask]
    residual_parts = []
    for class_index in class_indices:
        selected = group_rows[group_labels == class_index]
        residual_parts.append(selected - selected.mean(dim=0, keepdim=True))
    residuals = torch.cat(residual_parts, dim=0)
    dimension = rows.shape[1]
    covariance = residuals.T @ residuals / float(max(len(residuals), 1))
    covariance = 0.5 * (covariance + covariance.T)
    trace = torch.trace(covariance)
    mean_variance = trace / float(dimension)
    if shrinkage_override is None:
        alpha = torch.sum(covariance.square())
        trace_square = trace.square()
        numerator = (1.0 - 2.0 / float(dimension)) * alpha + trace_square
        denominator = (
            float(len(residuals) + 1) - 2.0 / float(dimension)
        ) * (alpha - trace_square / float(dimension))
        ratio = numerator / denominator.clamp_min(torch.finfo(rows.dtype).eps)
        shrinkage = torch.where(
            denominator > torch.finfo(rows.dtype).eps,
            ratio.clamp(0.0, 1.0),
            torch.ones_like(ratio),
        )
    else:
        value = float(shrinkage_override)
        if value < 0.0 or value > 1.0:
            raise BiNOVAD92Error("shrinkage_override must be in [0,1]")
        shrinkage = rows.new_tensor(value)
    identity = torch.eye(dimension, dtype=rows.dtype, device=rows.device)
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * mean_variance * identity
    shrunk = 0.5 * (shrunk + shrunk.T) + float(jitter) * identity
    return shrunk, shrinkage


def fit_differentiable_d92(
    rows: torch.Tensor,
    labels: torch.Tensor,
    *,
    old_class_count: int,
    shrinkage_override: float | None = None,
    jitter: float = 1.0e-5,
) -> DifferentiableD92State:
    values = torch.as_tensor(rows)
    targets = torch.as_tensor(labels, dtype=torch.long, device=values.device)
    if (
        values.ndim != 2
        or len(values) < 1
        or targets.shape != (len(values),)
        or not torch.is_floating_point(values)
        or not torch.isfinite(values).all()
        or jitter <= 0.0
    ):
        raise BiNOVAD92Error("D92 fit rows/labels are invalid")
    classes = torch.unique(targets, sorted=True)
    class_count = len(classes)
    if not torch.equal(classes, torch.arange(class_count, device=values.device)):
        raise BiNOVAD92Error("D92 labels must be a contiguous zero-based registry")
    old_count = int(old_class_count)
    if old_count < 1 or old_count > class_count:
        raise BiNOVAD92Error("old_class_count is outside the registry")
    if any(int((targets == index).sum()) < 1 for index in range(class_count)):
        raise BiNOVAD92Error("every D92 class needs support")
    centers = _class_centers(values, targets, class_count)
    old_indices = torch.arange(old_count, device=values.device)
    old_covariance, old_shrinkage = _oas_group_covariance(
        values,
        targets,
        old_indices,
        shrinkage_override=shrinkage_override,
        jitter=jitter,
    )
    if class_count > old_count:
        new_indices = torch.arange(old_count, class_count, device=values.device)
        new_covariance, new_shrinkage = _oas_group_covariance(
            values,
            targets,
            new_indices,
            shrinkage_override=shrinkage_override,
            jitter=jitter,
        )
        covariance = 0.5 * old_covariance + 0.5 * new_covariance
        old_weight, new_weight = 0.5, 0.5
    else:
        new_covariance = None
        new_shrinkage = None
        covariance = old_covariance
        old_weight, new_weight = 1.0, 0.0
    covariance = 0.5 * (covariance + covariance.T)
    cholesky = torch.linalg.cholesky(covariance)
    coefficient = torch.cholesky_solve(centers.T, cholesky).T
    prior = values.new_full((class_count,), 1.0 / float(class_count))
    intercept = -0.5 * torch.sum(centers * coefficient, dim=1) + torch.log(prior)
    return DifferentiableD92State(
        class_ids=tuple(range(class_count)),
        old_class_count=old_count,
        centers=centers,
        covariance=covariance,
        cholesky=cholesky,
        coefficient=coefficient,
        intercept=intercept,
        audit={
            "solver": "torch_cholesky",
            "shrinkage": "oas_analytic" if shrinkage_override is None else "fixed_test_override",
            "old_covariance_weight": old_weight,
            "new_covariance_weight": new_weight,
            "old_shrinkage": float(old_shrinkage.detach().cpu()),
            "new_shrinkage": (
                None if new_shrinkage is None else float(new_shrinkage.detach().cpu())
            ),
            "query_rows_used": 0,
        },
    )


def _d92_exact_metric(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Unroll the locked old-only B3 metric update without detaching support rows."""

    values = rows.to(dtype=torch.float32)
    targets = labels.to(dtype=torch.long, device=values.device)
    prototypes = torch.stack(
        [F.normalize(values[targets == index].mean(dim=0), dim=0) for index in range(6)]
    )
    log_diag = torch.zeros(256, dtype=values.dtype, device=values.device, requires_grad=True)
    weights = prototypes.clone().requires_grad_(True)
    log_m = torch.zeros_like(log_diag)
    log_v = torch.zeros_like(log_diag)
    weight_m = torch.zeros_like(weights)
    weight_v = torch.zeros_like(weights)
    lower = values.new_full((256,), -LOG_SCALE_LIMIT)
    upper = values.new_full((256,), LOG_SCALE_LIMIT)
    lower[160:] = -FFT_LOG_SCALE_LIMIT
    upper[160:] = FFT_LOG_SCALE_LIMIT
    generator = torch.Generator(device=values.device).manual_seed(_D92_SEED)
    beta1, beta2 = 0.9, 0.999
    for step in range(1, _D92_METRIC_EPOCHS + 1):
        noisy = values + FEATURE_NOISE_STD * torch.randn(
            values.shape, generator=generator, device=values.device, dtype=values.dtype
        )
        effective = torch.clamp(log_diag, min=lower, max=upper)
        logits = TEMPERATURE * (F.normalize(noisy * effective.exp(), dim=1) @ F.normalize(weights, dim=1).T)
        loss = F.cross_entropy(logits, targets) + PROTOTYPE_ANCHOR_WEIGHT * torch.mean(
            (F.normalize(weights, dim=1) - prototypes) ** 2
        )
        log_grad, weight_grad = torch.autograd.grad(
            loss,
            (log_diag, weights),
            create_graph=bool(rows.requires_grad),
        )
        norm = torch.sqrt(log_grad.square().sum() + weight_grad.square().sum())
        scale = (GRAD_CLIP / (norm + 1.0e-6)).clamp(max=1.0)
        log_grad, weight_grad = log_grad * scale, weight_grad * scale
        log_m = beta1 * log_m + (1.0 - beta1) * log_grad
        log_v = beta2 * log_v + (1.0 - beta2) * log_grad.square()
        weight_m = beta1 * weight_m + (1.0 - beta1) * weight_grad
        weight_v = beta2 * weight_v + (1.0 - beta2) * weight_grad.square()
        log_diag = log_diag * (1.0 - STAGE2B_LR * WEIGHT_DECAY) - STAGE2B_LR * (
            log_m / (1.0 - beta1**step)
        ) / (log_v / (1.0 - beta2**step)).sqrt().add(1.0e-8)
        weights = weights * (1.0 - STAGE2B_LR * WEIGHT_DECAY) - STAGE2B_LR * (
            weight_m / (1.0 - beta1**step)
        ) / (weight_v / (1.0 - beta2**step)).sqrt().add(1.0e-8)
        log_diag = torch.clamp(log_diag, min=lower, max=upper)
    return log_diag


def _d92_ledoit_wolf_covariance(rows: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of sklearn's StandardScaler plus Ledoit-Wolf auto covariance."""

    values = rows.to(dtype=torch.float64)
    count, dimension = values.shape
    centered = values - values.mean(dim=0, keepdim=True)
    scale = centered.square().mean(dim=0).sqrt()
    scale = torch.where(scale > torch.finfo(values.dtype).eps, scale, torch.ones_like(scale))
    standardized = centered / scale
    empirical = standardized.T @ standardized / float(count)
    diagonal = standardized.square().sum(dim=0) / float(count)
    mean_variance = diagonal.mean()
    beta_raw = standardized.square().T @ standardized.square()
    beta_raw = beta_raw.sum()
    gram = standardized.T @ standardized
    delta_raw = gram.square().sum() / float(count * count)
    beta = (beta_raw / float(count) - delta_raw) / float(dimension * count)
    delta = (delta_raw - 2.0 * mean_variance * diagonal.sum() + dimension * mean_variance.square()) / float(dimension)
    beta = torch.minimum(beta, delta)
    shrinkage = torch.where(delta == 0.0, torch.zeros_like(delta), beta / delta)
    covariance = (1.0 - shrinkage) * empirical
    covariance = covariance + shrinkage * mean_variance * torch.eye(
        dimension, dtype=values.dtype, device=values.device
    )
    return covariance * scale[:, None] * scale[None, :]


def _d92_means(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.stack([rows[labels == index].mean(dim=0) for index in range(6)])


def _d92_component(rows: torch.Tensor, labels: torch.Tensor, *, block: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Locked full or block-auto-shrinkage affine component in the exact score gauge."""

    values = rows.to(dtype=torch.float64)
    targets = labels.to(dtype=torch.long, device=values.device)
    means = _d92_means(values, targets)
    covariance = sum(
        _d92_ledoit_wolf_covariance(values[targets == index]) for index in range(6)
    ) / 6.0
    if block:
        mask = torch.zeros_like(covariance)
        mask[:160, :160] = 1.0
        mask[160:, 160:] = 1.0
        covariance = covariance * mask
    coefficients = torch.linalg.solve(covariance, means.T).T
    intercept = -0.5 * torch.sum(means * coefficients, dim=1) - torch.log(values.new_tensor(6.0))
    coefficients = coefficients - coefficients.mean(dim=0, keepdim=True)
    intercept = intercept - intercept.mean()
    return coefficients.to(dtype=torch.float32), intercept.to(dtype=torch.float32)


def _d92_canonical_component(rows: torch.Tensor, labels: torch.Tensor, *, block: bool) -> tuple[torch.Tensor, torch.Tensor]:
    coefficient, intercept = _d92_component(rows, labels, block=block)
    coefficient = coefficient.to(dtype=torch.float64)
    intercept = intercept.to(dtype=torch.float64)
    return coefficient - coefficient.mean(dim=0, keepdim=True), intercept - intercept.mean()


def _d92_rms(rows: torch.Tensor, coefficient: torch.Tensor, intercept: torch.Tensor) -> torch.Tensor:
    scores = rows.to(dtype=torch.float64) @ coefficient.T + intercept
    return (scores - scores.mean(dim=1, keepdim=True)).square().mean().sqrt()


def _d92_classwise_ce(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = scores.reshape(-1, 6)
    targets = labels.reshape(-1).to(dtype=torch.long, device=values.device)
    losses = F.cross_entropy(values, targets, reduction="none")
    return torch.stack([losses[targets == index].mean() for index in range(6)])


def _d92_loo_ce(rows: torch.Tensor, labels: torch.Tensor, *, block: bool) -> torch.Tensor:
    targets = labels.to(dtype=torch.long, device=rows.device)
    indices = [torch.nonzero(targets == index, as_tuple=False).flatten() for index in range(6)]
    count = len(indices[0])
    fold_scores, fold_labels = [], []
    for rank in range(count):
        held = torch.stack([index[rank] for index in indices])
        train_mask = torch.ones(len(rows), dtype=torch.bool, device=rows.device)
        train_mask[held] = False
        coefficient, intercept = _d92_canonical_component(rows[train_mask], targets[train_mask], block=block)
        scale = _d92_rms(rows[train_mask], coefficient, intercept)
        fold_scores.append((rows[held].to(dtype=torch.float64) @ coefficient.T + intercept) / scale)
        fold_labels.append(targets[held])
    return _d92_classwise_ce(torch.cat(fold_scores), torch.cat(fold_labels))


def _d92_reliability_weights(full_ce: torch.Tensor, block_ce: torch.Tensor, k_shot: int) -> torch.Tensor:
    return torch.softmax(-float(k_shot) * torch.stack([full_ce, block_ce], dim=1), dim=1)


def _d92_d46_affine(rows: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = labels.to(dtype=torch.long, device=rows.device)
    k_shot = int((targets == 0).sum())
    full_coef, full_intercept = _d92_canonical_component(rows, targets, block=False)
    block_coef, block_intercept = _d92_canonical_component(rows, targets, block=True)
    full_weight, block_weight = _d92_reliability_weights(
        _d92_loo_ce(rows, targets, block=False), _d92_loo_ce(rows, targets, block=True), k_shot
    ).unbind(dim=1)
    coefficient = full_weight[:, None] * full_coef / _d92_rms(rows, full_coef, full_intercept) + block_weight[:, None] * block_coef / _d92_rms(rows, block_coef, block_intercept)
    intercept = full_weight * full_intercept / _d92_rms(rows, full_coef, full_intercept) + block_weight * block_intercept / _d92_rms(rows, block_coef, block_intercept)
    coefficient, intercept = coefficient - coefficient.mean(dim=0, keepdim=True), intercept - intercept.mean()
    return coefficient.to(dtype=torch.float32), intercept.to(dtype=torch.float32), torch.stack([full_weight, block_weight], dim=1)


def _d92_fisher_transform(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = rows.to(dtype=torch.float64)
    targets = labels.to(dtype=torch.long, device=values.device)
    means = _d92_means(values, targets)
    centered = means - means.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * torch.finfo(values.dtype).eps * singular_values[0]
    rank = int((singular_values.detach() > tolerance.detach()).sum())
    if rank == 0:
        return torch.eye(values.shape[1], dtype=values.dtype, device=values.device)
    basis = vh[:rank].T
    between = (centered @ basis).square().mean(dim=0)
    within = ((values - means[targets]) @ basis).square().mean(dim=0)
    gain = between / (between + within).clamp_min(torch.finfo(values.dtype).tiny)
    gain = gain.clamp(0.0, 1.0)
    transform = torch.eye(values.shape[1], dtype=values.dtype, device=values.device) + (basis * gain) @ basis.T
    return 0.5 * (transform + transform.T)


def _d92_component_evidence(rows: torch.Tensor, labels: torch.Tensor, *, block: bool) -> dict[str, torch.Tensor]:
    targets = labels.to(dtype=torch.long, device=rows.device)
    outer_coef, outer_intercept = _d92_canonical_component(rows, targets, block=block)
    outer_residual = outer_coef @ _d92_fisher_transform(rows, targets).T
    outer_base_scale = _d92_rms(rows, outer_coef, outer_intercept)
    outer_residual_scale = _d92_rms(rows, outer_residual, outer_intercept)
    indices = [torch.nonzero(targets == index, as_tuple=False).flatten() for index in range(6)]
    base_scores, residual_scores, truth = [], [], []
    for rank in range(len(indices[0])):
        held = torch.stack([index[rank] for index in indices])
        train_mask = torch.ones(len(rows), dtype=torch.bool, device=rows.device)
        train_mask[held] = False
        coefficient, intercept = _d92_canonical_component(rows[train_mask], targets[train_mask], block=block)
        residual = coefficient @ _d92_fisher_transform(rows[train_mask], targets[train_mask]).T
        base_scores.append((rows[held].to(dtype=torch.float64) @ coefficient.T + intercept) / _d92_rms(rows[train_mask], coefficient, intercept))
        residual_scores.append((rows[held].to(dtype=torch.float64) @ residual.T + intercept) / _d92_rms(rows[train_mask], residual, intercept))
        truth.append(targets[held])
    return {
        "base_coefficient": outer_coef,
        "base_intercept": outer_intercept,
        "residual_coefficient": outer_residual,
        "base_scale": outer_base_scale,
        "residual_scale": outer_residual_scale,
        "base_ce": _d92_classwise_ce(torch.cat(base_scores), torch.cat(truth)),
        "residual_ce": _d92_classwise_ce(torch.cat(residual_scores), torch.cat(truth)),
        "base_scores": torch.stack(base_scores),
        "residual_scores": torch.stack(residual_scores),
        "truth": torch.stack(truth),
    }


def _d92_gate(base_scores: torch.Tensor, residual_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = labels.detach().to(dtype=torch.long, device=base_scores.device)
    base_prediction = base_scores.detach().argmax(dim=2)
    class_count = base_scores.shape[2]
    base_positive = torch.stack([((base_prediction == index) & (targets == index)).sum() for index in range(class_count)])
    base_false_positive = torch.stack([((base_prediction == index) & (targets != index)).sum() for index in range(class_count)])
    chosen = []
    for index in range(class_count):
        hybrid = base_scores.detach().clone()
        hybrid[:, :, index] = residual_scores.detach()[:, :, index]
        prediction = hybrid.argmax(dim=2)
        positive = ((prediction == index) & (targets == index)).sum()
        false_positive = ((prediction == index) & (targets != index)).sum()
        chosen.append((positive >= base_positive[index]) & (false_positive <= base_false_positive[index]) & ((positive > base_positive[index]) | (false_positive < base_false_positive[index])))
    initial = torch.stack(chosen)
    joint = base_scores.detach().clone()
    joint[:, :, initial] = residual_scores.detach()[:, :, initial]
    prediction = joint.argmax(dim=2)
    positive = torch.stack([((prediction == index) & (targets == index)).sum() for index in range(class_count)])
    false_positive = torch.stack([((prediction == index) & (targets != index)).sum() for index in range(class_count)])
    return initial if bool(((positive >= base_positive) & (false_positive <= base_false_positive)).all()) else torch.zeros_like(initial)


def _d92_exact_old_affine(rows: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_coefficient, base_intercept, base_weights = _d92_d46_affine(rows, labels)
    full = _d92_component_evidence(rows, labels, block=False)
    block = _d92_component_evidence(rows, labels, block=True)
    residual_weights = _d92_reliability_weights(full["residual_ce"], block["residual_ce"], int((labels == 0).sum()))
    base_scores = full["base_scores"] * base_weights[:, 0][None, None, :] + block["base_scores"] * base_weights[:, 1][None, None, :]
    residual_scores = full["residual_scores"] * residual_weights[:, 0][None, None, :] + block["residual_scores"] * residual_weights[:, 1][None, None, :]
    selected = _d92_gate(base_scores, residual_scores, full["truth"])
    if not bool(selected.any()):
        return base_coefficient, base_intercept
    residual_coefficient = residual_weights[:, 0, None] * full["residual_coefficient"] / full["residual_scale"] + residual_weights[:, 1, None] * block["residual_coefficient"] / block["residual_scale"]
    residual_intercept = residual_weights[:, 0] * full["base_intercept"] / full["residual_scale"] + residual_weights[:, 1] * block["base_intercept"] / block["residual_scale"]
    coefficient = torch.where(selected[:, None], residual_coefficient, base_coefficient.to(dtype=torch.float64))
    intercept = torch.where(selected, residual_intercept, base_intercept.to(dtype=torch.float64))
    return (coefficient - coefficient.mean(dim=0, keepdim=True)).to(dtype=torch.float32), (intercept - intercept.mean()).to(dtype=torch.float32)


def differentiable_old_d92_logits(
    fit_identity: torch.Tensor,
    fit_fft: torch.Tensor,
    fit_labels: torch.Tensor,
    eval_identity: torch.Tensor,
    eval_fft: torch.Tensor,
) -> torch.Tensor:
    """Compute exact old-only D92 logits and their local support/query Jacobian in one graph."""

    fit_rows = d92_geometry_features(fit_identity, fit_fft)
    eval_rows = d92_geometry_features(eval_identity, eval_fft)
    labels = torch.as_tensor(fit_labels, dtype=torch.long, device=fit_rows.device)
    if not torch.equal(torch.unique(labels, sorted=True), torch.arange(6, device=labels.device)):
        raise BiNOVAD92Error("old-only D92 requires the contiguous six-class registry")
    metric = _d92_exact_metric(fit_rows, labels)
    transformed_fit = F.normalize(fit_rows.to(dtype=torch.float32) * metric.exp(), dim=1)
    transformed_eval = F.normalize(eval_rows.to(dtype=torch.float32) * metric.exp(), dim=1)
    coefficient, intercept = _d92_exact_old_affine(transformed_fit, labels)
    return transformed_eval @ coefficient.T + intercept


def _top_two_smallest(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = torch.sort(values, dim=1).values
    first = ordered[:, 0]
    second = ordered[:, 1] if ordered.shape[1] > 1 else first
    return first, second


def d92_geometry_conditions(
    state: DifferentiableD92State,
    rows: torch.Tensor,
) -> torch.Tensor:
    values = torch.as_tensor(rows, dtype=state.centers.dtype, device=state.centers.device)
    differences = values[:, None, :] - state.centers[None, :, :]
    flat = differences.reshape(-1, differences.shape[-1]).T
    solved = torch.cholesky_solve(flat, state.cholesky).T.reshape_as(differences)
    distances = torch.sum(differences * solved, dim=2)
    old_distances = distances[:, : state.old_class_count]
    new_distances = distances[:, state.old_class_count :]
    if new_distances.shape[1] < 1:
        raise BiNOVAD92Error("geometry conditions require registered new classes")
    old_first, old_second = _top_two_smallest(old_distances)
    new_first, new_second = _top_two_smallest(new_distances)
    logits = state.score(values)
    old_top = logits[:, : state.old_class_count].max(dim=1).values
    new_top = logits[:, state.old_class_count :].max(dim=1).values
    probabilities = torch.softmax(logits, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=1)
    return torch.stack(
        [old_first, old_second, new_first, new_second, old_top - new_top, entropy],
        dim=1,
    )


def exact_d92_fit(
    identity160: Any,
    fft96: Any,
    labels: Any,
    *,
    class_ids: Sequence[int],
    old_class_count: int,
    seed: int,
    device: Any = "cpu",
) -> Any:
    """Fit the existing exact no-RF32 D92 state after BiNOVA adaptation."""

    registry = tuple(int(value) for value in class_ids)
    if len(registry) == int(old_class_count):
        from cvsrffi.stage2_sf_erbt_oldonly import fit_old_only_erbt

        return fit_old_only_erbt(
            identity160,
            fft96,
            labels,
            class_ids=registry,
            seed=int(seed),
            device=device,
        )
    from cvsrffi.stage2_sf_erbt_four_state import fit_registered_erbt

    return fit_registered_erbt(
        identity160,
        fft96,
        labels,
        class_ids=registry,
        old_class_count=int(old_class_count),
        seed=int(seed),
        device=device,
    )


__all__ = [
    "BiNOVAD92Error",
    "DifferentiableD92State",
    "d92_geometry_conditions",
    "d92_geometry_features",
    "differentiable_old_d92_logits",
    "exact_d92_fit",
    "fit_differentiable_d92",
]
