from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(device=values.device, dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp_min(1e-12)


def angular_sensitivity(base: torch.Tensor, perturbed: torch.Tensor, *, delta: float) -> torch.Tensor:
    delta = float(delta)
    if delta <= 0.0:
        raise ValueError("finite-difference delta must be positive")
    if base.shape != perturbed.shape:
        raise ValueError("base and perturbed features must have identical shape")
    cosine = (F.normalize(base.float(), dim=-1) * F.normalize(perturbed.float(), dim=-1)).sum(dim=-1)
    angle = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    return angle / delta


def selective_tangent_loss(
    sensitivity: torch.Tensor,
    *,
    budgets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    excess = (sensitivity.float() - budgets.to(device=sensitivity.device, dtype=torch.float32)).clamp_min(0.0)
    return _masked_mean(excess.square(), valid)


def fingerprint_keep_loss(
    fingerprint_sensitivity: torch.Tensor,
    *,
    minimum: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    deficit = (minimum.to(device=fingerprint_sensitivity.device, dtype=torch.float32) - fingerprint_sensitivity.float()).clamp_min(0.0)
    if valid is None:
        return deficit.square().mean()
    return _masked_mean(deficit.square(), valid)


def heteroscedastic_nuisance_loss(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    *,
    valid: torch.Tensor,
) -> torch.Tensor:
    if mean.shape != log_variance.shape or mean.shape != target.shape:
        raise ValueError("nuisance mean, log variance, and target must align")
    bounded_log_variance = log_variance.float().clamp(-8.0, 8.0)
    per_dimension = (target.float() - mean.float()).square() * torch.exp(-bounded_log_variance) + bounded_log_variance
    per_row = per_dimension.mean(dim=-1)
    return _masked_mean(per_row, valid)


def fingerprint_selectivity(
    fingerprint_sensitivity: torch.Tensor,
    nuisance_sensitivity: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    return fingerprint_sensitivity.float() / (nuisance_sensitivity.float() + float(eps))


def gradient_norm_ratio(
    auxiliary_loss: torch.Tensor,
    base_loss: torch.Tensor,
    parameters,
    *,
    eps: float = 1e-12,
) -> float:
    """Measure an auxiliary/base gradient ratio on one shared parameter set."""

    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not parameters or not auxiliary_loss.requires_grad or not base_loss.requires_grad:
        return float("nan")
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    base_gradients = torch.autograd.grad(
        base_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    def norm(gradients) -> float:
        total = sum(
            float(gradient.detach().float().square().sum().cpu().item())
            for gradient in gradients
            if gradient is not None
        )
        return total**0.5

    return norm(auxiliary_gradients) / max(norm(base_gradients), float(eps))


def worst_channel_bucket_accuracy(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    channel_values: torch.Tensor,
    bins: int = 4,
) -> dict[str, object]:
    """Quantile-bucket accuracy with explicit finite-value coverage."""

    predictions = torch.as_tensor(predictions).reshape(-1)
    labels = torch.as_tensor(labels, device=predictions.device).reshape(-1)
    values = torch.as_tensor(channel_values, device=predictions.device, dtype=torch.float32).reshape(-1)
    if predictions.numel() != labels.numel() or predictions.numel() != values.numel():
        raise ValueError("predictions, labels, and channel_values must align")
    bins = int(bins)
    if bins < 1:
        raise ValueError("bins must be positive")
    finite = torch.isfinite(values)
    valid_count = int(finite.sum().item())
    if valid_count == 0:
        return {"valid_count": 0, "bucket_count": 0, "worst_accuracy": float("nan"), "buckets": []}
    predictions = predictions[finite]
    labels = labels[finite]
    values = values[finite]
    quantiles = torch.linspace(0.0, 1.0, bins + 1, device=values.device)
    edges = torch.quantile(values, quantiles).unique(sorted=True)
    bucket_rows = []
    for index in range(max(0, int(edges.numel()) - 1)):
        lower = edges[index]
        upper = edges[index + 1]
        selected = (values >= lower) & ((values <= upper) if index == int(edges.numel()) - 2 else (values < upper))
        count = int(selected.sum().item())
        if count == 0:
            continue
        accuracy = float(predictions[selected].eq(labels[selected]).float().mean().item())
        bucket_rows.append(
            {
                "lower": float(lower.item()),
                "upper": float(upper.item()),
                "count": count,
                "accuracy": accuracy,
            }
        )
    worst = min((float(row["accuracy"]) for row in bucket_rows), default=float("nan"))
    return {
        "valid_count": valid_count,
        "bucket_count": len(bucket_rows),
        "worst_accuracy": worst,
        "buckets": bucket_rows,
    }
