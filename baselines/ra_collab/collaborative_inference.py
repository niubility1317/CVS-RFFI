from __future__ import annotations

import torch


def _as_probs(predictions: torch.Tensor) -> torch.Tensor:
    if predictions.dim() != 2:
        raise ValueError("Fusion expects [num_receivers, num_classes].")
    row_sums = predictions.sum(dim=1)
    if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
        return predictions.float()
    return torch.softmax(predictions.float(), dim=1)


def soft_fusion(predictions: torch.Tensor) -> torch.Tensor:
    probs = _as_probs(predictions)
    fused = probs.mean(dim=0)
    return fused / fused.sum().clamp_min(1e-8)


def adaptive_soft_fusion(predictions: torch.Tensor, snr: torch.Tensor, snr_scale: str = "linear") -> torch.Tensor:
    probs = _as_probs(predictions)
    weights = snr.float().view(-1)
    if weights.numel() != probs.size(0):
        raise ValueError("snr length must match number of receiver predictions.")
    if snr_scale == "linear":
        weights = torch.clamp(weights, min=0.0)
    elif snr_scale == "db_to_linear":
        weights = torch.pow(10.0, weights / 10.0)
    elif snr_scale == "db_direct":
        weights = torch.clamp(weights - weights.min(), min=0.0)
    else:
        raise ValueError("snr_scale must be one of: linear, db_to_linear, db_direct")
    if float(weights.sum()) <= 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-8)
    fused = torch.sum(probs * weights.unsqueeze(1), dim=0)
    return fused / fused.sum().clamp_min(1e-8)
