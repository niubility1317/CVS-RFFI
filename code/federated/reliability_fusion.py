from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch


def normalize_probabilities(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return p / p.sum(dim=1, keepdim=True).clamp_min(float(eps))


def conservative_probability_fusion(
    p_base: torch.Tensor,
    p_proto: torch.Tensor,
    *,
    rho: float,
    max_rho: float = 0.05,
    reliability: float = 1.0,
) -> torch.Tensor:
    if p_base.shape != p_proto.shape:
        raise ValueError(f"p_base and p_proto must have the same shape, got {tuple(p_base.shape)} and {tuple(p_proto.shape)}")
    base = normalize_probabilities(p_base)
    proto = normalize_probabilities(p_proto)
    gate = max(0.0, min(float(max_rho), float(rho))) * max(0.0, min(1.0, float(reliability)))
    return normalize_probabilities((1.0 - gate) * base + gate * proto)


def collaborative_reliability_from_probabilities(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = normalize_probabilities(p, eps=eps)
    if probs.dim() != 2:
        raise ValueError(f"p must be a [batch, classes] tensor, got shape {tuple(probs.shape)}")
    num_classes = int(probs.size(1))
    if num_classes <= 1:
        return torch.ones((int(probs.size(0)),), device=probs.device, dtype=probs.dtype)
    entropy = -(probs * probs.clamp_min(float(eps)).log()).sum(dim=1)
    confidence = 1.0 - entropy / max(float(math.log(num_classes)), float(eps))
    return torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _stack_aux_probabilities(aux_probabilities: torch.Tensor | Sequence[torch.Tensor], p_base: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(aux_probabilities):
        aux = aux_probabilities
    else:
        aux_items = list(aux_probabilities)
        if not aux_items:
            return p_base.new_zeros((0, int(p_base.size(0)), int(p_base.size(1))))
        aux = torch.stack([item.to(device=p_base.device, dtype=p_base.dtype) for item in aux_items], dim=0)
    if aux.dim() == 2:
        aux = aux.unsqueeze(0)
    if aux.dim() != 3:
        raise ValueError(f"aux_probabilities must have shape [views, batch, classes], got {tuple(aux.shape)}")
    if tuple(aux.shape[1:]) != tuple(p_base.shape):
        raise ValueError(f"aux_probabilities shape {tuple(aux.shape)} is incompatible with p_base {tuple(p_base.shape)}")
    return aux.to(device=p_base.device, dtype=p_base.dtype)


def _aux_reliability_tensor(
    aux_reliabilities: torch.Tensor | Sequence[float] | None,
    *,
    num_views: int,
    batch_size: int,
    device,
    dtype,
) -> torch.Tensor:
    if aux_reliabilities is None:
        return torch.ones((num_views, batch_size), device=device, dtype=dtype)
    rel = torch.as_tensor(aux_reliabilities, device=device, dtype=dtype)
    if rel.dim() == 0:
        rel = rel.view(1, 1).expand(num_views, batch_size)
    elif rel.dim() == 1:
        if int(rel.numel()) != int(num_views):
            raise ValueError(f"aux_reliabilities length must equal number of views ({num_views}), got {int(rel.numel())}")
        rel = rel.view(num_views, 1).expand(num_views, batch_size)
    elif rel.dim() == 2:
        if tuple(rel.shape) != (int(num_views), int(batch_size)):
            raise ValueError(f"aux_reliabilities must have shape [{num_views}, {batch_size}], got {tuple(rel.shape)}")
    else:
        raise ValueError(f"aux_reliabilities must be scalar, [views], or [views,batch], got {tuple(rel.shape)}")
    return torch.nan_to_num(rel, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def collaborative_probability_fusion(
    p_base: torch.Tensor,
    aux_probabilities: torch.Tensor | Sequence[torch.Tensor],
    *,
    mode: str = "adaptive",
    aux_reliabilities: torch.Tensor | Sequence[float] | None = None,
    base_weight: float = 1.0,
    max_aux_weight: float = 1.0,
) -> torch.Tensor:
    base = normalize_probabilities(p_base)
    if base.dim() != 2:
        raise ValueError(f"p_base must be a [batch, classes] tensor, got shape {tuple(base.shape)}")
    aux = _stack_aux_probabilities(aux_probabilities, base)
    if int(aux.size(0)) == 0:
        return base
    aux = normalize_probabilities(aux.reshape(-1, int(base.size(1)))).view_as(aux)
    mode = str(mode or "adaptive").lower()
    if mode == "soft":
        return normalize_probabilities(torch.cat([base.unsqueeze(0), aux], dim=0).mean(dim=0))
    if mode not in {"adaptive", "conservative"}:
        raise ValueError(f"Unknown collaborative fusion mode '{mode}'. Expected soft, adaptive, or conservative.")

    num_views = int(aux.size(0))
    batch_size = int(base.size(0))
    rel = _aux_reliability_tensor(
        aux_reliabilities,
        num_views=num_views,
        batch_size=batch_size,
        device=base.device,
        dtype=base.dtype,
    )
    confidence = torch.stack([collaborative_reliability_from_probabilities(aux[i]) for i in range(num_views)], dim=0)
    max_weight = max(0.0, float(max_aux_weight))
    aux_weight = (rel * confidence).clamp(0.0, max_weight)
    if mode == "conservative":
        aux_weight = aux_weight.clamp(0.0, min(max_weight, 0.5))
    base_w = max(0.0, float(base_weight))
    weighted = base * base_w
    denom = base.new_full((batch_size, 1), base_w)
    weighted = weighted + (aux * aux_weight.unsqueeze(-1)).sum(dim=0)
    denom = denom + aux_weight.sum(dim=0, keepdim=False).view(batch_size, 1)
    return normalize_probabilities(weighted / denom.clamp_min(1e-8))


def harm_rescue_report(p_base: torch.Tensor, p_fused: torch.Tensor, y: torch.Tensor) -> dict[str, int]:
    base_pred = normalize_probabilities(p_base).argmax(dim=1)
    fused_pred = normalize_probabilities(p_fused).argmax(dim=1)
    y = y.view(-1).long().to(base_pred.device)
    base_ok = base_pred == y
    fused_ok = fused_pred == y
    rescue = (~base_ok & fused_ok)
    harm = (base_ok & ~fused_ok)
    return {
        "total": int(y.numel()),
        "base_correct": int(base_ok.sum().item()),
        "fused_correct": int(fused_ok.sum().item()),
        "rescue": int(rescue.sum().item()),
        "harm": int(harm.sum().item()),
        "net_gain": int(rescue.sum().item() - harm.sum().item()),
    }
