"""Prototype diagnostics used by FJMP-v2 training logs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


TensorDict = dict[str, torch.Tensor]


def _safe_float(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return torch.quantile(x.float().reshape(-1), float(q))


@torch.no_grad()
def compute_proto_pairwise_cos(prototypes: torch.Tensor) -> TensorDict:
    prototypes = _safe_float(prototypes)
    if prototypes.dim() != 3:
        raise ValueError("prototypes must be shaped [C, K, D].")
    _, K, _ = prototypes.shape
    if K <= 1:
        zero = prototypes.new_tensor(0.0)
        return {
            "proto_pairwise_cos_mean": zero,
            "proto_pairwise_cos_max": zero,
            "proto_pairwise_cos_p95": zero,
        }
    P = F.normalize(prototypes, dim=-1, eps=1e-6)
    sim = torch.einsum("ckd,cld->ckl", P, P)
    mask = ~torch.eye(K, dtype=torch.bool, device=P.device)
    off = sim[:, mask].reshape(-1)
    return {
        "proto_pairwise_cos_mean": off.mean(),
        "proto_pairwise_cos_max": off.max(),
        "proto_pairwise_cos_p95": _quantile(off, 0.95),
    }


@torch.no_grad()
def compute_usage_entropy(usage: torch.Tensor) -> TensorDict:
    usage = _safe_float(usage)
    if usage.dim() == 1:
        usage = usage.unsqueeze(0)
    p = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
    return {
        "usage_entropy_mean": entropy.mean(),
        "usage_entropy_min": entropy.min() if entropy.numel() else usage.new_tensor(0.0),
        "usage_min": p.min() if p.numel() else usage.new_tensor(0.0),
        "usage_max": p.max() if p.numel() else usage.new_tensor(0.0),
    }


@torch.no_grad()
def compute_dead_proto_rate(usage: torch.Tensor, K: int | None = None, min_usage: float | None = None) -> TensorDict:
    usage = _safe_float(usage)
    num_proto = int(K or usage.size(-1))
    threshold = float(min_usage) if min_usage is not None else 0.05 / max(num_proto, 1)
    return {"dead_proto_rate": (usage < threshold).float().mean()}


@torch.no_grad()
def compute_harm_rescue(base_logits: torch.Tensor, fused_logits: torch.Tensor, y: torch.Tensor) -> TensorDict:
    base_logits = _safe_float(base_logits)
    fused_logits = _safe_float(fused_logits)
    y = y.to(device=fused_logits.device).long().view(-1)
    base_correct = base_logits.argmax(dim=-1).eq(y)
    fused_correct = fused_logits.argmax(dim=-1).eq(y)
    harm = base_correct & (~fused_correct)
    rescue = (~base_correct) & fused_correct
    harm_rate = harm.float().mean()
    rescue_rate = rescue.float().mean()
    return {
        "harm_rate": harm_rate,
        "rescue_rate": rescue_rate,
        "net_gain": rescue_rate - harm_rate,
    }


@torch.no_grad()
def compute_delta_stats(delta_logits: torch.Tensor, base_logits: torch.Tensor) -> TensorDict:
    delta = _safe_float(delta_logits)
    base = _safe_float(base_logits)
    delta_norm = delta.norm(dim=-1)
    ratio = delta_norm / base.norm(dim=-1).clamp_min(1e-6)
    return {
        "delta_norm_mean": delta_norm.mean(),
        "delta_norm_p95": _quantile(delta_norm, 0.95),
        "delta_ratio_mean": ratio.mean(),
        "delta_ratio_p95": _quantile(ratio, 0.95),
    }


@torch.no_grad()
def compute_rho_stats(rho: torch.Tensor) -> TensorDict:
    rho = _safe_float(rho).reshape(-1)
    return {
        "rho_mean": rho.mean() if rho.numel() else rho.new_tensor(0.0),
        "rho_std": rho.std(unbiased=False) if rho.numel() else rho.new_tensor(0.0),
        "rho_p50": _quantile(rho, 0.50),
        "rho_p90": _quantile(rho, 0.90),
        "rho_p95": _quantile(rho, 0.95),
        "rho_max": rho.max() if rho.numel() else rho.new_tensor(0.0),
    }


def compute_fjmp_v2_metrics(
    *,
    prototypes: torch.Tensor,
    usage: torch.Tensor,
    rho: torch.Tensor,
    delta_logits: torch.Tensor,
    base_logits: torch.Tensor,
    fused_logits: torch.Tensor,
    y: torch.Tensor,
) -> TensorDict:
    out: TensorDict = {}
    out.update(compute_proto_pairwise_cos(prototypes))
    out.update(compute_usage_entropy(usage))
    out.update(compute_dead_proto_rate(usage, K=usage.size(-1)))
    out.update(compute_rho_stats(rho))
    out.update(compute_delta_stats(delta_logits, base_logits))
    out.update(compute_harm_rescue(base_logits, fused_logits, y))
    return out


__all__ = [
    "compute_dead_proto_rate",
    "compute_delta_stats",
    "compute_fjmp_v2_metrics",
    "compute_harm_rescue",
    "compute_proto_pairwise_cos",
    "compute_rho_stats",
    "compute_usage_entropy",
]
