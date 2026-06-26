"""Metrics and warning checks for SGV-BP-FJMP."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F


def _margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    y = y.to(device=logits.device).long().view(-1)
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(y, logits.size(1)).bool(), float("-inf")).max(dim=1).values
    return true - other


def _head_metrics(prefix: str, logits: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    prob = F.softmax(logits.float(), dim=1)
    pred = prob.argmax(dim=1)
    correct = pred.eq(y.to(device=logits.device).long())
    return {
        f"{prefix}_acc": correct.float().mean(),
        f"{prefix}_margin": _margin(logits, y).mean(),
        f"{prefix}_conf": prob.max(dim=1).values.mean(),
    }


def _harm_rescue(prefix: str, base_logits: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    y = y.to(device=logits.device).long()
    base_ok = base_logits.argmax(dim=1).eq(y)
    pred_ok = logits.argmax(dim=1).eq(y)
    harm = base_ok & (~pred_ok)
    rescue = (~base_ok) & pred_ok
    return {
        f"{prefix}_harm": harm.float().mean(),
        f"{prefix}_rescue": rescue.float().mean(),
        f"{prefix}_net_gain": rescue.float().mean() - harm.float().mean(),
    }


def _usage_entropy(proto_scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(y.numel(), device=proto_scores.device)
    yy = y.to(device=proto_scores.device).long()
    assign = F.softmax(proto_scores[idx, yy, :].float(), dim=1)
    usage = assign.mean(dim=0)
    return -(usage * usage.clamp_min(1e-8).log()).sum()


def compute_sgv_bp_metrics(
    clean: Mapping[str, torch.Tensor],
    y: torch.Tensor,
    *,
    sat: Optional[Mapping[str, torch.Tensor]] = None,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    stem = f"{prefix}_" if prefix else ""
    out: dict[str, torch.Tensor] = {}
    out.update(_head_metrics(f"{stem}base_clean", clean["base_logits"], y))
    out.update(_head_metrics(f"{stem}head_clean", clean["head_logits"], y))
    out.update(_head_metrics(f"{stem}safe_clean", clean["safe_logits"], y))
    out.update(_harm_rescue(f"{stem}head_clean", clean["base_logits"], clean["head_logits"], y))
    out.update(_harm_rescue(f"{stem}safe_clean", clean["base_logits"], clean["safe_logits"], y))
    if "rho" in clean:
        out[f"{stem}rho_mean"] = clean["rho"].float().mean()
        out[f"{stem}rho_max"] = clean["rho"].float().max()
    if "gate" in clean:
        out[f"{stem}gate_mean_clean"] = clean["gate"].float().mean()
    if "delta" in clean:
        out[f"{stem}delta_norm_clean"] = clean["delta"].float().norm(dim=1).mean()
    if "proto_scores" in clean:
        out[f"{stem}proto_usage_entropy_clean"] = _usage_entropy(clean["proto_scores"], y)
    if sat is not None:
        out.update(_head_metrics(f"{stem}base_sat", sat["base_logits"], y))
        out.update(_head_metrics(f"{stem}head_sat", sat["head_logits"], y))
        out.update(_head_metrics(f"{stem}safe_sat", sat["safe_logits"], y))
        out.update(_harm_rescue(f"{stem}safe_sat", sat["base_logits"], sat["safe_logits"], y))
        if "gate" in sat:
            out[f"{stem}gate_mean_sat"] = sat["gate"].float().mean()
        if "delta" in sat:
            out[f"{stem}delta_norm_sat"] = sat["delta"].float().norm(dim=1).mean()
        if "proto_scores" in clean and "proto_scores" in sat:
            idx = torch.arange(y.numel(), device=clean["proto_scores"].device)
            yy = y.to(device=clean["proto_scores"].device).long()
            qc = F.softmax(clean["proto_scores"][idx, yy, :].float(), dim=1)
            qs = F.softmax(sat["proto_scores"][idx, yy, :].float(), dim=1)
            out[f"{stem}proto_assignment_kl_clean_sat"] = F.kl_div(qs.clamp_min(1e-8).log(), qc, reduction="batchmean")
    return out


def compute_proxy_safe_score(metrics: Mapping[str, float], rho_target: float = 0.25) -> float:
    """Document fallback proxy score, normalized terms expected in [0, 1]."""

    proxy = float(metrics.get("proxy_sat_mid_safe_acc", metrics.get("safe_sat_mid_acc", 0.0)))
    harm = float(metrics.get("proxy_sat_mid_harm", metrics.get("safe_sat_mid_harm", 0.0)))
    rescue = float(metrics.get("proxy_sat_mid_rescue", metrics.get("safe_sat_mid_rescue", 0.0)))
    clean_harm = float(metrics.get("clean_harm", metrics.get("safe_clean_harm", 0.0)))
    clean_drop = float(metrics.get("clean_acc_drop", 0.0))
    rho_penalty = max(0.0, float(metrics.get("rho_mean", 0.0)) - float(rho_target))
    return proxy - 2.0 * harm + rescue - clean_harm - 0.5 * clean_drop - 0.5 * rho_penalty


def warning_flags(metrics: Mapping[str, float]) -> list[str]:
    checks = [
        ("rho_mean_high", float(metrics.get("rho_mean", 0.0)) > 0.25),
        ("rho_easy_high", float(metrics.get("rho_easy", 0.0)) > 0.15),
        ("gate_easy_high", float(metrics.get("gate_easy", 0.0)) > 0.20),
        ("safe_clean_harm_high", float(metrics.get("safe_clean_harm", 0.0)) > 0.0),
        ("safe_clean_acc_drop", float(metrics.get("safe_clean_acc", 1.0)) < float(metrics.get("base_clean_acc", 0.0)) - 0.003),
        ("proto_usage_entropy_low", float(metrics.get("proto_usage_entropy_clean", 99.0)) < 0.2),
        ("base_sat_mid_drop_high", float(metrics.get("base_clean_acc", 0.0)) - float(metrics.get("base_sat_mid_acc", 0.0)) > 0.05),
    ]
    return [name for name, active in checks if active]


__all__ = ["compute_proxy_safe_score", "compute_sgv_bp_metrics", "warning_flags"]
