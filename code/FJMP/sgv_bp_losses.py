"""Loss functions and schedules for SGV-BP-FJMP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F


def _float(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _true_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = _float(logits)
    y = y.to(device=logits.device).long().view(-1)
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(y, logits.size(1)).bool(), float("-inf")).max(dim=1).values
    return true - other


@dataclass(frozen=True)
class SGVBPStageConfig:
    stage: int
    rho_max: float
    weights: dict[str, float]
    views: tuple[str, ...]


def sgv_bp_stage_config(epoch: int) -> SGVBPStageConfig:
    """Return the documented E1-E5/E6-E15/E16-E30 stage schedule."""

    epoch = int(epoch)
    if epoch <= 5:
        return SGVBPStageConfig(
            1,
            0.10,
            {
                "ce_head_clean": 0.50,
                "ce_head_sat": 0.10,
                "ce_safe_clean": 0.10,
                "ce_safe_sat": 0.02,
                "pres_clean": 3.0,
                "pres_sat": 0.5,
                "harm": 2.0,
                "kd_easy": 1.5,
                "kd_mid": 0.5,
                "kd_hard_low_margin": 0.05,
                "sgv_head": 0.2,
                "sgv_safe": 0.3,
                "gate_easy": 0.05,
                "gate_view_gap": 0.02,
                "delta": 0.03,
                "rho_reg": 0.05,
            },
            ("clean", "sat_low"),
        )
    if epoch <= 15:
        return SGVBPStageConfig(
            2,
            0.25,
            {
                "ce_head_clean": 0.30,
                "ce_head_sat": 0.15,
                "ce_safe_clean": 0.05,
                "ce_safe_sat": 0.02,
                "pres_clean": 3.0,
                "pres_sat": 1.5,
                "harm": 2.0,
                "kd_easy": 1.5,
                "kd_mid": 0.5,
                "kd_hard_low_margin": 0.05,
                "sgv_head": 0.5,
                "sgv_safe": 1.0,
                "sgv_margin": 0.3,
                "proto_sgv": 0.2,
                "worst": 0.3,
                "gate_easy": 0.08,
                "gate_view_gap": 0.03,
                "delta": 0.04,
                "rho_reg": 0.05,
            },
            ("clean", "sat_low", "sat_mid"),
        )
    return SGVBPStageConfig(
        3,
        0.30,
        {
            "ce_head_clean": 0.10,
            "ce_head_sat": 0.05,
            "ce_safe_clean": 0.02,
            "ce_safe_sat": 0.01,
            "pres_clean": 4.0,
            "pres_sat": 2.0,
            "harm": 3.0,
            "kd_easy": 1.5,
            "kd_mid": 0.7,
            "kd_hard_low_margin": 0.05,
            "sgv_head": 0.7,
            "sgv_safe": 1.0,
            "sgv_margin": 0.3,
            "proto_sgv": 0.3,
            "worst": 0.5,
            "domain_view": 0.05,
            "gate_easy": 0.10,
            "gate_view_gap": 0.05,
            "delta": 0.05,
            "rho_reg": 0.10,
        },
        ("clean", "sat_low", "sat_mid"),
    )


def compute_sample_strata(base_logits: torch.Tensor, y: torch.Tensor, tau_low: float = 0.5, tau_high: float = 2.0) -> dict[str, torch.Tensor]:
    margin = _true_margin(base_logits, y)
    pred = base_logits.argmax(dim=1)
    y = y.to(device=base_logits.device).long().view(-1)
    correct = pred.eq(y)
    return {
        "easy": correct & (margin > float(tau_high)),
        "mid": correct & (margin <= float(tau_high)) & (margin >= float(tau_low)),
        "hard_low_margin": correct & (margin < float(tau_low)),
        "hard_wrong": ~correct,
        "base_correct": correct,
        "base_margin": margin,
    }


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    values = values.float()
    if mask is not None:
        values = values * mask.to(device=values.device, dtype=values.dtype)
    if weight is not None:
        values = values * weight.to(device=values.device, dtype=values.dtype).view_as(values)
    denom = torch.ones_like(values)
    if mask is not None:
        denom = denom * mask.to(device=values.device, dtype=values.dtype)
    if weight is not None:
        denom = denom * weight.to(device=values.device, dtype=values.dtype).view_as(values)
    return values.sum() / denom.sum().clamp_min(1.0)


def stratified_ce(logits: torch.Tensor, y: torch.Tensor, strata: Mapping[str, torch.Tensor]) -> torch.Tensor:
    ce = F.cross_entropy(_float(logits), y.long(), reduction="none")
    weights = torch.ones_like(ce)
    weights[strata["easy"]] = 0.2
    weights[strata["mid"]] = 0.5
    weights[strata["hard_low_margin"]] = 1.0
    weights[strata["hard_wrong"]] = 1.0
    return (weights * ce).sum() / weights.sum().clamp_min(1.0)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: Optional[torch.Tensor] = None, temperature: float = 4.0) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    loss = F.kl_div(
        F.log_softmax(_float(student_logits) / t, dim=1),
        F.softmax(_float(teacher_logits).detach() / t, dim=1),
        reduction="none",
    ).sum(dim=1) * (t * t)
    return _masked_mean(loss, mask)


def margin_preserve_loss(base_logits: torch.Tensor, safe_logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, delta: float = 0.05, cap: Optional[float] = None) -> torch.Tensor:
    base_margin = _true_margin(base_logits, y).detach()
    if cap is not None:
        base_margin = base_margin.clamp(max=float(cap))
    safe_margin = _true_margin(safe_logits, y)
    return _masked_mean(F.relu(base_margin - safe_margin + float(delta)), mask)


def harm_constraint_loss(base_logits: torch.Tensor, safe_logits: torch.Tensor, y: torch.Tensor, tau_harm: float = 0.5) -> torch.Tensor:
    base_correct = base_logits.argmax(dim=1).eq(y.to(device=base_logits.device).long())
    drop = (_true_margin(base_logits, y).detach() - _true_margin(safe_logits, y))
    ce = F.cross_entropy(_float(safe_logits), y.long(), reduction="none")
    return _masked_mean(torch.sigmoid(drop / max(float(tau_harm), 1e-6)) * ce, base_correct)


def gate_easy_loss(gate_clean: torch.Tensor, gate_sat: Optional[torch.Tensor], easy: torch.Tensor) -> torch.Tensor:
    values = gate_clean.float().view(-1)
    if gate_sat is not None:
        values = torch.cat([values, gate_sat.float().view(-1)], dim=0)
        easy = torch.cat([easy, easy], dim=0)
    return _masked_mean(values, easy)


def gate_view_gap_loss(gate_clean: torch.Tensor, gate_sat: torch.Tensor) -> torch.Tensor:
    return (gate_clean.float().mean() - gate_sat.float().mean()).abs()


def proto_assignment_consistency(proto_clean: torch.Tensor, proto_sat: torch.Tensor, y: torch.Tensor, temperature: float = 1.5) -> torch.Tensor:
    idx = torch.arange(y.numel(), device=proto_clean.device)
    yy = y.to(device=proto_clean.device).long()
    qc = F.softmax(proto_clean[idx, yy, :] / max(float(temperature), 1e-6), dim=1)
    qs = F.softmax(proto_sat[idx, yy, :] / max(float(temperature), 1e-6), dim=1)
    return 0.5 * (
        F.kl_div(qc.clamp_min(1e-8).log(), qs.detach(), reduction="batchmean")
        + F.kl_div(qs.clamp_min(1e-8).log(), qc.detach(), reduction="batchmean")
    )


def worst_domain_view_loss(logits: torch.Tensor, y: torch.Tensor, group: torch.Tensor, tau: float = 0.5, min_group_count: int = 8) -> torch.Tensor:
    ce = F.cross_entropy(_float(logits), y.long(), reduction="none")
    losses = []
    for g in torch.unique(group):
        mask = group.eq(g)
        if int(mask.sum().item()) >= int(min_group_count):
            losses.append(ce[mask].mean())
    if not losses:
        return ce.mean() * 0.0
    stacked = torch.stack(losses)
    return float(tau) * torch.logsumexp(stacked / max(float(tau), 1e-6), dim=0)


def compute_sgv_bp_losses(
    clean: Mapping[str, torch.Tensor],
    sat: Optional[Mapping[str, torch.Tensor]],
    y: torch.Tensor,
    *,
    epoch: int = 1,
    weights: Optional[Mapping[str, float]] = None,
    sat_reliability: Optional[torch.Tensor] = None,
    group: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    stage = sgv_bp_stage_config(epoch)
    w = dict(stage.weights)
    if weights:
        w.update({k: float(v) for k, v in weights.items()})
    strata = compute_sample_strata(clean["base_logits"], y)
    zero = clean["base_logits"].new_tensor(0.0)
    losses = {
        "ce_head_clean": stratified_ce(clean["head_logits"], y, strata),
        "ce_safe_clean": stratified_ce(clean["safe_logits"], y, strata),
        "pres_clean": margin_preserve_loss(clean["base_logits"], clean["safe_logits"], y, strata["base_correct"], delta=0.05),
        "harm": harm_constraint_loss(clean["base_logits"], clean["safe_logits"], y),
        "kd_easy": kd_loss(clean["safe_logits"], clean["base_logits"], strata["easy"]),
        "kd_mid": kd_loss(clean["safe_logits"], clean["base_logits"], strata["mid"]),
        "kd_hard_low_margin": kd_loss(clean["safe_logits"], clean["base_logits"], strata["hard_low_margin"]),
        "gate_easy": gate_easy_loss(clean.get("gate", zero.expand_as(y.float())), sat.get("gate") if sat else None, strata["easy"]),
        "delta": clean.get("delta", zero).float().norm(dim=1).mean() if "delta" in clean else zero,
        "rho_reg": clean.get("rho", zero).float().mean() if "rho" in clean else zero,
    }
    if sat is not None:
        r = sat_reliability if sat_reliability is not None else torch.ones_like(y, dtype=torch.float32, device=y.device)
        losses["ce_head_sat"] = _masked_mean(F.cross_entropy(_float(sat["head_logits"]), y.long(), reduction="none"), weight=r)
        losses["ce_safe_sat"] = _masked_mean(F.cross_entropy(_float(sat["safe_logits"]), y.long(), reduction="none"), weight=r)
        losses["pres_sat"] = margin_preserve_loss(clean["base_logits"], sat["safe_logits"], y, strata["easy"] | strata["mid"], delta=0.0, cap=3.0)
        losses["sgv_head"] = kd_loss(sat["head_logits"], clean["head_logits"].detach(), strata["easy"] | strata["mid"] | strata["hard_low_margin"])
        losses["sgv_safe"] = kd_loss(sat["safe_logits"], clean["base_logits"].detach(), strata["easy"] | strata["mid"])
        losses["sgv_margin"] = _masked_mean(F.relu(_true_margin(clean["safe_logits"], y) - _true_margin(sat["safe_logits"], y) - 0.3), strata["easy"] | strata["mid"], r)
        losses["gate_view_gap"] = gate_view_gap_loss(clean.get("gate", r), sat.get("gate", r))
        if "proto_scores" in clean and "proto_scores" in sat:
            losses["proto_sgv"] = proto_assignment_consistency(clean["proto_scores"], sat["proto_scores"], y)
    if group is not None:
        losses["worst"] = worst_domain_view_loss(clean["safe_logits"], y, group)
    total = zero
    for name, value in losses.items():
        total = total + float(w.get(name, 0.0)) * value
    losses["loss"] = total
    return losses


__all__ = [
    "SGVBPStageConfig",
    "compute_sample_strata",
    "compute_sgv_bp_losses",
    "gate_easy_loss",
    "gate_view_gap_loss",
    "harm_constraint_loss",
    "kd_loss",
    "margin_preserve_loss",
    "proto_assignment_consistency",
    "sgv_bp_stage_config",
    "stratified_ce",
    "worst_domain_view_loss",
]
