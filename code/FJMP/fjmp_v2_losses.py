"""Losses and stage schedule for FJMP-v2 safe residual prototypes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F


def _safe_float(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def compute_boundary_weight(
    base_logits: torch.Tensor,
    y: torch.Tensor,
    m_ce: float = 3.0,
    tau_ce: float = 0.75,
    rescue_weight: float = 0.5,
) -> torch.Tensor:
    base_logits = _safe_float(base_logits)
    y = y.to(device=base_logits.device).long().view(-1)
    if base_logits.size(1) > 1:
        top2 = torch.topk(base_logits, k=2, dim=-1).values
        base_margin = top2[:, 0] - top2[:, 1]
    else:
        base_margin = torch.zeros(base_logits.size(0), device=base_logits.device)
    base_pred = base_logits.argmax(dim=-1)
    base_correct = base_pred.eq(y)
    w_boundary = torch.sigmoid((float(m_ce) - base_margin) / max(float(tau_ce), 1e-6))
    w_rescue = float(rescue_weight) * (~base_correct).float()
    return torch.clamp(w_boundary + w_rescue, 0.0, 1.0)


def boundary_trimmed_ce_loss(
    fused_logits: torch.Tensor,
    base_logits: torch.Tensor,
    y: torch.Tensor,
    m_ce: float = 3.0,
    tau_ce: float = 0.75,
    rescue_weight: float = 0.5,
    return_weight: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    y = y.to(device=fused_logits.device).long().view(-1)
    ce_each = F.cross_entropy(_safe_float(fused_logits), y, reduction="none")
    weight = compute_boundary_weight(base_logits.detach(), y, m_ce=m_ce, tau_ce=tau_ce, rescue_weight=rescue_weight)
    loss = (weight.to(device=fused_logits.device) * ce_each).mean()
    return (loss, weight) if return_weight else loss


@torch.no_grad()
def compute_kd_weight(
    base_logits: torch.Tensor,
    y: torch.Tensor,
    conf_threshold: float = 0.0,
    margin_threshold: float = -1.0,
) -> torch.Tensor:
    base_logits = _safe_float(base_logits)
    y = y.to(device=base_logits.device).long().view(-1)
    base_prob = base_logits.softmax(dim=-1)
    base_conf = base_prob.max(dim=-1).values
    base_correct = base_logits.argmax(dim=-1).eq(y).float()
    if base_logits.size(1) > 1:
        top2 = torch.topk(base_logits, k=2, dim=-1).values
        base_margin = top2[:, 0] - top2[:, 1]
    else:
        base_margin = torch.zeros_like(base_conf)
    keep = (base_conf >= float(conf_threshold)) & (base_margin >= float(margin_threshold))
    return base_correct * base_conf * keep.float()


def selective_kd_loss(
    fused_logits: torch.Tensor,
    base_logits: torch.Tensor,
    y: torch.Tensor,
    T: float = 2.0,
    conf_threshold: float = 0.0,
    margin_threshold: float = -1.0,
    return_weight: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    temperature = max(float(T), 1e-6)
    kd_each = F.kl_div(
        F.log_softmax(_safe_float(fused_logits) / temperature, dim=-1),
        F.softmax(_safe_float(base_logits).detach() / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (temperature * temperature)
    weight = compute_kd_weight(
        base_logits.detach(),
        y,
        conf_threshold=conf_threshold,
        margin_threshold=margin_threshold,
    ).to(device=fused_logits.device)
    loss = (weight * kd_each).mean()
    return (loss, weight) if return_weight else loss


def angular_diversity_loss(prototypes: torch.Tensor, tau_div: float = 0.55) -> torch.Tensor:
    prototypes = _safe_float(prototypes)
    if prototypes.dim() != 3:
        raise ValueError("prototypes must be shaped [C, K, D].")
    _, K, _ = prototypes.shape
    if K <= 1:
        return prototypes.new_tensor(0.0)
    P = F.normalize(prototypes, dim=-1, eps=1e-6)
    sim = torch.einsum("ckd,cld->ckl", P, P)
    mask = ~torch.eye(K, dtype=torch.bool, device=P.device)
    sim_off = sim[:, mask].view(P.size(0), K, K - 1)
    return F.relu(sim_off - float(tau_div)).pow(2).mean()


def prototype_usage_loss(
    sim: torch.Tensor,
    y: torch.Tensor,
    K: Optional[int] = None,
    tau_assign: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sim.dim() != 3:
        raise ValueError("sim must be shaped [B, C, K].")
    bsz, classes, num_proto = sim.shape
    expected_k = int(K or num_proto)
    if expected_k != num_proto:
        raise ValueError(f"K={expected_k} does not match sim K={num_proto}.")
    y = y.to(device=sim.device).long().view(-1)
    valid = (y >= 0) & (y < classes)
    if not bool(valid.any()):
        q = sim.new_zeros((0, num_proto))
        usage = sim.new_full((classes, num_proto), 1.0 / num_proto)
        return sim.new_tensor(0.0), usage, q

    idx = torch.arange(bsz, device=sim.device)[valid]
    yy = y[valid]
    q = F.softmax(_safe_float(sim[idx, yy, :]) / max(float(tau_assign), 1e-6), dim=-1)
    usage_sum = sim.new_zeros((classes, num_proto), dtype=torch.float32)
    count = sim.new_zeros((classes,), dtype=torch.float32)
    usage_sum.index_add_(0, yy, q)
    count.index_add_(0, yy, torch.ones_like(yy, dtype=torch.float32))
    usage = usage_sum / count.clamp_min(1.0).unsqueeze(1)

    class_mask = count > 0
    target = torch.full_like(usage[class_mask], 1.0 / num_proto)
    loss = F.kl_div((usage[class_mask] + 1e-8).log(), target, reduction="batchmean")
    return loss, usage, q


def assignment_entropy_loss(q: torch.Tensor, K: int, target_ratio: float = 0.5) -> torch.Tensor:
    if q.numel() == 0:
        return q.new_tensor(0.0)
    entropy = -(q * (q + 1e-8).log()).sum(dim=-1)
    target_entropy = math.log(max(int(K), 1)) * float(target_ratio)
    return (entropy.mean() - target_entropy).pow(2)


def delta_ratio_loss(
    delta_logits: torch.Tensor,
    base_logits: torch.Tensor,
    r_max: float = 0.10,
    bound_weight: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = _safe_float(delta_logits)
    base = _safe_float(base_logits).detach()
    ratio = delta.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-6)
    soft = ratio.pow(2).mean()
    bound = F.relu(ratio - float(r_max)).pow(2).mean()
    return soft + float(bound_weight) * bound, ratio.detach()


def logit_residual_loss(delta_logits: torch.Tensor, target_norm: float = 3.0) -> torch.Tensor:
    delta_norm = _safe_float(delta_logits).norm(dim=-1)
    return F.relu(delta_norm - float(target_norm)).pow(2).mean()


def gate_regularization(rho: torch.Tensor) -> torch.Tensor:
    return _safe_float(rho).pow(2).mean()


@dataclass(frozen=True)
class FJMPV2StageWeights:
    stage: str
    rho_max: float
    ce_trim: float
    kd: float
    div: float
    usage: float
    entropy: float
    delta: float
    logit: float
    gate: float
    freeze_prototypes: bool = False
    train_gate_only: bool = False


def get_fjmp_v2_stage_weights(epoch: int, cfg: Optional[Mapping[str, float]] = None) -> FJMPV2StageWeights:
    cfg = dict(cfg or {})
    epoch = int(epoch)
    rho_final = float(cfg.get("rho_max", 0.15))
    if epoch <= 2:
        return FJMPV2StageWeights(
            stage="stage1",
            rho_max=float(cfg.get("stage1_rho_max", 0.03)),
            ce_trim=float(cfg.get("stage1_ce_trim", 0.3)),
            kd=float(cfg.get("lambda_kd_selective", 0.3)),
            div=float(cfg.get("lambda_proto_div", 0.01)),
            usage=float(cfg.get("lambda_proto_usage", 0.005)),
            entropy=0.0,
            delta=float(cfg.get("lambda_delta", 0.01)),
            logit=0.0,
            gate=0.0,
        )
    if epoch <= 8:
        progress = (epoch - 3) / max(8 - 3, 1)
        rho = 0.03 + max(0.0, min(1.0, progress)) * (rho_final - 0.03)
        return FJMPV2StageWeights(
            stage="stage2",
            rho_max=float(rho),
            ce_trim=float(cfg.get("lambda_ce_trim", 1.0)),
            kd=float(cfg.get("lambda_kd_selective", 0.3)),
            div=float(cfg.get("lambda_proto_div", 0.01)),
            usage=float(cfg.get("lambda_proto_usage", 0.005)),
            entropy=float(cfg.get("lambda_assign_entropy", 0.001)),
            delta=float(cfg.get("lambda_delta", 0.01)),
            logit=float(cfg.get("lambda_logit_residual", 0.01)),
            gate=float(cfg.get("lambda_gate", 0.003)),
        )
    return FJMPV2StageWeights(
        stage="stage3",
        rho_max=rho_final,
        ce_trim=float(cfg.get("stage3_ce_trim", 0.2)),
        kd=float(cfg.get("lambda_kd_selective", 0.3)),
        div=0.0,
        usage=0.0,
        entropy=0.0,
        delta=0.0,
        logit=float(cfg.get("lambda_logit_residual", 0.01)),
        gate=float(cfg.get("lambda_gate", 0.003)),
        freeze_prototypes=True,
        train_gate_only=True,
    )


def compute_fjmp_v2_loss(
    fused_logits: torch.Tensor,
    base_logits: torch.Tensor,
    y: torch.Tensor,
    aux: Mapping[str, torch.Tensor],
    *,
    epoch: int = 1,
    cfg: Optional[Mapping[str, float]] = None,
) -> dict[str, torch.Tensor]:
    cfg = dict(cfg or {})
    weights = get_fjmp_v2_stage_weights(epoch, cfg)
    ce, w_ce = boundary_trimmed_ce_loss(
        fused_logits,
        base_logits,
        y,
        m_ce=float(cfg.get("ce_trim_margin", 3.0)),
        tau_ce=float(cfg.get("ce_trim_tau", 0.75)),
        rescue_weight=float(cfg.get("ce_trim_rescue_weight", 0.5)),
        return_weight=True,
    )
    kd, w_kd = selective_kd_loss(
        fused_logits,
        base_logits,
        y,
        T=float(cfg.get("kd_selective_temperature", 2.0)),
        conf_threshold=float(cfg.get("kd_conf_threshold", 0.0)),
        margin_threshold=float(cfg.get("kd_margin_threshold", -1.0)),
        return_weight=True,
    )
    proto_ce = F.cross_entropy(_safe_float(aux["proto_logits"]), y.to(device=fused_logits.device).long())
    div = angular_diversity_loss(aux["prototypes"], tau_div=float(cfg.get("tau_div", 0.55)))
    usage, usage_vec, q = prototype_usage_loss(
        aux["sim"],
        y,
        K=int(cfg.get("K", aux["sim"].size(-1))),
        tau_assign=float(cfg.get("tau_assign", 0.75)),
    )
    entropy = assignment_entropy_loss(q, int(cfg.get("K", aux["sim"].size(-1))), float(cfg.get("target_entropy_ratio", 0.5)))
    delta, delta_ratio = delta_ratio_loss(
        aux.get("delta", aux["rho"] * aux["delta_logits"]),
        base_logits,
        r_max=float(cfg.get("delta_ratio_max", 0.10)),
        bound_weight=float(cfg.get("delta_bound_weight", 5.0)),
    )
    logit = logit_residual_loss(aux["delta_logits"], target_norm=float(cfg.get("logit_residual_target_norm", 3.0)))
    gate = gate_regularization(aux["rho"])
    total = (
        weights.ce_trim * ce
        + weights.kd * kd
        + weights.div * div
        + weights.usage * usage
        + weights.entropy * entropy
        + weights.delta * delta
        + weights.logit * logit
        + weights.gate * gate
    )
    if int(epoch) <= int(cfg.get("proto_warmup_epochs", 0)):
        total = total + float(cfg.get("lambda_ce_proto_warmup", 0.0)) * proto_ce
    return {
        "loss": total,
        "loss_ce_proto_warmup": proto_ce,
        "loss_ce_trim": ce,
        "loss_kd_selective": kd,
        "loss_proto_div": div,
        "loss_proto_usage": usage,
        "loss_assign_entropy": entropy,
        "loss_delta": delta,
        "loss_logit_residual": logit,
        "loss_gate": gate,
        "w_ce_mean": w_ce.mean().detach(),
        "w_kd_mean": w_kd.mean().detach(),
        "delta_ratio_mean": delta_ratio.mean().detach(),
        "usage": usage_vec.detach(),
        "assignment_q": q.detach(),
    }


__all__ = [
    "FJMPV2StageWeights",
    "angular_diversity_loss",
    "assignment_entropy_loss",
    "boundary_trimmed_ce_loss",
    "compute_boundary_weight",
    "compute_fjmp_v2_loss",
    "compute_kd_weight",
    "delta_ratio_loss",
    "gate_regularization",
    "get_fjmp_v2_stage_weights",
    "logit_residual_loss",
    "prototype_usage_loss",
    "selective_kd_loss",
]
