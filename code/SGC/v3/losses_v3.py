from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F

from .metrics_v3 import compute_flip_metrics
from .target_adaptation import PseudoLabelFilterConfig, build_pseudo_label_mask, probability_margin


TensorDict = Dict[str, torch.Tensor]


DEFAULT_LOSS_WEIGHTS = {
    "clean_kl": 2.0,
    "clean_feat": 1.0,
    "clean_margin": 1.0,
    "clean_gate": 1.0,
    "clean_res": 0.1,
    "pair_feat": 1.0,
    "pair_logit": 1.0,
    "proto": 0.5,
    "sat_ce": 0.3,
    "gate_sat": 1.0,
    "gate_sparse": 0.01,
    "safe": 1.0,
    "target_pl": 1.0,
    "target_ent": 0.01,
    "target_div": 0.01,
    "target_anchor": 0.1,
}


def _kl_student_to_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    T = float(temperature)
    student_logp = F.log_softmax(student_logits / T, dim=-1)
    teacher_p = F.softmax(teacher_logits.detach() / T, dim=-1)
    return F.kl_div(student_logp, teacher_p, reduction="batchmean") * (T * T)


def _margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.topk(min(2, logits.size(-1)), dim=-1).values
    if top2.size(-1) == 1:
        return top2[:, 0]
    return top2[:, 0] - top2[:, 1]


def _weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    out = dict(DEFAULT_LOSS_WEIGHTS)
    if weights:
        out.update({key: float(value) for key, value in weights.items()})
    return out


def compute_sgc_v3_losses(
    model,
    batch: Mapping[str, torch.Tensor],
    *,
    weights: Mapping[str, float] | None = None,
    pseudo_cfg: PseudoLabelFilterConfig | None = None,
) -> Tuple[torch.Tensor, TensorDict]:
    w = _weights(weights)
    x_clean = batch["x_clean"]
    x_sat = batch["x_sat"]
    y = batch["y"].long().view(-1)

    out_clean = model(x_clean)
    out_sat = model(x_sat)

    loss_clean_kl = _kl_student_to_teacher(out_clean["logits_final"], out_clean["logits_base"])
    loss_clean_feat = F.mse_loss(out_clean["z_sgc"], out_clean["z_base"].detach())
    loss_clean_margin = F.relu(_margin(out_clean["logits_base"]).detach() - _margin(out_clean["logits_final"])).mean()
    loss_clean_gate = out_clean["gate"].square().mean()
    loss_clean_res = out_clean["delta_z_ratio"].square().mean() + out_clean["delta_logit_norm"].square().mean()

    loss_pair_feat = 1.0 - F.cosine_similarity(out_sat["z_sgc"], out_clean["z_base"].detach(), dim=-1).mean()
    loss_pair_logit = _kl_student_to_teacher(out_sat["logits_final"], out_clean["logits_base"])
    proto_bank = getattr(model, "prototype_bank", None)
    if proto_bank is not None:
        loss_proto = proto_bank.pull_push_loss(out_sat["z_sgc"], y)
    else:
        loss_proto = x_clean.new_tensor(0.0)
    loss_sat_ce = F.cross_entropy(out_sat["logits_final"], y)

    scenario = batch.get("scenario")
    if scenario is not None:
        sat_target = (scenario.to(device=x_sat.device).long().view(-1) > 0).float().view(-1, 1)
        loss_gate_sat = F.binary_cross_entropy_with_logits(out_sat["sat_logit"].float(), sat_target)
    else:
        loss_gate_sat = x_clean.new_tensor(0.0)
    loss_gate_sparse = out_sat["gate"].mean()
    loss_safe = (
        F.relu(out_sat["delta_z_ratio"] - float(getattr(model.cfg, "epsilon_z", 0.02))).mean()
        + F.relu(out_sat["delta_logit_norm"] - float(getattr(model.cfg, "epsilon_logit", 0.5))).mean()
    )

    target_logs: TensorDict = {}
    loss_target = x_clean.new_tensor(0.0)
    if "x_target" in batch and batch["x_target"] is not None:
        out_t = model(batch["x_target"])
        prob_t = out_t["prob_final"]
        pseudo_y = out_t["pseudo_y"].detach()
        mask = build_pseudo_label_mask(prob_t.detach(), cfg=pseudo_cfg)
        if mask.any():
            ce = F.cross_entropy(out_t["logits_final"], pseudo_y, reduction="none")
            weighted_ce = (ce * out_t["pseudo_weight"].detach())[mask].mean()
        else:
            weighted_ce = x_clean.new_tensor(0.0)
        ent = -(prob_t * (prob_t + 1e-8).log()).sum(dim=-1).mean()
        mean_prob = prob_t.mean(dim=0)
        div = -(mean_prob * (mean_prob + 1e-8).log()).sum()
        anchor = out_t["delta_z_ratio"].square().mean() + out_t["delta_logit_norm"].square().mean()
        loss_target = (
            w["target_pl"] * weighted_ce
            + w["target_ent"] * ent
            - w["target_div"] * div
            + w["target_anchor"] * anchor
        )
        target_logs.update(
            {
                "target/loss_pl": weighted_ce.detach(),
                "target/loss_entropy": ent.detach(),
                "target/loss_diversity": div.detach(),
                "target/pseudo_coverage": mask.float().mean().detach(),
                "target/pseudo_confidence": out_t["pseudo_weight"].detach().mean(),
                "target/pseudo_margin": probability_margin(prob_t.detach()).mean(),
            }
        )
    else:
        target_logs["target/pseudo_coverage"] = x_clean.new_tensor(0.0)

    loss = (
        w["clean_kl"] * loss_clean_kl
        + w["clean_feat"] * loss_clean_feat
        + w["clean_margin"] * loss_clean_margin
        + w["clean_gate"] * loss_clean_gate
        + w["clean_res"] * loss_clean_res
        + w["pair_feat"] * loss_pair_feat
        + w["pair_logit"] * loss_pair_logit
        + w["proto"] * loss_proto
        + w["sat_ce"] * loss_sat_ce
        + w["gate_sat"] * loss_gate_sat
        + w["gate_sparse"] * loss_gate_sparse
        + w["safe"] * loss_safe
        + loss_target
    )

    flips = compute_flip_metrics(out_sat["logits_base"].detach(), out_sat["logits_final"].detach(), y)
    logs: TensorDict = {
        "train/loss_total": loss.detach(),
        "train/loss_clean_kl": loss_clean_kl.detach(),
        "train/loss_clean_feat": loss_clean_feat.detach(),
        "train/loss_clean_margin": loss_clean_margin.detach(),
        "train/loss_clean_gate": loss_clean_gate.detach(),
        "train/loss_pair_feat": loss_pair_feat.detach(),
        "train/loss_pair_logit": loss_pair_logit.detach(),
        "train/loss_proto": loss_proto.detach(),
        "train/loss_sat_ce": loss_sat_ce.detach(),
        "train/loss_gate_safety": (loss_gate_sat + loss_gate_sparse + loss_safe).detach(),
        "sgc/gate_clean_mean": out_clean["gate"].detach().mean(),
        "sgc/gate_sat_mean": out_sat["gate"].detach().mean(),
        "sgc/delta_z_ratio_mean": out_sat["delta_z_ratio"].detach().mean(),
        "sgc/delta_z_ratio_p95": torch.quantile(out_sat["delta_z_ratio"].detach().float(), 0.95),
        "sgc/delta_logit_norm_mean": out_sat["delta_logit_norm"].detach().mean(),
        "sgc/wrong_to_right": flips["wrong_to_right_rate"].detach(),
        "sgc/right_to_wrong": flips["right_to_wrong_rate"].detach(),
        "sgc/net_gain": flips["net_gain"].detach(),
        "sgc/top1_flip_rate": flips["top1_flip_rate"].detach(),
    }
    logs.update(target_logs)
    return loss, logs
