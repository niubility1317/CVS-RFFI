from __future__ import annotations

import torch
import torch.nn.functional as F


def mutual_independence_loss(z_e: torch.Tensor, z_r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dim = min(z_e.size(1), z_r.size(1))
    ze = F.normalize(z_e[:, :dim], dim=1, eps=eps)
    zr = F.normalize(z_r[:, :dim], dim=1, eps=eps)
    return torch.sum(ze * zr, dim=1).mean()


def entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1).mean()


def riei_total_loss(
    outputs: dict,
    emitter_labels: torch.Tensor,
    receiver_labels: torch.Tensor,
    lambda_mi: float = 0.1,
    lambda_ie: float = 0.1,
):
    loss_ce_e = F.cross_entropy(outputs["emitter_logits"], emitter_labels)
    loss_ce_r = F.cross_entropy(outputs["receiver_logits"], receiver_labels)
    loss_ce = loss_ce_e + loss_ce_r
    loss_mi = mutual_independence_loss(outputs["z_e"], outputs["z_r"])
    loss_ie = entropy_from_logits(outputs["cross_emitter_logits"]) + entropy_from_logits(outputs["cross_receiver_logits"])
    loss = loss_ce + float(lambda_mi) * loss_mi - float(lambda_ie) * loss_ie
    return {
        "loss": loss,
        "loss_ce": loss_ce,
        "loss_ce_e": loss_ce_e,
        "loss_ce_r": loss_ce_r,
        "loss_mi": loss_mi,
        "loss_ie": loss_ie,
    }
