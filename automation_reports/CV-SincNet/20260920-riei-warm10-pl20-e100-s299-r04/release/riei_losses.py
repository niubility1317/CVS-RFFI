from __future__ import annotations

import torch
import torch.nn.functional as F


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def mutual_independence_loss(
    z_e: torch.Tensor,
    z_r: torch.Tensor,
    eps: float = 1e-8,
    mode: str = "cosine_abs",
    reduction: str = "mean",
) -> torch.Tensor:
    dim = min(z_e.size(1), z_r.size(1))
    ze = F.normalize(z_e[:, :dim], dim=1, eps=eps)
    zr = F.normalize(z_r[:, :dim], dim=1, eps=eps)
    if mode == "cosine_abs":
        return _reduce(torch.sum(ze * zr, dim=1).abs(), reduction)
    if mode == "cosine_square":
        return _reduce(torch.sum(ze * zr, dim=1).square(), reduction)
    if mode == "cross_cov":
        ze = ze - ze.mean(dim=0, keepdim=True)
        zr = zr - zr.mean(dim=0, keepdim=True)
        denom = max(1, ze.size(0) - 1)
        cov = ze.transpose(0, 1).matmul(zr) / float(denom)
        values = cov.square()
        return values.mean() if reduction == "mean" else _reduce(values, reduction)
    raise ValueError(f"Unsupported RIEI MI mode: {mode}")


def entropy_from_logits(
    logits: torch.Tensor,
    eps: float = 1e-8,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    temp = max(float(temperature), eps)
    probs = F.softmax(logits / temp, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)
    return _reduce(entropy, reduction)


def riei_total_loss(
    outputs: dict,
    emitter_labels: torch.Tensor,
    receiver_labels: torch.Tensor,
    lambda_mi: float = 1.2,
    lambda_ie: float = 1.2,
    mi_mode: str = "cosine_abs",
    ie_temperature: float = 1.0,
    ce_reduction: str = "mean",
    mi_reduction: str = "mean",
    ie_reduction: str = "mean",
):
    loss_ce_e = F.cross_entropy(outputs["emitter_logits"], emitter_labels, reduction=ce_reduction)
    loss_ce_r = F.cross_entropy(outputs["receiver_logits"], receiver_labels, reduction=ce_reduction)
    loss_ce = loss_ce_e + loss_ce_r
    loss_mi = mutual_independence_loss(outputs["z_e"], outputs["z_r"], mode=mi_mode, reduction=mi_reduction)
    loss_ie = entropy_from_logits(
        outputs["cross_emitter_logits"],
        temperature=ie_temperature,
        reduction=ie_reduction,
    ) + entropy_from_logits(
        outputs["cross_receiver_logits"],
        temperature=ie_temperature,
        reduction=ie_reduction,
    )
    loss = loss_ce + float(lambda_mi) * loss_mi - float(lambda_ie) * loss_ie
    return {
        "loss": loss,
        "loss_ce": loss_ce,
        "loss_ce_e": loss_ce_e,
        "loss_ce_r": loss_ce_r,
        "loss_mi": loss_mi,
        "loss_ie": loss_ie,
    }
