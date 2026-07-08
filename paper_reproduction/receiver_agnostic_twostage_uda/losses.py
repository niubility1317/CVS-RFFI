from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def transmitter_ce_loss(tx_logits: torch.Tensor, tx_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(tx_logits, tx_labels.long())


def domain_bce_loss(domain_logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
    labels = domain_labels.float().view_as(domain_logits)
    return F.binary_cross_entropy_with_logits(domain_logits, labels)


def dann_loss(
    source_outputs: dict[str, torch.Tensor],
    target_outputs: dict[str, torch.Tensor],
    source_labels: torch.Tensor,
    *,
    domain_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    source_domain = torch.zeros_like(source_outputs["domain_logits"])
    target_domain = torch.ones_like(target_outputs["domain_logits"])
    domain_logits = torch.cat([source_outputs["domain_logits"], target_outputs["domain_logits"]], dim=0)
    domain_labels = torch.cat([source_domain, target_domain], dim=0)
    loss_tx = transmitter_ce_loss(source_outputs["tx_logits"], source_labels)
    loss_domain = domain_bce_loss(domain_logits, domain_labels)
    return {
        "loss": loss_tx + float(domain_weight) * loss_domain,
        "loss_tx": loss_tx,
        "loss_domain": loss_domain,
    }


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError("features must include batch and feature dimensions")
    return features.flatten(1)


def _gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
) -> torch.Tensor:
    total = torch.cat([x, y], dim=0)
    dist = torch.cdist(total, total, p=2).pow(2)
    if fix_sigma is None:
        denom = max(total.shape[0] * total.shape[0] - total.shape[0], 1)
        bandwidth = dist.detach().sum() / denom
    else:
        bandwidth = torch.as_tensor(float(fix_sigma), dtype=dist.dtype, device=dist.device)
    bandwidth = bandwidth.clamp_min(1e-6) / (float(kernel_mul) ** (int(kernel_num) // 2))
    kernels = [torch.exp(-dist / (bandwidth * (float(kernel_mul) ** i))) for i in range(int(kernel_num))]
    return sum(kernels)


def _class_weights(labels_or_probs: torch.Tensor, num_classes: int, *, eps: float = 1e-8) -> torch.Tensor:
    if labels_or_probs.ndim == 1:
        probs = F.one_hot(labels_or_probs.long(), num_classes=int(num_classes)).float()
    elif labels_or_probs.ndim == 2:
        if labels_or_probs.shape[1] != int(num_classes):
            raise ValueError("probability columns must match num_classes")
        probs = labels_or_probs.float()
    else:
        raise ValueError("labels_or_probs must be 1-D labels or 2-D probabilities")
    return probs / probs.sum(dim=0, keepdim=True).clamp_min(eps)


def lmmd_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    num_classes: int,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Local MMD from Eq.14-Eq.16 using source labels and target predicted probabilities."""
    source = _flatten_features(source_features)
    target = _flatten_features(target_features)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    source_w = _class_weights(source_labels, num_classes, eps=eps).to(source.device)
    target_w = _class_weights(target_probs, num_classes, eps=eps).to(target.device)
    kernels = _gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    ns = source.shape[0]
    k_ss = kernels[:ns, :ns]
    k_tt = kernels[ns:, ns:]
    k_st = kernels[:ns, ns:]
    losses: list[torch.Tensor] = []
    for class_idx in range(int(num_classes)):
        ws = source_w[:, class_idx]
        wt = target_w[:, class_idx]
        if ws.sum() <= eps or wt.sum() <= eps:
            continue
        ss = (ws[:, None] * ws[None, :] * k_ss).sum()
        tt = (wt[:, None] * wt[None, :] * k_tt).sum()
        st = (ws[:, None] * wt[None, :] * k_st).sum()
        losses.append(ss + tt - 2.0 * st)
    if not losses:
        return source.sum() * 0.0
    return torch.stack(losses).mean().clamp_min(0.0)


def multi_layer_lmmd_loss(
    source_activations: Sequence[torch.Tensor],
    target_activations: Sequence[torch.Tensor],
    source_labels: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    num_classes: int,
    lmmd_weight: float = 1.0,
) -> torch.Tensor:
    if len(source_activations) != len(target_activations):
        raise ValueError("source and target activations must have the same number of layers")
    terms = [
        lmmd_loss(src, tgt, source_labels, target_probs, num_classes=num_classes)
        for src, tgt in zip(source_activations, target_activations)
    ]
    if not terms:
        raise ValueError("at least one activation layer is required for LMMD")
    return float(lmmd_weight) * torch.stack(terms).sum()


def dv_kl_domain_alignment(source_estimates: torch.Tensor, target_estimates: torch.Tensor) -> torch.Tensor:
    """Donsker-Varadhan KL estimate used for source/target domain alignment."""
    source = source_estimates.flatten()
    target = target_estimates.flatten()
    if source.numel() == 0 or target.numel() == 0:
        raise ValueError("source and target estimate batches must be non-empty")
    return source.mean() - torch.logsumexp(target, dim=0) + torch.log(
        torch.as_tensor(float(target.numel()), dtype=target.dtype, device=target.device)
    )


def curriculum_thresholds(pseudo_counts: torch.Tensor, *, base_tau: float = 0.7, eps: float = 1e-8) -> torch.Tensor:
    """CPL class thresholds: classes with more pseudo labels keep a higher threshold."""
    counts = pseudo_counts.float()
    if counts.ndim != 1:
        raise ValueError("pseudo_counts must be a 1-D tensor")
    if not (0.0 < float(base_tau) < 1.0):
        raise ValueError("base_tau must be in (0,1)")
    max_count = counts.max()
    if max_count <= eps:
        return torch.full_like(counts, float(base_tau))
    return (counts / max_count.clamp_min(eps) * float(base_tau)).clamp(min=eps, max=float(base_tau))


def adaptive_pseudo_labels(target_probs: torch.Tensor, thresholds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select target pseudo labels whose predicted class probability clears the CPL threshold."""
    if target_probs.ndim != 2:
        raise ValueError("target_probs must have shape [batch,num_classes]")
    if thresholds.ndim != 1 or thresholds.numel() != target_probs.shape[1]:
        raise ValueError("thresholds must have one value per class")
    confidence, labels = target_probs.max(dim=1)
    selected_threshold = thresholds.to(target_probs.device)[labels]
    return labels, confidence > selected_threshold


def class_balance_weights(
    predicted_counts: torch.Tensor,
    *,
    total_seen: int | float,
    prior: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Class weighting from prior probability over estimated target class frequency."""
    counts = predicted_counts.float()
    if counts.ndim != 1:
        raise ValueError("predicted_counts must be a 1-D tensor")
    total = float(total_seen)
    if total <= 0:
        raise ValueError("total_seen must be positive")
    if prior is None:
        prior_probs = torch.full_like(counts, 1.0 / max(counts.numel(), 1))
    else:
        prior_probs = prior.float().to(counts.device)
        if prior_probs.shape != counts.shape:
            raise ValueError("prior must match predicted_counts")
        prior_probs = prior_probs / prior_probs.sum().clamp_min(eps)
    estimated = counts / max(total, eps)
    weights = prior_probs / estimated.clamp_min(eps)
    return weights / weights.mean().clamp_min(eps)


def _weighted_ce(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels.long(), reduction="none")
    weights = class_weights.to(logits.device)[labels.long()]
    return (losses * weights).mean()


def gada_minimax_objective(
    source_outputs: dict[str, torch.Tensor],
    target_outputs: dict[str, torch.Tensor],
    *,
    source_labels: torch.Tensor,
    target_pseudo_labels: torch.Tensor,
    target_mask: torch.Tensor,
    class_weights: torch.Tensor,
    mu: float = 0.5,
    kl_weight: float = 0.005,
) -> dict[str, torch.Tensor]:
    """Paper Eq.10-Eq.11 objective for E/C minimization with T's DV-KL term."""
    if not (0.0 < float(mu) < 1.0):
        raise ValueError("mu must be in (0,1)")
    loss_source = _weighted_ce(source_outputs["tx_logits"], source_labels, class_weights)
    if target_mask.any():
        target_logits = target_outputs["tx_logits"][target_mask]
        target_labels = target_pseudo_labels.to(target_logits.device)[target_mask]
        loss_target = _weighted_ce(target_logits, target_labels, class_weights)
    else:
        loss_target = source_outputs["tx_logits"].sum() * 0.0
    loss_weighted_ce = float(mu) * loss_source + (1.0 - float(mu)) * loss_target
    loss_kl = dv_kl_domain_alignment(source_outputs["estimate_logits"], target_outputs["estimate_logits"])
    loss = loss_weighted_ce + float(kl_weight) * loss_kl
    return {
        "loss": loss,
        "loss_weighted_ce": loss_weighted_ce,
        "loss_source": loss_source,
        "loss_target": loss_target,
        "loss_kl": loss_kl,
    }
