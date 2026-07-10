from __future__ import annotations

import torch
import torch.nn.functional as F


def dv_kl_domain_alignment(source_estimates: torch.Tensor, target_estimates: torch.Tensor) -> torch.Tensor:
    """Donsker-Varadhan KL estimate used for source/target domain alignment."""
    source = source_estimates.flatten()
    target = target_estimates.flatten()
    if source.numel() == 0 or target.numel() == 0:
        raise ValueError("source and target estimate batches must be non-empty")
    return source.mean() - torch.logsumexp(target, dim=0) + torch.log(
        torch.as_tensor(float(target.numel()), dtype=target.dtype, device=target.device)
    )


def mine_kl_stabilized_objective(
    source_estimates: torch.Tensor,
    target_estimates: torch.Tensor,
    *,
    ma_et: torch.Tensor | float = 1.0,
    ma_rate: float = 0.01,
    eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Official MINE objective used by the released trainer.

    The released code logs the DV estimate, but optimizes the moving-average
    surrogate from MINE to reduce instability in the exponential target term.
    """
    source = source_estimates.flatten()
    target = target_estimates.flatten()
    if source.numel() == 0 or target.numel() == 0:
        raise ValueError("source and target estimate batches must be non-empty")
    if not (0.0 < float(ma_rate) <= 1.0):
        raise ValueError("ma_rate must be in (0,1]")
    source_mean = source.mean()
    target_exp_mean = torch.exp(target).mean()
    ma_prev = torch.as_tensor(ma_et, dtype=target_exp_mean.dtype, device=target_exp_mean.device)
    ma_next = (1.0 - float(ma_rate)) * ma_prev + float(ma_rate) * target_exp_mean
    kl = source_mean - torch.log(target_exp_mean + float(eps))
    loss = source_mean - (1.0 / ma_next.mean()).detach() * target_exp_mean
    return {"kl": kl, "ma_et": ma_next.detach(), "loss": loss}


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
    smoothing: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
    mean_normalize: bool = False,
) -> torch.Tensor:
    """Class weighting from prior probability over estimated target class frequency."""
    counts = predicted_counts.float()
    if counts.ndim != 1:
        raise ValueError("predicted_counts must be a 1-D tensor")
    total = float(total_seen)
    if total <= 0:
        raise ValueError("total_seen must be positive")
    if float(smoothing) < 0.0:
        raise ValueError("smoothing must be non-negative")
    if (clip_min is None) != (clip_max is None):
        raise ValueError("clip_min and clip_max must be provided together")
    if clip_min is not None and not (0.0 < float(clip_min) <= float(clip_max)):
        raise ValueError("clip_min and clip_max must satisfy 0 < clip_min <= clip_max")
    if prior is None:
        prior_probs = torch.full_like(counts, 1.0 / max(counts.numel(), 1))
    else:
        prior_probs = prior.float().to(counts.device)
        if prior_probs.shape != counts.shape:
            raise ValueError("prior must match predicted_counts")
        prior_probs = prior_probs / prior_probs.sum().clamp_min(eps)
    smoothed_counts = counts + float(smoothing)
    estimated_total = total + float(smoothing) * float(counts.numel())
    estimated = smoothed_counts / torch.as_tensor(estimated_total, dtype=counts.dtype, device=counts.device).clamp_min(eps)
    weights = prior_probs / estimated.clamp_min(eps)
    if clip_min is not None:
        weights = weights.clamp(min=float(clip_min), max=float(clip_max))
    if mean_normalize:
        weights = weights / weights.mean().clamp_min(eps)
    return weights


def _weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    reduction_mode: str,
) -> torch.Tensor:
    normalized_mode = str(reduction_mode).strip().lower()
    if normalized_mode == "pytorch_weighted_mean":
        return F.cross_entropy(logits, labels.long(), weight=class_weights.to(logits.device))
    if normalized_mode != "paper_sample_mean":
        raise ValueError("weighted_ce_reduction must be one of: paper_sample_mean, pytorch_weighted_mean")
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
    kl_loss_override: torch.Tensor | None = None,
    weighted_ce_reduction: str = "paper_sample_mean",
) -> dict[str, torch.Tensor]:
    """Paper Eq.10-Eq.11 objective for E/C minimization with T's DV-KL term."""
    if not (0.0 < float(mu) < 1.0):
        raise ValueError("mu must be in (0,1)")
    loss_source = _weighted_ce(
        source_outputs["tx_logits"],
        source_labels,
        class_weights,
        reduction_mode=weighted_ce_reduction,
    )
    if target_mask.any():
        target_logits = target_outputs["tx_logits"][target_mask]
        target_labels = target_pseudo_labels.to(target_logits.device)[target_mask]
        loss_target = _weighted_ce(
            target_logits,
            target_labels,
            class_weights,
            reduction_mode=weighted_ce_reduction,
        )
    else:
        loss_target = source_outputs["tx_logits"].sum() * 0.0
    loss_weighted_ce = float(mu) * loss_source + (1.0 - float(mu)) * loss_target
    loss_kl = (
        dv_kl_domain_alignment(source_outputs["estimate_logits"], target_outputs["estimate_logits"])
        if kl_loss_override is None
        else kl_loss_override
    )
    loss = loss_weighted_ce + float(kl_weight) * loss_kl
    return {
        "loss": loss,
        "loss_weighted_ce": loss_weighted_ce,
        "loss_source": loss_source,
        "loss_target": loss_target,
        "loss_kl": loss_kl,
    }
