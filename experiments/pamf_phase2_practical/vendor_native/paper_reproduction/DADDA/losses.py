from __future__ import annotations

import torch
import torch.nn.functional as F


def _squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2).pow(2)


def estimate_rbf_bandwidth(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("kernel inputs must be 2-D tensors")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("kernel inputs must have matching feature dimension")
    with torch.no_grad():
        combined = torch.cat([source_features, target_features], dim=0)
        pairwise = _squared_distances(combined, combined)
        positive = pairwise[pairwise > 0]
        sigma2 = positive.median() if positive.numel() else torch.tensor(1.0, device=source_features.device, dtype=source_features.dtype)
        return sigma2.clamp_min(1e-6)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, *, bandwidth: float | torch.Tensor | None = None) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("kernel inputs must be 2-D tensors")
    if x.shape[1] != y.shape[1]:
        raise ValueError("kernel inputs must have matching feature dimension")
    distances = _squared_distances(x, y)
    sigma2 = estimate_rbf_bandwidth(x, y) if bandwidth is None else torch.as_tensor(bandwidth, dtype=x.dtype, device=x.device).clamp_min(1e-6)
    return torch.exp(-distances / (2.0 * sigma2))


def mmd_loss(source_features: torch.Tensor, target_features: torch.Tensor, *, bandwidth: float | None = None) -> torch.Tensor:
    """Paper Eq. (2): global maximum mean discrepancy between source and target."""
    if source_features.numel() == 0 or target_features.numel() == 0:
        raise ValueError("MMD requires non-empty source and target features")
    resolved_bandwidth = estimate_rbf_bandwidth(source_features, target_features) if bandwidth is None else bandwidth
    k_ss = rbf_kernel(source_features, source_features, bandwidth=resolved_bandwidth).mean()
    k_tt = rbf_kernel(target_features, target_features, bandwidth=resolved_bandwidth).mean()
    k_st = rbf_kernel(source_features, target_features, bandwidth=resolved_bandwidth).mean()
    return (k_ss + k_tt - 2.0 * k_st).clamp_min(0.0)


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=int(num_classes)).to(dtype=torch.float32, device=labels.device)


def _as_probabilities(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if (values >= -eps).all() and torch.allclose(values.sum(dim=1), torch.ones_like(values.sum(dim=1)), atol=1e-4):
        return values.clamp_min(0.0)
    return F.softmax(values, dim=1)


def _class_weights(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return probabilities / probabilities.sum(dim=0, keepdim=True).clamp_min(eps)


def lmmd_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_logits_or_probs: torch.Tensor,
    *,
    num_classes: int | None = None,
    bandwidth: float | None = None,
    reduction: str = "mean",
    target_is_probabilities: bool | None = None,
    detach_target_probabilities: bool = False,
) -> torch.Tensor:
    """Paper Eq. (3)-(4): local MMD weighted by source labels and target soft labels."""
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("LMMD features must be 2-D tensors")
    if source_features.shape[0] != source_labels.shape[0]:
        raise ValueError("source_labels must match source feature batch")
    if target_features.shape[0] != target_logits_or_probs.shape[0]:
        raise ValueError("target probabilities must match target feature batch")
    resolved_classes = int(num_classes or target_logits_or_probs.shape[1])
    source_probs = _one_hot(source_labels, resolved_classes)
    if target_logits_or_probs.ndim != 2 or target_logits_or_probs.shape[1] != resolved_classes:
        raise ValueError("target logits/probabilities must have shape [batch,num_classes]")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")
    if target_is_probabilities is True:
        target_probs = target_logits_or_probs.clamp_min(0.0)
    elif target_is_probabilities is False:
        target_probs = F.softmax(target_logits_or_probs, dim=1)
    else:
        target_probs = _as_probabilities(target_logits_or_probs)
    if detach_target_probabilities:
        target_probs = target_probs.detach()
    source_weights = _class_weights(source_probs)
    target_weights = _class_weights(target_probs)
    resolved_bandwidth = estimate_rbf_bandwidth(source_features, target_features) if bandwidth is None else bandwidth
    k_ss = rbf_kernel(source_features, source_features, bandwidth=resolved_bandwidth)
    k_tt = rbf_kernel(target_features, target_features, bandwidth=resolved_bandwidth)
    k_st = rbf_kernel(source_features, target_features, bandwidth=resolved_bandwidth)
    losses = []
    for class_i in range(resolved_classes):
        ws = source_weights[:, class_i]
        wt = target_weights[:, class_i]
        source_mass = float(source_probs[:, class_i].sum().detach().cpu())
        target_mass = float(target_probs[:, class_i].sum().detach().cpu())
        if source_mass <= 0.0 or target_mass <= 0.0:
            continue
        term = ws @ k_ss @ ws + wt @ k_tt @ wt - 2.0 * (ws @ k_st @ wt)
        losses.append(term)
    if not losses:
        return source_features.sum() * 0.0
    stacked = torch.stack(losses)
    if reduction == "sum":
        return stacked.sum().clamp_min(0.0)
    return stacked.mean().clamp_min(0.0)


def dynamic_adaptive_factor(global_mmd: torch.Tensor, local_lmmd: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Paper Eq. (5): alpha balances global MMD and the class-wise LMMD sum."""
    alpha = global_mmd / (global_mmd + local_lmmd + float(eps))
    return alpha.clamp(0.0, 1.0)


def dadda_objective(
    source_outputs: dict[str, torch.Tensor],
    target_outputs: dict[str, torch.Tensor],
    source_labels: torch.Tensor,
    *,
    tradeoff_lambda: float,
    bandwidth: float | None = None,
    alpha_mode: str = "dynamic",
    fixed_alpha: float = 0.5,
    detach_target_probabilities: bool = False,
    detach_dynamic_alpha: bool = False,
) -> dict[str, torch.Tensor]:
    """Paper Eq. (6)-(9): CE plus dynamic MMD/LMMD alignment."""
    ce = F.cross_entropy(source_outputs["logits"], source_labels.long())
    global_mmd = mmd_loss(source_outputs["global_features"], target_outputs["global_features"], bandwidth=bandwidth)
    local_lmmd = lmmd_loss(
        source_outputs["local_features"],
        target_outputs["local_features"],
        source_labels,
        target_outputs["logits"],
        num_classes=source_outputs["logits"].shape[1],
        bandwidth=bandwidth,
        target_is_probabilities=False,
        detach_target_probabilities=detach_target_probabilities,
    )
    local_lmmd_sum = lmmd_loss(
        source_outputs["local_features"],
        target_outputs["local_features"],
        source_labels,
        target_outputs["logits"],
        num_classes=source_outputs["logits"].shape[1],
        bandwidth=bandwidth,
        reduction="sum",
        target_is_probabilities=False,
        detach_target_probabilities=detach_target_probabilities,
    )
    if alpha_mode == "dynamic":
        alpha = dynamic_adaptive_factor(global_mmd, local_lmmd_sum)
        if detach_dynamic_alpha:
            alpha = alpha.detach()
    elif alpha_mode == "fixed":
        alpha = torch.as_tensor(float(fixed_alpha), dtype=global_mmd.dtype, device=global_mmd.device).clamp(0.0, 1.0)
    else:
        raise ValueError("alpha_mode must be 'dynamic' or 'fixed'")
    # Eq. (5) defines alpha against the class-wise LMMD sum; use the same local
    # distance in Eq. (9) so local alignment is not reduced by the class count.
    dynamic_joint = (1.0 - alpha) * global_mmd + alpha * local_lmmd_sum
    total = ce + float(tradeoff_lambda) * dynamic_joint
    return {
        "loss": total,
        "cross_entropy": ce.detach(),
        "mmd": global_mmd.detach(),
        "lmmd": local_lmmd.detach(),
        "lmmd_sum": local_lmmd_sum.detach(),
        "alpha": alpha.detach(),
        "dynamic_joint": dynamic_joint.detach(),
    }
