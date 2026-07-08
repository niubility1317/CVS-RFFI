from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def _assigned_labels_and_weights(assigned_targets: dict[int, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.tensor(sorted(assigned_targets), dtype=torch.long, device=device)
    weights = torch.stack([assigned_targets[int(label)].to(device=device) for label in labels.tolist()], dim=0)
    return labels, _normalize(weights)


def pseudo_target_cross_entropy(
    features: torch.Tensor,
    labels: torch.Tensor,
    assigned_targets: dict[int, torch.Tensor],
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, feature_dim]")
    label_ids, weights = _assigned_labels_and_weights(assigned_targets, features.device)
    logits = _normalize(features) @ weights.t()
    target = torch.empty_like(labels, dtype=torch.long)
    for compact_index, label in enumerate(label_ids.tolist()):
        target[labels == int(label)] = compact_index
    if not torch.isin(labels, label_ids).all():
        raise ValueError("labels contain classes without assigned pseudo targets")
    return F.cross_entropy(logits, target)


def _log_prob(anchor: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, temperature: float) -> torch.Tensor:
    if positives.numel() == 0:
        raise ValueError("contrastive positives must not be empty")
    if negatives.numel() == 0:
        raise ValueError("contrastive negatives must not be empty")
    pos_logits = positives @ anchor / temperature
    neg_logits = negatives @ anchor / temperature
    return -(pos_logits - torch.logsumexp(neg_logits, dim=0)).mean()


def supervised_anchor_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    assigned_targets: dict[int, torch.Tensor],
    pseudo_targets: torch.Tensor,
    perturbed_targets: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = _normalize(features)
    all_targets = _normalize(pseudo_targets.to(device=features.device))
    all_perturbed = _normalize(perturbed_targets.to(device=features.device))
    assigned_values = {label: target.to(device=features.device) for label, target in assigned_targets.items()}
    losses = []

    for idx, (anchor, label_tensor) in enumerate(zip(z, labels)):
        label = int(label_tensor.item())
        same_mask = labels == label
        same_mask[idx] = False
        positives = [z[same_mask]]
        positives.append(_normalize(assigned_values[label].view(1, -1)))
        target_index = sorted(assigned_targets).index(label)
        positives.append(all_perturbed[target_index : target_index + 1])
        pos = torch.cat([p for p in positives if p.numel() > 0], dim=0)
        neg = z[labels != label]
        losses.append(_log_prob(anchor, pos, neg, temperature))

    assigned_count = len(assigned_targets)
    for target_index in range(assigned_count, all_targets.size(0)):
        anchor = all_targets[target_index]
        pos = all_perturbed[target_index : target_index + 1]
        neg = torch.cat([z, all_targets[:assigned_count], all_perturbed[:assigned_count]], dim=0)
        losses.append(_log_prob(anchor, pos, neg, temperature))

    return torch.stack(losses).mean()


def class_center_separation_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    assigned_targets: dict[int, torch.Tensor],
    pseudo_targets: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    centers = []
    device = features.device
    for label in sorted(assigned_targets):
        mask = labels == int(label)
        if bool(mask.any()):
            centers.append(features[mask].mean(dim=0))
        else:
            centers.append(assigned_targets[label].to(device=device))
    unassigned = pseudo_targets[len(assigned_targets) :].to(device=device)
    if unassigned.numel() > 0:
        centers.extend(unassigned)
    m = _normalize(torch.stack(centers, dim=0))
    if m.size(0) <= 1:
        return m.new_tensor(0.0)
    sim = m @ m.t()
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    logits = sim[mask].view(sim.size(0), sim.size(0) - 1) / temperature
    return torch.logsumexp(logits, dim=1).mean()


def base_training_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    assigned_targets: dict[int, torch.Tensor],
    pseudo_targets: torch.Tensor,
    perturbed_targets: torch.Tensor,
    *,
    contrast_temperature: float = 0.1,
    center_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ce = pseudo_target_cross_entropy(features, labels, assigned_targets)
    contrastive = supervised_anchor_contrastive_loss(
        features,
        labels,
        assigned_targets,
        pseudo_targets,
        perturbed_targets,
        temperature=contrast_temperature,
    )
    center = class_center_separation_loss(
        features,
        labels,
        assigned_targets,
        pseudo_targets,
        temperature=center_temperature,
    )
    total = ce + contrastive + center
    return total, {"ce": ce.detach(), "contrastive": contrastive.detach(), "center": center.detach(), "total": total.detach()}


def incremental_calibration_loss(
    new_features: torch.Tensor,
    new_labels: torch.Tensor,
    old_weights: torch.Tensor,
    new_weights: torch.Tensor,
    *,
    new_class_ids: torch.Tensor,
    prototypes: torch.Tensor,
    top_k: int = 60,
    margin: float = 0.2,
    tau_fuse: float = 0.01,
    lambda_align: float = 1.6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if tau_fuse <= 0:
        raise ValueError("tau_fuse must be positive")
    device = new_features.device
    if prototypes.shape != new_weights.shape:
        raise ValueError("prototypes and new_weights must have matching shape")
    if new_class_ids.numel() != new_weights.size(0):
        raise ValueError("new_class_ids must have one id per new weight row")
    z = _normalize(new_features)
    old_w = _normalize(old_weights.to(device=device))
    new_w = _normalize(new_weights.to(device=device))
    all_w = torch.cat([old_w, new_w], dim=0)
    old_count = old_w.size(0)
    class_to_row = {int(label): idx for idx, label in enumerate(new_class_ids.detach().cpu().tolist())}

    margin_terms = []
    for row, label_tensor in enumerate(new_labels.detach().cpu().tolist()):
        class_row = class_to_row[int(label_tensor)]
        true_index = old_count + class_row
        scores = z[row] @ all_w.t()
        self_score = scores[true_index]
        competitor_scores = torch.cat([scores[:true_index], scores[true_index + 1 :]], dim=0)
        k = min(int(top_k), competitor_scores.numel())
        top_scores = torch.topk(competitor_scores, k=k).values
        comp_score = tau_fuse * torch.logsumexp(top_scores / tau_fuse, dim=0)
        margin_terms.append(F.relu(comp_score - self_score + margin))

    margin_tensor = torch.stack(margin_terms)
    hard_mask = margin_tensor > 0
    margin_loss = margin_tensor[hard_mask].mean() if bool(hard_mask.any()) else margin_tensor.sum() * 0.0
    proto = _normalize(prototypes.to(device=device))
    align_loss = (1.0 - (proto * new_w).sum(dim=1)).mean()
    total = margin_loss + float(lambda_align) * align_loss
    terms = {
        "margin": margin_loss.detach(),
        "align": align_loss.detach(),
        "total": total.detach(),
        "hard_count": torch.tensor(float(hard_mask.sum().item()), device=new_weights.device),
    }
    return total, terms
