"""Vectorized equivalent of the existing project contrastive formulation.

This preserves the project formulation, including normalization and averaging;
it is not independent evidence of numerical or scientific parity with the paper.
"""
import torch
import torch.nn.functional as F


def contrastive_loss(features, labels, assigned_targets, pseudo_targets, perturbed_targets, *, temperature=0.1):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if features.ndim != 2 or labels.ndim != 1 or labels.numel() != features.size(0):
        raise ValueError('incompatible feature or label shape')
    if pseudo_targets.ndim != 2 or perturbed_targets.shape != pseudo_targets.shape:
        raise ValueError('incompatible pseudo-target shape')
    if pseudo_targets.size(1) != features.size(1):
        raise ValueError('pseudo-target feature dimension mismatch')
    label_ids = torch.tensor(list(assigned_targets), device=labels.device, dtype=labels.dtype)
    if not torch.isin(labels, label_ids).all():
        raise ValueError('labels contain classes without assigned pseudo targets')
    if torch.unique(labels).numel() < 2:
        raise ValueError('contrastive batch must contain at least two classes')
    count = len(assigned_targets)
    if count > pseudo_targets.size(0):
        raise ValueError('too few pseudo targets')
    z = F.normalize(features.float(), dim=1)
    targets = F.normalize(pseudo_targets.to(features.device).float(), dim=1)
    perturb = F.normalize(perturbed_targets.to(features.device).float(), dim=1)
    assigned = F.normalize(torch.stack([t.to(features.device) for t in assigned_targets.values()]).float(), dim=1)
    same = labels[:, None] == labels[None, :]
    label_index = (labels[:, None] == label_ids[None, :]).to(torch.int64).argmax(dim=1)
    scores = (z @ z.t()) / temperature
    negatives = torch.logsumexp(scores.masked_fill(same, -torch.inf), dim=1)
    same_sum = scores.masked_fill(~same, 0).sum(dim=1)
    assigned_scores = (z * assigned[label_index]).sum(dim=1) / temperature
    perturb_scores = (z * perturb[label_index]).sum(dim=1) / temperature
    positive_mean = (same_sum + assigned_scores + perturb_scores) / (same.sum(dim=1) + 2)
    sample_losses = negatives - positive_mean
    unused = targets[count:]
    if unused.size(0):
        unused_negatives = torch.cat([z, targets[:count], perturb[:count]], dim=0)
        neg_logits = unused @ unused_negatives.t() / temperature
        pos_logits = (unused * perturb[count:]).sum(dim=1) / temperature
        reserved_losses = torch.logsumexp(neg_logits, dim=1) - pos_logits
        return torch.cat([sample_losses, reserved_losses]).mean()
    return sample_losses.mean()
