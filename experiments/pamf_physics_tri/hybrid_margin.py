"""Capped, detached raw-view classification margins for confident source U."""
import torch
from torch.nn import functional as F


def capped_margin_loss(raw, augmented, weights, pseudo, accepted, maximum):
    """Penalize confident packets only when augmentation loses their raw margin."""
    weights = F.normalize(weights.detach(), dim=1)

    def margin(features):
        logits = F.normalize(features, dim=1) @ weights.T
        selected = logits.gather(1, pseudo[:, None]).squeeze(1)
        other = logits.masked_fill(F.one_hot(pseudo, logits.shape[1]).bool(), -torch.inf).max(dim=1).values
        return selected - other

    reference = margin(raw.detach()).clamp(min=0., max=maximum).detach()
    altered = margin(augmented)
    violation = (reference - altered).clamp_min(0.) * accepted
    return violation.mean(), dict(reference=reference, augmented_margin=altered.detach(),
                                  violating=(violation > 0).detach(), amount=violation.detach())
