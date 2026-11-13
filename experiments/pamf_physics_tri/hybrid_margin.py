"""Capped, detached raw-view classification margins for confident source U."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def source_margin_cap(features, labels, weights):
    """Use the median correct-label cosine margin on the unchanged labeled source."""
    scores = F.normalize(features, dim=1, eps=1e-4) @ F.normalize(weights, dim=1, eps=1e-4).T
    correct = scores.argmax(dim=1) == labels
    if not correct.any():
        raise ValueError('No correctly classified labeled source examples for margin cap')
    chosen = scores.gather(1, labels[:, None]).squeeze(1)
    other = scores.masked_fill(F.one_hot(labels, scores.shape[1]).bool(), -torch.inf).max(dim=1).values
    return torch.quantile((chosen - other)[correct], .5), int(correct.sum())


def capped_margin_loss(raw, augmented, weights, pseudo, accepted, maximum):
    """Penalize confident packets only when augmentation loses their raw margin."""
    weights = F.normalize(weights.detach(), dim=1, eps=1e-4)

    def margin(features):
        logits = F.normalize(features, dim=1, eps=1e-4) @ weights.T
        selected = logits.gather(1, pseudo[:, None]).squeeze(1)
        other = logits.masked_fill(F.one_hot(pseudo, logits.shape[1]).bool(), -torch.inf).max(dim=1).values
        return selected - other

    reference = margin(raw.detach()).clamp(min=0., max=maximum).detach()
    altered = margin(augmented)
    violation = (reference - altered).clamp_min(0.) * accepted
    return violation.mean(), dict(reference=reference, augmented_margin=altered.detach(),
                                  violating=(violation > 0).detach(), amount=violation.detach())
