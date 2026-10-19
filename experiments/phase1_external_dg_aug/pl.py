"""Ordinary hard pseudo-label confidence masking, without a teacher or U truth."""
import torch
import torch.nn.functional as F


def pseudo_targets(net, identity_forward, x, threshold=.95):
    if threshold != .95:
        raise ValueError('PL threshold was fixed before target access')
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            logits = identity_forward(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite PL confidence')
            confidence, labels = logits.softmax(1).max(1)
            accepted = confidence >= threshold
    finally:
        net.train(was_training)
    return labels.detach(), accepted.detach(), confidence.detach()


def masked_pl_loss(logits, labels, accepted):
    if len(logits) != len(labels) or accepted.shape != labels.shape:
        raise ValueError('PL batch shape mismatch')
    # Whole-U-batch normalization makes the effective contribution track coverage.
    return (F.cross_entropy(logits, labels.detach(), reduction='none') * accepted.detach().to(logits.dtype)).mean()
