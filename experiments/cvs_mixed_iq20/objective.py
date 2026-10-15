"""Reuse RDrop separately within each already-generated IQ view."""
import torch
from torch.nn import functional as F


def rdrop_per_sample(first, second, labels):
    if (labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long
            or first.shape != second.shape or first.shape != (len(labels), 6)
            or not bool(((labels >= 0) & (labels < 6)).all())
            or not bool(torch.isfinite(first).all())
            or not bool(torch.isfinite(second).all())):
        raise ValueError('Two finite default-head logits and legal L TX labels required')
    ce = .5 * (F.cross_entropy(first, labels, reduction='none', label_smoothing=.01)
               + F.cross_entropy(second, labels, reduction='none', label_smoothing=.01))
    p, q = first.log_softmax(1), second.log_softmax(1)
    symmetric_kl = .5 * ((p.exp() - q.exp()) * (p - q)).sum(1)
    return ce + symmetric_kl, ce, symmetric_kl


def mixed_objective(clean_pair, generalization_pair, satellite_pair, labels):
    """.25 RDrop(raw) + .25 RDrop(g) + .5 RDrop(s), equal sample weights.

    Each pair must be produced from the same IQ; view generation belongs outside
    this function. There is no clean/urban KL and no student-confidence gate.
    """
    clean, clean_ce, clean_kl = rdrop_per_sample(*clean_pair, labels)
    general, general_ce, general_kl = rdrop_per_sample(*generalization_pair, labels)
    sat, sat_ce, sat_kl = rdrop_per_sample(*satellite_pair, labels)
    total = (.25 * clean + .25 * general + .5 * sat).mean()
    return total, dict(clean_CE=clean_ce.mean(), generalization_CE=general_ce.mean(),
                       satellite_CE=sat_ce.mean(), clean_RDrop_KL=clean_kl.mean(),
                       generalization_RDrop_KL=general_kl.mean(), satellite_RDrop_KL=sat_kl.mean(),
                       total_L_scale=1., physical_packets=len(labels),
                       labeled_forward_packets=6 * len(labels))
