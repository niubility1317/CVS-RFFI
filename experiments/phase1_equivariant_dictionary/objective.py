"""Receiver-balanced source loss with optional legal L_s supervision."""

import torch
from torch.nn import functional as F


def source_loss(reader, output, clean, clean_features, clean_soft, labels, groups):
    n = len(output)
    if (reader.training or any(p.requires_grad for p in reader.parameters())
            or clean.shape != output.shape or labels.shape != (n,) or groups.shape != (n,)
            or clean_features.shape[0] != n or clean_soft.shape != (n, 6)
            or (labels < -1).any() or (labels >= 6).any()):
        raise ValueError('Frozen reader, aligned source examples and legal labels are required')
    clean = clean.detach()
    features = reader.features(output)
    logits = reader.classify_features(features)
    if logits.shape != (n, 6) or not torch.isfinite(logits).all():
        raise FloatingPointError('Finite six-class frozen-reader scores are required')
    energy = clean.flatten(1).square().sum(1).clamp_min(1e-8)
    wave = (output-clean).flatten(1).square().sum(1)/energy
    feature = 1-F.cosine_similarity(features, clean_features.detach(), dim=1, eps=1e-8)
    semantic = F.kl_div(F.log_softmax(logits/2, dim=1), clean_soft.detach(), reduction='none').sum(1)*4
    per = wave+0.2*feature+0.1*semantic
    loss = torch.stack([per[groups == group].mean() for group in groups.unique()]).mean()
    known = labels >= 0
    supervised = F.cross_entropy(logits[known], labels[known].long()) if known.any() else logits.new_zeros(())
    loss = loss+0.2*supervised
    return loss, dict(wave=wave, feature=feature, semantic=semantic, supervised=supervised)
