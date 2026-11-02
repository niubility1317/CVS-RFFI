"""Receiver-balanced paired-source loss with frozen reader features."""

import torch
from torch.nn import functional as F


def source_loss(reader, output, clean, clean_features, groups):
    n = len(output)
    if (reader.training or any(p.requires_grad for p in reader.parameters())
            or clean.shape != output.shape or groups.shape != (n,)
            or clean_features.shape[0] != n):
        raise ValueError('Frozen reader and aligned paired-source examples are required')
    clean = clean.detach()
    features = reader.features(output)
    if not torch.isfinite(features).all():
        raise FloatingPointError('Finite frozen-reader features are required')
    energy = clean.flatten(1).square().sum(1).clamp_min(1e-8)
    wave = (output-clean).flatten(1).square().sum(1)/energy
    feature = 1-F.cosine_similarity(features, clean_features.detach(), dim=1, eps=1e-8)
    per = wave+0.2*feature
    loss = torch.stack([per[groups == group].mean() for group in groups.unique()]).mean()
    return loss, dict(wave=wave, feature=feature)
