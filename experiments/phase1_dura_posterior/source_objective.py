"""Source-only waveform and frozen-reader objectives for the DURA correction."""

import torch
from torch.nn import functional as F


def source_objective(reader, output, clean, baseline, clean_features, clean_logits,
                     baseline_logits, groups):
    """Balance source receivers and preserve baseline decisions when they agree with clean F."""
    n = len(output)
    if (reader.training or any(p.requires_grad for p in reader.parameters())
            or output.shape != clean.shape or baseline.shape != clean.shape
            or groups.shape != (n,) or clean_features.shape[0] != n
            or clean_logits.shape != baseline_logits.shape or clean_logits.shape[0] != n):
        raise ValueError('Frozen source reader and aligned source batches are required')
    clean = clean.detach()
    baseline = baseline.detach()
    clean_features = clean_features.detach()
    clean_logits = clean_logits.detach()
    baseline_logits = baseline_logits.detach()
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    wave = (output-clean).square().mean((1, 2))/energy
    trust = (output-baseline).square().mean((1, 2))/energy
    features = reader.features(output)
    logits = reader.classify_features(features)
    feature = 1-F.cosine_similarity(features, clean_features, dim=1, eps=1e-8)
    soft = F.softmax(clean_logits/2, dim=1)
    semantic = F.kl_div(F.log_softmax(logits/2, dim=1), soft, reduction='none').sum(1)*4
    baseline_logp = F.log_softmax(baseline_logits, dim=1)
    preserve = (baseline_logp.exp()*(baseline_logp-F.log_softmax(logits, dim=1))).sum(1)
    clean_probability = F.softmax(clean_logits, dim=1)
    agreed = (clean_probability.max(1).values >= 0.95) & (clean_logits.argmax(1) == baseline_logits.argmax(1))
    per = wave + 0.2*feature + 0.1*semantic + 0.01*trust + 0.05*preserve*agreed.to(preserve)
    loss = torch.stack([per[groups == group].mean() for group in groups.unique()]).mean()
    return loss, dict(wave=wave, feature=feature, semantic=semantic, trust=trust,
                      preserve=preserve, preserve_enabled=agreed)
