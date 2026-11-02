"""Frozen-reader source disagreement emphasis with DURA waveform protection."""
import torch

from experiments.phase1_volterra_aligned.objective import aligned_iq_error, source_loss

SOURCE_WEIGHT_COUNTS = [0, 0]


@torch.no_grad()
def difficult_weights(reader, anchor, clean_features):
    clean_logits = reader.classify_features(clean_features.detach())
    anchor_logits = reader.classify_features(reader.features(anchor.detach()))
    if (clean_logits.shape != anchor_logits.shape or clean_logits.ndim != 2
            or clean_logits.shape[1] != 6
            or not torch.isfinite(clean_logits).all() or not torch.isfinite(anchor_logits).all()):
        raise ValueError('Finite six-class frozen-reader predictions are required')
    confidence, clean_class = clean_logits.softmax(1).max(1)
    disagreement = (confidence >= 0.95) & (anchor_logits.argmax(1) != clean_class)
    return 1.0 + 7.0 * disagreement.to(anchor_logits.dtype)


def group_weighted_mean(per, weights, groups):
    if per.ndim != 1 or weights.shape != per.shape or groups.shape != per.shape or len(per) == 0:
        raise ValueError('One loss, weight and receiver group per source example are required')
    if not torch.isfinite(per).all() or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Finite source losses and positive weights are required')
    return torch.stack([(per[groups == g] * weights[groups == g]).sum() / weights[groups == g].sum()
                        for g in groups.unique()]).mean()


def focused_source_objective(reader, output, anchor, clean, clean_features, groups):
    _, terms = source_loss(reader, output, clean, clean_features, groups)
    with torch.no_grad():
        weights = difficult_weights(reader, anchor, clean_features)
        SOURCE_WEIGHT_COUNTS[0] += len(weights)
        SOURCE_WEIGHT_COUNTS[1] += int((weights > 1).sum())
        baseline_error = aligned_iq_error(anchor.detach(), clean)
    degradation = (terms['aligned'] - baseline_error).clamp_min(0)
    per = (terms['aligned'] + 0.05 * terms['power'] + 0.2 * terms['feature']
           + 0.2 * terms['distill'] + 0.05 * terms['margin'] + 5.0 * degradation)
    return group_weighted_mean(per, weights, groups), dict(terms, degradation=degradation, weights=weights)
