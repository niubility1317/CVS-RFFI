"""Source-only DURA-relative waveform protection for memory correction."""
import torch

from experiments.phase1_volterra_aligned.objective import aligned_iq_error, source_loss


def degradation_penalty(output, anchor, clean, groups):
    if groups.shape != (len(output),):
        raise ValueError('One receiver group per source waveform is required')
    candidate_error = aligned_iq_error(output, clean)
    with torch.no_grad():
        anchor_error = aligned_iq_error(anchor.detach(), clean)
    excess = (candidate_error - anchor_error).clamp_min(0)
    return torch.stack([excess[groups == group].mean() for group in groups.unique()]).mean()


def protected_source_objective(reader, output, anchor, clean, clean_features, groups):
    loss, terms = source_loss(reader, output, clean, clean_features, groups)
    penalty = degradation_penalty(output, anchor, clean, groups)
    return loss + 5.0 * penalty, dict(terms, degradation_penalty=penalty)
