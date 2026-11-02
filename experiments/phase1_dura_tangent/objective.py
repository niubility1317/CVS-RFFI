"""Source-only local feature-response matching without identity supervision."""
import torch
from torch.nn import functional as F

from experiments.phase1_dura_safe_memory.objective import protected_source_objective


def tangent_consistency_loss(reader, output, clean, clean_features, groups, *, direction=None, epsilon=0.02):
    if (reader.training or any(p.requires_grad for p in reader.parameters())
            or output.ndim != 3 or output.shape[1:] != (2, 256)
            or clean.shape != output.shape or groups.shape != (len(output),)
            or clean_features.shape[0] != len(output) or epsilon <= 0):
        raise ValueError('Frozen reader, paired IQ and receiver groups required')
    clean = clean.detach()
    if direction is None:
        direction = torch.randn_like(clean)
    if direction.shape != clean.shape or not torch.isfinite(direction).all():
        raise ValueError('A finite perturbation direction per source waveform is required')
    direction = direction.detach()
    direction_rms = direction.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    clean_rms = clean.square().mean((1, 2), keepdim=True).sqrt()
    delta = epsilon * clean_rms * direction / direction_rms
    with torch.no_grad():
        teacher_base = F.normalize(clean_features.detach(), dim=1, eps=1e-8)
        teacher_perturbed = F.normalize(reader.features(clean + delta), dim=1, eps=1e-8)
        teacher_response = (teacher_perturbed - teacher_base) / epsilon
    student_base = F.normalize(reader.features(output), dim=1, eps=1e-8)
    student_perturbed = F.normalize(reader.features(output + delta), dim=1, eps=1e-8)
    student_response = (student_perturbed - student_base) / epsilon
    per = (student_response - teacher_response).square().sum(dim=1)
    loss = torch.stack([per[groups == group].mean() for group in groups.unique()]).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Finite feature-response consistency loss required')
    return loss


def tangent_source_objective(reader, output, anchor, clean, clean_features, groups):
    protected, terms = protected_source_objective(reader, output, anchor, clean, clean_features, groups)
    tangent = tangent_consistency_loss(reader, output, clean, clean_features, groups)
    return protected + 0.2 * tangent, dict(terms, feature_response=tangent)
