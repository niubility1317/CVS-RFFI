"""Source-only bounded reader-stability loss with no identity labels."""
import torch
from torch.nn import functional as F
from experiments.phase1_dura_safe_memory.objective import protected_source_objective


def adversarial_reader_loss(reader, output, clean_features, groups, epsilon=0.01):
    if (output.ndim != 3 or output.shape[1:] != (2, 256) or groups.shape != (len(output),)
            or not output.requires_grad or not 0 <= epsilon <= 1):
        raise ValueError('Differentiable IQ windows, source groups and bounded epsilon required')
    with torch.no_grad():
        teacher = reader.classify_features(clean_features).softmax(1)
        confidence, pseudo = teacher.max(1)
    logits = reader.classify_features(reader.features(output))
    loss0 = F.cross_entropy(logits, pseudo, reduction='sum')
    gradient = torch.autograd.grad(loss0, output, retain_graph=True, create_graph=False)[0]
    grad_rms = gradient.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    amplitude = output.detach().square().mean((1, 2), keepdim=True).sqrt()
    delta = (epsilon * amplitude * gradient / grad_rms).detach()
    perturbed_logits = reader.classify_features(reader.features(output + delta))
    per = F.cross_entropy(perturbed_logits, pseudo, reduction='none')
    group_losses = []
    for group in groups.unique():
        selected = groups == group
        weight = confidence[selected]
        group_losses.append((per[selected] * weight).sum() / weight.sum().clamp_min(1e-12))
    robust_loss = torch.stack(group_losses).mean()
    if not torch.isfinite(robust_loss) or not torch.isfinite(delta).all():
        raise FloatingPointError('Finite source reader stability loss required')
    return robust_loss, delta


def robust_source_objective(reader, output, anchor, clean, clean_features, groups):
    protected, terms = protected_source_objective(reader, output, anchor, clean, clean_features, groups)
    stability, _ = adversarial_reader_loss(reader, output, clean_features, groups)
    return protected + 0.1 * stability, dict(terms, reader_stability=stability)
