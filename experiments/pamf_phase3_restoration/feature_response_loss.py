"""Continuous source feature response preservation without classifier outputs."""
import torch


def feature_response_loss(base, changed, source, changed_source, scale):
    """Compare paired source perturbation responses in fixed variance units.

    A fixed 1%-of-reference feature RMS floor limits small-denominator
    amplification. Reference features and source-only scales are not optimized.
    """
    if (base.ndim != 2 or base.shape[1] != 160 or changed.shape != base.shape
            or source.shape != base.shape or changed_source.shape != base.shape
            or scale.shape != (160,)):
        raise ValueError('Matched original fused160 feature responses required')
    if not all(torch.isfinite(value).all() for value in (base, changed, source, changed_source, scale)):
        raise ValueError('Finite source feature responses required')
    if not (scale > 0).all():
        raise ValueError('Positive source feature variances required')
    source = source.detach()
    reference = changed_source.detach() - source
    variance = scale.detach()
    numerator = ((changed - base - reference).square() / variance).mean(1)
    denominator = ((reference.square() / variance).mean(1)
                   + 1e-4 * (source.square() / variance).mean(1)).clamp_min(1e-12)
    return numerator / denominator
