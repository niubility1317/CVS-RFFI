"""Source-only oracle probe of a proposed physical innovation direction."""
import torch


def oracle_line_probe(anchor, clean, innovation, *, limit=0.5):
    """Measure bounded best per-record steps; these oracle steps are not deployable.

    The direction is normalized by its RMS and scaled to the anchor RMS, so
    the bound has the same meaning for all records.  No labels or models are
    used here; ``clean`` is available only for source-side diagnostics.
    """
    if (anchor.ndim != 3 or anchor.shape[1:] != (2, 256)
            or clean.shape != anchor.shape or innovation.shape != anchor.shape):
        raise ValueError('Matching 256-sample source IQ windows required')
    if not isinstance(limit, (int, float)) or not 0 < limit <= 1:
        raise ValueError('A positive bounded oracle step is required')
    if not all(torch.isfinite(x).all() for x in (anchor, clean, innovation)):
        raise FloatingPointError('Finite source waveforms required')
    axes = (1, 2)
    energy = clean.square().mean(axes).clamp_min(1e-12)
    base_scale = anchor.square().mean(axes, keepdim=True).sqrt()
    direction_scale = innovation.square().mean(axes, keepdim=True).sqrt()
    direction = innovation / direction_scale.clamp_min(1e-12) * base_scale
    denominator = direction.square().sum(axes)
    numerator = ((clean-anchor)*direction).sum(axes)
    alpha = torch.where(denominator > 1e-20, numerator / denominator.clamp_min(1e-20), 0.)
    positive = alpha.clamp(0, limit)
    signed = alpha.clamp(-limit, limit)

    def error(step):
        candidate = anchor + step[:, None, None] * direction
        return (candidate-clean).square().mean(axes) / energy

    return dict(anchor_error=error(torch.zeros_like(alpha)), positive_error=error(positive),
                signed_error=error(signed), optimal_alpha=alpha,
                innovation_rms=direction_scale.reshape(-1),
                positive_clipped=(alpha > limit), negative_clipped=(alpha < -limit))
