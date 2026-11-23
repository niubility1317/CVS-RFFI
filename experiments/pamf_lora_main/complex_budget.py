"""Compare complex-vector bandwidths at the same physical update amplitude."""

import torch


BUDGETS = {'rms02': .02, 'rms05': .05, 'rms10': .10, 'rms18': .18}
LEARNED_SCALES = {'1x': 1., '1_4x': 1.4, '2x': 2.}


def chirp_rms(value):
    return value.abs().square().mean(-1, keepdim=True).sqrt()


def known_vectors(error):
    """Preserve complex direction within each support and never overshoot."""
    f = torch.fft.fftfreq(error.shape[-1], device=error.device).abs()
    narrow, middle, broad = (error*support for support in
                             ((f > 0) & (f <= 1/8), (f > 0) & (f <= 1/4), f > 0))
    narrow_size, broad_size = chirp_rms(narrow), chirp_rms(broad)
    pairs = {}
    for label, cap in BUDGETS.items():
        shared = torch.minimum(narrow_size, broad_size).clamp_max(cap)
        pairs[f'narrow_{label}'] = narrow * (shared/narrow_size.clamp_min(1e-8))
        pairs[f'broad_{label}'] = broad * (shared/broad_size.clamp_min(1e-8))
    return dict(narrow_exact=narrow, mid_exact=middle, broad_exact=broad, all_exact=error, **pairs)
