"""Separate complex signal response and orthogonal error on a source replay."""

import torch


def split_along(value, direction):
    energy = direction.abs().square().sum(-1, keepdim=True)
    safe = energy > 1e-8
    product = (value*direction.conj()).sum(-1, keepdim=True)
    coefficient = torch.where(safe, product/energy.clamp_min(1e-8), 0.)
    parallel = coefficient*direction
    return parallel, value-parallel


def bands(n, device):
    freq = torch.fft.fftfreq(n, device=device).abs()
    return ((freq > 0) & (freq <= 1/32), (freq > 1/32) & (freq <= 1/8))


def equal_rms(parallel, orthogonal):
    length = lambda x: x.abs().square().mean(-1, keepdim=True).sqrt()
    a, b = length(parallel), length(orthogonal)
    shared = torch.minimum(a, b).clamp_max(.06)
    return parallel*(shared/a.clamp_min(1e-8)), orthogonal*(shared/b.clamp_min(1e-8))


def observed_gain(ordinary, retained):
    """Apply only a bounded change along the ordinary estimate in each band."""
    out = ordinary.clone()
    for selected in bands(ordinary.shape[-1], ordinary.device):
        base = ordinary[..., selected]
        project, _ = split_along(retained[..., selected]-base, base)
        norm = project.abs().square().sum(-1, keepdim=True).sqrt()
        bound = .15*base.abs().square().sum(-1, keepdim=True).sqrt()
        out[..., selected] = base + project*(bound/norm.clamp_min(1e-8)).clamp_max(1.)
    return out


def component_readouts(ordinary, retained, signal):
    """Change one physical error component at a time, with fixed out-of-band IQ."""
    named = ('retention_gain', 'retention_residual', 'retention_bands',
             'matched_gain', 'matched_residual', 'matched_both')
    out = {k: ordinary.clone() for k in named}
    mismatch = 0.
    for selected in bands(ordinary.shape[-1], ordinary.device):
        b, a, truth = (w[..., selected] for w in (ordinary, retained, signal))
        base_bias, base_noise = split_along(b-truth, truth)
        new_bias, new_noise = split_along(a-truth, truth)
        gain_fix, noise_fix = equal_rms(-base_bias, -base_noise)
        out['retention_gain'][..., selected] = truth + new_bias + base_noise
        out['retention_residual'][..., selected] = truth + base_bias + new_noise
        out['retention_bands'][..., selected] = a
        out['matched_gain'][..., selected] = b + gain_fix
        out['matched_residual'][..., selected] = b + noise_fix
        out['matched_both'][..., selected] = b + (gain_fix+noise_fix) * (.5**.5)
        lengths = torch.stack([(v-b).abs().square().mean(-1).sqrt() for v in
                               (out['matched_gain'][..., selected], out['matched_residual'][..., selected],
                                out['matched_both'][..., selected])])
        mismatch = max(mismatch, float((lengths.max(0).values-lengths.min(0).values).max()))
    return out, mismatch
