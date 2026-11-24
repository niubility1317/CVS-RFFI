"""Average LoRa recovery edits across plausible received tracking coordinates."""

import torch

from .antithetic_detail_source import read_stages
from .channel_blind import zero_carrier
from .fission_risk import band_masks
from .frame_leakage import carrier_peak
from .private_dose import match_band_energy
from .refinement_bound import bound_refinement


def split_uncertainty(wave):
    """Estimate joint center-phase and slope disagreement from interleaved pilots."""
    time = torch.linspace(-1, 1, wave.shape[-1], device=wave.device, dtype=wave.real.dtype)
    grid = torch.linspace(-.8, .8, 17, device=wave.device, dtype=wave.real.dtype)
    weight = 1 - time.square()
    kernel = torch.polar(weight.expand(len(grid), -1), -grid[:, None] * time)
    choices = []
    for start in (0, 1):
        sample = wave[..., start::2]
        score = (sample @ kernel[:, start::2].T).abs().square()
        slope = carrier_peak(score, grid)
        phase = -slope * time[start::2]
        tone = (sample * torch.polar(weight[start::2].expand_as(phase), phase)).sum(-1, keepdim=True)
        choices.append((slope, tone))
    (first, tone_a), (second, tone_b) = choices
    phase = (.5 * (tone_a * tone_b.conj()).angle()).clamp(-.25, .25)
    slope = (.5 * (first - second)).clamp(-.6, .6)
    return phase, slope


def multipliers(wave, phase, slope):
    time = torch.linspace(-1, 1, wave.shape[-1], device=wave.device, dtype=wave.real.dtype)
    side = torch.polar(torch.ones_like(wave.real), 2**.5 * (phase + slope * time))
    return torch.stack((torch.ones_like(side), side, side.conj()), dim=1)


def marginal_readouts(y, rotation, quality, rms, model, recipe, *, disagreement=None):
    """Use unchanged physical IQ and install one edit in the original frame."""
    phase, slope = split_uncertainty(y) if disagreement is None else disagreement
    relative = multipliers(y, phase, slope)
    framed = y[:, None] * relative.conj()
    transformed = rotation[:, None] * relative
    perturbed = framed[:, 1:]
    probe_quality = (perturbed.mean(-1, keepdim=True).abs().square() /
                     perturbed.abs().square().mean(-1, keepdim=True).clamp_min(1e-8)).clamp(0, 1)
    q = torch.cat((quality[:, None], probe_quality), dim=1)
    scale = rms[:, None].expand_as(q)
    coarse, full = read_stages(model, framed.flatten(0, 1), q.flatten(0, 1),
                               transformed.flatten(0, 1), scale.flatten(0, 1), recipe)
    coarse = coarse.unflatten(0, framed.shape[:2])[:, 0]
    edits = (full.unflatten(0, framed.shape[:2]) - framed) * relative
    nominal, above, below = edits.unbind(1)
    mixed = y + zero_carrier(.5 * nominal + .25 * (above + below))
    _, bounded = bound_refinement(y, coarse, mixed, rms)
    residual = bounded - y
    target, proposal = (torch.fft.fft(edit, norm='ortho') for edit in (nominal, residual))
    masks = band_masks(target).float()
    fixed = torch.fft.ifft(match_band_energy(target, proposal, rms, masks), norm='ortho')
    return (nominal, residual, fixed), (phase, slope)
