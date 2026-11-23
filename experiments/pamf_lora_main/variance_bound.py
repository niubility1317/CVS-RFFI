"""Use split-observation noise evidence to limit per-band waveform edits."""

import torch

from .channel_blind import ChannelBlind, zero_carrier


def spectral_contrast(spectrum, quality):
    """Weighted across-chirp spread and the corresponding noise weights."""
    weights = quality.square().clamp_min(1e-6)
    weights = weights / weights.sum(1, keepdim=True)
    common = (weights * spectrum).sum(1, keepdim=True)
    contrast = (weights * (spectrum - common).abs().square()).sum(1)
    return contrast, weights * (1 - weights)


def variance_interval(outer, contrast, contrast_weights, selectors):
    """Estimate a centered two-point variance distribution from disjoint bands."""
    expected = (outer * contrast_weights).sum(1).clamp_min(1e-8)
    means = contrast @ selectors.T
    ratios = (means / expected).clamp(.25, 4.).reshape(len(outer), -1, 2)
    center_ratio = ratios.mean(-1)
    center = outer * center_ratio.sqrt()[:, None, :]
    split_disagreement = (ratios[..., 0].log() - ratios[..., 1].log()).abs()
    outside_disagreement = center_ratio.log().abs()
    spread = torch.maximum(split_disagreement, outside_disagreement)
    fraction = -torch.expm1(-spread)
    return center, fraction[:, None, :]


def point_gain(variance, proposal_power, chirps):
    """Observation/proposal fusion with an expected-noise edit budget."""
    posterior = variance / (variance + 1. / chirps)
    budget = (variance.sqrt() / proposal_power.clamp_min(1e-8).sqrt()).clamp_max(1.)
    return torch.minimum(posterior, budget)


def noise_gain(variance, proposal_power, chirps, fraction=None):
    if fraction is None:
        return point_gain(variance, proposal_power, chirps)
    lower = variance * (1 - fraction)
    upper = variance * (1 + fraction)
    return .5 * (point_gain(lower, proposal_power, chirps) + point_gain(upper, proposal_power, chirps))


class VarianceBound(ChannelBlind):
    def __init__(self, widths, *, uncertain):
        super().__init__(widths=widths, companion='mirror')
        self.uncertain = uncertain
        frequency = (256 * torch.fft.fftfreq(256)).round()
        absolute = frequency.abs()
        regions = torch.stack(((absolute > 0) & (absolute <= 8), (absolute > 8) & (absolute <= 32),
                               (absolute > 32) & (absolute <= 64), absolute > 64))
        select = torch.stack((regions & (frequency > 0), regions & (frequency < 0)), dim=1).flatten(0, 1)
        # The unpaired Nyquist coefficient belongs to the negative band.
        self.register_buffer('selectors', select.float() / select.sum(1, keepdim=True), persistent=False)
        self.register_buffer('regions', regions.float(), persistent=False)
        self.register_buffer('bands', regions.float() / regions.sum(1, keepdim=True), persistent=False)

    def forward(self, observed, quality, _template, *, controls=False):
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        coarse, fine, outer = self.propose(observed, quality, mixed_conv=mixed)
        spectrum = torch.fft.fft(observed, norm='ortho')
        contrast, weights = spectral_contrast(spectrum, quality)
        variance, fraction = variance_interval(outer, contrast, weights, self.selectors)

        def project(proposal, integrate):
            edit = torch.fft.fft(proposal - observed, norm='ortho')
            power = edit.abs().square() @ self.bands.T
            bound = noise_gain(variance, power, observed.shape[1], fraction if integrate else None)
            correction = torch.fft.ifft(edit * (bound @ self.regions), norm='ortho')
            return observed + zero_carrier(correction), bound

        coarse, _ = project(coarse, self.uncertain)
        output, mixed_gain = project(fine, self.uncertain)
        if controls:
            point, nominal_gain = project(fine, False) if self.uncertain else (output, mixed_gain)
            return coarse, output, point, fraction, (nominal_gain - mixed_gain).mean(dim=1)
        return coarse, output
