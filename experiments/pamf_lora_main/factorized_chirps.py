"""Joint Gaussian restoration of shared and per-chirp LoRa detail."""

import torch
from torch import nn

from .channel_blind import ChannelBlind, zero_carrier
from .variance_bound import spectral_contrast, variance_interval


def packet_posterior(residual, noise, shared, individual, *, coupled):
    """Compute the posterior increment given centered complex observations."""
    if not coupled:
        variance = shared + individual
        return residual * (variance / (variance + noise).clamp_min(1e-8))
    total = (individual + noise).clamp_min(1e-8)
    precision = total.reciprocal()
    # Marginalize the individual innovations when estimating the shared mode.
    common = shared * (precision * residual).sum(1, keepdim=True)
    common = common / (1 + shared * precision.sum(1, keepdim=True))
    return common + (individual * precision) * (residual - common)


class ChirpPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(256, dtype=torch.complex64))
        self.register_buffer('shared', torch.ones(256))
        self.register_buffer('individual', torch.ones(256))

    @torch.no_grad()
    def fit(self, spectrum):
        common = spectrum.mean(1, keepdim=True)
        mean = common.mean(0).squeeze(0)
        chirps = spectrum.shape[1]
        independent = (spectrum - common).abs().square().sum(1).mean(0) / (chirps - 1)
        common_power = (common.squeeze(1) - mean).abs().square().mean(0)
        self.mean.copy_(mean)
        self.shared.copy_((common_power - independent / chirps).clamp_min(0.))
        self.individual.copy_(independent.clamp_min(1e-7))

    def posterior(self, measured, center, noise, *, coupled):
        return center + packet_posterior(measured - center, noise, self.shared, self.individual,
                                         coupled=coupled)


class FactorizedChirps(ChannelBlind):
    def __init__(self, widths, *, coupled):
        super().__init__(widths=widths, companion='mirror')
        self.coupled = coupled
        self.clean_detail = ChirpPrior()
        frequency = (256 * torch.fft.fftfreq(256)).round()
        absolute = frequency.abs()
        bins = torch.stack((absolute <= 8, (absolute > 8) & (absolute <= 32),
                            (absolute > 32) & (absolute <= 64), absolute > 64))
        segments = torch.stack((bins & (frequency > 0), bins & (frequency < 0)), dim=1).flatten(0, 1)
        self.register_buffer('selectors', segments.float() / segments.sum(1, keepdim=True), persistent=False)
        self.register_buffer('bands', bins.float(), persistent=False)

    def forward(self, observed, quality, _template, *, controls=False):
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        coarse, fine, outside = self.propose(observed, quality, mixed_conv=mixed)
        measured = torch.fft.fft(observed, norm='ortho')
        moment, weights = spectral_contrast(measured, quality)
        band_noise, _ = variance_interval(outside, moment, weights, self.selectors)
        noise = band_noise @ self.bands

        def restore(center, coupled):
            spectrum = self.clean_detail.posterior(measured, center, noise, coupled=coupled)
            edit = torch.fft.ifft(spectrum - measured, norm='ortho')
            return observed + zero_carrier(edit)

        initial = restore(torch.fft.fft(coarse, norm='ortho'), self.coupled)
        proposal = torch.fft.fft(fine, norm='ortho')
        result = restore(proposal, self.coupled)
        if controls:
            engineering_a = restore(self.clean_detail.mean, False)
            engineering_b = restore(self.clean_detail.mean, True)
            unlinked = restore(proposal, False) if self.coupled else result
            return initial, result, engineering_a, engineering_b, unlinked
        return initial, result
