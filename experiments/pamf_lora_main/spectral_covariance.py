"""Preserve the shared real/imaginary LoRa shape with a rank-four covariance."""

import torch

from .channel_blind import zero_carrier
from .factorized_chirps import ChirpPrior, FactorizedChirps, packet_posterior
from .variance_bound import spectral_contrast, variance_interval


def covariance_factors(covariance, rank):
    """Approximate off-diagonal structure without changing marginal power."""
    marginal = covariance.diagonal().clamp_min(1e-7)
    scale = marginal.sqrt()
    correlation = covariance / (scale[:, None] * scale[None, :])
    correlation.diagonal().fill_(1.)
    values, vectors = torch.linalg.eigh(correlation)
    strength = (.7 * (values[-rank:] - 1).clamp_min(0)).sqrt()
    factors = scale[:, None] * vectors[:, -rank:] * strength
    power = factors.square().sum(-1).clamp_min(1e-12)
    factors = factors * (.95 * marginal / power).clamp(max=1).sqrt()[:, None]
    return marginal - factors.square().sum(-1), factors


def residual_posterior(residual, noise, private, shared, factors=None):
    """Condition a common low-rank plus diagonal prior and diagonal private priors."""
    precision = (private + noise).clamp_min(1e-8).reciprocal()
    pooled_uncertainty = precision.sum(1, keepdim=True).reciprocal()
    pooled = (precision * residual).sum(1, keepdim=True) * pooled_uncertainty
    total = shared + pooled_uncertainty
    base_solution = pooled / total
    common = shared * base_solution
    if factors is not None:
        weighted = factors / total.transpose(1, 2)
        system = torch.eye(factors.shape[1], dtype=residual.dtype, device=residual.device) + factors.T @ weighted
        latent = torch.linalg.solve_ex(system, factors.T @ base_solution.transpose(1, 2)).result
        common = common + pooled_uncertainty * (weighted @ latent).transpose(1, 2)
    return common + private * precision * (residual - common)


def real_coordinates(spectrum):
    return torch.cat((spectrum.real, spectrum.imag), dim=-1)


class SharedSpectralPrior(ChirpPrior):
    def __init__(self, rank=4):
        super().__init__()
        bins = torch.cat((torch.arange(1, 65), torch.arange(192, 256)))
        self.register_buffer('correlated_bins', bins, persistent=False)
        features = 2 * len(bins)
        self.register_buffer('private_real', torch.ones(features))
        self.register_buffer('shared_real', torch.ones(features))
        self.register_buffer('factors', torch.zeros(features, rank))

    @torch.no_grad()
    def fit(self, spectrum):
        super().fit(spectrum)
        values = real_coordinates(spectrum.index_select(-1, self.correlated_bins)).double()
        c = values.mean(1)
        shared = c - c.mean(0)
        private = values - c[:, None]
        private_power = private.square().sum((0, 1)) / (len(values) * (values.shape[1] - 1))
        common = shared.T @ shared / (len(values) - 1) - torch.diag(private_power / values.shape[1])
        diagonal, factors = covariance_factors(common, self.factors.shape[1])
        self.private_real.copy_(private_power.clamp_min(1e-7))
        self.shared_real.copy_(diagonal + factors.square().sum(-1))
        self.factors.copy_(factors)

    def innovation(self, residual, noise, correlated):
        outside = packet_posterior(residual, noise, self.shared, self.individual, coupled=True)
        bins = self.correlated_bins
        selected = real_coordinates(residual.index_select(-1, bins))
        variance = noise.index_select(-1, bins) * .5
        variance = torch.cat((variance, variance), dim=-1)
        shared = self.shared_real
        factors = self.factors if correlated else None
        if correlated:
            shared = shared - factors.square().sum(-1)
        update = residual_posterior(selected, variance, self.private_real, shared, factors)
        in_phase, quadrature = update.chunk(2, dim=-1)
        return outside.index_copy(-1, bins, torch.complex(in_phase, quadrature))


class SharedSpectrumRecovery(FactorizedChirps):
    def __init__(self, widths, *, correlated):
        super().__init__(widths, coupled=True)
        self.correlated = correlated
        self.clean_detail = SharedSpectralPrior()

    def forward(self, observed, quality, _rotation, *, controls=False):
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        coarse, fine, outer = self.propose(observed, quality, mixed_conv=mixed)
        measured = torch.fft.fft(observed, norm='ortho')
        contrast, weights = spectral_contrast(measured, quality)
        band_noise, _ = variance_interval(outer, contrast, weights, self.selectors)
        noise = band_noise @ self.bands

        def restore(center, correlated):
            updated = center + self.clean_detail.innovation(measured - center, noise, correlated)
            edit = torch.fft.ifft(updated - measured, norm='ortho')
            return observed + zero_carrier(edit)

        first = restore(torch.fft.fft(coarse, norm='ortho'), self.correlated)
        center = torch.fft.fft(fine, norm='ortho')
        last = restore(center, self.correlated)
        if controls:
            mean = self.clean_detail.mean
            return first, last, restore(mean, False), restore(mean, True), restore(center, False)
        return first, last
