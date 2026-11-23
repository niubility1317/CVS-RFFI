"""Restore shared LoRa detail while integrating uncertain short echoes."""

import torch
from torch import nn

from .channel_blind import ChannelBlind
from .crossfit_pilot import LAGS
from .weak_signal_probe import observed_noise_rms


def common_weights(quality):
    weights = quality.square().clamp_min(1e-6)
    return weights / weights.sum(1, keepdim=True)


def design(observed, template, weights, bins, taper):
    carrier = observed.mean(-1, keepdim=True)
    delayed = torch.stack([template.roll(lag, -1) for lag in LAGS], dim=-1)
    modulation = carrier[..., None] * delayed * template.conj()[..., None]
    common = (modulation * weights[..., None]).sum(1)
    spectrum = torch.fft.fft(common, dim=1, norm='ortho').index_select(1, bins)
    real = torch.cat((spectrum.real, spectrum.imag), dim=1)
    imaginary = torch.cat((-spectrum.imag, spectrum.real), dim=1)
    return torch.cat((real, imaginary), dim=-1) * taper


def marginal_system(prior_variance, noise, columns):
    diagonal = prior_variance + noise
    scaled = columns / diagonal[..., None]
    small = torch.eye(columns.shape[-1], dtype=columns.dtype, device=columns.device) + columns.mT @ scaled
    solve = torch.linalg.solve_ex(small, scaled.mT).result
    return diagonal, scaled, small, solve


def posterior_update(residual, prior_variance, system):
    diagonal, scaled, _, solve = system
    point = prior_variance * residual / diagonal
    coefficient = solve @ residual[..., None]
    echo = prior_variance * (scaled @ coefficient).squeeze(-1)
    return point, echo


def echo_log_evidence(residual, system):
    _, scaled, small, solve = system
    rhs = scaled.mT @ residual[..., None]
    improvement = (rhs.mT @ (solve @ residual[..., None])).flatten()
    return .5 * (improvement - torch.linalg.slogdet(small).logabsdet)


class CommonSpectrum(nn.Module):
    def __init__(self):
        super().__init__()
        bins = torch.cat((torch.arange(1, 33), torch.arange(224, 256)))
        self.register_buffer('bins', bins, persistent=False)
        self.register_buffer('taper', torch.tensor([.06 / (1 + abs(lag)) for lag in LAGS] * 2),
                             persistent=False)
        width = 2 * len(bins)
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('axes', torch.eye(width))
        self.register_buffer('variance', torch.ones(width))

    def vector(self, value, weights):
        waveform = (value * weights).sum(1)
        spectrum = torch.fft.fft(waveform, norm='ortho').index_select(-1, self.bins)
        return torch.cat((spectrum.real, spectrum.imag), dim=-1)

    @torch.no_grad()
    def fit(self, vectors):
        vectors = vectors.cpu()
        mean = vectors.mean(0)
        centered = vectors - mean
        eigenvalues, axes = torch.linalg.eigh(centered.T @ centered / (len(centered) - 1))
        self.mean.copy_(mean)
        self.axes.copy_(axes)
        self.variance.copy_(eigenvalues.clamp_min(1e-8))

    def install(self, base, old_vector, new_vector):
        delta = (new_vector - old_vector) @ self.axes.T
        i, q = delta.chunk(2, dim=-1)
        spectrum = base.new_zeros((len(base), base.shape[-1]))
        spectrum = spectrum.index_copy(-1, self.bins, torch.complex(i, q))
        return base + torch.fft.ifft(spectrum, norm='ortho')[:, None]


class MarginalPilot(ChannelBlind):
    def __init__(self, widths, *, echo):
        super().__init__(widths=widths, companion='mirror')
        self.common = CommonSpectrum()
        self.echo = echo
        self.reliability = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, observed, quality, template, *, controls=False):
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        coarse, prior, noise = self.propose(observed, quality, mixed_conv=mixed)
        strength = observed.shape[1] * noise
        refined = (observed + strength * prior) / (1 + strength)

        weights = common_weights(quality)
        variance = self.common.variance
        measurement = self.common.vector(observed, weights) @ self.common.axes
        noise_common = (.5 * (weights.square() * noise).sum(1)).clamp_min(1e-7)
        columns = design(observed, template, weights, self.common.bins, self.common.taper)
        columns = self.common.axes.T @ columns
        system = marginal_system(variance, noise_common, columns)
        fixed_center = self.common.mean @ self.common.axes
        evidence = echo_log_evidence(measurement - fixed_center, system)
        # Temper the evidence because neighboring frequency errors are correlated.
        belief = (.25 * evidence[:, None]).clamp(-12., 12.).sigmoid()
        energy = measurement.square().mean(dim=1, keepdim=True)
        features = torch.cat((evidence[:, None].div(8).tanh(), (30 * noise_common).log1p(),
                              quality.mean(1), (10 * energy).log1p()), dim=1)
        trust = self.reliability(features).sigmoid()

        def update(base, prior_iq, include_echo):
            prior_vector = self.common.vector(prior_iq, weights) @ self.common.axes
            base_vector = self.common.vector(base, weights) @ self.common.axes
            point, echo = posterior_update(measurement - prior_vector, variance, system)
            proposal = prior_vector + point
            if include_echo:
                proposal = proposal - belief * echo
            return self.common.install(base, base_vector, torch.lerp(base_vector, proposal, trust))

        initial = update(coarse, coarse, self.echo)
        final = update(refined, prior, self.echo)
        if controls:
            point, echo = posterior_update(measurement - fixed_center, variance, system)
            additive = self.common.install(observed, measurement, fixed_center + point)
            uncertain = self.common.install(observed, measurement, fixed_center + point - belief * echo)
            disable_echo = update(refined, prior, False) if self.echo else final
            return initial, final, additive, uncertain, disable_echo, belief
        return initial, final
