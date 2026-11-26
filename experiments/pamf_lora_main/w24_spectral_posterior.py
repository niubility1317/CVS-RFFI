"""Complex packet spectral priors with an observation-conditioned posterior readout."""

import torch
from torch import nn
from torch.nn import functional as F

from .w24_packet_unet import CENTER


WIDTH = 80
CHIRPS = 8


class DirectionalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(WIDTH, WIDTH, 3, dilation=dilation)
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self, hidden):
        context = F.pad(hidden, (2 * self.dilation, 0), mode='circular')
        update = self.norm(self.conv(context).transpose(-2, -1)).transpose(-2, -1)
        return hidden + F.silu(update)


class DirectionalPrior(nn.Module):
    """One-sided frequency context with pointwise channel normalization."""

    def __init__(self):
        super().__init__()
        self.entry = nn.Conv1d(2 * CHIRPS, WIDTH, 1)
        self.blocks = nn.Sequential(*(DirectionalBlock(d) for d in (1, 2, 4, 8, 16, 32)))

    def forward(self, spec, *, exclude_current):
        values = torch.cat((spec.real, spec.imag), dim=1)
        if exclude_current:
            values = values.roll(1, dims=-1)
        return self.blocks(self.entry(values))


def weighted_posterior(observation, prediction, clean_variance, noise_variance):
    gain = noise_variance / (noise_variance + clean_variance)
    return observation + gain * (prediction - observation)


class SpectralPosterior(nn.Module):
    def __init__(self, *, exclude_current):
        super().__init__()
        self.exclude_current = exclude_current
        self.lower = DirectionalPrior()
        self.upper = DirectionalPrior()
        self.prior = nn.Sequential(
            nn.Conv1d(2 * WIDTH + 2 * CHIRPS + 2, WIDTH, 1), nn.SiLU(),
            nn.Conv1d(WIDTH, WIDTH, 1), nn.SiLU(), nn.Conv1d(WIDTH, 3 * (CHIRPS + 1), 1),
        )
        nn.init.normal_(self.prior[-1].weight, std=.002)
        nn.init.zeros_(self.prior[-1].bias)
        freq = torch.fft.fftfreq(CENTER)
        self.register_buffer('frequency', 2 * freq, persistent=False)
        self.register_buffer('noise_band', freq.abs() >= 1 / 3, persistent=False)
        self.register_buffer('editable', freq != 0, persistent=False)

    def prior_spectrum(self, spectrum, quality, edge_power, noise_level):
        before = self.lower(spectrum, exclude_current=self.exclude_current)
        after = self.upper(spectrum.flip(-1), exclude_current=self.exclude_current).flip(-1)
        condition = torch.cat((quality, edge_power, noise_level), dim=1).expand(-1, -1, CENTER)
        frequency = self.frequency.expand(len(spectrum), 1, -1)
        hints = self.prior(torch.cat((before, after, condition, frequency), dim=1))
        hints = hints.float().unflatten(1, (CHIRPS + 1, 3))
        real, imag, log_variance = hints.unbind(2)
        shape = torch.complex(real, imag)
        variance = .008 * log_variance.clamp(-7., 7.).exp()
        private = shape[:, 1:] - shape[:, 1:].mean(1, keepdim=True)
        return shape[:, :1], private, variance[:, :1], variance[:, 1:]

    def forward(self, y, quality, rms, edges):
        packet_rms = rms.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        scale = (rms / packet_rms).clamp_min(1e-8)
        spectrum = torch.fft.fft(y * scale, norm='ortho')
        noise = (spectrum[..., self.noise_band].abs().square().flatten(1).median(-1).values
                 / .69314718056).clamp(1e-6, 4.)[:, None, None]
        edge = (edges.abs().square().mean(-1, keepdim=True) / packet_rms.square()).clamp_min(1e-8)
        expected_c, expected_d, var_c, var_d = self.prior_spectrum(spectrum, quality, edge.log() / 5.,
                                                                  noise.log() / 5.)
        observed_c = spectrum.mean(1, keepdim=True)
        observed_d = spectrum - observed_c
        common = weighted_posterior(observed_c, expected_c, var_c, noise / CHIRPS)
        private = weighted_posterior(observed_d, expected_d, var_d, noise * (1. - 1. / CHIRPS))
        private = private - private.mean(1, keepdim=True)

        def encoded_stage(target):
            correction = torch.fft.ifft((target - spectrum) * self.editable, norm='ortho') / scale
            # The existing protocol renders half of the endpoint delta.
            return y + 2. * correction

        return encoded_stage(expected_c + expected_d), encoded_stage(common + private)

    def propose(self, y, quality, rms, edges):
        return self(y, quality, rms, edges),
