"""Packet spectral shrinkage in the observed LoRa preamble coordinates."""

from math import log

import torch
from torch import nn


def power(z):
    return z.real.square() + z.imag.square()


def components(spectrum):
    common = spectrum.mean(dim=1, keepdim=True)
    return common, spectrum - common


def statistics(common, distinct):
    return torch.cat((power(common), power(distinct).mean(dim=1, keepdim=True)), dim=1)


def shrinkage_head(inputs):
    return nn.Sequential(nn.Conv1d(inputs, 32, 9, padding=4), nn.SiLU(),
                         nn.Conv1d(32, 32, 5, padding=2), nn.SiLU(),
                         nn.Conv1d(32, 2, 1))


class ChirpSpectrumRecovery(nn.Module):
    """One observed spectral estimate with a learned, bounded refinement."""

    def __init__(self, separate):
        super().__init__()
        self.separate = separate
        self.coarse = shrinkage_head(5)
        self.fine = shrinkage_head(9)
        for head in (self.coarse, self.fine):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        frequency = torch.fft.fftshift(torch.fft.fftfreq(256)).abs()
        self.register_buffer('frequency', frequency, persistent=False)

    def reconstruct(self, common, distinct, shrink, protection):
        if not self.separate:
            shrink = shrink.mean(dim=1, keepdim=True).expand_as(shrink)
        keep = 1 - shrink * protection
        return common * keep[:, :1] + distinct * keep[:, 1:]

    def forward(self, observed, coherence):
        spectrum = torch.fft.fftshift(torch.fft.fft(observed, norm='ortho'), dim=-1)
        common, distinct = components(spectrum)
        powers = statistics(common, distinct)
        # An observed outer-band floor, in normalized chirp coordinates.
        # Non-white disturbance and true energy in the band can bias this proxy.
        outer = power(spectrum[..., self.frequency > .25]).flatten(1)
        noise = outer.median(dim=-1).values[:, None, None] / log(2.)
        n = observed.shape[1]
        group_noise = noise * noise.new_tensor([1 / n, 1 - 1 / n])[None, :, None]
        relative = (powers / (group_noise + 1e-4)).log1p()
        prior = ((group_noise + 1e-5) / (powers + group_noise + 2e-5)).clamp(.02, .98).logit()
        # Pass the entire tracked carrier; approach an identity mapping at low measured noise.
        protection = (noise / (noise + .002)) * (self.frequency > 0)
        shape = common.real
        features = torch.cat((relative, (noise + 1e-4).log().expand_as(shape),
                              coherence.mean(dim=1, keepdim=True).expand_as(shape),
                              self.frequency.expand_as(shape)), dim=1)
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            coarse_change = self.coarse(features)
        coarse_logit = prior + coarse_change.float()
        coarse = self.reconstruct(common, distinct, coarse_logit.sigmoid(), protection)
        coarse_power = statistics(*components(coarse))
        residual_power = statistics(*components(spectrum - coarse))
        evidence = torch.cat((coarse_power, residual_power), dim=1)
        fine_features = torch.cat((features, (evidence / (group_noise.repeat(1, 2, 1) + 1e-4)).log1p()), dim=1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            fine_change = self.fine(fine_features)
        refined = self.reconstruct(common, distinct, (coarse_logit + fine_change.float()).sigmoid(), protection)
        waves = (coarse, refined)
        return tuple(torch.fft.ifft(torch.fft.ifftshift(z, dim=-1), norm='ortho') for z in waves)
