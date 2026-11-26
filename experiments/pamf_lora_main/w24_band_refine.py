"""Learn a bounded 9–32 kHz edit on a frozen W24 DURA backbone."""

import torch
from torch import nn


EDIT_WEIGHT = .25
POSITIVE = slice(9, 33)
NEGATIVE = slice(-32, -8)


def sidebands(spectrum):
    return torch.stack((spectrum[..., POSITIVE], spectrum[..., NEGATIVE]), dim=-2)


def power(bands):
    return bands.abs().square().mean(dim=(-3, -2, -1), keepdim=True)


def match_energy(proposed, original):
    a, b = power(proposed), power(original)
    usable = (a > 1e-12) & (b > 1e-12)
    factor = (b / a.clamp_min(1e-12)).sqrt()
    return torch.where(usable, proposed * factor, original)


def band_edit_risk(edit, desired):
    current, exact = (sidebands(torch.fft.fft(x, norm='ortho')) for x in (edit, desired))
    current_power, exact_power = power(current.detach()), power(exact)
    usable = ((current_power > 1e-12) & (exact_power > 1e-12)).flatten()
    exact = exact * (current_power / exact_power.clamp_min(1e-12)).sqrt()
    scale = exact.abs().square() + .1 * current_power + 1e-5
    risk = ((current - exact).abs().square() / scale).log1p().mean((-3, -2, -1))
    return (risk * usable).sum() / usable.sum().clamp_min(1)


class BandRefiner(nn.Module):
    def __init__(self, training, *, polar):
        super().__init__()
        self.backbone = training.DuraW24().eval().requires_grad_(False)
        self.polar = polar
        self.head = nn.Sequential(
            nn.Conv1d(6, 32, 5, padding=2), nn.SiLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.SiLU(), nn.Conv1d(32, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @property
    def priors(self):
        return self.backbone.priors

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def propose(self, y, quality, rms, edges):
        (coarse, first, final), *physical = self.backbone.propose(y, quality, rms, edges)
        spectrum = torch.fft.fft(final - y, norm='ortho')
        observed = sidebands(torch.fft.fft(y, norm='ortho'))
        original = sidebands(spectrum)
        fields = torch.stack((observed.real, observed.imag, original.real, original.imag,
                              original.abs(), quality.unsqueeze(-1).expand_as(original.real)), dim=-2)
        outputs = .5 * self.head(fields.flatten(0, 2)).tanh()
        re, im = (part.reshape_as(original.real) for part in outputs.unbind(dim=-2))
        factor = torch.polar(re.exp(), im) if self.polar else torch.complex(1 + re, im)
        scale = rms.unsqueeze(-1)
        changed = match_energy(original * factor * scale, original * scale) / scale
        difference = torch.zeros_like(spectrum)
        difference[..., POSITIVE], difference[..., NEGATIVE] = (changed - original).unbind(dim=-2)
        adjusted = final + torch.fft.ifft(difference, norm='ortho')
        return (coarse, first, adjusted), *physical
