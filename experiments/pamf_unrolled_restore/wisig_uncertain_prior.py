"""Phase-coordinate-preserving WiSig recovery and source-only training objective."""

import torch
from torch import nn
import torch.nn.functional as F

from experiments.pamf_reception_restore.models import Core


def as_complex(iq):
    return torch.complex(iq[:, 0], iq[:, 1])


def as_iq(z):
    return torch.stack((z.real, z.imag), dim=1)


def received_features(y):
    symbols = y[:, 80:160].reshape(-1, 4, 20)
    a, b = symbols[:, :-1], symbols[:, 1:]
    product = (a.conj() * b).sum(dim=(1, 2))
    energy = a.abs().square().sum(dim=(1, 2)) * b.abs().square().sum(dim=(1, 2))
    coherence = (product.abs() / energy.clamp_min(1e-16).sqrt()).clamp(0, 1)
    phase = torch.angle(product) / 20
    index = torch.arange(80, 160, device=y.device, dtype=y.real.dtype)
    rotation = torch.polar(torch.ones_like(phase[:, None] * index), phase[:, None] * index).reshape(-1, 4, 20)
    aligned = symbols * rotation.conj()
    difference = aligned[:, 1::2] - aligned[:, ::2]
    variance = .5 * difference.abs().square().mean(dim=(1, 2))
    stf_power = aligned.abs().square().mean(dim=(1, 2))
    return (variance / stf_power.clamp_min(1e-8)).clamp(0, 1), coherence, rotation


class UncertaintyPrior(nn.Module):
    """Joint spectral coarse restore, IQ detail prior and soft observation anchor."""

    def __init__(self):
        super().__init__()
        self.spectral = nn.Sequential(nn.Conv1d(4, 24, 7, padding=3), nn.SiLU(),
                                      nn.Conv1d(24, 24, 5, padding=2), nn.SiLU(), nn.Conv1d(24, 2, 1))
        self.detail = Core(incoming=8, widths=(24, 32, 48, 64))
        nn.init.zeros_(self.spectral[-1].bias)
        nn.init.zeros_(self.spectral[-1].weight)

    def forward(self, y):
        noise, confidence, rotation = received_features(y)
        spectrum = torch.fft.fftshift(torch.fft.fft(y, norm='ortho'), dim=-1)
        conditioning = torch.stack((noise, confidence), dim=1)[:, :, None].expand(-1, -1, y.shape[-1])
        spectral_change = self.spectral(torch.cat((as_iq(spectrum), conditioning), dim=1))
        correction = torch.fft.ifft(torch.fft.ifftshift(as_complex(spectral_change), dim=-1), norm='ortho')
        coarse = y + correction
        local = self.detail(torch.cat((as_iq(y), as_iq(coarse), as_iq(correction), conditioning), dim=1))
        proposed_change = correction + as_complex(local)

        # Penalize alteration of the coherent short-training component more than
        # the innovation. The rotated coordinates are internal to the projection.
        stf_change = proposed_change[:, 80:160].reshape(-1, 4, 20)
        shared = (stf_change * rotation.conj()).mean(dim=1, keepdim=True) * rotation
        taper = torch.ones(80, device=y.device, dtype=y.real.dtype)
        taper[:20] = torch.linspace(0, 1, 20, device=y.device, dtype=y.real.dtype)
        taper[-20:] = taper[:20].flip(0)
        protected = F.pad(shared.flatten(1) * taper, (80, y.shape[-1] - 160))
        d = noise[:, None] / (noise[:, None] + .05)
        refined = y + d * (proposed_change - confidence[:, None] * protected)
        return coarse, refined, noise


def perturb(signal, generator):
    """Apply continuous generic distortions; return only the waveform(s)."""
    batch, _, length = signal.shape
    device = signal.device

    def uniform(*size):
        return torch.rand(size, device=device, generator=generator)

    def complex_normal(*size):
        return torch.complex(torch.randn(size, device=device, generator=generator),
                             torch.randn(size, device=device, generator=generator)) * (2 ** -.5)

    # A short causal echo and a leading echo cover local convolution variation.
    # Shared within a source difference pair, independent for repeated observations.
    left = F.pad(signal[..., :-1], (1, 0), mode='reflect')
    right = F.pad(signal[..., 3:], (0, 3), mode='reflect')
    echoed = signal + .18 * complex_normal(batch, 1, 1) * left + .13 * complex_normal(batch, 1, 1) * right
    t = torch.linspace(-1, 1, length, device=device)
    phase = .5 * (2 * uniform(batch, 1, 1) - 1) + .4 * (2 * uniform(batch, 1, 1) - 1) * t
    gain = (.75 * (2 * uniform(batch, 1, 1) - 1)).exp()
    observed = torch.polar(gain.expand_as(phase), phase) * echoed
    noise = complex_normal(batch, 1, length)
    noise = noise + .55 * (2 * uniform(batch, 1, 1) - 1) * noise.roll(1, dims=-1)
    noise_scale = uniform(batch, 1, 1).square()
    return observed + noise_scale * noise


def normalized_error(output, reference):
    return ((output - reference).abs().square().mean(-1) / reference.abs().square().mean(-1).clamp_min(.01)).mean()


def natural_delta(source):
    """Keep the smaller complex differences from adjacent unlabeled source windows."""
    difference = torch.fft.fft(source.roll(1, dims=0) - source, norm='ortho')
    fraction = abs(difference).detach().quantile(.75, dim=-1, keepdim=True)
    spectrum = .25 * difference * (abs(difference) <= fraction)
    return torch.fft.ifft(spectrum, norm='ortho')


def fit_loss(model, source, generator, preserve, observe=None, perturbation=perturb):
    delta = natural_delta(source)
    targets = torch.stack((source, source, source + delta), dim=1)
    shared = perturbation(targets[:, (0, 2)], generator)
    independent = perturbation(targets[:, 1:2], generator)
    inputs = torch.stack((shared[:, 0], independent[:, 0], shared[:, 1]), dim=1)
    inputs[-4:] = targets[-4:]
    received = inputs.flatten(0, 1) if observe is None else observe(inputs, generator)
    coarse, refined, _ = model(received)
    target = targets.flatten(0, 1)
    difference = refined.unflatten(0, targets.shape[:2])
    transported = difference[:, 2] - difference[:, 0]
    transfer = ((transported - delta).abs().square().mean(-1)
                / (delta.abs().square().mean(-1) + .002)).log1p().mean()
    repeated = (difference[:, 1] - difference[:, 0]).abs().square().mean()
    losses = torch.stack((normalized_error(coarse, target), normalized_error(refined, target), repeated, transfer))
    weights = losses.new_tensor((.35, 1., .1, preserve))
    return (losses * weights).sum(), losses.detach()
