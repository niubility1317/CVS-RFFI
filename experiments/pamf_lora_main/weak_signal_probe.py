"""Train LoRa denoising to pass weak source signal changes with noise held fixed."""

import torch

from experiments.pamf_unrolled_restore.denoise import Denoise

from .broadband_vector import partitions, supported
from .complex_vector import fft
from .denoise_components import bands


class ProjectedDenoise(Denoise):
    """Estimate noise within the existing bounded noncarrier support."""

    def forward(self, received, coherence, *, packet=True):
        noise = super().forward(received, coherence, packet=packet)
        return torch.fft.ifft(supported(fft(noise), True), norm='ortho')


def source_probe(signal, to_first, indices):
    """Use the same smooth, weak complex change in both reception frames."""
    spec = fft(signal[:, 0])
    freq = torch.fft.fftfreq(spec.shape[-1], device=spec.device)
    delay = 2.*indices.remainder(5).to(freq.dtype)
    angle = (torch.pi/2.)*indices.remainder(4).to(freq.dtype)
    ramp = 2.*torch.pi*delay[:, None]*freq + angle[:, None]
    mask = (freq.abs() > 0) & (freq.abs() <= 1/8)
    delta = .20*spec*mask*torch.polar(torch.ones_like(ramp), ramp)[:, None]
    return torch.fft.ifft(delta, norm='ortho')[:, None] / to_first


def reconstruction(model, input_wave, quality, signal):
    received, q, exact = (x.flatten(0, 1) for x in (input_wave, quality, signal))
    restored = received-model(received, q)
    reference = fft(exact)
    error = fft(restored-exact)
    by_band = []
    weights = reference.real.new_tensor((.35, .35, .15, .15))
    for band in partitions(reference.shape[-1], reference.device, True):
        power = reference[..., band].abs().square().mean(-1)
        residual = error[..., band].abs().square().mean(-1)
        by_band.append((residual/(.002+power)).log1p().mean())
    return (weights*torch.stack(by_band)).sum(), restored.reshape_as(input_wave)


def probe_response(model, received, quality, delta):
    y, q, v = (x.flatten(0, 1) for x in (received, quality, delta))
    # R(y+v)-R(y-v) = 2v when the estimated noise ignores this signal change.
    plus, minus = model(torch.cat((y+v, y-v)), torch.cat((q, q))).chunk(2)
    difference = fft(.5*(minus-plus))
    expected = fft(v)
    loss, gain = [], []
    for band in bands(expected.shape[-1], expected.device):
        ref, error = expected[..., band], difference[..., band]
        power = ref.abs().square().sum(-1)
        floor = .0002*band.sum()
        loss.append((error.abs().square().sum(-1)/(floor+power)).mean())
        gain.append(1.+(error*ref.conj()).real.sum() / power.sum().clamp_min(1e-8))
    return torch.stack(loss), torch.stack(gain)


def held_readouts(restored, measured, signal):
    """Measure pooled residual energy and physical signal transfer in each band."""
    output, observation, wanted = (fft(a) for a in (restored, measured, signal))
    sums = []
    for band in bands(output.shape[-1], output.device):
        a, b, c = (v[..., band] for v in (output, observation, wanted))
        sums.append(torch.stack(((a-c).abs().square().sum(), (b-c).abs().square().sum(),
                                 (a*c.conj()).real.sum(), c.abs().square().sum())))
    return torch.stack(sums)
