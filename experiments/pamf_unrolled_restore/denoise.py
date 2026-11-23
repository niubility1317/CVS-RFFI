"""Learn the signal before added noise, without undoing the channel itself."""

import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core


def complementary_context(spectrum, coherence):
    weights = coherence.square().clamp_min(.005)
    weight = (weights.sum(1, keepdim=True) - weights).clamp_min(.005)
    mean = ((weights*spectrum).sum(1, keepdim=True) - weights*spectrum) / weight
    power = spectrum.abs().square()
    second = ((weights*power).sum(1, keepdim=True) - weights*power) / weight
    spread = (second-mean.abs().square()).clamp_min(1e-8).sqrt()
    return mean, spread


class Denoise(nn.Module):
    """Estimate additive noise from each waveform and the other same-packet symbols."""

    def __init__(self):
        super().__init__()
        self.spectral = Core(incoming=7, outgoing=3)
        self.temporal = Core(incoming=5, outgoing=2)

    def forward(self, received, coherence, *, packet):
        spec = torch.fft.fftshift(torch.fft.fft(received, norm='ortho'), dim=-1)
        context, spread = complementary_context(spec, coherence)
        if not packet:
            context, spread = torch.zeros_like(context), torch.zeros_like(spread)
        confidence = coherence.expand_as(received.real)
        features = torch.stack((spec.real, spec.imag, spec.abs().log1p(),
                                context.real, context.imag, spread.log1p(), confidence), dim=2)
        spectral = self.spectral(features.flatten(0, 1)).unflatten(0, received.shape[:2])
        amplitude, real, imag = spectral.unbind(2)
        # Keep the initialization exactly at the received IQ.
        weight = torch.sigmoid(amplitude - 3.) - torch.sigmoid(amplitude.new_tensor(-3.))
        estimate = weight*spec + .15*torch.complex(real, imag)
        spectral_noise = torch.fft.ifft(torch.fft.ifftshift(estimate, dim=-1), norm='ortho')
        features = torch.stack((received.real, received.imag, spectral_noise.real,
                                spectral_noise.imag, confidence), dim=2)
        temporal = self.temporal(features.flatten(0, 1)).unflatten(0, received.shape[:2])
        return spectral_noise + .10*torch.complex(temporal[:, :, 0], temporal[:, :, 1])


def noise_loss(estimate, exact, received):
    error = estimate - exact
    signal = received - exact
    nrmse = (error.abs().square().mean(-1) / (.08 + signal.abs().square().mean(-1))).mean()
    e = torch.fft.fft(error, norm='ortho').abs().square()
    p = torch.fft.fft(signal, norm='ortho').abs().square()
    frequency = torch.fft.fftfreq(error.shape[-1], device=error.device).abs()
    near = (frequency > 0) & (frequency <= 1/8)
    near_shape = (e[..., near].mean(-1) / (.04 + p[..., near].mean(-1))).mean()
    return nrmse + .20*near_shape, nrmse
