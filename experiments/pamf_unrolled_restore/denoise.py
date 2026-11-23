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


def spectral_companion(spectrum, mode):
    """Get a conjugated companion for each bin of an even, centered spectrum."""
    size = spectrum.shape[-1]
    midpoint = size // 2
    index = torch.arange(size, device=spectrum.device)
    if mode == 'mirror':
        partner = (-index) % size
    else:
        # Use the next bin away from zero, with the preceding bin at the edges.
        near = torch.where(index < midpoint, index - 1, index + 1)
        partner = torch.where(index <= 1, index + 1, torch.where(index == size - 1, index - 1, near))
    companion = spectrum.index_select(-1, partner).conj()
    # DC and Nyquist have no separate conjugate partner.
    return companion * ((index != 0) & (index != midpoint))


class Denoise(nn.Module):
    """Estimate additive noise from each waveform and the other same-packet symbols."""

    def __init__(self, *, widths=(32, 64, 96, 128), companion=None):
        super().__init__()
        if companion not in (None, 'adjacent', 'mirror'):
            raise ValueError(f'Unknown spectral companion {companion}')
        self.companion = companion
        self.spectral = Core(incoming=7 if companion is None else 11, outgoing=3, widths=widths)
        self.temporal = Core(incoming=5, outgoing=2, widths=widths)

    def forward(self, received, coherence, *, packet, mixed_conv=False):
        spec = torch.fft.fftshift(torch.fft.fft(received, norm='ortho'), dim=-1)
        context, spread = complementary_context(spec, coherence)
        if not packet:
            context, spread = torch.zeros_like(context), torch.zeros_like(spread)
        confidence = coherence.expand_as(received.real)
        fields = (spec.real, spec.imag, spec.abs().log1p(),
                  context.real, context.imag, spread.log1p(), confidence)
        if self.companion is not None:
            own_pair, packet_pair = (spectral_companion(x, self.companion) for x in (spec, context))
            fields += (own_pair.real, own_pair.imag, packet_pair.real, packet_pair.imag)
        features = torch.stack(fields, dim=2)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed_conv):
            spectral = self.spectral(features.flatten(0, 1)).unflatten(0, received.shape[:2])
        if mixed_conv:
            spectral = spectral.float()
        amplitude, real, imag = spectral.unbind(2)
        # Keep the initialization exactly at the received IQ.
        weight = torch.sigmoid(amplitude - 3.) - torch.sigmoid(amplitude.new_tensor(-3.))
        estimate = weight*spec + .15*torch.complex(real, imag)
        spectral_noise = torch.fft.ifft(torch.fft.ifftshift(estimate, dim=-1), norm='ortho')
        features = torch.stack((received.real, received.imag, spectral_noise.real,
                                spectral_noise.imag, confidence), dim=2)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed_conv):
            temporal = self.temporal(features.flatten(0, 1)).unflatten(0, received.shape[:2])
        if mixed_conv:
            temporal = temporal.float()
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
