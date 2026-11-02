"""Input-only phase-equivariant neural coupling between neighboring FFT bins."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SpectralCoupling(nn.Module):
    """Condition complex frequency-mixing filters on one received IQ window."""
    OFFSETS = (-16, -4, -1, 0, 1, 4, 16)

    def __init__(self, width=24):
        super().__init__()
        self.entry = nn.Conv1d(16, width, 7, padding=3, padding_mode='circular')
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 5, dilation=d, padding=2*d,
                                              padding_mode='circular') for d in (1, 2, 4, 8))
        self.gate_output = nn.Conv1d(width, 14, 1)
        nn.init.zeros_(self.gate_output.weight); nn.init.zeros_(self.gate_output.bias)

    def forward(self, received, initial, log_variance):
        shape = received.shape
        if (received.ndim != 3 or shape[1:] != (2, 256) or initial.shape != shape
                or log_variance.shape != (shape[0],)):
            raise ValueError('Matching IQ windows and one blind log variance are required')
        if not all(torch.isfinite(v).all() for v in (received, initial, log_variance)):
            raise FloatingPointError('Finite received evidence is required')
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        z = torch.complex(received[:, 0], received[:, 1])/scale[:, 0, :]
        spectrum = torch.fft.fft(z, dim=-1, norm='ortho')
        neighbors = torch.stack([torch.roll(spectrum, shifts=k, dims=-1) for k in self.OFFSETS], dim=1)
        cross = neighbors*spectrum[:, None].conj()
        magnitude = torch.log1p(spectrum.abs().square().clamp_max(16))[:, None]
        noise = ((log_variance-power[:, 0, 0].log()).clamp(-16, 8)/8)[:, None, None]
        h = F.silu(self.entry(torch.cat((magnitude, cross.real.clamp(-16, 16), cross.imag.clamp(-16, 16),
                                        noise.expand(-1, 1, 256)), dim=1)))
        for layer in self.layers: h = h+0.2*F.silu(layer(h))
        weights = self.gate_output(h).tanh().reshape(len(received), 2, 7, 256)
        delta_f = (torch.complex(weights[:, 0], weights[:, 1])*neighbors).sum(1)
        delta = torch.fft.ifft(delta_f, dim=-1, norm='ortho')
        return initial+(3.0/math.sqrt(7))*torch.stack((delta.real, delta.imag), 1)*scale
