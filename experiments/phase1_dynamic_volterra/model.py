"""Input-only phase-equivariant dynamic memory polynomial restoration."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class DynamicVolterra(nn.Module):
    """Predict local complex linear and cubic memory filters from invariant IQ evidence."""
    LAGS = (-4, -2, -1, 0, 1, 2, 4)

    def __init__(self, width=24):
        super().__init__()
        self.entry = nn.Conv1d(16, width, 7, padding=3)
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 5, dilation=d, padding=2*d) for d in (1, 2, 4, 8))
        self.gate_output = nn.Conv1d(width, 28, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def forward(self, received, initial, log_variance):
        shape = received.shape
        if (received.ndim != 3 or shape[1:] != (2, 256) or initial.shape != shape
                or log_variance.shape != (shape[0],)):
            raise ValueError('Matching IQ windows and one blind log variance are required')
        if not all(torch.isfinite(v).all() for v in (received, initial, log_variance)):
            raise FloatingPointError('Finite received evidence is required')
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        normalized = received/scale
        padded = F.pad(normalized, (4, 4), mode='reflect')
        shifted = torch.stack([padded[:, :, 4+k:4+k+256] for k in self.LAGS], dim=1)
        sr, si = shifted[:, :, 0], shifted[:, :, 1]
        rr, ri = normalized[:, 0:1], normalized[:, 1:2]
        real = (sr*rr+si*ri).clamp(-16, 16)
        imag = (si*rr-sr*ri).clamp(-16, 16)
        magnitude = torch.log1p((rr.square()+ri.square()).clamp_max(16))
        noise = ((log_variance-power[:, 0, 0].log()).clamp(-16, 8)/8)[:, None, None]
        h = F.silu(self.entry(torch.cat((magnitude, real, imag, noise.expand(-1, 1, 256)), dim=1)))
        for layer in self.layers: h = h+0.2*F.silu(layer(h))
        coefficients = self.gate_output(h).tanh().reshape(len(received), 2, 14, 256)
        shifted_complex = torch.complex(sr, si)
        intensity = (sr.square()+si.square()).clamp_max(8)
        basis = torch.cat((shifted_complex, shifted_complex*intensity), dim=1)
        weights = torch.complex(coefficients[:, 0], coefficients[:, 1])
        delta = (weights*basis).sum(1)
        correction = torch.stack((delta.real, delta.imag), 1)*scale
        return initial+(0.25/math.sqrt(14))*correction
