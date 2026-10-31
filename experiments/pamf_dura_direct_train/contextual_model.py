"""An input-conditioned widely-linear spectral and time-domain restorer.

Only the current received IQ window is available at inference.  The spectral
operator is learned from that window; it is not a true-channel oracle.  Both
branches produce corrections to the same IQ estimate and start at zero.
"""
import math

import torch
from torch import nn


class ContextBlock(nn.Module):
    def __init__(self, width, context_dim, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, width)
        self.norm2 = nn.GroupNorm(4, width)
        self.condition = nn.Linear(context_dim, 2 * width)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, value, context):
        scale, bias = self.condition(context).chunk(2, dim=1)
        h = self.norm1(self.conv1(value))
        h = torch.nn.functional.silu(h * (1 + 0.25 * torch.tanh(scale[..., None]))
                                     + 0.25 * torch.tanh(bias[..., None]))
        h = self.norm2(self.conv2(h))
        return (value + torch.nn.functional.silu(h)) / math.sqrt(2)


class ContextualSpectralRestorer(nn.Module):
    """Adaptive complex and conjugate-complex frequency paths plus a TCN.

    The FFT uses all 256 received samples from one window; it must never be
    described as causal streaming inference.  A bounded spectral correction
    limits initial gain but does not claim a guarantee on final restoration.
    """
    def __init__(self, width=32, context_dim=64):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(64, context_dim, 5, stride=2, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.spectral_head = nn.Linear(context_dim, 4 * 256)
        nn.init.zeros_(self.spectral_head.weight)
        nn.init.zeros_(self.spectral_head.bias)
        self.time_input = nn.Conv1d(2, width, 5, padding=2)
        self.blocks = nn.ModuleList(ContextBlock(width, context_dim, d) for d in (1, 2, 4, 8, 16, 32))
        self.time_output = nn.Conv1d(width, 2, 1)
        nn.init.zeros_(self.time_output.weight)
        nn.init.zeros_(self.time_output.bias)

    def forward(self, received):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('Expected received IQ with shape [batch, 2, 256]')
        context = self.context(received)
        coeff = 2 * torch.tanh(self.spectral_head(context).reshape(-1, 4, 256))
        direct = torch.complex(coeff[:, 0], coeff[:, 1])
        image = torch.complex(coeff[:, 2], coeff[:, 3])
        z = torch.complex(received[:, 0], received[:, 1])
        spectrum = torch.fft.fft(z, norm='ortho')
        conjugate_spectrum = torch.fft.fft(z.conj(), norm='ortho')
        correction = torch.fft.ifft(direct * spectrum + image * conjugate_spectrum, norm='ortho')
        spectral = torch.stack((correction.real, correction.imag), dim=1)
        time = self.time_input(received)
        for block in self.blocks:
            time = block(time, context)
        return received + spectral + self.time_output(time)
