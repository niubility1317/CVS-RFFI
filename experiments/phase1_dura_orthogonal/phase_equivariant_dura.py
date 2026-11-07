"""Phase-equivariant complex residual conditioned on frozen DURA context."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate

RESIDUAL_SCALE = .25


class ComplexConv1d(nn.Module):
    """Bias-free complex convolution represented as one real block convolution."""
    def __init__(self, inputs, outputs, kernel, dilation=1, *, zero=False):
        super().__init__()
        if min(inputs, outputs, kernel, dilation) < 1 or kernel % 2 != 1:
            raise ValueError('Positive channels, dilation and odd kernel required')
        self.real = nn.Parameter(torch.empty(outputs, inputs, kernel))
        self.imag = nn.Parameter(torch.empty(outputs, inputs, kernel))
        self.padding = dilation*(kernel//2)
        self.dilation = dilation
        if zero:
            nn.init.zeros_(self.real)
            nn.init.zeros_(self.imag)
        else:
            nn.init.kaiming_uniform_(self.real, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.imag, a=math.sqrt(5))

    def forward(self, real, imag):
        kernel = torch.cat((torch.cat((self.real, -self.imag), 1),
                            torch.cat((self.imag, self.real), 1)), 0)
        output = F.conv1d(torch.cat((real, imag), 1), kernel, padding=self.padding, dilation=self.dilation)
        return output.chunk(2, dim=1)


class PhaseGate(nn.Module):
    """Scale complex activations using phase-invariant magnitude and noise."""
    def __init__(self, width):
        super().__init__()
        self.magnitude = nn.Parameter(torch.zeros(1, width, 1))
        self.noise = nn.Parameter(torch.zeros(1, width, 1))
        self.bias = nn.Parameter(torch.zeros(1, width, 1))

    def forward(self, real, imag, noise):
        magnitude = (real.square()+imag.square()+1e-8).sqrt()
        gain = torch.sigmoid(self.magnitude*magnitude + self.noise*noise + self.bias)
        return real*gain, imag*gain


class PhaseEquivariantDura(nn.Module):
    """Conditional residual equivariant to a common phase rotation of all IQ inputs."""
    def __init__(self, width=16):
        super().__init__()
        self.entry = ComplexConv1d(3, width, 5)
        self.entry_gate = PhaseGate(width)
        self.blocks = nn.ModuleList(ComplexConv1d(width, width, 5, dilation=d) for d in (1, 2, 4))
        self.gates = nn.ModuleList(PhaseGate(width) for _ in self.blocks)
        self.head = ComplexConv1d(width, 1, 1, zero=True)

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        if validate and (received.ndim != 3 or received.shape[1:] != (2, 256)
                         or anchor.shape != received.shape or innovation.shape != received.shape
                         or variance.shape != (len(received),)):
            raise ValueError('Matching IQ, DURA context and variance required')
        power = anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12)
        scale = power.sqrt()
        noise = torch.log1p(variance[:, None, None].clamp_min(0.)/power)
        values = (received/scale, anchor/scale, innovation/scale)
        real = torch.cat([value[:, 0:1] for value in values], dim=1)
        imag = torch.cat([value[:, 1:2] for value in values], dim=1)
        real, imag = self.entry_gate(*self.entry(real, imag), noise)
        for block, gate in zip(self.blocks, self.gates):
            update_real, update_imag = gate(*block(real, imag), noise)
            real, imag = real+.2*update_real, imag+.2*update_imag
        real, imag = self.head(real, imag)
        bound = torch.rsqrt(1.+real.square()+imag.square())
        delta = torch.cat((real*bound, imag*bound), 1)*scale*RESIDUAL_SCALE
        return anchor + noise_gate(anchor, variance)*delta
