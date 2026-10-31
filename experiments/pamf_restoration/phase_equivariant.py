"""Source-trainable complex IQ restorer with global phase equivariance.

This research candidate uses only the current [B, 2, 256] received window.
No classifier, channel metadata, other windows or online state are consumed.
Equivariance is an architectural assumption, not an identity guarantee.
"""
import math

import torch
from torch import nn


class ComplexConv(nn.Module):
    """Complex-linear convolution, implemented with real-valued kernels."""

    def __init__(self, incoming, outgoing, kernel, dilation=1):
        super().__init__()
        padding = dilation * (kernel // 2)
        self.real = nn.Conv1d(incoming, outgoing, kernel, padding=padding,
                              dilation=dilation, bias=False)
        self.imag = nn.Conv1d(incoming, outgoing, kernel, padding=padding,
                              dilation=dilation, bias=False)

    def forward(self, real, imag):
        return self.real(real) - self.imag(imag), self.real(imag) + self.imag(real)


class RadialGate(nn.Module):
    """Nonlinear gain depends on magnitude only; phase is not discarded."""

    def __init__(self, width):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, width, 1))
        self.slope = nn.Parameter(torch.ones(1, width, 1))

    def forward(self, real, imag):
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        gain = 2 * torch.sigmoid(self.slope * magnitude + self.bias)
        return real * gain, imag * gain


class ComplexBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.first = ComplexConv(width, width, 5, dilation)
        self.gate = RadialGate(width)
        self.second = ComplexConv(width, width, 3)

    def forward(self, real, imag):
        a, b = self.first(real, imag)
        a, b = self.gate(a, b)
        a, b = self.second(a, b)
        return (real + a) / math.sqrt(2), (imag + b) / math.sqrt(2)


class PhaseEquivariantRestorer(nn.Module):
    """Residual multiscale complex TCN; identity at initialization."""

    def __init__(self, width=24):
        super().__init__()
        self.stem = ComplexConv(1, width, 5)
        self.blocks = nn.ModuleList(ComplexBlock(width, d) for d in (1, 2, 4, 8, 16))
        self.output = ComplexConv(width, 1, 1)
        nn.init.zeros_(self.output.real.weight)
        nn.init.zeros_(self.output.imag.weight)

    def forward(self, received):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('Expected received IQ with shape [batch, 2, 256]')
        real, imag = self.stem(received[:, :1], received[:, 1:])
        for block in self.blocks:
            real, imag = block(real, imag)
        real, imag = self.output(real, imag)
        return received + torch.cat((real, imag), dim=1)


class WidelyLinearRestorer(PhaseEquivariantRestorer):
    """Phase-equivariant path plus learnable conjugate-IQ image correction.

    The image branch deliberately breaks global phase equivariance: real IQ
    imbalance can have a fixed phase reference and cannot in general be
    inverted by a strictly equivariant mapping.
    """

    def __init__(self, width=24):
        super().__init__(width)
        self.image = ComplexConv(1, 1, 5)
        nn.init.zeros_(self.image.real.weight)
        nn.init.zeros_(self.image.imag.weight)

    def forward(self, received):
        estimate = super().forward(received)
        real, imag = self.image(received[:, :1], -received[:, 1:])
        return estimate + torch.cat((real, imag), dim=1)
