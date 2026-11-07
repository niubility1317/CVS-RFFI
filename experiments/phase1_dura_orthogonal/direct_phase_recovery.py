"""Direct complex-IQ recovery with exact global phase equivariance."""
import torch
from torch import nn

from experiments.phase1_dura_orthogonal.phase_equivariant_dura import ComplexConv1d, PhaseGate

DILATIONS = (1, 2, 4, 8, 16, 32)
RESIDUAL_SCALE = 2.0


class DirectPhaseRecovery(nn.Module):
    """Predict a complex residual from one received IQ window, without DURA context."""
    def __init__(self, width=32):
        super().__init__()
        self.entry = ComplexConv1d(1, width, 5)
        self.entry_gate = PhaseGate(width)
        self.blocks = nn.ModuleList(ComplexConv1d(width, width, 5, dilation=d) for d in DILATIONS)
        self.gates = nn.ModuleList(PhaseGate(width) for _ in DILATIONS)
        self.head = ComplexConv1d(width, 1, 1, zero=True)

    def forward(self, received, anchor=None, innovation=None, variance=None, *, validate=True):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('Received IQ windows required')
        if validate and not torch.isfinite(received).all():
            raise ValueError('Finite received IQ required')
        scale = received.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        normalized = received/scale
        invariant_noise = torch.zeros_like(scale)
        real, imag = self.entry_gate(*self.entry(normalized[:, :1], normalized[:, 1:]), invariant_noise)
        for block, gate in zip(self.blocks, self.gates):
            r, i = gate(*block(real, imag), invariant_noise)
            real, imag = real + .2*r, imag + .2*i
        real, imag = self.head(real, imag)
        bound = torch.rsqrt(1 + real.square() + imag.square())
        correction = torch.cat((real*bound, imag*bound), 1)*scale*RESIDUAL_SCALE
        return received + correction
