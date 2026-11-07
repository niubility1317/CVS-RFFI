"""Frozen DURA correction with phase-equivariant local conjugate features."""
import torch

from experiments.phase1_dura_orthogonal.direct_circular_recovery import circular_features, WINDOWS
from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate
from experiments.phase1_dura_orthogonal.phase_equivariant_dura import (
    ComplexConv1d, PhaseEquivariantDura, RESIDUAL_SCALE,
)


class DuraCircularRecovery(PhaseEquivariantDura):
    """Use received, DURA anchor, and physical innovation, with co-rotating features."""
    def __init__(self, width=32):
        super().__init__(width=width)
        self.entry = ComplexConv1d(3*(1+len(WINDOWS)), width, 5)

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape
                or innovation.shape != received.shape or variance.shape != (len(received),)):
            raise ValueError('Matching IQ, DURA context and variance required')
        if validate and (not all(torch.isfinite(v).all() for v in (received, anchor, innovation, variance))
                         or (variance <= 0).any()):
            raise ValueError('Finite IQ and positive variance required')
        power = anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12)
        scale = power.sqrt()
        noise = torch.log1p(variance[:, None, None].clamp_min(0.)/power)
        features = [circular_features(value/scale) for value in (received, anchor, innovation)]
        real = torch.cat([pair[0] for pair in features], 1)
        imag = torch.cat([pair[1] for pair in features], 1)
        real, imag = self.entry_gate(*self.entry(real, imag), noise)
        for block, gate in zip(self.blocks, self.gates):
            r, i = gate(*block(real, imag), noise)
            real, imag = real+.2*r, imag+.2*i
        real, imag = self.head(real, imag)
        bound = torch.rsqrt(1+real.square()+imag.square())
        delta = torch.cat((real*bound, imag*bound), 1)*scale*RESIDUAL_SCALE
        return anchor + noise_gate(anchor, variance)*delta
