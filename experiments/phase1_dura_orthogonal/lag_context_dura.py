"""Frozen DURA correction using non-circular, phase-aligned lag context."""
import torch
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate
from experiments.phase1_dura_orthogonal.phase_equivariant_dura import (
    ComplexConv1d, PhaseEquivariantDura, RESIDUAL_SCALE,
)

LAGS = (20, 40)


def lag_context(anchor):
    """Aligned neighbor-minus-current IQ; out-of-window locations are zero."""
    if anchor.ndim != 3 or anchor.shape[1:] != (2, 256):
        raise ValueError('IQ windows [batch, 2, 256] required')
    z = torch.complex(anchor[:, 0], anchor[:, 1])
    features = []
    for lag in LAGS:
        previous, current = z[:, :-lag], z[:, lag:]
        cross = (current * previous.conj()).sum(-1)
        phase = cross / cross.abs().clamp_min(1e-12)
        before = phase[:, None] * previous - current
        after = phase.conj()[:, None] * current - previous
        features.extend((F.pad(before, (lag, 0)), F.pad(after, (0, lag))))
    return torch.stack([torch.stack((v.real, v.imag), 1) for v in features], 1)


class LagContextDura(PhaseEquivariantDura):
    """Learn a noise-gated residual from received, DURA, and lag differences."""
    def __init__(self, width=32):
        super().__init__(width=width)
        self.entry = ComplexConv1d(3 + 2*len(LAGS), width, 5)

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
        original = (received/scale, anchor/scale, innovation/scale)
        aligned = lag_context(anchor/scale)
        real = torch.cat([*(v[:, 0:1] for v in original), aligned[:, :, 0]], 1)
        imag = torch.cat([*(v[:, 1:2] for v in original), aligned[:, :, 1]], 1)
        real, imag = self.entry_gate(*self.entry(real, imag), noise)
        for block, gate in zip(self.blocks, self.gates):
            dr, di = gate(*block(real, imag), noise)
            real, imag = real+.2*dr, imag+.2*di
        real, imag = self.head(real, imag)
        bound = torch.rsqrt(1+real.square()+imag.square())
        delta = torch.cat((real*bound, imag*bound), 1)*scale*RESIDUAL_SCALE
        return anchor + noise_gate(anchor, variance)*delta
