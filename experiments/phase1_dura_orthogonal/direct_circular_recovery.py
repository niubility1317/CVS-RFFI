"""Direct phase-equivariant recovery with local conjugate IQ features."""
import torch
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.direct_phase_recovery import DirectPhaseRecovery, RESIDUAL_SCALE
from experiments.phase1_dura_orthogonal.phase_equivariant_dura import ComplexConv1d

WINDOWS = (17, 65, 129)


def circular_features(normalized):
    """Build x and local E[x²]/E[|x|²] times conjugate(x), all co-rotating."""
    if normalized.ndim != 3 or normalized.shape[1:] != (2, 256):
        raise ValueError('Normalized IQ windows required')
    i, q = normalized[:, :1], normalized[:, 1:]
    power = i.square() + q.square()
    second_real, second_imag = i.square()-q.square(), 2*i*q
    reals, imags = [i], [q]
    for size in WINDOWS:
        average = lambda value: F.avg_pool1d(value, size, stride=1, padding=size//2)
        denominator = average(power).clamp_min(1e-8)
        pr, pi = average(second_real)/denominator, average(second_imag)/denominator
        reals.append(pr*i + pi*q)
        imags.append(pi*i - pr*q)
    return torch.cat(reals, 1), torch.cat(imags, 1)


class DirectCircularRecovery(DirectPhaseRecovery):
    """Extend direct phase recovery using local second-order conjugate structure."""
    def __init__(self, width=32):
        super().__init__(width=width)
        self.entry = ComplexConv1d(1+len(WINDOWS), width, 5)

    def forward(self, received, anchor=None, innovation=None, variance=None, *, validate=True):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('Received IQ windows required')
        if validate and not torch.isfinite(received).all():
            raise ValueError('Finite received IQ required')
        scale = received.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        real, imag = circular_features(received/scale)
        invariant_noise = torch.zeros_like(scale)
        real, imag = self.entry_gate(*self.entry(real, imag), invariant_noise)
        for block, gate in zip(self.blocks, self.gates):
            r, i = gate(*block(real, imag), invariant_noise)
            real, imag = real+.2*r, imag+.2*i
        real, imag = self.head(real, imag)
        bound = torch.rsqrt(1+real.square()+imag.square())
        return received + torch.cat((real*bound, imag*bound), 1)*scale*RESIDUAL_SCALE
