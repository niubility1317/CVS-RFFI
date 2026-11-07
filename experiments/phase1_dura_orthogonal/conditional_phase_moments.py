"""Condition DURA phase correction on normalized complex signal moments."""
import torch
from torch import nn

from experiments.phase1_dura_orthogonal.conditional_phase_dura import ConditionalPhaseDura


def complex_moments(signal):
    real, imag = signal[:, 0], signal[:, 1]
    power = real.square()+imag.square()
    second_real = real.square()-imag.square()
    second_imag = 2*real*imag
    norm2 = power.mean(-1).clamp_min(1e-12)
    norm4 = power.square().mean(-1).clamp_min(1e-12)
    return torch.stack((second_real.mean(-1)/norm2, second_imag.mean(-1)/norm2,
                        (second_real.square()-second_imag.square()).mean(-1)/norm4,
                        (2*second_real*second_imag).mean(-1)/norm4), dim=1)


class ConditionalPhaseMomentsDura(ConditionalPhaseDura):
    def __init__(self, width=32):
        super().__init__(width)
        self.summary = nn.Sequential(nn.Linear(19, 32), nn.SiLU(), nn.Linear(32, 3))
        nn.init.zeros_(self.summary[-1].weight)
        nn.init.zeros_(self.summary[-1].bias)

    def condition_features(self, received, anchor, innovation, variance, power):
        base = super().condition_features(received, anchor, innovation, variance, power)
        return torch.cat((base, *(complex_moments(v) for v in (received, anchor, innovation))), dim=1)
