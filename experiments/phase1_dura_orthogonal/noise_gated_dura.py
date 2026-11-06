"""Source-calibrated noise-gated time-varying correction of frozen DURA."""
import torch

from experiments.phase1_dura_orthogonal.dura_timevarying import DuraTimeVaryingRecovery

GATE_LOG10_LOW = -3.8
GATE_LOG10_HIGH = -2.8


def noise_gate(anchor, variance):
    if anchor.ndim != 3 or anchor.shape[1:] != (2, 256) or variance.shape != (len(anchor),):
        raise ValueError('Matching anchor IQ and per-record variance required')
    power = anchor.square().mean((1, 2)).clamp_min(1e-12)
    statistic = (variance/power).clamp_min(1e-12).log10()
    return ((statistic-GATE_LOG10_LOW)/(GATE_LOG10_HIGH-GATE_LOG10_LOW)).clamp(0., 1.)[:, None, None]


class NoiseGatedDuraTimeVarying(DuraTimeVaryingRecovery):
    def forward(self, received, anchor, innovation, variance, *, validate=True):
        corrected = super().forward(received, anchor, innovation, variance, validate=validate)
        return anchor + noise_gate(anchor, variance)*(corrected-anchor)
