"""Source-only phase supervision for an independently trained DURA correction."""
import math

import torch

from experiments.phase1_dura_orthogonal.conditional_phase_moments import ConditionalPhaseMomentsDura
from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate

PHASE_WEIGHT = 1.0


class PhaseSupervisedDura(ConditionalPhaseMomentsDura):
    pass


def phase_supervision_loss(output, anchor, variance, clean):
    if output.shape != anchor.shape or clean.shape != anchor.shape:
        raise ValueError('Matching output, source clean and anchor IQ required')
    ar, ai, cr, ci = anchor[:, 0], anchor[:, 1], clean[:, 0], clean[:, 1]
    target = torch.atan2((ci*ar-cr*ai).sum(-1), (cr*ar+ci*ai).sum(-1)).clamp(-math.pi/2, math.pi/2)
    out_r, out_i = output[:, 0], output[:, 1]
    effective = torch.atan2((out_i*ar-out_r*ai).sum(-1), (out_r*ar+out_i*ai).sum(-1))
    gate = noise_gate(anchor, variance)[:, 0, 0]
    difference = torch.atan2((effective-target).sin(), (effective-target).cos())
    return (gate*difference.square()).sum()/gate.sum().clamp_min(1e-8)
