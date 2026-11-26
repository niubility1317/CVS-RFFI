"""Redistribute half-readout reconstruction risk over paired source impairment."""

import torch


def disturbance(received, clean):
    reference = clean[:, None]
    dimensions = (-2, -1)
    power = reference.abs().square().mean(dimensions)
    return (received - reference).abs().square().mean(dimensions) / power.clamp_min(1e-8)


def waveform_per_chirp(output, target):
    # Preserve the archive's risk and retain its leading chirp dimensions.
    reference = torch.fft.fft(target, norm='ortho')[..., 1:]
    error = torch.fft.fft(output - target, norm='ortho')[..., 1:]
    power = reference.abs().square()
    return (error.abs().square() / (power + .03 * power.mean(-1, keepdim=True) + .001)).log1p().mean(-1)


def redistribution(risk, severity):
    severity = severity.detach().flatten()
    values = risk.flatten()
    weights = 1 + 3 * severity / (.1 + severity)
    correction = (values * weights).sum() / weights.sum() - values.mean()
    strongest = severity.topk((len(severity) + 3) // 4).indices
    return correction, values[strongest].mean()
