"""Packet-local estimates from the visible WiSig short training field."""

import torch


def receive_coordinates(iq):
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError('Expected WiSig IQ with shape (batch, 2, 256)')
    received = torch.complex(iq[:, 0], iq[:, 1])
    scale = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    normalized = received / scale
    a, b = normalized[:, 48:128], normalized[:, 68:148]
    cross = (b * a.conj()).sum(-1, keepdim=True)
    rotation = torch.polar(torch.ones_like(cross.real), cross.angle())
    variance = .5 * (b - a * rotation).abs().square().mean(-1, keepdim=True)
    denominator = (a.abs().square().sum(-1, keepdim=True) * b.abs().square().sum(-1, keepdim=True)).sqrt()
    quality = (cross.abs() / denominator.clamp_min(1e-8)).clamp(0, 1)
    return normalized, scale, quality, variance


def guard_features(iq):
    """Eight normalized repeat-mismatch features from four visible STF windows."""
    normalized, _, _, _ = receive_coordinates(iq)
    output = []
    for start in (48, 60, 72, 84):
        a, b = normalized[:, start:start + 32], normalized[:, start + 20:start + 52]
        cross = (b * a.conj()).sum(-1)
        rotation = torch.polar(torch.ones_like(cross.real), cross.angle())
        residual = .5 * (b - a * rotation[:, None]).abs().square().mean(-1)
        norm = (a.abs().square().sum(-1) * b.abs().square().sum(-1)).sqrt().clamp_min(1e-8)
        coherence = (cross.abs() / norm).clamp(1e-8, 1.)
        output.extend((residual.clamp_min(1e-10).log(), -coherence.log()))
    return torch.stack(output, dim=1)


def guard_score(iq, record):
    features = guard_features(iq)
    midpoint, dispersion, weight = (features.new_tensor(record[key]) for key in ('midpoint', 'dispersion', 'weight'))
    return ((features - midpoint) / dispersion).clamp(-10, 10) @ weight + record['offset']
