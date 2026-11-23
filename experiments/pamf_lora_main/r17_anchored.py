"""Rotate a frozen R17 correction within observed LoRa bands at fixed dose."""

import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core

from .dura_repair import no_tone
from .residual_dura_bandfit import band_masks, correction, fft, power


MIN_NORM = 1e-7
MAX_TANGENT = .35


def real_dot(left, right):
    return (left.conj() * right).real.sum(-1)


def norm(iq):
    return torch.linalg.vector_norm(iq, dim=-1, keepdim=True)


@torch.no_grad()
def trajectory(received, r17, projector):
    coordinates = projector.coordinates(received)
    direct = no_tone(coordinates[0])
    stages = r17.stage_outputs(received)[0]
    features = torch.stack([fft(direct), *(fft(correction(projector, stage, coordinates, direct))
                                            for stage in stages)], dim=1)
    return features, coordinates


class AnchoredDirection(nn.Module):
    """Inspect the current chirp and all three unchanged R17 corrections."""

    def __init__(self):
        super().__init__()
        self.head = Core(incoming=8, outgoing=2)

    def forward(self, spectra):
        scale = (spectra.abs().square().mean(-1, keepdim=True) + 1e-6).sqrt()
        values = spectra / scale
        raw = self.head(torch.cat((values.real, values.imag), dim=1))
        update = torch.complex(raw[:, 0], raw[:, 1])
        base = spectra[:, -1]
        out = base.clone()
        for band in band_masks(base.device):
            anchor = base[:, band]
            length = norm(anchor)
            unit = anchor / length.clamp_min(MIN_NORM)
            proposed = update[:, band]
            orthogonal = proposed - real_dot(unit, proposed)[:, None] * unit
            tangent = MAX_TANGENT * orthogonal / (1. + power(orthogonal)[:, None]).sqrt()
            direction = (unit + tangent) / (1. + power(tangent)[:, None]).sqrt()
            out[:, band] = torch.where(length > MIN_NORM, length * direction, anchor)
        return out


def direction_loss(proposed, base, truth):
    """Learn band directions with neither the carrier nor raw IQ power as a shortcut."""
    losses = []
    for band in band_masks(base.device):
        a, b, t = (value[:, band] for value in (proposed, base, truth))
        base_norm = norm(b)[:, 0]
        target_norm = norm(t)[:, 0]
        valid = (base_norm > MIN_NORM) & (target_norm > MIN_NORM)
        denom = base_norm.clamp_min(MIN_NORM) * target_norm.clamp_min(MIN_NORM)
        losses.append(1. - real_dot(a, t) / denom)
        losses[-1] = torch.where(valid, losses[-1], 0.)
    return torch.stack(losses).mean(0)
