"""Cancel observable nonaffine phase on the verified repeated preamble."""

import torch

from experiments.pamf_lora_main.received_quality import clean_geometry_weight
from experiments.pamf_lora_main.repeated_edit import PREAMBLE


def preamble(iq):
    start, count, period = PREAMBLE[iq.shape[-1]]
    return iq[:, start:start + count * period].unflatten(-1, (count, period))


def correlation(a, b):
    scale = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
    return (a * b.conj()).sum(-1) / scale


def curvature_from_links(links):
    second = links[:,1:] * links[:,:-1].conj()
    magnitude = second.abs().sum(-1)
    aggregate = second.sum(-1)
    confidence = aggregate.abs() / magnitude.clamp_min(1e-12)
    return .5 * aggregate.angle(), confidence


def receive_curvature(iq):
    periods = preamble(iq)
    linked = correlation(periods[:,1:], periods[:,:-1])
    curve, agreement = curvature_from_links(linked)
    strength = ((linked.abs().median(-1).values - .20) / .45).clamp(0.,1.)
    return curve, strength * agreement


def paired_curvature(iq, clean):
    """Isolated source-reference estimate. A deployment reader has no clean input."""
    relation = correlation(preamble(iq),preamble(clean))
    links = relation[:,1:] * relation[:,:-1].conj()
    curve, _ = curvature_from_links(links)
    return curve


def apply_curvature(iq, curvature):
    start, count, period = PREAMBLE[iq.shape[-1]]
    stop = start + count * period
    position = torch.linspace(-.5*count,.5*count,count*period,device=iq.device,dtype=iq.real.dtype)
    edge = (count - 2 * position.abs()).clamp(0.,1.)
    square = position.square()
    basis = edge * (square-(edge*square).sum()/edge.sum())
    cap = torch.minimum(curvature.new_tensor(.18), .8 / basis.abs().max())
    curve = curvature.clamp(-cap,cap)
    angle = -basis * curve[:,None]
    changed = iq[:,start:stop] * torch.polar(torch.ones_like(angle),angle)
    return torch.cat((iq[:,:start],changed,iq[:,stop:]),dim=-1)


def restore_received(iq, calibration):
    estimate, confidence = receive_curvature(iq)
    return apply_curvature(iq, estimate * confidence * clean_geometry_weight(iq,calibration))


def diagnostic_reference(iq, clean, calibration):
    """Apply the same correction family using an isolated paired reference."""
    if iq.shape != clean.shape or not clean.is_complex():
        raise ValueError('Paired source reference must match real capture shape')
    return apply_curvature(iq, paired_curvature(iq,clean)*clean_geometry_weight(iq,calibration))
