"""Cross-fit the relative mirror and echo action of the current LoRa packet."""

import torch

from .dura_repair import real_iq
from .joint_unroll_source import coordinates
from .relative_echo import shift


DELAY_US = 8


def delay(value):
    offset = value.shape[-1] * (DELAY_US/1000)
    lower = int(offset)
    fraction = offset-lower
    return (1.-fraction)*shift(value, lower) + fraction*shift(value, lower+1)


def features(value, phase, *, raw_frame):
    if raw_frame:
        mirror = (phase*value).conj()*phase.conj()
        echo = delay(phase*value)*phase.conj()
    else:
        mirror = value.conj()
        echo = delay(value)
    return torch.stack((value, mirror, echo), -1)


def fit(design, innovation, index, *, relative):
    x, y = design[..., index, :], innovation[..., index, None]
    ridge = .025 * torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    coefficient = torch.linalg.solve(x.mH @ x/len(index)+ridge, x.mH @ y/len(index)).squeeze(-1)
    if relative:
        coefficient = coefficient-coefficient.mean(1, keepdim=True)
    radius = coefficient.abs().sum(-1, keepdim=True)
    if relative:
        # Preserve the zero-mean relative action when enforcing the packet bound.
        radius = radius.amax(1, keepdim=True)
    scale = (.55 / radius.clamp_min(1e-8)).clamp_max(1.)
    return coefficient*scale


def opposite_fold(y):
    ref = torch.cat((y[:, 1::2].mean(1, keepdim=True), y[:, ::2].mean(1, keepdim=True)), 1).repeat(1, 3, 1)
    samples = torch.arange(y.shape[-1], device=y.device)
    inside = (samples % 32 >= 5) & (samples % 32 < 27)
    folds = tuple(samples[inside & (samples//32 % 2 == i)] for i in range(2))
    indices = torch.cat(folds)
    unit = (ref-y)[..., indices].abs().square().mean((1, 2))
    return ref, folds, unit


def estimate(y, phase, reference, held_reference, folds, *, relative, raw_frame):
    design = features(reference, phase, raw_frame=raw_frame)
    innovation = y-reference
    parameters = [fit(design, innovation, index, relative=relative) for index in folds]
    held = []
    for index, coeff in zip(folds, reversed(parameters), strict=True):
        forward = reference[..., index] + (design[..., index, :] * coeff.unsqueeze(-2)).sum(-1)
        held.append((forward-held_reference[..., index]).abs().square().mean((1, 2)))
    coefficient = .5*(parameters[0]+parameters[1])
    return coefficient, .5*(held[0]+held[1])


def invert_half(y, phase, coeff, raw_frame):
    x = y
    for _ in range(3):
        action = (features(x, phase, raw_frame=raw_frame)*coeff.unsqueeze(-2)).sum(-1)
        x = y-action
    edit = x-y
    return y+.5*(edit-edit.mean(-1, keepdim=True))


def receive_fit(y, phase):
    ref, folds, unit = opposite_fold(y)
    coefficient, held = estimate(y, phase, ref, y, folds, relative=True, raw_frame=True)
    evidence = ((unit-held)/unit.clamp_min(1e-9)).clamp_min(0.)
    return coefficient, evidence, ref, folds


def restore_pair(iq, tracker, threshold):
    y, phase, _, rms = coordinates(tracker, iq)
    physical, evidence, ref, folds = receive_fit(y, phase)
    active = evidence > threshold
    if not active.any():
        return iq, iq, active
    value, angle = y[active], phase[active]
    direct, _ = estimate(value, angle, ref[active], value, folds, relative=True, raw_frame=False)
    outputs = []
    for coefficient, raw in ((direct, False), (physical[active], True)):
        edit = invert_half(value, angle, coefficient, raw)-value
        result = iq.clone()
        result[active] += real_iq(edit*angle*rms[active])
        outputs.append(result)
    return (*outputs, active)
