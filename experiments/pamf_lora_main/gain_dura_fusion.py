"""DURA observation update in the complement of the anchor restoration."""

import torch

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import real_dot


def remove_axis(vector, axis, axis_norm):
    gain = (vector * axis.conj()).sum(-1, keepdim=True) / axis_norm.clamp_min(1e-12)
    return vector - gain * axis


def bounded_delta(delta, axis_norm):
    budget = .25 * axis_norm.sqrt()
    scale = (budget / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
    return scale * delta


def companion_only(observed, anchor, companion):
    """Exact solution when only the companion prior is used."""
    axis = anchor - observed
    norm = axis.abs().square().sum(-1, keepdim=True)
    direction = .5 * remove_axis(companion - anchor, axis, norm)
    return anchor + bounded_delta(direction, norm)


def _cg_update(residual, normal_operator):
    delta, direction = torch.zeros_like(residual), residual
    power = real_dot(residual, residual)
    for _ in range(2):
        normal = normal_operator(direction)
        step = power / real_dot(direction, normal).clamp_min(1e-20)
        delta = delta + step[:, None] * direction
        residual = residual - step[:, None] * normal
        remaining = real_dot(residual, residual)
        direction = residual + (remaining / power.clamp_min(1e-20))[:, None] * direction
        power = remaining
    return delta


def fuse(observed, anchor, companion, action, variance):
    """Take two constrained MAP steps with a companion IQ prior and a fixed edit budget."""
    axis = anchor - observed
    axis_norm = axis.abs().square().sum(-1, keepdim=True)

    def project(vector):
        return remove_axis(vector, axis, axis_norm)

    penalty = variance.reshape(len(anchor), 1).clamp_min(1e-8) / SCALE**2
    prior = .5 * project(companion - anchor)
    residual = project(action.adjoint(observed - action(anchor)) + penalty * prior)
    def normal_operator(direction):
        return project(action.adjoint(action(direction)) + penalty * direction)

    delta = _cg_update(residual, normal_operator)
    return anchor + bounded_delta(project(delta), axis_norm)


def frequency_real(observed, anchor, action, variance):
    """Solve the same observation problem using nonnegative spectral gains of the anchor."""
    spectrum = torch.fft.fft(anchor, norm='ortho')
    amplitude = spectrum.abs()
    phase = spectrum / amplitude.clamp_min(1e-12)
    phase = torch.where(amplitude >= 1e-5 * amplitude.amax(-1, keepdim=True), phase, 0)
    edit = anchor - observed
    edit_f = torch.fft.fft(edit, norm='ortho')
    edit_norm = edit_f.norm(dim=-1, keepdim=True)
    floor = (1e-6 * edit_norm).clamp_min(1e-12)

    def to_real(freq):
        return (freq * phase.conj()).real

    def to_time(coeff):
        return torch.fft.ifft(phase * coeff, norm='ortho')

    def unit(vector):
        norm = vector.norm(dim=-1, keepdim=True)
        return torch.where(norm > floor, vector / norm.clamp_min(1e-12), 0)

    # Orthogonalize the two old edit constraints within the real gain space.
    first = unit(to_real(edit_f))
    second = to_real(1j * edit_f)
    second = unit(second - (second * first).sum(-1, keepdim=True) * first)

    def project(coeff):
        return coeff - (coeff * first).sum(-1, keepdim=True) * first \
                     - (coeff * second).sum(-1, keepdim=True) * second

    def adjoint_real(value):
        return to_real(torch.fft.fft(action.adjoint(value), norm='ortho'))

    penalty = variance.reshape(len(anchor), 1).clamp_min(1e-8) / SCALE**2

    def normal_operator(coeff):
        return project(adjoint_real(action(to_time(coeff))) + penalty * coeff)

    residual = project(adjoint_real(observed - action(anchor)))
    coeff = project(_cg_update(residual, normal_operator))
    gain_limit = torch.where(coeff < 0, .99 * amplitude / (-coeff).clamp_min(1e-12), 1.)
    delta = to_time(gain_limit.amin(-1, keepdim=True).clamp(max=1.) * coeff)
    return anchor + bounded_delta(delta, edit_norm.square())
