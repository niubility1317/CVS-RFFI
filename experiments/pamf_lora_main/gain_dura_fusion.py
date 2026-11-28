"""DURA observation update in the complement of the anchor restoration."""

import torch

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import real_dot


def fuse(observed, anchor, companion, action, variance):
    """Take two constrained MAP steps with a companion IQ prior and a fixed edit budget."""
    axis = anchor - observed
    axis_norm = axis.abs().square().sum(-1, keepdim=True)

    def project(vector):
        gain = (vector * axis.conj()).sum(-1, keepdim=True) / axis_norm.clamp_min(1e-12)
        return vector - gain * axis

    penalty = variance.reshape(len(anchor), 1).clamp_min(1e-8) / SCALE**2
    prior = .5 * project(companion - anchor)
    residual = project(action.adjoint(observed - action(anchor)) + penalty * prior)
    delta = torch.zeros_like(anchor)
    direction = residual
    power = real_dot(residual, residual)
    for _ in range(2):
        normal = project(action.adjoint(action(direction)) + penalty * direction)
        step = power / real_dot(direction, normal).clamp_min(1e-20)
        delta = delta + step[:, None] * direction
        residual = residual - step[:, None] * normal
        remaining = real_dot(residual, residual)
        direction = residual + (remaining / power.clamp_min(1e-20))[:, None] * direction
        power = remaining
    delta = project(delta)
    budget = .25 * axis_norm.sqrt()
    factor = (budget / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
    return anchor + factor * delta
