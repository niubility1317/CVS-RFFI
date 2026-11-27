"""Separate dose error and orthogonal edit energy on authorized additive source pairs."""

import torch


ALIGNMENT_WEIGHT = .5
ORTHOGONAL_WEIGHT = 3.


def edit_parts(edit, desired, selected):
    """Use the real inner product of complex IQ in the linear W24 center."""
    dims = (-2, -1)
    power = desired.abs().square().mean(dims)
    projection = (edit * desired.conj()).real.mean(dims)
    gain = projection / power.clamp_min(1e-12)
    perpendicular = edit - gain[:, None, None] * desired
    parallel_error = (gain - 1).square() * power
    perpendicular_error = perpendicular.abs().square().mean(dims)
    support = selected.to(power.dtype) / (power + .001)
    count = selected.sum().clamp_min(1)
    return torch.stack(((parallel_error * support).sum() / count,
                        (perpendicular_error * support).sum() / count))
