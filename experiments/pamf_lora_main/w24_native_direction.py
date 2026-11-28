"""Separate the amount of coarse restoration from a new complex direction."""

import torch

from experiments.pamf_lora_main.w24_native_noise_pair import NoisePairRefiner


COARSE_DOSE = 1.5
DIRECTION_WEIGHT = .35


class DirectionRefiner(NoisePairRefiner):
    """Propose a time/frequency residual with zero DC in the tracked frame."""

    def __init__(self, *, projected):
        super().__init__()
        self.projected = projected

    def forward(self, observed, coarse):
        edit = super().forward(observed, coarse)
        return edit - edit.mean(-1, keepdim=True)


def direction_coefficient(edit, axis):
    dims = (-2, -1)
    return (edit * axis.conj()).mean(dims) / axis.abs().square().mean(dims).clamp_min(1e-12)


def remove_coarse_component(proposal, coarse, frontend):
    edit, axis = (frontend.aperture(wave)[..., :384] for wave in (proposal, coarse))
    coefficient = direction_coefficient(edit, axis)
    residual = proposal - coefficient[:, None] * coarse
    # A weak W24 axis can imply a large correction outside the center window.
    # A packetwise scalar limits that correction without changing the direction.
    old_power, new_power = (v.abs().square().mean(-1, keepdim=True) for v in (proposal, residual))
    return residual * ((old_power + 1e-12) / (new_power + 1e-12)).sqrt().clamp(max=1.)


def projected_pair_loss(edit, axis, remaining, selected):
    coefficient = direction_coefficient(remaining, axis)
    target = remaining - coefficient[:, None, None] * axis
    error = (edit - target).abs().square().mean((-2, -1))
    power = remaining.abs().square().mean((-2, -1))
    return (error.masked_fill(~selected, 0.) / (power + .001)).sum() / selected.sum().clamp_min(1)
