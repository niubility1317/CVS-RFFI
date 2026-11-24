"""Fit matched LoRa phase allocation under generic local and correlated bursts."""

from dataclasses import replace
from functools import partial

import torch

from .joint_spectrum_source import evaluate, smoke
from .local_phase import rotation
from .phase_dose import objective
from .phase_dose_source import COMPARISONS, FIELDS, MODES, RECIPES as BASE, recover, render
from .reference_cancel import generic_observation
from .spectral_block_source import main


RUN = '20261124-phase1-lora-phase-dose-bursts-source-m4-r01'


def burst_observation(wave, rng):
    """Add burst-local observations with independently and jointly affected chirps."""
    batch, chirps, samples = wave.shape
    base = generic_observation(wave, rng)

    def draw(shape):
        return torch.rand(shape, device=wave.device, dtype=wave.real.dtype, generator=rng)

    positions = torch.linspace(0., 1., samples, device=wave.device, dtype=wave.real.dtype)
    shape = (batch, 1, 1)
    family = draw(shape)
    shared = family >= .70
    common_center = .08 + .84 * draw(shape)
    centers = .08 + .84 * draw((batch, chirps, 1))
    centers = torch.where(shared, common_center, centers)
    radius = .055 + .23 * draw(shape)
    distance = ((positions - centers).abs() / radius).clamp_max(1.)
    envelope = .5 + .5 * (torch.pi * distance).cos()
    ranks = draw((batch, chirps, 1)).argsort(dim=1).argsort(dim=1)
    count = torch.randint(2, 5, shape, device=wave.device, generator=rng)
    members = ranks < count
    group_phase = .40 + 1.40 * draw(shape)
    group_phase = torch.where(draw(shape) < .50, -group_phase, group_phase)
    phase = torch.where(shared, group_phase * members, group_phase * (2 * draw((batch, chirps, 1)) - 1))
    attenuation = .12 + .45 * draw(shape)
    mixed = wave + (attenuation + (1 - attenuation) * envelope) * (base - wave)
    burst = mixed * rotation(phase * envelope)
    return torch.where(family < .40, base, burst)


RECIPES = {name: replace(recipe, perturb=burst_observation) for name, recipe in BASE.items()}


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=partial(smoke, render_fn=render, recover_fn=recover),
         run_evaluate=partial(evaluate, fields=FIELDS, render_fn=render))
