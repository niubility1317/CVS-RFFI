"""Train matched base-width and expanded DURA source-only LoRa restorers."""

from functools import partial

import numpy as np
import torch

from .channel_blind import corruption_score
from .dura_capacity import DuraCapacity, WIDTHS, objective
from .joint_spectrum_source import evaluate, smoke
from .joint_unroll_source import coordinates
from .posterior_transport import condition_waveforms
from .spectral_block_source import FitRecipe, main
from .variance_transport import physical_norm


RUN = '20261124-phase1-lora-dura-capacity-source-m4-r01'
ARMS = tuple(WIDTHS)
READOUTS = ('coarse', 'stage_one', 'stage_two', 'half')
RECIPES = {
    size: FitRecipe(size, 40_000, partial(DuraCapacity, size), condition_waveforms, None,
                    coarse_first=True, batch=16, fit_per_rx=4000,
                    validation_steps=(1800, 6000, 12000, 20000, 30000))
    for size in ARMS
}
MODES = ('direct', 'reference', *(f'{arm}_{name}' for arm in ARMS for name in READOUTS),
         'expanded_unit_action', 'expanded_fixed_noise', 'expanded_no_cg')
COMPARISONS = (('expanded_half', 'base_half'), ('expanded_stage_two', 'base_stage_two'),
               ('expanded_coarse', 'base_coarse'), ('expanded_stage_two', 'expanded_stage_one'),
               ('expanded_half', 'expanded_unit_action'), ('expanded_half', 'expanded_fixed_noise'),
               ('expanded_half', 'expanded_no_cg'),
               *((name, base) for name in MODES[2:] for base in ('direct', 'reference')))
FIELDS = ('active', 'outer_variance', 'used_variance', 'held_mismatch',
          'relative_action', 'mirror_ratio', 'echo_ratio', 'coarse_dose', 'first_dose', 'final_dose')


def recover(iq, tracker, models, threshold, *, compare=False, robust=False, activation_score=corruption_score):
    y, phase, quality, rms = coordinates(tracker, iq)
    active = activation_score(y) > threshold
    count = int(active.sum())
    output = iq.expand(4*len(models)+(3 if compare else 0), *iq.shape).clone()
    records = iq.new_zeros((len(iq), len(FIELDS)))
    if not count:
        return output, count, records
    y, phase, quality, rms = [x[active] for x in (y, phase, quality, rms)]
    edits = []
    for model in models:
        (coarse, first, final), operator, variance, noise, mismatch = model.stages(y, quality, rms, robust=robust)
        changes = (coarse-y, first-y, final-y)
        edits.extend((*changes, .5*changes[-1]))
    if compare:
        for override in ({'use_action': False}, {'use_noise': False}, {'solve': False}):
            stage, *_ = model.stages(y, quality, rms, **override)
            edits.append(.5*(stage[-1]-y))
        dose = physical_norm(y, rms).flatten().clamp_min(1e-7)
        measure = [physical_norm(edit, rms).flatten()/dose for edit in changes]
        fields = (noise, variance, mismatch, (operator.main-1.).abs(),
                  operator.image.abs(), operator.echo.abs())
        records[active] = torch.stack((torch.ones_like(dose),
                                      *(value.mean((1, 2)) for value in fields), *measure), 1)
    physical = torch.stack(edits)*phase*rms
    output[:, active] = iq[active] + torch.stack((physical.real, physical.imag), -2)
    return output, count, records


def render(iq, _unused, reference, tracker, models, *, threshold, measurements=None, switch=None):
    base, _ = reference(iq)
    output, count, records = recover(iq, tracker, models, threshold, compare=True)
    if measurements is not None:
        measurements.append(np.asarray(records.tolist()))
    return torch.cat((torch.stack((iq, base)), output)), count


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES,
         comparisons=COMPARISONS, calculate=objective,
         run_smoke=partial(smoke, render_fn=render, recover_fn=recover),
         run_evaluate=partial(evaluate, fields=FIELDS, render_fn=render))
