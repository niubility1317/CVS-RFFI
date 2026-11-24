"""Compare one and two phase correction directions under the same long training budget."""

from dataclasses import replace
from functools import partial

import numpy as np
import torch

from .channel_blind import corruption_score
from .dura_repair import complex_iq
from .joint_spectrum_source import evaluate, smoke
from .joint_unroll_source import coordinates
from .local_phase import rotation
from .neighbor_phase import NeighborPhaseDose
from .phase_dose import objective
from .phase_dose_bursts_source import RECIPES as BURST_RECIPES
from .phase_observation_diagnostic import phase_budget, phase_ratio
from .spectral_block_source import main


RUN = '20261124-phase1-lora-neighbor-phase-source-m4-r01'
RECIPES = {name: replace(recipe, name=name, model=partial(NeighborPhaseDose, use_neighbor=name == 'neighbor'))
           for name, recipe in zip(('repeat', 'neighbor'), BURST_RECIPES.values(), strict=True)}
STAGES = ('coarse', 'full', 'half')
MODES = ('direct', 'reference', *(f'{arm}_{stage}' for arm in RECIPES for stage in STAGES),
         'neighbor_equal', 'repeat_equal', 'neighbor_off', 'unit_action', 'fixed_noise')
COMPARISONS = (('neighbor_equal', 'repeat_equal'), ('neighbor_full', 'neighbor_off'),
               ('neighbor_full', 'unit_action'), ('neighbor_full', 'fixed_noise'),
               *((f'neighbor_{stage}', f'repeat_{stage}') for stage in STAGES),
               *((mode, base) for mode in MODES[2:] for base in ('reference', 'direct')))
FIELDS = ('active', 'outer_power', 'held_residual', 'relative_action', 'neighbor_agreement',
          'neighbor_phase_rms_before_joint_cap', 'neighbor_edit_rms', 'equal_edit_rms')


def recover(iq, tracker, models, threshold, *, compare=False):
    y, _, quality, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    output = iq.expand(3 * len(models) + (5 if compare else 0), *iq.shape).clone()
    measurements = iq.new_zeros((len(iq), len(FIELDS)))
    if not count:
        return output, count, measurements
    raw = complex_iq(iq[active])
    y, quality, rms = y[active], quality[active], rms[active]
    waves = []
    for model in models:
        coarse, phase, evidence = model.phases(y, quality, rms)
        waves.extend(raw * rotation(angle) for angle in (coarse, phase, .5 * phase))
    if compare:
        no_neighbor = model.phases(y, quality, rms, use_neighbor=False)[1]
        repeat = raw * rotation(no_neighbor)
        dose = phase_ratio(raw, waves[-2])
        equal = torch.minimum(dose, phase_ratio(raw, repeat))[:, None, None]
        waves.extend((phase_budget(raw, phase, equal), phase_budget(raw, no_neighbor, equal), repeat))
        for override in (dict(use_action=False), dict(use_noise=False)):
            waves.append(raw * rotation(model.phases(y, quality, rms, **override)[1]))
        info = evidence[-1]
        measurements[active] = torch.cat((torch.ones_like(info[:, :1]), info, dose[:, None], equal[:, 0]), 1)
    stacked = torch.stack(waves)
    output[:, active] = torch.stack((stacked.real, stacked.imag), dim=-2)
    return output, count, measurements


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None, switch='use_neighbor'):
    reference, _ = baseline(iq)
    output, count, evidence = recover(iq, tracker, models, threshold, compare=True)
    if measurements is not None:
        measurements.append(np.asarray(evidence.tolist()))
    return torch.cat((torch.stack((iq, reference)), output)), count


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=partial(smoke, render_fn=render, recover_fn=recover),
         run_evaluate=partial(evaluate, fields=FIELDS, render_fn=render))
