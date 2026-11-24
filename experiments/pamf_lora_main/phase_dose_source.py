"""Fit and evaluate both LoRa placements of a physically matched phase dose."""

from functools import partial

import numpy as np
import torch

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe
from .dura_repair import complex_iq
from .joint_spectrum_source import evaluate, smoke
from .joint_unroll_source import coordinates
from .local_phase import rotation
from .phase_dose import PhaseDose, objective
from .phase_observation_diagnostic import phase_budget, phase_ratio, protected_angle, repeat_angle
from .reference_cancel import generic_observation
from .spectral_block_source import main


RUN = '20261124-phase1-lora-phase-dose-source-m4-r01'
RECIPES = {name: FitRecipe(name, 40_000, partial(PhaseDose, local=name == 'local'),
                           generic_observation, None, coarse_first=True, batch=24, fit_per_rx=4000,
                           validation_steps=(1800, 6000, 12000, 20000, 30000))
           for name in ('pooled', 'local')}
READOUTS = ('coarse', 'full', 'half')
MODES = ('direct', 'reference', *(f'{arm}_{stage}' for arm in RECIPES for stage in READOUTS),
         'local_equal', 'pooled_equal', 'repeat_time', 'unit_action', 'fixed_noise')
COMPARISONS = (('local_equal', 'pooled_equal'), ('local_full', 'local_coarse'),
               ('local_full', 'repeat_time'), ('local_full', 'unit_action'), ('local_full', 'fixed_noise'),
               *((f'local_{stage}', f'pooled_{stage}') for stage in READOUTS),
               *((mode, base) for mode in MODES[2:] for base in ('reference', 'direct')))
FIELDS = ('active', 'outer_power', 'held_residual', 'relative_action', 'dose_spread',
          'local_edit_rms', 'equal_edit_rms', 'repeat_edit_rms')


def recover(iq, tracker, models, threshold, *, compare=False):
    y, _, quality, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    output = iq.expand(3 * len(models) + (5 if compare else 0), *iq.shape).clone()
    values = iq.new_zeros((len(iq), len(FIELDS)))
    if not count:
        return output, count, values
    raw = complex_iq(iq[active])
    y, quality, rms = y[active], quality[active], rms[active]
    waves = []
    for model in models:
        coarse, final, estimates = model.phases(y, quality, rms)
        waves.extend(raw * rotation(angle) for angle in (coarse, final, .5 * final))
    if compare:
        pooled = model.phases(y, quality, rms, local=False)[1]
        local_rms, pooled_rms = phase_ratio(raw, waves[-2]), phase_ratio(raw, raw * rotation(pooled))
        common = torch.minimum(local_rms, pooled_rms)[:, None, None]
        shared = y * (rms / rms.square().mean(1, keepdim=True).sqrt())
        repeated = raw * rotation(protected_angle(repeat_angle(shared, quality), raw))
        waves.extend((phase_budget(raw, final, common), phase_budget(raw, pooled, common), repeated))
        for options in (dict(use_action=False), dict(use_noise=False)):
            variant = model.phases(y, quality, rms, **options)[1]
            waves.append(raw * rotation(variant))
        evidence = estimates[-1]
        values[active] = torch.cat((torch.ones_like(evidence[:, :1]), evidence, local_rms[:, None],
                                    common[:, 0], phase_ratio(raw, repeated)[:, None]), dim=1)
    output[:, active] = torch.stack((torch.stack(waves).real, torch.stack(waves).imag), dim=-2)
    return output, count, values


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None, switch='local'):
    reference, _ = baseline(iq)
    output, count, values = recover(iq, tracker, models, threshold, compare=True)
    if measurements is not None:
        measurements.append(np.asarray(values.tolist()))
    return torch.cat((torch.stack((iq, reference)), output)), count


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=partial(smoke, render_fn=render, recover_fn=recover),
         run_evaluate=partial(evaluate, fields=FIELDS, render_fn=render))
