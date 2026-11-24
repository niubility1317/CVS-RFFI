"""Train and score matched LoRa covariance readout strengths."""

from functools import partial

import numpy as np
import torch

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe
from .cross_frequency_dose import CrossFrequencyDose, initialize, objective
from .joint_spectrum_source import evaluate, smoke
from .joint_unroll_source import coordinates
from .phase_dose_bursts_source import burst_observation
from .spectral_block_source import main
from .variance_transport import match_dose, physical_norm


RUN = '20261124-phase1-lora-cross-frequency-dose-source-m4-r01'
RECIPES = {
    name: FitRecipe(name, 40_000, partial(CrossFrequencyDose, separate=name == 'separate'),
                    burst_observation, None, coarse_first=True, batch=24, fit_per_rx=4000,
                    validation_steps=(1800, 6000, 12000, 20000, 30000), initialize=initialize)
    for name in ('tied', 'separate')
}
STAGES = ('coarse', 'full', 'half')
MODES = ('direct', 'reference', *(f'{arm}_{stage}' for arm in RECIPES for stage in STAGES),
         'separate_equal', 'tied_equal', 'analytic_fixed', 'unit_action', 'fixed_noise')
COMPARISONS = (('separate_equal', 'tied_equal'), ('separate_full', 'analytic_fixed'),
               ('separate_full', 'unit_action'), ('separate_full', 'fixed_noise'),
               ('separate_full', 'separate_coarse'),
               *((f'separate_{s}', f'tied_{s}') for s in STAGES),
               *((mode, control) for mode in MODES[2:] for control in ('reference', 'direct')))
FIELDS = ('active', 'common_quadrature_variance', 'private_quadrature_variance', 'held_residual',
          'relative_action', 'common_gain', 'private_gain', 'edit_rms', 'matched_rms', 'analytic_edit_rms')


def recover(iq, tracker, models, threshold, *, compare=False):
    y, phase, q, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    output = iq.expand(3 * len(models) + (5 if compare else 0), *iq.shape).clone()
    values = iq.new_zeros((len(iq), len(FIELDS)))
    if not count:
        return output, count, values
    y, phase, q, rms = (value[active] for value in (y, phase, q, rms))
    edits = []
    for model in models:
        observed = model.observe(y, q, rms)
        coarse, final, doses = model.solve(y, rms, observed)
        edits.extend((coarse, final, .5 * final))
    if compare:
        tied = model.solve(y, rms, observed, separate=False)[1]
        matched = match_dose(final, tied, rms)
        edits.extend((matched, match_dose(tied, final, rms)))
        _, rays, action, _, _, noise, mismatch = observed
        analytic = model.apply(y, rms, rays, torch.ones_like(doses))
        edits.append(analytic)
        for override in (dict(use_action=False), dict(use_noise=False)):
            state = model.observe(y, q, rms, **override)
            edits.append(model.solve(y, rms, state)[1])
        evidence = torch.stack((torch.ones_like(noise[:, 0]), *noise.unbind(-1),
                                mismatch.mean((1, 2)), (action - 1.).abs().mean((1, 2)),
                                *doses.unbind(-1)), dim=-1)
        raw = physical_norm(y, rms).flatten().clamp_min(1e-8)
        strength = torch.stack([physical_norm(edit, rms).flatten() / raw for edit in (final, matched, analytic)], -1)
        values[active] = torch.cat((evidence, strength), dim=1)
    physical = torch.stack(edits) * (phase * rms)
    output[:, active] = iq[active] + torch.stack((physical.real, physical.imag), dim=-2)
    return output, count, values


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None, switch='separate'):
    reference, _ = baseline(iq)
    output, active, values = recover(iq, tracker, models, threshold, compare=True)
    if measurements is not None:
        measurements.append(np.asarray(values.tolist()))
    return torch.cat((torch.stack((iq, reference)), output)), active


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=partial(smoke, render_fn=render, recover_fn=recover),
         run_evaluate=partial(evaluate, fields=FIELDS, render_fn=render))
