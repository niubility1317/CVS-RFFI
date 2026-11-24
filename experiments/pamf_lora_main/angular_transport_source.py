"""Train physically matched LoRa complex-direction and parallel readouts."""

from functools import partial

import numpy as np
import torch

from .angular_transport import AngularTransport, initialize, inner, objective
from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe
from .joint_spectrum_source import evaluate, smoke
from .joint_unroll_source import coordinates
from .phase_dose_bursts_source import burst_observation
from .spectral_block_source import main
from .variance_transport import physical_norm


RUN = '20261124-phase1-lora-angular-transport-source-m4-r01'
RECIPES = {
    name: FitRecipe(name, 40_000, partial(AngularTransport, angular=name == 'angular'),
                    burst_observation, None, coarse_first=True, batch=24, fit_per_rx=4000,
                    validation_steps=(1800, 6000, 12000, 20000, 30000), initialize=initialize)
    for name in ('parallel', 'angular')
}
STAGES = ('coarse', 'full', 'half')
MODES = ('direct', 'reference', *(f'{arm}_{stage}' for arm in RECIPES for stage in STAGES),
         'same_weight_parallel', 'analytic_half', 'angular_double_dose', 'unit_action', 'fixed_noise')
COMPARISONS = (('angular_full', 'same_weight_parallel'), ('angular_full', 'analytic_half'),
               ('angular_full', 'unit_action'), ('angular_full', 'fixed_noise'),
               ('angular_full', 'angular_coarse'), ('angular_double_dose', 'angular_full'),
               *((f'angular_{stage}', f'parallel_{stage}') for stage in STAGES),
               *((mode, base) for mode in MODES[2:] for base in ('direct', 'reference')))
FIELDS = ('active', 'outer_common', 'outer_private', 'held_mismatch', 'relative_action',
          'anchor_rms', 'angular_rms', 'same_weight_parallel_rms', 'angular_anchor_real_cosine',
          'same_weight_parallel_anchor_real_cosine')


def recover(iq, tracker, models, threshold, *, compare=False):
    y, angle, quality, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    output = iq.expand(3 * len(models) + (5 if compare else 0), *iq.shape).clone()
    statistics = iq.new_zeros((len(iq), len(FIELDS)))
    if not count:
        return output, count, statistics
    y, angle, q, rms = (value[active] for value in (y, angle, quality, rms))
    edits = []
    for model in models:
        observed = model.observe(y, q, rms)
        coarse, final = model.stages(y, rms, observed)
        edits.extend((coarse, final, .5 * final))
    if compare:
        anchor, _, _, _, action, _, noise, mismatch = observed
        shared = model.stages(y, rms, observed, angular=False)[1]
        edits.extend((shared, anchor, 2. * final))
        for override in (dict(use_action=False), dict(use_noise=False)):
            state = model.observe(y, q, rms, **override)
            edits.append(model.stages(y, rms, state)[1])
        size = physical_norm(y, rms).flatten().clamp_min(1e-8)
        norms = torch.stack([physical_norm(plan, rms).flatten() / size for plan in (anchor, final, shared)], 1)
        projection = torch.stack([(inner(edit, anchor, rms)
                                  / (physical_norm(edit, rms) * physical_norm(anchor, rms)).clamp_min(1e-10))
                                  .flatten() for edit in (final, shared)], 1)
        statistics[active] = torch.cat((torch.ones_like(noise[:, :1]), noise, mismatch.mean((1, 2))[:, None],
                                       (action - 1.).abs().mean((1, 2))[:, None], norms, projection), dim=1)
    physical = torch.stack(edits) * angle * rms
    output[:, active] = iq[active] + torch.stack((physical.real, physical.imag), dim=-2)
    return output, count, statistics


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None, switch='angular'):
    reference, _ = baseline(iq)
    output, count, evidence = recover(iq, tracker, models, threshold, compare=True)
    if measurements is not None:
        measurements.append(np.asarray(evidence.tolist()))
    return torch.cat((torch.stack((iq, reference)), output)), count


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=partial(smoke, render_fn=render, recover_fn=recover),
         run_evaluate=partial(evaluate, fields=FIELDS, render_fn=render))
