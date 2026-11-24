"""Train a matched LoRa target-coordinate ablation and freeze both source readouts."""

from functools import partial

import numpy as np
import torch

from .angular_transport import initialize
from .band_allocation import band_energy, fft
from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe
from .joint_spectrum_source import evaluate, smoke
from .joint_unroll_source import coordinates
from .phase_dose_bursts_source import burst_observation
from .spectral_block_source import main
from .weak_sideband import WeakSideband, objective, physical_inner


RUN = '20261124-phase1-lora-weak-sideband-source-m4-r01'
RECIPES = {
    name: FitRecipe(name, 40_000, partial(WeakSideband, separated=name == 'separate'),
                    burst_observation, None, coarse_first=True, batch=24, fit_per_rx=4000,
                    validation_steps=(1800, 6000, 12000, 20000, 30000), initialize=initialize)
    for name in ('fullref', 'separate')
}
STAGES = ('coarse', 'final')
MODES = ('direct', 'reference', *(f'{name}_{stage}' for name in RECIPES for stage in STAGES),
         'anchor', 'unit_action', 'fixed_noise', 'no_feedback')
COMPARISONS = (('separate_coarse', 'fullref_coarse'), ('separate_final', 'fullref_final'),
               ('separate_final', 'separate_coarse'),
               *((mode, control) for mode in MODES[2:] for control in ('direct', 'reference', 'anchor')
                 if mode != control))
FIELDS = ('active', 'outer_common', 'outer_private', 'held_residual', 'anchor_rms_fraction',
          'fullref_cosine', 'separate_cosine', 'near_cosine', 'middle_cosine',
          'maximum_band_energy_error')


def recover(iq, tracker, models, threshold, *, compare=False):
    observed, phase, quality, rms = coordinates(tracker, iq)
    active = corruption_score(observed) > threshold
    count = int(active.sum())
    outputs = iq.expand(2*len(models) + (4 if compare else 0), *iq.shape).clone()
    telemetry = iq.new_zeros((len(iq), len(FIELDS)))
    if not count:
        return outputs, count, telemetry
    y, rotation, q, rms = (value[active] for value in (observed, phase, quality, rms))
    edits = []
    for model in models:
        state = model.observe(y, q, rms)
        edits.extend(model.stages(y, rms, state))
    if compare:
        anchor, _, _, _, _, _, noise, mismatch = state
        values = fft(torch.stack(edits))
        mask = model.weak_bands.to(y.real)
        reference = fft(anchor)
        band_base = band_energy(reference, rms, mask)
        error = (band_energy(values, rms, mask) - band_base).abs().amax((0,2))
        relative_error = error / band_base.sum(-1).clamp_min(1e-8)
        if (relative_error > 1e-4).any() or not torch.isfinite(values).all():
            raise ValueError('Expected shared per-band physical dose')
        overlap = physical_inner(values, reference, rms)
        total = (physical_inner(values, values, rms) * physical_inner(reference, reference, rms))
        cosine = (overlap / total.clamp_min(1e-16).sqrt()).squeeze(-1).squeeze(-1)
        band_overlap = ((values[-1] * reference.conj()).real * rms.square()).sum(-2) @ mask.T
        band_cosine = band_overlap / band_base.clamp_min(1e-8)
        dose = (physical_inner(anchor, anchor, rms) / physical_inner(y, y, rms).clamp_min(1e-8)).sqrt()
        telemetry[active] = torch.cat((torch.ones_like(noise[:, :1]), noise,
                                       mismatch.mean((1,2))[:,None], dose.flatten()[:,None],
                                       cosine[[1,3]].T, band_cosine, relative_error[:,None]), 1)
        edits.append(anchor)
        for override in (dict(use_action=False), dict(use_noise=False)):
            other = model.observe(y, q, rms, **override)
            edits.append(model.stages(y, rms, other)[1])
        edits.append(model.stages(y, rms, state, use_action=False)[1])
    adjustment = torch.stack(edits) * rotation * rms
    outputs[:,active] = iq[active] + torch.stack((adjustment.real, adjustment.imag), dim=-2)
    return outputs, count, telemetry


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None, switch=None):
    prior, _ = baseline(iq)
    outputs, count, telemetry = recover(iq, tracker, models, threshold, compare=True)
    if measurements is not None:
        measurements.append(np.asarray(telemetry.tolist()))
    return torch.cat((torch.stack((iq, prior)), outputs)), count


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=partial(smoke, render_fn=render, recover_fn=recover),
         run_evaluate=partial(evaluate, fields=FIELDS, render_fn=render))
