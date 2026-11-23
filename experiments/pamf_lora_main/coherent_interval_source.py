"""Fit and score matched LoRa common-spectrum constraint arms."""

from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, panel, receive_only, setup
from .coherent_interval import CoherentInterval, initialize, objective
from .complex_vector_source import load_reader, score
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze
from .reference_cancel import generic_observation
from .spectral_block_source import load_models, main
from .variance_transport import match_dose, physical_norm


RUN = '20261124-phase1-lora-coherent-interval-source-m4-r01'
RECIPES = {
    name: FitRecipe(name, 40_000, partial(CoherentInterval, directional=name == 'interval'),
                    generic_observation, None, coarse_first=True, batch=24, fit_per_rx=4000,
                    validation_steps=(1800, 6000, 12000, 20000, 30000), initialize=initialize)
    for name in ('scalar', 'interval')
}
MODES = ('direct', 'reference', 'scalar_coarse', 'scalar_full', 'scalar_half',
         'interval_coarse', 'interval_full', 'interval_half', 'interval_matched', 'scalar_matched',
         'interval_off', 'interval_without_action', 'interval_without_noise')
COMPARISONS = (('interval_matched', 'scalar_matched'),
               *((f'interval_{name}', f'scalar_{name}') for name in ('coarse', 'full', 'half')),
               ('interval_full', 'interval_coarse'),
               *(('interval_full', name) for name in MODES[10:]),
               *((mode, 'reference') for mode in MODES[2:]))
FIELDS = ('active', 'weak_confidence', 'support_fraction', 'relative_action_confidence',
          'outer_variance', 'interval_full_dose', 'matched_dose')
DEVICE = 'cuda:0'


def recovery(iq, tracker, models, threshold, *, ablate=False):
    y, rotation, quality, rms = coordinates(tracker, iq)
    enabled = corruption_score(y) > threshold
    n_output = 3 * len(models) + (5 if ablate else 0)
    result = iq.expand(n_output, *iq.shape).clone()
    metrics = iq.new_zeros((len(iq), len(FIELDS)))
    active = int(enabled.sum())
    if not active:
        return result, active, metrics
    y, phase, q, rms = (value[enabled] for value in (y, rotation, quality, rms))
    raw = iq[enabled]
    updates = []
    for model in models:
        coarse, fine, _, _, evidence, trust = model.stages(y, q, rms)
        updates.extend((coarse - y, fine - y, .5 * (fine - y)))
    if ablate:
        interval = models[-1]
        full = fine - y
        scalar = interval.stages(y, q, rms, constraint='scalar')[1] - y
        matched_directional, matched_scalar = match_dose(full, scalar, rms), match_dose(scalar, full, rms)
        updates.extend((matched_directional, matched_scalar))
        for options in (
            {'constraint': 'off'}, {'use_action': False}, {'use_noise': False},
        ):
            updates.append(interval.stages(y, q, rms, **options)[1] - y)
        noise = interval.estimate_power(interval.transformed(y)).mean((1, 2, 3))
        ratio = physical_norm(torch.stack((full, matched_directional)), rms) / physical_norm(y, rms).clamp_min(1e-7)
        support = evidence[..., interval.weak]
        metrics[enabled] = torch.stack((torch.ones_like(noise), support.mean((1, 2)),
                                      (support > .1).float().mean((1, 2)), trust.mean((1, 2)),
                                      noise, *ratio[..., 0, 0].unbind(0)), dim=1)
    physical = torch.stack(updates) * phase * rms
    result[:, enabled] = raw + torch.stack((physical.real, physical.imag), dim=-2)
    return result, active, metrics


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None):
    ref, _ = baseline(iq)
    out, active, metric = recovery(iq, tracker, models, threshold, ablate=True)
    if measurements is not None:
        measurements.append(np.asarray(metric.tolist()))
    return torch.cat((torch.stack((iq, ref)), out)), active


@torch.no_grad()
def smoke(rt, recipes, modes):
    root = Path(rt['run_root'])
    (_, source, _), contract = setup(rt, DEVICE, smoke=True)
    models, threshold = load_models(root, contract, DEVICE, recipes)
    _, _, expected, baseline, tracker = source_control(rt, 'full', DEVICE)
    if contract != expected:
        raise ValueError('Expected a matching source contract for LoRa smoke')
    reader = load_reader(root, contract, DEVICE)
    rng = torch.Generator(device=DEVICE).manual_seed(2026112414)
    perturbed = real_iq(generic_observation(complex_iq(source), rng))
    disturbed = perturbed[corruption_score(coordinates(tracker, perturbed)[0]).argmax().view(1)]
    clean = source[corruption_score(coordinates(tracker, source)[0]).argmin().view(1)]
    incoming = torch.cat((clean, disturbed))
    outputs, active = render(incoming, None, baseline, tracker, models, threshold=threshold)
    logits = reader(outputs.flatten(0, 1))[0]
    if active != 1 or logits.shape != (2 * len(modes), 10) or not torch.isfinite(logits).all():
        raise ValueError('Need a real frozen no-query recognition result and active restoration')
    if not torch.equal(outputs[2:, 0], clean.expand_as(outputs[2:, 0])):
        raise ValueError('Expected exact clean bypass')
    timing = {}
    for label, iq in (('clean', clean), ('generic_disturbed', disturbed)):
        records = []
        for i in range(28):
            torch.cuda.synchronize()
            began = perf_counter()
            recovery(iq, tracker, models[-1:], threshold)
            torch.cuda.synchronize()
            if i >= 8:
                records.append(1000 * (perf_counter() - began))
        timing[label] = float(np.median(records))
    write(root/'smoke.json', dict(status='PASS', V_reads=0, target_reads=0, timing_b1_ms=timing,
          params={name:sum(p.numel() for p in m.parameters()) for name, m in zip(recipes, models, strict=True)},
          device=torch.cuda.get_device_name()))


def evaluate(rt, route, recipes, modes):
    binding = source_control(rt, route, DEVICE)
    models, threshold = load_models(Path(rt['run_root']), binding[2], DEVICE, recipes)
    measurements = []

    def finish(root, route, source, chosen, ids, predictions, metrics, modes):
        values = np.concatenate(measurements).reshape(len(VIEWS), len(chosen), len(FIELDS))
        by_view = {}
        for view, receivers in zip(VIEWS, values, strict=True):
            by_rx = {}
            for rx, group in zip(SOURCE, np.split(receivers, len(SOURCE)), strict=True):
                retained = group[group[:, 0] > 0]
                stats = {'active': len(retained)}
                if len(retained):
                    summary = np.quantile(retained[:, 1:], (.1, .5, .9), axis=0).T.tolist()
                    stats.update(zip(FIELDS[1:], summary, strict=True))
                by_rx[rx] = stats
            by_view[view] = by_rx
        write(root/f'{route}-estimates.json', dict(quantiles=[.1, .5, .9], by_view=by_view))
        freeze(root, route, source, chosen, ids, predictions, metrics, modes)

    score(rt, route, panel=panel, modes=modes, binding=binding, frozen_models=models, replay_fn=receive_only,
          render=partial(render, threshold=threshold, measurements=measurements), require_clean_bypass=False,
          finalize=finish)


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=smoke, run_evaluate=evaluate)
