"""Compare point and split-evidence variance fusion on LoRa source IQ."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, panel, receive_only, setup
from .chirp_spectrum_source import dispatch, freeze, score_frozen
from .complex_vector_source import load_reader, publish, score
from .crossfit_pilot import paired_action_inputs
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .mirror_refine_source import load_model
from .natural_detail import natural_objective
from .rf_noise_transport import colored_observation
from .rf_noise_transport_source import timing
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, TAIL_WEIGHT, WIDTHS
from .variance_bound import VarianceBound


RUN = '20261124-phase1-lora-variance-envelope-source-m4-r01'
RECIPES = {
    arm: FitRecipe(arm, STEPS, partial(VarianceBound, WIDTHS, uncertain=uncertain), colored_observation,
                   VarianceBound.forward)
    for arm, uncertain in (('point', False), ('interval', True))
}
MODES = ('direct', 'reference', 'point_coarse', 'point', 'point_half',
         'interval_coarse', 'interval', 'interval_half', 'interval_center_half')
PAIRS = (('interval_coarse', 'point_coarse'), ('interval', 'point'), ('interval_half', 'point_half'),
         ('interval_half', 'interval_center_half'))


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold, statistics=None):
    reference, _ = baseline(iq)
    observed, phase, quality, rms = coordinates(tracker, iq)
    disturbance = corruption_score(observed)
    active = disturbance > threshold
    if not active.any():
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    y, phase, quality, rms = (v[active] for v in (observed, phase, quality, rms))
    point, interval = models
    a, b = point(y, quality, phase)
    c, d, alternate, width, reduction = interval(y, quality, phase, controls=True)
    if statistics is not None:
        mild = disturbance[active] <= .15
        for i, selected in enumerate((mild, ~mild)):
            statistics[i, 0] += selected.sum()
            statistics[i, 1:5] += (width[:, 0] * selected[:, None]).sum(0)
            statistics[i, 5:9] += (reduction * selected[:, None]).sum(0)
    values = (a, b, torch.lerp(y, b, .5), c, d, torch.lerp(y, d, .5), torch.lerp(y, alternate, .5))
    outputs = [iq, reference]
    for value in values:
        restored = iq.clone()
        restored[active] += real_iq((value - y) * phase * rms)
        outputs.append(restored)
    return torch.stack(outputs), int(active.sum())


@torch.no_grad()
def checkpoint_smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, calibration, _), contract = setup(rt, device, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if contract != expected:
        raise ValueError('The source U recovery and actual recognition checkpoint require the same IQ contract')
    models = [recipe.model().to(device).eval() for recipe in RECIPES.values()]
    rng = torch.Generator(device=device).manual_seed(2026112403)
    batch = real_iq(colored_observation(complex_iq(calibration[:2]), rng))
    values, active = render(batch, None, baseline, tracker, models, threshold=-1.)
    logits = load_reader(root, contract, device)(values.flatten(0, 1))[0]
    if active != len(batch) or logits.shape != (len(MODES) * len(batch), 10) or not torch.isfinite(logits).all():
        raise ValueError('Split-variance real-checkpoint smoke failed')
    write(root / 'checkpoint_smoke.json', dict(status='PASS', variants=len(MODES), source_U=len(batch),
                                              target_reads=0))
    print('LORA_SPLIT_VARIANCE_REAL_CHECKPOINT_PASS', flush=True)


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    models, thresholds = zip(*(load_model(root, r, binding[2], device) for r in RECIPES.values()), strict=True)
    if thresholds[0] != thresholds[1]:
        raise ValueError('Both arms require the same source U gate')
    stats = torch.zeros((2, 9), device=device)
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=thresholds[0], statistics=stats),
          binding=binding, frozen_models=models, replay_fn=receive_only, finalize=freeze, reader_chunk=144)
    keys = ('active_at_most_0.15', 'active_above_0.15')
    output = {}
    for key, (n, *components) in zip(keys, stats.tolist(), strict=True):
        values = [x / max(1, n) for x in components]
        output[key] = dict(exposures=int(n), mean_width=values[:4], mean_gain_reduction=values[4:])
    write(root / f'{route}-variance-width.json', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for action in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        stage = commands.add_parser(action)
        stage.add_argument('runtime', type=Path)
        if action != 'dispatch':
            names = RECIPES if action in ('smoke', 'fit') else ROUTES if action == 'evaluate' else ('source',)
            stage.add_argument('name', choices=names)
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        publish(**options, run=RUN, runner=__spec__.name)
        return
    rt = read(options['runtime'])
    if action == 'dispatch':
        dispatch(rt, RECIPES, initial_stages=(('checkpoint_smoke', ('source',)),))
    elif action == 'score':
        score_frozen(rt, modes=MODES, comparisons=PAIRS)
    else:
        activate(rt)
        if action == 'checkpoint_smoke':
            checkpoint_smoke(rt)
        elif action == 'evaluate':
            evaluate(rt, options['name'])
        elif action == 'timing':
            timing(rt, RECIPES['interval'])
        else:
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[options['name']], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35,
                                  make_pair=partial(paired_action_inputs, echo_probability=.5),
                                  tail_weight=TAIL_WEIGHT))


if __name__ == '__main__':
    main()
