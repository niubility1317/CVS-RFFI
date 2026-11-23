"""Compare held-out preamble equalization with a matched no-response network."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, panel, receive_only, setup
from .chirp_spectrum_source import dispatch, freeze, score_frozen
from .complex_vector_source import load_reader, publish, score
from .crossfit_pilot import CrossfitPilot, paired_action_inputs
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .mirror_refine_source import load_model
from .natural_detail import natural_objective
from .pilot_joint import PilotJoint
from .rf_noise_transport import colored_observation
from .rf_noise_transport_source import timing
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, TAIL_WEIGHT, WIDTHS


RUN = '20261124-phase1-lora-crossfit-pilot-response-source-m4-r01'
RECIPES = {
    'identity': FitRecipe('identity', STEPS,
                          partial(PilotJoint, equalize=False, widths=WIDTHS, companion='mirror'),
                          colored_observation, PilotJoint.forward),
    'crossfit': FitRecipe('crossfit', STEPS, partial(CrossfitPilot, WIDTHS),
                          colored_observation, CrossfitPilot.forward),
}
MODES = ('direct', 'reference', 'engineering', 'identity_coarse', 'identity', 'identity_half',
         'crossfit_coarse', 'crossfit', 'crossfit_half')
PAIRS = (('engineering', 'direct'), ('crossfit_coarse', 'identity_coarse'),
         ('crossfit', 'identity'), ('crossfit_half', 'identity_half'))


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold, statistics=None):
    reference, _ = baseline(iq)
    y, rotation, quality, rms = coordinates(tracker, iq)
    disturbance = corruption_score(y)
    active = disturbance > threshold
    if not bool(active.any()):
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    selected = tuple(tensor[active] for tensor in (y, rotation, quality, rms))
    observed, basis, q, scale = selected
    identity, crossfit = models
    a, b = identity(observed, q, basis)
    c, d, engineering, support = crossfit(observed, q, basis, diagnostics=True)
    if statistics is not None:
        mild = disturbance[active] <= .15
        for i, mask in enumerate((mild, ~mild)):
            statistics[i, 0] += mask.sum()
            statistics[i, 1] += (mask & (support > 0)).sum()
            statistics[i, 2] += (support * mask).sum()
    values = (engineering, a, b, torch.lerp(observed, b, .5), c, d, torch.lerp(observed, d, .5))
    outputs = [iq, reference]
    for value in values:
        out = iq.clone()
        out[active] += real_iq((value - observed) * basis * scale)
        outputs.append(out)
    return torch.stack(outputs), int(active.sum())


@torch.no_grad()
def checkpoint_smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, calibration, _), contract = setup(rt, device, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if contract != expected:
        raise ValueError('Unexpected source U and frozen recognition contract')
    models = [recipe.model().to(device).eval() for recipe in RECIPES.values()]
    rng = torch.Generator(device=device).manual_seed(2026112403)
    batch = real_iq(colored_observation(complex_iq(calibration[:2]), rng))
    outputs, active = render(batch, None, baseline, tracker, models, threshold=-1.)
    logits = load_reader(root, contract, device)(outputs.flatten(0, 1))[0]
    if active != len(batch) or logits.shape != (len(MODES) * len(batch), 10) or not torch.isfinite(logits).all():
        raise ValueError('Cross-fitted source U checkpoint smoke failed')
    write(root / 'checkpoint_smoke.json', dict(status='PASS', variants=len(MODES), source_U=len(batch),
                                              target_reads=0))
    print('LORA_CROSSFIT_REAL_CHECKPOINT_PASS', flush=True)


def evaluate(rt, route):
    root = Path(rt['run_root'])
    binding = source_control(rt, route, 'cuda:0')
    results = [load_model(root, recipe, binding[2], 'cuda:0') for recipe in RECIPES.values()]
    models, thresholds = zip(*results, strict=True)
    if thresholds[0] != thresholds[1]:
        raise ValueError('Different source U bypass gates in a matched comparison')
    stats = torch.zeros((2, 3), device='cuda:0')
    renderer = partial(render, threshold=thresholds[0], statistics=stats)
    score(rt, route, panel=panel, modes=MODES, render=renderer, binding=binding,
          frozen_models=models, replay_fn=receive_only, finalize=freeze, reader_chunk=144)
    output = {}
    for label, (n, held, total) in zip(('active_at_most_0.15', 'active_above_0.15'), stats.tolist(), strict=True):
        output[label] = dict(exposures=int(n), crossfit_positive=int(held), mean_support=total/max(n, 1))
    write(root / f'{route}-crossfit-support.json', output)


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
            timing(rt, RECIPES['crossfit'])
        else:
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[options['name']], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35, make_pair=paired_action_inputs,
                                  tail_weight=TAIL_WEIGHT))


if __name__ == '__main__':
    main()
