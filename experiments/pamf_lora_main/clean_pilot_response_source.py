"""Fit clean-source response protection and coarse corroboration in one loop."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, panel, receive_only, setup
from .chirp_spectrum_source import dispatch, freeze, score_frozen
from .clean_pilot_response import ProtectedPilot
from .complex_vector_source import load_reader, publish, score
from .crossfit_pilot import estimate_shared, paired_action_inputs
from .crossfit_pilot_source import RECIPES as PREVIOUS_RECIPES, RUN as PREVIOUS_RUN
from .data import SOURCE
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .mirror_refine_source import load_model
from .natural_detail import natural_objective
from .rf_noise_transport import colored_observation
from .rf_noise_transport_source import timing
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, TAIL_WEIGHT, WIDTHS
from .weak_signal_probe import observed_noise_rms


RUN = '20261124-phase1-lora-clean-pilot-protected-source-m4-r01'
PACKETS_PER_RECEIVER = 100


@torch.no_grad()
def calibrate(model, tracker, source_u):
    balanced = torch.cat([packets[:PACKETS_PER_RECEIVER] for packets in source_u.chunk(len(SOURCE))])
    clean_responses = []
    for packet in balanced.split(20):
        observed, rotation, quality, _ = coordinates(tracker, packet)
        response, _ = estimate_shared(observed, rotation, quality, observed_noise_rms(observed).square())
        clean_responses.append(response)
    model.calibration.fit(torch.cat(clean_responses))


RECIPES = {
    arm: FitRecipe(arm, STEPS, partial(ProtectedPilot, WIDTHS, anchor=anchor), colored_observation,
                   ProtectedPilot.forward, initialize=calibrate)
    for arm, anchor in (('protected', False), ('anchored', True))
}
ARMS = ('crossfit', *RECIPES)
MODES = ('direct', 'reference', 'engineering', 'protected_engineering',
         *(name for arm in ARMS for name in (f'{arm}_coarse', arm, f'{arm}_half')))
PAIRS = (
    ('engineering','direct'), ('protected_engineering','engineering'),
    *((f'protected{suffix}', f'crossfit{suffix}') for suffix in ('_coarse', '', '_half')),
    *((f'anchored{suffix}', f'protected{suffix}') for suffix in ('_coarse', '', '_half')),
)


def parent_model(root, contract, device):
    parent = root.parent / PREVIOUS_RUN
    if read(parent / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the completed uncalibrated source-only restoration')
    return load_model(parent, PREVIOUS_RECIPES['crossfit'], contract, device)


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference, _ = baseline(iq)
    y, rotation, quality, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    if not active.any():
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    y, rotation, quality, rms = (value[active] for value in (y, rotation, quality, rms))
    previous, protected, anchored = models
    a, b, engineering, _ = previous(y, quality, rotation, diagnostics=True)
    c, d, calibrated, _ = protected(y, quality, rotation, diagnostics=True)
    e, f = anchored(y, quality, rotation)
    values = (engineering, calibrated, a, b, torch.lerp(y, b, .5), c, d, torch.lerp(y, d, .5),
              e, f, torch.lerp(y, f, .5))
    outputs = [iq, reference]
    for value in values:
        restored = iq.clone()
        restored[active] += real_iq((value - y) * rotation * rms)
        outputs.append(restored)
    return torch.stack(outputs), int(active.sum())


@torch.no_grad()
def checkpoint_smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (training, calibration, _), contract = setup(rt, device, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if contract != expected:
        raise ValueError('The source U fit and actual recognizer require the same IQ contract')
    previous, _ = parent_model(root, contract, device)
    models = [previous]
    for recipe in RECIPES.values():
        model = recipe.model().to(device).eval()
        calibrate(model, tracker, training)
        models.append(model)
    rng = torch.Generator(device=device).manual_seed(2026112403)
    batch = real_iq(colored_observation(complex_iq(calibration[:2]), rng))
    values, active = render(batch, None, baseline, tracker, models, threshold=-1.)
    logits = load_reader(root, contract, device)(values.flatten(0, 1))[0]
    if active != len(batch) or logits.shape != (len(MODES) * len(batch), 10) or not torch.isfinite(logits).all():
        raise ValueError('Protected pilot real-checkpoint smoke failed')
    write(root / 'checkpoint_smoke.json', dict(status='PASS', variants=len(MODES), source_U=len(batch),
                                              target_reads=0))
    print('LORA_PROTECTED_PILOT_REAL_CHECKPOINT_PASS', flush=True)


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    previous = parent_model(root, binding[2], device)
    models, thresholds = zip(previous, *(load_model(root, r, binding[2], device) for r in RECIPES.values()),
                             strict=True)
    if len(set(thresholds)) != 1:
        raise ValueError('The source U bypass threshold must match the registered control')
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=thresholds[0]),
          binding=binding, frozen_models=models, replay_fn=receive_only, finalize=freeze, reader_chunk=144)


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
            timing(rt, RECIPES['anchored'])
        else:
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[options['name']], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35, make_pair=paired_action_inputs,
                                  tail_weight=TAIL_WEIGHT))


if __name__ == '__main__':
    main()
