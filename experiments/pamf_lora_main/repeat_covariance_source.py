"""Matched LoRa source-only comparison of complex cross-chirp noise covariance."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write
from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, setup
from .chirp_spectrum_source import dispatch, evaluate, measure, score_frozen
from .complex_vector_source import load_reader, publish
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .natural_detail import natural_objective, paired_echo_inputs
from .repeat_covariance import RepeatCovariance, repeated_noise
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-repeat-covariance-source-m4-r01'
STEPS = 1800
MODES = ('direct', 'reference', 'diagonal_coarse', 'diagonal', 'diagonal_half',
         'covariance_coarse', 'covariance', 'covariance_half')
PAIRS = (('covariance_coarse','diagonal_coarse'), ('covariance','diagonal'),
         ('covariance_half','diagonal_half'))


def restore(model, y, quality, _rotation):
    return model(y, quality)


RECIPES = {arm: FitRecipe(arm, STEPS, partial(RepeatCovariance, correlated), repeated_noise, restore)
           for arm, correlated in (('diagonal', False), ('covariance', True))}


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference, _ = baseline(iq)
    observed, rotation, quality, rms = coordinates(tracker, iq)
    active = corruption_score(observed) > threshold
    outputs = []
    if not bool(active.any()):
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    y, basis, q = observed[active], (rotation * rms)[active], quality[active]
    for model in models:
        coarse, fine = model(y, q)
        for value in (coarse, fine, torch.lerp(y, fine, .5)):
            output = iq.clone()
            output[active] += real_iq((value - y) * basis)
            outputs.append(output)
    return torch.stack((iq, reference, *outputs)), int(active.sum())


@torch.no_grad()
def checkpoint_smoke(rt):
    root = Path(rt['run_root'])
    (_, calibration, _), contract = setup(rt, 'cuda:0', smoke=True)
    _, _, binding, baseline, tracker = source_control(rt, 'full', 'cuda:0')
    if contract != binding:
        raise ValueError('Source U and the real LoRa classifier differ in contract')
    reader = load_reader(root, contract, 'cuda:0')
    models = [r.model().cuda().eval() for r in RECIPES.values()]
    rng = torch.Generator(device='cuda:0').manual_seed(2026112401)
    batch = real_iq(repeated_noise(complex_iq(calibration[:2]), rng))
    output, _ = render(batch, None, baseline, tracker, models, threshold=-1.)
    logits = reader(output.flatten(0, 1))[0]
    if logits.shape != (len(MODES) * len(batch), 10) or not torch.isfinite(logits).all():
        raise ValueError('New covariance path did not pass frozen source-only real checkpoint smoke')
    write(root / 'checkpoint_smoke.json', dict(status='PASS', source_U=len(batch), target_reads=0))
    print('LORA_REPEAT_COVARIANCE_REAL_CHECKPOINT_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + key, type=Path, required=True)
    for action in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        cmd = commands.add_parser(action)
        cmd.add_argument('runtime', type=Path)
        if action != 'dispatch':
            cmd.add_argument('name', choices=RECIPES if action in ('smoke', 'fit') else
                             ROUTES if action == 'evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args, run=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action == 'dispatch':
        dispatch(rt, RECIPES, initial_stages=(('checkpoint_smoke', ('source',)),))
    elif action == 'score':
        score_frozen(rt, modes=MODES, comparisons=PAIRS)
    else:
        activate(rt)
        if action == 'checkpoint_smoke':
            checkpoint_smoke(rt)
        elif action == 'evaluate':
            evaluate(rt, args['name'], recipes=RECIPES, modes=MODES, renderer=render)
        elif action == 'timing':
            measure(rt, RECIPES['covariance'], verify_reader=False)
        else:
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[args['name']], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35, make_pair=paired_echo_inputs))


if __name__ == '__main__':
    main()
