"""Train matched joint and independent posteriors for LoRa chirp detail."""

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
from .data import SOURCE
from .dura_repair import complex_iq, real_iq
from .factorized_chirps import FactorizedChirps
from .joint_unroll_source import coordinates, source_control
from .mirror_refine_source import load_model
from .natural_detail import natural_objective
from .rf_noise_transport import colored_observation
from .rf_noise_transport_source import timing
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, TAIL_WEIGHT, WIDTHS


RUN = '20261124-phase1-lora-factorized-chirps-source-m4-r01'
PACKETS_PER_RECEIVER = 100


@torch.no_grad()
def calibrate(model, tracker, source_u):
    balanced = torch.cat([packets[:PACKETS_PER_RECEIVER] for packets in source_u.chunk(len(SOURCE))])
    spectra = []
    for packets in balanced.split(20):
        observed, _, _, _ = coordinates(tracker, packets)
        spectra.append(torch.fft.fft(observed, norm='ortho'))
    model.clean_detail.fit(torch.cat(spectra))


RECIPES = {
    arm: FitRecipe(arm, STEPS, partial(FactorizedChirps, WIDTHS, coupled=coupled), colored_observation,
                   FactorizedChirps.forward, initialize=calibrate)
    for arm, coupled in (('independent', False), ('joint', True))
}
MODES = ('direct', 'reference', 'engineering_independent', 'engineering_joint',
         'independent_coarse', 'independent', 'independent_half', 'joint_coarse', 'joint', 'joint_half',
         'joint_unlinked_half', 'joint_common_half', 'joint_private_half')
PAIRS = (('engineering_joint', 'engineering_independent'), ('joint_coarse', 'independent_coarse'),
         ('joint', 'independent'), ('joint_half', 'independent_half'), ('joint_half', 'joint_unlinked_half'),
         ('joint_common_half', 'direct'), ('joint_private_half', 'direct'))


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference, _ = baseline(iq)
    observed, phase, quality, rms = coordinates(tracker, iq)
    active = corruption_score(observed) > threshold
    if not active.any():
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    y, phase, quality, rms = (v[active] for v in (observed, phase, quality, rms))
    independent, joint = models
    a, b = independent(y, quality, phase)
    c, d, analytic_independent, analytic_joint, unlinked = joint(y, quality, phase, controls=True)
    shared_edit = (d - y).mean(1, keepdim=True)
    values = (analytic_independent, analytic_joint, a, b, torch.lerp(y, b, .5),
              c, d, torch.lerp(y, d, .5), torch.lerp(y, unlinked, .5),
              y + .5 * shared_edit, torch.lerp(y, d, .5) - .5 * shared_edit)
    outputs = [iq, reference]
    for value in values:
        restored = iq.clone()
        restored[active] += real_iq((value - y) * phase * rms)
        outputs.append(restored)
    return torch.stack(outputs), int(active.sum())


@torch.no_grad()
def checkpoint_smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (training, calibration, _), contract = setup(rt, device, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if contract != expected:
        raise ValueError('The source clean statistics and frozen recognizer need the same IQ contract')
    models = [recipe.model().to(device).eval() for recipe in RECIPES.values()]
    for model in models:
        calibrate(model, tracker, training)
    rng = torch.Generator(device=device).manual_seed(2026112403)
    batch = real_iq(colored_observation(complex_iq(calibration[:2]), rng))
    values, active = render(batch, None, baseline, tracker, models, threshold=-1.)
    logits = load_reader(root, contract, device)(values.flatten(0, 1))[0]
    if active != len(batch) or logits.shape != (len(MODES) * len(batch), 10) or not torch.isfinite(logits).all():
        raise ValueError('Factorized chirp real-checkpoint smoke failed')
    write(root / 'checkpoint_smoke.json', dict(status='PASS', variants=len(MODES), source_U=len(batch),
                                              target_reads=0))
    print('LORA_FACTORIZED_CHIRPS_REAL_CHECKPOINT_PASS', flush=True)


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    models, thresholds = zip(*(load_model(root, r, binding[2], device) for r in RECIPES.values()), strict=True)
    if thresholds[0] != thresholds[1]:
        raise ValueError('Matched source U clean thresholds required')
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
            timing(rt, RECIPES['joint'])
        else:
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[options['name']], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35,
                                  make_pair=partial(paired_action_inputs, echo_probability=.5),
                                  tail_weight=TAIL_WEIGHT))


if __name__ == '__main__':
    main()
