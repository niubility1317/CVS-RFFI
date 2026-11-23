"""Test an echo-aware joint LoRa posterior against the same fit without echo."""

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
from .coupled_echo import CoupledEcho
from .factorized_chirps_source import calibrate
from .joint_unroll_source import coordinates, source_control
from .mirror_refine_source import load_model
from .natural_detail import natural_objective
from .rf_noise_transport import colored_observation
from .rf_noise_transport_source import timing
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, TAIL_WEIGHT, WIDTHS


RUN = '20261124-phase1-lora-coupled-echo-source-m4-r01'
RECIPES = {
    arm: FitRecipe(arm, STEPS, partial(CoupledEcho, WIDTHS, echo=echo), colored_observation,
                   CoupledEcho.forward, initialize=calibrate)
    for arm, echo in (('linked', False), ('echo', True))
}
MODES = ('direct', 'reference', 'engineering_linked', 'engineering_echo',
         'linked_coarse', 'linked', 'linked_half', 'echo_coarse', 'echo', 'echo_half',
         'echo_off_half', 'echo_common_half', 'echo_private_half')
PAIRS = (('engineering_echo', 'engineering_linked'), ('echo_coarse', 'linked_coarse'),
         ('echo', 'linked'), ('echo_half', 'linked_half'), ('echo_half', 'echo_off_half'),
         ('echo_common_half', 'direct'), ('echo_private_half', 'direct'))


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold, statistics=None):
    reference, _ = baseline(iq)
    observed, phase, quality, rms = coordinates(tracker, iq)
    disturbance = corruption_score(observed)
    active = disturbance > threshold
    if not active.any():
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    y, phase, quality, rms = (v[active] for v in (observed, phase, quality, rms))
    linked, echo = models
    a, b = linked(y, quality, phase)
    c, d, analytic_linked, analytic_echo, echo_off, belief = echo(y, quality, phase, controls=True)
    if statistics is not None:
        mild = disturbance[active] <= .15
        weight = belief.flatten()
        effect = (d - echo_off).abs().square().mean(dim=(1, 2)).sqrt()
        for i, group in enumerate((mild, ~mild)):
            statistics[i, 0] += group.sum()
            statistics[i, 1] += (weight * group).sum()
            statistics[i, 2] += ((weight > .5) & group).sum()
            statistics[i, 3] += (effect * group).sum()
    shared_edit = (d - y).mean(1, keepdim=True)
    values = (analytic_linked, analytic_echo, a, b, torch.lerp(y, b, .5),
              c, d, torch.lerp(y, d, .5), torch.lerp(y, echo_off, .5),
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
        raise ValueError('Coupled echo real-checkpoint smoke failed')
    write(root / 'checkpoint_smoke.json', dict(status='PASS', variants=len(MODES), source_U=len(batch),
                                              target_reads=0))
    print('LORA_COUPLED_ECHO_REAL_CHECKPOINT_PASS', flush=True)


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    models, thresholds = zip(*(load_model(root, r, binding[2], device) for r in RECIPES.values()), strict=True)
    if thresholds[0] != thresholds[1]:
        raise ValueError('Matched source U clean thresholds required')
    stats = torch.zeros((2, 4), device=device)
    renderer = partial(render, threshold=thresholds[0], statistics=stats)
    score(rt, route, panel=panel, modes=MODES, render=renderer,
          binding=binding, frozen_models=models, replay_fn=receive_only, finalize=freeze, reader_chunk=144)
    keys = ('active_at_most_0.15', 'active_above_0.15')
    result = {key: dict(exposures=int(n), mean_echo_weight=total/max(1,n), above_half=int(above),
                        mean_echo_effect_rms=effect/max(1,n))
              for key,(n,total,above,effect) in zip(keys,stats.tolist(),strict=True)}
    write(root / f'{route}-echo-weight.json', result)


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
            timing(rt, RECIPES['echo'])
        else:
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[options['name']], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35,
                                  make_pair=partial(paired_action_inputs, echo_probability=.5),
                                  tail_weight=TAIL_WEIGHT))


if __name__ == '__main__':
    main()
