"""Train matched single-observation LoRa restorers using antithetic source IQ."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .action_support_source import independent_score, predict_clean_reserve
from .antithetic_detail import objective
from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .factorized_chirps_source import calibrate
from .gated_slope import THRESHOLD
from .gated_slope_source import dispatch, restore
from .joint_unroll_source import coordinates, source_control
from .mirror_refine_source import load_model
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_covariance import SharedSpectrumRecovery
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, WIDTHS
from .weak_guard_source import parent_model


RUN = '20261124-phase1-lora-antithetic-noise-response-source-m4-r01'
RECIPES = {
    arm: FitRecipe(arm, STEPS, partial(SharedSpectrumRecovery, WIDTHS, correlated=True),
                   colored_observation, SharedSpectrumRecovery.forward, initialize=calibrate)
    for arm in ('pointwise', 'antithetic')
}
MODES = ('direct', 'reference', 'rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse', 'half', 'full')))
PAIRS = tuple((f'antithetic_{stage}', f'pointwise_{stage}') for stage in ('half', 'coarse', 'full'))
PAIRS += (('antithetic_half', 'rank_parent_half'), ('pointwise_half', 'rank_parent_half'))


def bind_models(root, contract, device):
    parent, threshold = parent_model(root, contract, device)
    arms = [load_model(root, arm, contract, device) for arm in RECIPES.values()]
    if abs(threshold - THRESHOLD) > 1e-6 or any(abs(v - threshold) > 1e-6 for _, v in arms):
        raise ValueError('All restoration arms must share the legal clean calibration')
    return parent, tuple(model for model, _ in arms), threshold


@torch.no_grad()
def render(iq, _, baseline, tracker, models):
    parent, two_arms, threshold = models
    reference, _ = baseline(iq)
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    n = int(active.sum())
    if not n:
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    frame = tuple(v[active] for v in frame)
    rank, = restore(iq, active, parent, frame, (.5,))
    y, rot, q, rms = frame
    rows = [iq, reference, rank]
    for model in two_arms:
        coarse, full = model(y, q, rot)
        for wave in (coarse, torch.lerp(y, full, .5), full):
            restored = iq.clone()
            restored[active] += real_iq((wave - y) * rot * rms)
            rows.append(restored)
    return torch.stack(rows), n


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    score(rt, route, panel=panel, modes=MODES, render=render, binding=binding,
          frozen_models=bind_models(root, binding[2], device), require_clean_bypass=False,
          replay_fn=receive_only, finalize=freeze, reader_chunk=144)


def reserve(rt):
    def bind(root, contract, device, baseline, tracker):
        models = bind_models(root, contract, device)
        return lambda iq: render(iq, None, baseline, tracker, models)
    predict_clean_reserve(rt, modes=MODES, bind=bind)


def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (training, cal, _), contract = setup(rt, device, smoke=True)
    _, _, matched, baseline, tracker = source_control(rt, 'full', device)
    if contract != matched:
        raise ValueError('Expected matched blind source and frozen recognition input')
    parent, threshold = parent_model(root, contract, device)
    models = []
    for recipe, split in zip(RECIPES.values(), (False, True), strict=True):
        model = recipe.model().to(device)
        calibrate(model, tracker, training)
        rng = torch.Generator(device=device).manual_seed(2026114990)
        example = cal[:6]
        loss, terms = objective(model, tracker, example, rng, with_identity=True, recipe=recipe,
                                step=1, split_risk=split)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not terms.isfinite().all() or float(gradient) <= 0:
            raise ValueError('Expected finite antithetic source training gradient')
        models.append(model.eval().requires_grad_(False))
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupted = real_iq(colored_observation(complex_iq(cal), rng))
    active = corruption_score(coordinates(tracker, corrupted)[0]) > threshold
    batch = torch.cat((cal[:2], corrupted[active][:2]))
    waves, n = render(batch, None, baseline, tracker, (parent, tuple(models), threshold))
    with torch.no_grad():
        logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    if (n != 2 or logits.shape != (len(MODES)*len(batch), 10) or not logits.isfinite().all()
            or not torch.equal(waves[2:, :2], batch[:2].expand_as(waves[2:, :2]))):
        raise ValueError('Real source antithetic no-query checkpoint smoke failed')
    write(root/'smoke.json', dict(status='PASS', source_U=len(batch), finite_gradient_arms=2,
                                 active=n, clean_bypassed=2, target_reads=0))
    print('LORA_ANTITHETIC_REAL_CHECKPOINT_PASS', flush=True)


@torch.no_grad()
def timing(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    _, arms, threshold = bind_models(root, contract, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupt = real_iq(colored_observation(complex_iq(cal), rng))
    strongest = corruption_score(coordinates(tracker, corrupt)[0]).argmax().view(1)
    results = {}
    for name, model in zip(RECIPES, arms, strict=True):
        periods = {}
        for label, iq in (('clean', cal[:1]), ('generic', corrupt[strongest])):
            durations = []
            for _ in range(32):
                torch.cuda.synchronize()
                started = perf_counter()
                frame = coordinates(tracker, iq)
                active = corruption_score(frame[0]) > threshold
                if active.any():
                    restore(iq, active, model, tuple(v[active] for v in frame), (.5,))
                torch.cuda.synchronize()
                durations.append(1000*(perf_counter() - started))
            periods[label] = dict(median_ms=float(np.median(durations[8:])), active=int(active.sum()))
        results[name] = dict(parameters=sum(p.numel() for p in model.parameters()), timing=periods)
    write(root/'b1.json', dict(gpu=torch.cuda.get_device_name(), n_measurements=24,
                              scope='single half strength arm, excludes DURA and F', arms=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        stage = actions.add_parser(action)
        stage.add_argument('runtime', type=Path)
        if action != 'dispatch':
            stage.add_argument('row', choices=RECIPES if action == 'fit' else
                               (*ROUTES, 'reserve') if action == 'evaluate' else ('source',))
    args = vars(parser.parse_args())
    phase = args.pop('action')
    if phase == 'publish':
        return publish(**args, run=RUN, runner=__spec__.name)
    rt = read(args['runtime'])
    if phase == 'dispatch':
        return dispatch(rt, fit_arms=RECIPES)
    if phase == 'score':
        return independent_score(rt, modes=MODES, comparisons=PAIRS)
    activate(rt)
    if phase == 'smoke':
        return smoke(rt)
    if phase == 'fit':
        recipe = RECIPES[args['row']]
        return fit(rt, recipe=recipe, receiver_batches=True,
                   calculate=partial(objective, split_risk=recipe.name == 'antithetic'))
    if phase == 'timing':
        return timing(rt)
    return reserve(rt) if args['row'] == 'reserve' else evaluate(rt, args['row'])


if __name__ == '__main__':
    main()
