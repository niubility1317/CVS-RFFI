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


def load_arm(root, recipe, contract, device, reused):
    if reused and recipe.name in reused:
        root = root.parent / reused[recipe.name]
        if read(root / 'state.json')['status'] != 'SOURCE_SCORED':
            raise ValueError(f'Expected a completed source-only run for {recipe.name}')
    return load_model(root, recipe, contract, device)


def bind_models(root, contract, device, *, recipes=RECIPES, reused=None):
    parent, threshold = parent_model(root, contract, device)
    arms = [load_arm(root, arm, contract, device, reused) for arm in recipes.values()]
    if abs(threshold - THRESHOLD) > 1e-6 or any(abs(v - threshold) > 1e-6 for _, v in arms):
        raise ValueError('All restoration arms must share the legal clean calibration')
    return parent, tuple(model for model, _ in arms), threshold


def read_stages(model, observed, quality, rotation, rms, recipe):
    stages = model(observed, quality, rotation)
    return stages if recipe is None else recipe.transform_stages(observed, stages, rms)


@torch.no_grad()
def render(iq, _, baseline, tracker, models, *, modes=MODES, recipes=None, reframe=None, mixer=torch.lerp):
    parent, two_arms, threshold = models
    reference, _ = baseline(iq)
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    n = int(active.sum())
    if not n:
        return torch.stack((iq, reference, *([iq] * (len(modes) - 2)))), 0
    frame = tuple(v[active] for v in frame)
    rank, = restore(iq, active, parent, frame, (.5,))
    if reframe is not None:
        frame = reframe(frame)
    y, rot, q, rms = frame
    rows = [iq, reference, rank]
    stage_recipes = recipes.values() if recipes is not None else (None,) * len(two_arms)
    for model, recipe in zip(two_arms, stage_recipes, strict=True):
        coarse, full = read_stages(model, y, q, rot, rms, recipe)
        for wave in (coarse, mixer(y, full, .5), full):
            restored = iq.clone()
            restored[active] += real_iq((wave - y) * rot * rms)
            rows.append(restored)
    return torch.stack(rows), n


def evaluate(rt, route, *, recipes=RECIPES, modes=MODES, reused=None, reframe=None, mixer=torch.lerp):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    renderer = partial(render, modes=modes, recipes=recipes, reframe=reframe, mixer=mixer)
    score(rt, route, panel=panel, modes=modes, render=renderer, binding=binding,
          frozen_models=bind_models(root, binding[2], device, recipes=recipes, reused=reused), require_clean_bypass=False,
          replay_fn=receive_only, finalize=freeze, reader_chunk=144)


def reserve(rt, *, recipes=RECIPES, modes=MODES, reused=None, reframe=None, mixer=torch.lerp):
    def bind(root, contract, device, baseline, tracker):
        models = bind_models(root, contract, device, recipes=recipes, reused=reused)
        return lambda iq: render(
            iq, None, baseline, tracker, models, modes=modes, recipes=recipes, reframe=reframe, mixer=mixer)
    predict_clean_reserve(rt, modes=modes, bind=bind)


def smoke(rt, *, recipes=RECIPES, modes=MODES, split_risk=(False, True), calculations=None, reused=None,
          reframe=None, calibrate_fn=calibrate, mixer=torch.lerp):
    root, device = Path(rt['run_root']), 'cuda:0'
    (training, cal, _), contract = setup(rt, device, smoke=True)
    _, _, matched, baseline, tracker = source_control(rt, 'full', device)
    if contract != matched:
        raise ValueError('Expected matched blind source and frozen recognition input')
    parent, threshold = parent_model(root, contract, device)
    models = []
    finite_gradient_arms = 0
    for recipe, split in zip(recipes.values(), split_risk, strict=True):
        if reused and recipe.name in reused:
            model, reused_threshold = load_arm(root, recipe, contract, device, reused)
            if abs(reused_threshold - threshold) > 1e-6:
                raise ValueError('Expected matched calibration from the frozen control')
            models.append(model)
            continue
        model = recipe.model().to(device)
        if calibrate_fn is not None:
            calibrate_fn(model, tracker, training)
        rng = torch.Generator(device=device).manual_seed(2026114990)
        example = cal[:6]
        calculate = (calculations or {}).get(recipe.name, partial(objective, split_risk=split))
        loss, terms = calculate(model, tracker, example, rng, with_identity=True, recipe=recipe, step=1)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not terms.isfinite().all() or float(gradient) <= 0:
            raise ValueError('Expected finite antithetic source training gradient')
        finite_gradient_arms += 1
        models.append(model.eval().requires_grad_(False))
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupted = real_iq(colored_observation(complex_iq(cal), rng))
    active = corruption_score(coordinates(tracker, corrupted)[0]) > threshold
    batch = torch.cat((cal[:2], corrupted[active][:2]))
    waves, n = render(batch, None, baseline, tracker, (parent, tuple(models), threshold),
                      modes=modes, recipes=recipes, reframe=reframe, mixer=mixer)
    with torch.no_grad():
        logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    if (n != 2 or logits.shape != (len(modes)*len(batch), 10) or not logits.isfinite().all()
            or not torch.equal(waves[2:, :2], batch[:2].expand_as(waves[2:, :2]))):
        raise ValueError('Real source antithetic no-query checkpoint smoke failed')
    write(root/'smoke.json', dict(status='PASS', source_U=len(batch), finite_gradient_arms=finite_gradient_arms,
                                 active=n, clean_bypassed=2, target_reads=0))
    print('LORA_ANTITHETIC_REAL_CHECKPOINT_PASS', flush=True)


@torch.no_grad()
def timing(rt, *, recipes=RECIPES, reused=None, reframe=None, mixer=None):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    _, arms, threshold = bind_models(root, contract, device, recipes=recipes, reused=reused)
    tracker = PacketRestorer('consensus').to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupt = real_iq(colored_observation(complex_iq(cal), rng))
    strongest = corruption_score(coordinates(tracker, corrupt)[0]).argmax().view(1)
    results = {}
    for (name, recipe), model in zip(recipes.items(), arms, strict=True):
        periods = {}
        for label, iq in (('clean', cal[:1]), ('generic', corrupt[strongest])):
            durations = []
            for _ in range(32):
                torch.cuda.synchronize()
                started = perf_counter()
                frame = coordinates(tracker, iq)
                active = corruption_score(frame[0]) > threshold
                if active.any():
                    chosen = tuple(v[active] for v in frame)
                    y, rot, q, rms = chosen if reframe is None else reframe(chosen)
                    _, full = read_stages(model, y, q, rot, rms, recipe)
                    processed = iq.clone()
                    update = .5 * (full - y) if mixer is None else mixer(y, full, .5) - y
                    processed[active] += real_iq(update * rot * rms)
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
