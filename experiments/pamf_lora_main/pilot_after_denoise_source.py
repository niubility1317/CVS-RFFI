"""Fit identical LoRa recovery budgets with raw or denoised preamble action estimation."""
import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write
from .action_support_source import independent_score, predict_clean_reserve
from .antithetic_detail_source import bind_models, timing
from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .clean_pilot_response_source import calibrate
from .complex_vector_source import load_reader, publish, score
from .dura_repair import complex_iq, real_iq
from .gated_slope_source import dispatch, restore
from .joint_unroll_source import coordinates, source_control
from .pilot_after_denoise import PilotAfterDenoise, objective
from .refinement_bound import bound_refinement
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, WIDTHS
from .weak_guard_source import parent_model


RUN = '20261124-phase1-lora-pilot-after-denoise-source-m4-r01'
RECIPES = {
    arm: FitRecipe(arm, STEPS, partial(PilotAfterDenoise, WIDTHS, denoised_fit=filtered),
                   colored_observation, PilotAfterDenoise.forward, initialize=calibrate,
                   stage_transform=bound_refinement)
    for arm, filtered in (('raw_pilot', False), ('denoised_pilot', True))
}
MODES = ('direct', 'reference', 'rank_half', 'raw_pilot_coarse', 'raw_pilot_half',
         'denoised_pilot_coarse', 'denoised_pilot_half', 'denoised_no_action_half')
PAIRS = (('denoised_pilot_half', 'raw_pilot_half'),
         ('denoised_pilot_half', 'denoised_no_action_half'),
         ('denoised_pilot_half', 'rank_half'),
         ('denoised_pilot_coarse', 'raw_pilot_coarse'))


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, statistics=None):
    rank_model, arms, threshold = models
    reference, _ = baseline(iq)
    frame = coordinates(tracker, iq)
    disturbance = corruption_score(frame[0])
    active = disturbance > threshold
    n = int(active.sum())
    if not n:
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    frame = tuple(v[active] for v in frame)
    rank, = restore(iq, active, rank_model, frame, (.5,))
    y, rotation, quality, rms = frame
    values = []
    for i, (model, recipe) in enumerate(zip(arms, RECIPES.values(), strict=True)):
        coarse, fine, support = model(y, quality, rotation, diagnostics=True)
        coarse, fine = recipe.transform_stages(y, (coarse, fine), rms)
        values.extend((coarse, torch.lerp(y, fine, .5)))
        if statistics is not None:
            mild = disturbance[active] <= .15
            for level, mask in enumerate((mild, ~mild)):
                statistics[i, level, 0] += mask.sum()
                statistics[i, level, 1] += (mask & (support > 0)).sum()
                statistics[i, level, 2] += (support * mask).sum()
    model = arms[1]
    recipe = RECIPES['denoised_pilot']
    identity = recipe.transform_stages(y, model(y, quality, rotation, action=False), rms)[1]
    values.append(torch.lerp(y, identity, .5))
    rows = [iq, reference, rank]
    for wave in values:
        out = iq.clone()
        out[active] += real_iq((wave-y)*rotation*rms)
        rows.append(out)
    return torch.stack(rows), n


def smoke_gradients(model, tracker, clean, rng, recipe):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        loss, terms = objective(model, tracker, clean, rng, with_identity=True, recipe=recipe, step=step)
        loss.backward()
        if not terms.isfinite().all():
            raise FloatingPointError('Nonfinite received pilot training objective')
        if step == 1:
            # A zero-initialized coarse edit sets the fine edit cap to zero.
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if grad <= 0:
                raise ValueError('Expected an initial coarse gradient')
            optimizer.step()
    norms = [torch.nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True).item()
             for module in (model.coarse, model.prior)]
    if min(norms) <= 0:
        raise ValueError('Expected coarse and refinement gradients after the initial update')
    return norms


def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (training, cal, _), contract = setup(rt, device, smoke=True)
    _, _, bound, baseline, tracker = source_control(rt, 'full', device)
    if bound != contract:
        raise ValueError('Unexpected source U and frozen reader contract')
    models = []
    gradients = []
    for recipe in RECIPES.values():
        model = recipe.model().to(device)
        calibrate(model, tracker, training)
        rng = torch.Generator(device=device).manual_seed(2026113703)
        gradients.append(smoke_gradients(model, tracker, cal[:6], rng, recipe))
        models.append(model.eval().requires_grad_(False))
    rank, threshold = parent_model(root, contract, device)
    disturbed = real_iq(colored_observation(complex_iq(cal), rng))
    accepted = corruption_score(coordinates(tracker, disturbed)[0]) > threshold
    inputs = torch.cat((cal[:2], disturbed[accepted][:2]))
    waves, active = render(inputs, None, baseline, tracker, (rank, models, threshold))
    with torch.no_grad():
        logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    if (active != 2 or logits.shape != (len(MODES)*len(inputs), 10) or not logits.isfinite().all()
            or not torch.equal(waves[2:, :2], inputs[:2].expand_as(waves[2:, :2]))):
        raise ValueError('Received pilot real-checkpoint smoke failed')
    write(root/'smoke.json', dict(status='PASS', gradient_norms=gradients,
          clean_bypassed=2, active=active, target_reads=0))
    print('LORA_PILOT_AFTER_DENOISE_REAL_CHECKPOINT_PASS', flush=True)


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    stats = torch.zeros((2, 2, 3), device=device)
    score(rt, route, panel=panel, modes=MODES, render=partial(render, statistics=stats),
          binding=binding, frozen_models=bind_models(root, binding[2], device, recipes=RECIPES),
          require_clean_bypass=False, replay_fn=receive_only, finalize=freeze, reader_chunk=128)
    report = {}
    for arm, records in zip(RECIPES, stats.tolist(), strict=True):
        report[arm] = {}
        for label, (n, positive, total) in zip(('active_at_most_0.15', 'active_above_0.15'), records, strict=True):
            report[arm][label] = dict(exposures=int(n), positive=int(positive), mean_support=total/max(n, 1))
    write(root/f'{route}-pilot-support.json', report)


def reserve(rt):
    def build(root, contract, device, baseline, tracker):
        models = bind_models(root, contract, device, recipes=RECIPES)
        return lambda iq: render(iq, None, baseline, tracker, models)
    predict_clean_reserve(rt, modes=MODES, bind=build)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        step = actions.add_parser(action)
        step.add_argument('runtime', type=Path)
        if action != 'dispatch':
            choices = RECIPES if action == 'fit' else (*ROUTES, 'reserve') if action == 'evaluate' else ('source',)
            step.add_argument('row', choices=choices)
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        return publish(**options, run=RUN, runner=__spec__.name)
    rt = read(options['runtime'])
    if action == 'dispatch':
        return dispatch(rt, fit_arms=RECIPES)
    if action == 'score':
        return independent_score(rt, modes=MODES, comparisons=PAIRS)
    activate(rt)
    if action == 'smoke':
        return smoke(rt)
    if action == 'fit':
        return fit(rt, recipe=RECIPES[options['row']], calculate=objective, receiver_batches=True)
    if action == 'timing':
        return timing(rt, recipes=RECIPES)
    if options['row'] == 'reserve':
        return reserve(rt)
    return evaluate(rt, options['row'])


if __name__ == '__main__':
    main()
