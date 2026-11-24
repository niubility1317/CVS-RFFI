"""Train paired full-vector and balanced-vector LoRa DURA corrections."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Callable

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .complex_vector import ComplexVector, teacher_vectors, vector_risk
from .data import SOURCE, VIEWS
from .denoise_components_source import selection as attribution_panel
from .denoise_source import EVAL_PER_RX, normalized, replay, summarize
from .dura_repair import complex_iq, real_iq
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_supervised import construct as recognizer
from .joint_unroll_source import coordinates, source_control
from .packet_runner import schedule
from .physical_inverse_source import source_trace_implementation
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import RUN as RESIDUAL_RUN, SEED
from .runner import activate
from .signal_retention_source import build_arrays, panel as earlier_panel
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-complex-vector-source-m2-r01'
KEY = 2026112738
TRAIN_PER_RX, HELD_PER_RX = 650, 80
STEPS, BATCH = 1500, 12
ARMS = ('direct', 'balanced')
MODES = ('direct', 'reference', 'direct_vector', 'balanced_vector', 'balanced_half',
         'ideal_direct', 'ideal_balanced', 'known_noise')
ISOLATION = dict(F_reads=0, TX_reads=0, V_reads=0, target_reads=0)


@dataclass(frozen=True)
class VectorRecipe:
    seed: int
    steps: int
    train_per_rx: int
    held_per_rx: int
    arms: tuple[str, str]
    model: Callable
    targets: Callable
    loss: Callable


DEFAULT = VectorRecipe(KEY, STEPS, TRAIN_PER_RX, HELD_PER_RX, ARMS, lambda _: ComplexVector(),
                       teacher_vectors, lambda value, target, _: vector_risk(value, target))


@torch.no_grad()
def training_pairs(arrays, create_targets=teacher_vectors):
    measured, q, foreground, baseline, supported, clean, cq = arrays
    take = supported.flatten()
    x, b, q, foreground = (a.flatten(0, 1)[take] for a in (measured, baseline, q, foreground))
    if len(x) < 16:
        raise ValueError('Missing source practical exposures at the mature gate')
    return (x, b, q, create_targets(b, foreground)), (clean, cq)


def record(contract, route, recipe=DEFAULT):
    return dict(status='SOURCE_FROZEN', route=route, contract=contract, seed=recipe.seed, steps=recipe.steps,
                train_per_rx=recipe.train_per_rx, held_per_rx=recipe.held_per_rx,
                arms=list(recipe.arms), **ISOLATION)


def fit(rt, route, *, smoke=False, recipe=DEFAULT):
    device = 'cuda:0'
    torch.manual_seed(recipe.seed)
    _, _, original, baseline, tracker = source_control(rt, route, device)
    fit_per, held_per = (6, 4) if smoke else (recipe.train_per_rx, recipe.held_per_rx)
    raw_fit, raw_held, contract = build_arrays(rt, route, tracker, fit_per, held_per, baseline=baseline)
    if original != contract:
        raise ValueError('The mature restoration path and source exposure must agree')
    (y, b, q, targets), (clean, clean_q) = training_pairs(raw_fit, recipe.targets)
    (yh, bh, qh, th), _ = training_pairs(raw_held, recipe.targets)
    del raw_fit, raw_held
    models = {k: recipe.model(k).to(device) for k in recipe.arms}
    first, second = models.values()
    second.load_state_dict(first.state_dict())
    optim = torch.optim.AdamW([p for m in models.values() for p in m.parameters()], lr=2e-4, weight_decay=1e-4)
    steps = 1 if smoke else recipe.steps
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps, eta_min=2e-5)
    rng = torch.Generator(device=device).manual_seed(recipe.seed)
    for step in range(1, steps+1):
        ids = torch.randint(len(y), (BATCH,), device=device, generator=rng)
        clean_ids = torch.randint(len(clean), (2,), device=device, generator=rng)
        reset, rq = clean[clean_ids], clean_q[clean_ids]
        v = torch.cat((y[ids], reset))
        w = torch.cat((b[ids], reset))
        trust = torch.cat((q[ids], rq))
        losses = torch.stack([recipe.loss(m(v, w, trust), torch.cat((targets[k][ids],
                                   torch.zeros_like(targets[k][ids[:2]]))), k) for k, m in models.items()])
        if not torch.isfinite(losses).all():
            raise FloatingPointError('Nonfinite source complex-vector fitting')
        optim.zero_grad(set_to_none=True)
        losses.sum().backward()
        for m in models.values():
            norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
            if step == 1 and (norm <= 0 or any(f.out.weight.grad.norm() <= 0 for f in (m.spectral, m.temporal))):
                raise ValueError('Expected finite nonzero gradients in both complex paths')
        optim.step()
        lr.step()
        if step == 1 or step % 50 == 0:
            print('COMPLEX_VECTOR_FIT', route, step, *losses.detach().tolist(), flush=True)
    held = {}
    with torch.no_grad():
        indices = torch.arange(len(yh), device=device).split(20)
        for k, m in models.items():
            m.eval()
            held[k] = sum(len(i)*float(recipe.loss(m(yh[i], bh[i], qh[i]), th[k][i], k))
                          for i in indices)/len(yh)
    evidence = record(contract, route, recipe)
    root = Path(rt['run_root'])
    report = {k: v for k, v in evidence.items() if k != 'contract'}
    report.update(fit_active=len(y), held_active=len(yh), held_loss=held,
                  zero_baseline={k: float(recipe.loss(torch.zeros_like(t), t, k)) for k,t in th.items()})
    if smoke:
        write(root/f'smoke-{route}.json', dict(report, status='PASS', steps=1,
                                               train_per_rx=fit_per, held_per_rx=held_per))
    else:
        torch.save(dict(proof=evidence, models={k:m.state_dict() for k,m in models.items()}), root/f'{route}-model.pt')
        write(root/f'{route}-complete.json', report)


def result_panel(rx):
    excluded = np.union1d(earlier_panel(rx), attribution_panel(rx, EVAL_PER_RX))
    rng = np.random.default_rng(KEY+1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == k), excluded),
                                             EVAL_PER_RX, replace=False)) for k in SOURCE])


@torch.no_grad()
def readouts(received, noise, baseline, tracker, models):
    mature, active = baseline(received)
    copies = {k: mature.clone() for k in MODES[2:]}
    if active.any():
        observed, rotation, q, rms = coordinates(tracker, received[active])
        basis = rotation*rms
        b = complex_iq(mature[active])/basis
        source_signal = observed - complex_iq(noise[active])/basis
        teachers = teacher_vectors(b, source_signal)
        d, correction = (m(observed, b, q) for m in models)
        updates = (d, correction, .5*correction, teachers['direct'], teachers['balanced'])
        for key, delta in zip(MODES[2:-1], updates, strict=True):
            time = torch.fft.ifft(delta, norm='ortho')*basis
            copies[key][active] = mature[active] + real_iq(time)
        y = received[active]
        copies['known_noise'][active] = normalized(y-noise[active], y)
    return torch.stack((received, mature, *(copies[k] for k in MODES[2:]))), int(active.sum())


def load_models(root, route, contract, recipe, device):
    saved = torch.load(root/f'{route}-model.pt', map_location=device, weights_only=True)
    if saved['proof'] != record(contract, route, recipe) or set(saved['models']) != set(recipe.arms):
        raise ValueError('Wrong individually frozen complex-vector source weights')
    models = [recipe.model(k).to(device).eval().requires_grad_(False) for k in recipe.arms]
    for m, k in zip(models, recipe.arms, strict=True):
        m.load_state_dict(saved['models'][k], strict=True)
    return models


def load_reader(root, contract, device):
    model = recognizer(device).eval().requires_grad_(False)
    weights = complete_f(root.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    return model


@torch.no_grad()
def score(rt, route, *, recipe=DEFAULT, panel=result_panel, modes=MODES, render=readouts,
          checkpoint_run=None, binding=None, frozen_models=None, require_clean_bypass=True,
          replay_fn=replay, finalize=summarize, reader_chunk=None):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    source, cache, contract, baseline, tracker = binding or source_control(rt, route, device)
    parent = root if checkpoint_run is None else root.parent/checkpoint_run
    if checkpoint_run is not None and read(parent/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected complete source-selected vector checkpoints')
    models = load_models(parent, route, contract, recipe, device) if frozen_models is None else frozen_models
    f = load_reader(root, contract, device)
    raw, rx, ids, _ = source_role(source, contract, 'V')
    chosen = panel(rx)
    rows = list(zip(raw[chosen], rx[chosen], ids[chosen], strict=True))
    official = np.load(cache/'V.npy', mmap_mode='r', allow_pickle=False) if cache is not None else None
    pred, metrics = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for index, view in enumerate(VIEWS):
            capture, noise = (np.stack(v) for v in zip(*pool.map(partial(replay_fn, route, view), rows,
                                                                chunksize=8), strict=True))
            if official is not None and not np.array_equal(capture, official[index, chosen]):
                raise ValueError('Full source view mismatch')
            n_active, parts = 0, []
            for y, epsilon in zip(tensor(capture, device).split(18), tensor(noise, device).split(18), strict=True):
                x, active = render(y, epsilon, baseline, tracker, models)
                inputs = x.flatten(0, 1)
                logits = (f(inputs)[0] if reader_chunk is None else
                          torch.cat([f(batch)[0] for batch in inputs.split(reader_chunk)]))
                if not torch.isfinite(logits).all() or (require_clean_bypass and view=='clean' and active):
                    raise ValueError('Invalid clean guard or nonfinite frozen classification')
                n_active += active
                parts.append(logits.argmax(1).reshape(len(modes), len(y)).cpu().tolist())
            pred[view] = np.concatenate(parts, axis=1)
            metrics[view] = dict(active=n_active)
            print('COMPLEX_VECTOR_SOURCE', route, view, n_active, flush=True)
    finalize(root, route, source, chosen, ids, pred, metrics, modes)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root/'state.json'
    try:
        for phase in ('smoke', 'fit', 'score'):
            write(state, dict(status=phase.upper()), exclusive=False)
            schedule(rt, [(phase, route) for route in ROUTES], partial(start, module=rt['runner']))
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit, *, run=RUN, runner=None,
            allowed_gpus=(3,6,2,0,4,1,5)):
    cfg = read(context)
    rt = {k: cfg[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project/'runs'/RESIDUAL_RUN
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Exact source receipt required')
    rt.update(run_id=run, runner=runner or __spec__.name, code_commit=release_commit(repo),
              run_root=str(project/'runs'/run), release_root=str(project/'releases'/f'{run}-release'),
              source_root=str(project/'releases'/f'{DATA_RUN}-release'/'data'/'source'), parent_root=str(parent),
              allowed_gpus=list(allowed_gpus), max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+option, type=Path, required=True)
    for stage in ('dispatch', 'smoke', 'fit', 'score'):
        entry = commands.add_parser(stage)
        entry.add_argument('runtime', type=Path)
        if stage != 'dispatch':
            entry.add_argument('route', choices=ROUTES)
    options = vars(cli.parse_args())
    action = options.pop('action')
    if action == 'publish':
        publish(**options)
    else:
        runtime = read(options['runtime'])
        if action == 'dispatch':
            dispatch(runtime)
        else:
            activate(runtime)
            if action == 'score':
                score(runtime, options['route'])
            else:
                source_trace_implementation()
                fit(runtime, options['route'], smoke=action == 'smoke')
