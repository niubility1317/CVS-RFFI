"""Test source-only signal-retaining denoising with a shared reception operator."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_unrolled_restore.denoise import Denoise, noise_loss
from experiments.pamf_unrolled_restore.signal_retention import retention_loss

from .data import SOURCE, VIEWS
from .denoise_source import EVAL_PER_RX, panel as prior_panel, score_models
from .dura_repair import PacketRestorer, complex_iq
from .fixed_action_noise import paired
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_supervised import handles
from .joint_unroll_source import coordinates, source_control
from .packet_runner import schedule
from .physical_inverse_source import source_trace_implementation
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import L_COUNT, RUN as RESIDUAL_RUN, ResidualView
from .residual_dura_calibration import source_state
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-signal-retention-source-m2-r01'
KEY = 2026112709
FIT_PER_RX, HELD_PER_RX = 500, 80
BATCH, STEPS = 12, 1500
RETAIN = .12
ARMS = ('noise_only', 'retention')
MODES = ('direct', 'reference', 'noise_only', 'retention', 'retention_half', 'known_noise')
ISOLATION = dict(F_reads=0, TX_reads=0, V_reads=0, target_reads=0)


@torch.no_grad()
def build_arrays(rt, route, model, count, hold, *, baseline=None):
    parent = Path(rt['parent_root'])
    source, cache, contract = handles(dict(rt, parent_root=str(parent.parent / DATA_RUN)))
    raw, rx, ids, slots = source_role(source, contract, 'U_s')
    clean = np.load(cache/'U_s.npy', mmap_mode='r', allow_pickle=False)[0]
    if route == 'full':
        cache_received = np.load(cache/'U_s_practical.npy', mmap_mode='r', allow_pickle=False)
    else:
        _, folder, source_contract, _ = source_state(rt)
        streams = ResidualView(folder)
        if source_contract != contract or not streams.ready():
            raise ValueError('Source residual contract differs')
        cache_received = streams.received[0, L_COUNT:]
    rng = np.random.default_rng(KEY)
    groups = [rng.permutation(np.flatnonzero(rx == k))[:count+hold] for k in SOURCE]
    cohort_fit, cohort_hold = [np.concatenate(x) for x in zip(*[(x[:count], x[count:]) for x in groups])]
    if (len(cohort_fit), len(cohort_hold)) != (len(SOURCE)*count, len(SOURCE)*hold):
        raise ValueError('Missing receiver-balanced source training or held records')
    parts = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for cohort in (cohort_fit, cohort_hold):
            jobs = ((raw[i], VIEWS[1+int(slots[i])%6], rx[i], ids[i]) for i in cohort)
            records = pool.map(partial(paired, route=route), jobs, chunksize=8)
            exact, components = (np.stack(v) for v in zip(*records, strict=True))
            if not np.array_equal(exact, cache_received[cohort]) or not np.isfinite(components).all():
                raise ValueError('First noise draw must equal the registered source IQ')
            wave = tensor(np.stack((exact, components[:, 0]), axis=1), 'cuda:0')
            background = complex_iq(tensor(components[:, 1], 'cuda:0'))
            received, rot, q, rms = coordinates(model, wave.flatten(0, 1))
            received, rot, q, rms = (x.unflatten(0, (len(cohort), 2)) for x in (received, rot, q, rms))
            basis = rot * rms
            foreground = background[:, None] / basis
            to_first = basis / basis[:, :1]
            epsilon = received-foreground
            numerator = (epsilon[:, 0].conj()*epsilon[:, 1]*to_first[:, 1]).sum().abs()
            denom = (epsilon[:, 0].abs().square().sum() *
                     (epsilon[:, 1]*to_first[:, 1]).abs().square().sum()).sqrt().clamp_min(1e-10)
            correlation = float(numerator/denom)
            if correlation > .10:
                raise ValueError('Independent re-noising has excessive aggregate correlation')
            clean_frame, _, clean_q, _ = coordinates(model, tensor(clean[cohort], 'cuda:0'))
            if baseline is None:
                parts.append((received, q, foreground, to_first, clean_frame, clean_q))
            else:
                estimates, active = zip(*(baseline(group) for group in wave.flatten(0, 1).split(40)), strict=True)
                supported = torch.cat(active).unflatten(0, (len(cohort), 2))
                base = complex_iq(torch.cat(estimates)).unflatten(0, (len(cohort), 2)) / basis
                parts.append((received, q, foreground, base, supported, clean_frame, clean_q))
            print('RETENTION_NOISE_REPLAYS', route, len(cohort), correlation, flush=True)
    return (*parts, contract)


def risk(model, arrays, indices, *, identity):
    received, q, signal, common, clean, clean_q = (v[indices] for v in arrays)
    if identity:
        received[-2:] = clean[-2:, None]
        q[-2:] = clean_q[-2:, None]
        signal[-2:] = clean[-2:, None]
        common[-2:] = 1.
    y = received.flatten(0, 1)
    epsilon = (received-signal).flatten(0, 1)
    predicted = model(y, q.flatten(0, 1), packet=True)
    reconstruction, nmse = noise_loss(predicted, epsilon, y)
    restored = (y-predicted).reshape_as(received) * common
    retain = retention_loss(restored, signal[:, 0])
    return reconstruction, nmse, retain


def proof(contract, route):
    return dict(status='SOURCE_FROZEN', route=route, contract=contract, seed=KEY, steps=STEPS,
                retention=RETAIN, fit_per_rx=FIT_PER_RX, held_per_rx=HELD_PER_RX, arms=list(ARMS), **ISOLATION)


def fit(rt, route, *, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(KEY)
    tracker = PacketRestorer('consensus').to(device).eval()
    count, hold = (5, 3) if smoke else (FIT_PER_RX, HELD_PER_RX)
    fitting, held, contract = build_arrays(rt, route, tracker, count, hold)
    models = {arm: Denoise().to(device) for arm in ARMS}
    models['retention'].load_state_dict(models['noise_only'].state_dict())
    optimizer = torch.optim.AdamW([p for model in models.values() for p in model.parameters()],
                                 lr=2e-4, weight_decay=1e-4)
    rng = torch.Generator(device=device).manual_seed(KEY)
    steps = 1 if smoke else STEPS
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=2e-5)
    for step in range(1, steps+1):
        i = torch.randint(len(fitting[0]), (BATCH,), generator=rng, device=device)
        all_values = torch.stack([torch.stack(risk(m, fitting, i, identity=True)) for m in models.values()])
        objective = all_values[:, 0].sum() + RETAIN * all_values[1, 2]
        if not torch.isfinite(all_values).all():
            raise FloatingPointError('Nonfinite fixed-action training loss')
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        for m in models.values():
            grad = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
            if step == 1 and grad <= 0:
                raise ValueError('Expected both model gradients in exact source trace smoke')
        optimizer.step()
        lr.step()
        if step == 1 or step % 50 == 0:
            print('RETENTION_FIT', route, step, *all_values.detach().flatten().tolist(), flush=True)
    scores = {}
    with torch.no_grad():
        for arm, model in models.items():
            model.eval()
            scores[arm] = (torch.stack([torch.stack(risk(model, held, i, identity=False))*len(i)
                           for i in torch.arange(len(held[0]), device=device).split(BATCH)]).sum(0)/len(held[0])).tolist()
    root = Path(rt['run_root'])
    record = proof(contract, route)
    report = {k: v for k, v in record.items() if k != 'contract'}
    report['held_error'] = scores
    if smoke:
        write(root/f'smoke-{route}.json', dict(report, status='PASS', steps=steps,
                                               fit_per_rx=count, held_per_rx=hold))
    else:
        torch.save(dict(proof=record, models={k: m.state_dict() for k, m in models.items()}), root/f'{route}-model.pt')
        write(root/f'{route}-complete.json', report)


def panel(rx):
    excluded = prior_panel(rx)
    rng = np.random.default_rng(KEY+1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == k), excluded),
                                             EVAL_PER_RX, replace=False)) for k in SOURCE])


def score(rt, route):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    binding = source_control(rt, route, device)
    saved = torch.load(root / f'{route}-model.pt', map_location=device, weights_only=True)
    if saved['proof'] != proof(binding[2], route) or set(saved['models']) != set(ARMS):
        raise ValueError('Wrong frozen signal-retaining source model')
    models = [Denoise().to(device).eval().requires_grad_(False) for k in ARMS]
    for model, arm in zip(models, ARMS, strict=True):
        model.load_state_dict(saved['models'][arm], strict=True)
    score_models(root, route, binding, models, selection=panel, modes=MODES,
                 packet_arms=(True, True), device=device)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    try:
        for stage in ('smoke', 'fit', 'score'):
            write(state, dict(status=stage.upper()), exclusive=False)
            schedule(rt, [(stage, route) for route in ROUTES], partial(start, module=rt['runner']))
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as e:
        write(state, dict(status='FAILED', error=str(e)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    cfg = read(context)
    rt = {k: cfg[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / RESIDUAL_RUN
    sub = read(parent_submit)
    if sub['archive_transfer'] != 'VERIFIED' or PurePosixPath(sub['run_root']) != parent:
        raise ValueError('Matching source datasets required')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest='action', required=True)
    command = sub.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        command.add_argument('--'+option, type=Path, required=True)
    for stage in ('dispatch', 'smoke', 'fit', 'score'):
        entry = sub.add_parser(stage)
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
