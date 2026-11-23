"""Measure bandwidth, dose and learned-vector scaling after mature LoRa DURA."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .complex_budget import BUDGETS, LEARNED_SCALES, chirp_rms, known_vectors
from .complex_vector import ComplexVector, fft
from .complex_vector_source import ARMS, RUN as PARENT, record, result_panel
from .data import SOURCE, VIEWS
from .denoise_source import EVAL_PER_RX, normalized, replay, summarize
from .dura_repair import complex_iq, real_iq
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_supervised import construct as recognizer
from .joint_unroll_source import coordinates, source_control
from .packet_runner import schedule
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import RUN as RESIDUAL_RUN, SEED
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-complex-budget-source-m2-r01'
KEY = 2026112751
LEARNED = tuple(f'{k}_{tag}' for k in ARMS for tag in LEARNED_SCALES)
MATCHED = tuple(f'{scope}_{cap}' for cap in BUDGETS for scope in ('narrow', 'broad'))
MODES = ('direct', 'reference', *LEARNED, *MATCHED, 'narrow_exact', 'mid_exact',
         'broad_exact', 'all_exact', 'known_noise')


def panel(rx, samples):
    excluded = result_panel(rx)
    rng = np.random.default_rng(KEY)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == key), excluded),
                                             samples, replace=False)) for key in SOURCE])


@torch.no_grad()
def readouts(y, noise, baseline, tracker, models):
    ref, active = baseline(y)
    output = {k: ref.clone() for k in MODES[2:]}
    sums = {k: 0. for k in MODES[2:]}
    if not active.any():
        return torch.stack((y, ref, *output.values())), 0, sums, 0.
    selected, mature = y[active], ref[active]
    obs, rotation, q, rms = coordinates(tracker, selected)
    basis = rotation*rms
    base = complex_iq(mature)/basis
    desired = obs-complex_iq(noise[active])/basis
    options = known_vectors(fft(desired-base))
    for arm, model in zip(ARMS, models, strict=True):
        actual = model(obs, base, q)
        options.update({f'{arm}_{key}': factor*actual for key, factor in LEARNED_SCALES.items()})
    delivered = {}
    for key, change in options.items():
        output[key][active] = mature + real_iq(torch.fft.ifft(change, norm='ortho')*basis)
        amount = chirp_rms(complex_iq(output[key][active]-mature)/basis)[..., 0]
        sums[key] = float(amount.mean(dim=1).sum())
        if key in MATCHED:
            delivered[key] = amount
    output['known_noise'][active] = normalized(selected-noise[active], selected)
    sums['known_noise'] = float(chirp_rms(complex_iq(output['known_noise'][active]-mature)/basis)
                                .flatten(1).mean(dim=1).sum())
    discrepancy = max(float((delivered[f'narrow_{key}']-delivered[f'broad_{key}']).abs().max())
                      for key in BUDGETS)
    return torch.stack((y, ref, *output.values())), int(active.sum()), sums, discrepancy


@torch.no_grad()
def score(rt, route, smoke=False):
    root = Path(rt['run_root'])
    device = 'cuda:0'
    source, cache, contract, baseline, tracker = source_control(rt, route, device)
    parent = root.parent / PARENT
    if read(parent/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected complete source-frozen vector candidates')
    state = torch.load(parent/f'{route}-model.pt', map_location=device, weights_only=True)
    if state['proof'] != record(contract, route) or set(state['models']) != set(ARMS):
        raise ValueError('Expected matching checkpoint ancestry and an exact LoRa route')
    models = [ComplexVector().to(device).eval().requires_grad_(False) for k in ARMS]
    for name, m in zip(ARMS, models, strict=True):
        m.load_state_dict(state['models'][name], strict=True)
    iq, rx, ids, _ = source_role(source, contract, 'V')
    chosen = panel(rx, 3 if smoke else EVAL_PER_RX)
    rows = list(zip(iq[chosen], rx[chosen], ids[chosen], strict=True))
    official = np.load(cache/'V.npy', mmap_mode='r', allow_pickle=False) if cache is not None else None
    if not smoke:
        f = recognizer(device).eval().requires_grad_(False)
        path = complete_f(root.parent/SET_RUN, contract, case(SEED, SET_METHOD))/'model.pt'
        f.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    views = (VIEWS[0], VIEWS[-1]) if smoke else VIEWS
    metrics, predictions = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for view in views:
            wave, epsilon = (np.stack(v) for v in zip(*pool.map(partial(replay, route, view), rows,
                                                                  chunksize=8), strict=True))
            if official is not None and not np.array_equal(wave, official[VIEWS.index(view), chosen]):
                raise ValueError('Source observations differ from the registered Full route')
            count, gap, parts = 0, 0., []
            sums = {k: 0. for k in MODES[2:]}
            for y, n in zip(tensor(wave, device).split(9), tensor(epsilon, device).split(9), strict=True):
                values, ne, ds, match = readouts(y, n, baseline, tracker, models)
                if not torch.isfinite(values).all() or match > 1e-6 or (view == 'clean' and ne):
                    raise ValueError('Invalid waveform, clean guard, or paired correction energy')
                count += ne
                gap = max(gap, match)
                for k in ds:
                    sums[k] += ds[k]
                if not smoke:
                    logits = f(values.flatten(0, 1))[0]
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite frozen Tri-set logits')
                    parts.append(logits.argmax(1).reshape(len(MODES), len(y)).cpu().tolist())
            metrics[view] = dict(active=count, max_dose_difference=gap,
                                 mean_active_dose={k: v/max(count, 1) for k,v in sums.items()})
            if not smoke:
                predictions[view] = np.concatenate(parts, axis=1)
            print('COMPLEX_BUDGET_SOURCE', route, view, count, gap, flush=True)
    if smoke:
        if not metrics[VIEWS[-1]]['active']:
            raise ValueError('Expected practical corrections during the physical source smoke')
        write(root/f'smoke-{route}.json', dict(status='PASS', route=route, n=len(chosen),
              F_reads=0, TX_reads=0, target_reads=0, metrics=metrics))
    else:
        summarize(root, route, source, chosen, ids, predictions, metrics, MODES)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root/'state.json'
    try:
        for phase in ('smoke', 'score'):
            write(state, dict(status=phase.upper()), exclusive=False)
            schedule(rt, [(phase, route) for route in ROUTES], partial(start, module=rt['runner']))
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as e:
        write(state, dict(status='FAILED', error=str(e)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    cfg = read(context)
    rt = {k: cfg[k] for k in ('project', 'python', 'frozen_sources')}
    base = PurePosixPath(rt['project'])
    parent = base/'runs'/RESIDUAL_RUN
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Expected the frozen source contract')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(base/'runs'/RUN), release_root=str(base/'releases'/f'{RUN}-release'),
              source_root=str(base/'releases'/f'{DATA_RUN}-release'/'data'/'source'), parent_root=str(parent),
              allowed_gpus=[3,6,2,0,4,1,5], max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'score'):
        tool = commands.add_parser(action)
        tool.add_argument('runtime', type=Path)
        if action != 'dispatch':
            tool.add_argument('route', choices=ROUTES)
    opts = vars(cli.parse_args())
    action = opts.pop('action')
    if action == 'publish':
        publish(**opts)
    else:
        runtime = read(opts['runtime'])
        if action == 'dispatch':
            dispatch(runtime)
        else:
            activate(runtime)
            score(runtime, opts['route'], smoke=action=='smoke')
