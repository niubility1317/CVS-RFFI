"""Score gain and orthogonal-noise swaps on a held-out LoRa source panel."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_unrolled_restore.denoise import Denoise

from .data import SOURCE, VIEWS
from .denoise_components import bands, component_readouts, observed_gain
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
from .signal_retention_source import ARMS, RUN as PARENT, panel as earlier_panel, proof
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-noise-components-source-m2-r01'
DRAW = 2026112721
MODES = ('direct', 'reference', 'ordinary', 'retained', 'retention_gain', 'retention_residual',
         'retention_bands', 'matched_gain', 'matched_residual', 'matched_both', 'observed_gain', 'known_noise')


def selection(rx, count):
    excluded = earlier_panel(rx)
    rng = np.random.default_rng(DRAW)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == k), excluded),
                                             count, replace=False)) for k in SOURCE])


@torch.no_grad()
def readouts(iq, noise, reference, tracker, models):
    baseline, allowed = reference(iq)
    output = [iq, baseline, *(iq.clone() for _ in MODES[2:])]
    if not allowed.any():
        return torch.stack(output), 0, 0.
    y = iq[allowed]
    measured, phase, quality, rms = coordinates(tracker, y)
    guesses = [m(measured, quality, packet=True) for m in models]
    b, a = (torch.fft.fft(measured-value, norm='ortho') for value in guesses)
    eps = complex_iq(noise[allowed])*phase.conj()/rms
    target = torch.fft.fft(measured-eps, norm='ortho')
    mixed, mismatch = component_readouts(b, a, target)
    ordinary = y-real_iq(guesses[0]*phase*rms)
    nominal = normalized(ordinary, y)
    output[MODES.index('ordinary')][allowed] = nominal
    output[MODES.index('retained')][allowed] = normalized(y-real_iq(guesses[1]*phase*rms), y)
    output[MODES.index('known_noise')][allowed] = normalized(y-noise[allowed], y)
    dims = (-3, -2, -1)
    scale = (y.square().sum(dims, keepdim=True) /
             ordinary.square().sum(dims, keepdim=True).clamp_min(1e-7)).sqrt()
    for arm, spectrum in dict(observed_gain=observed_gain(b, a), **mixed).items():
        frame = torch.fft.ifft(spectrum-b, norm='ortho')*phase*rms
        output[MODES.index(arm)][allowed] = nominal + real_iq(frame)*scale
    # Pull delivered IQ back into the common tracked frame for the dose check.
    edited = [output[MODES.index(k)][allowed] for k in ('matched_gain', 'matched_residual', 'matched_both')]
    delta = torch.stack([complex_iq((value-nominal) / scale)*phase.conj()/rms for value in edited])
    delta = torch.fft.fft(delta, norm='ortho')
    for selected in bands(delta.shape[-1], delta.device):
        dose = delta[..., selected].abs().square().mean(-1).sqrt()
        mismatch = max(mismatch, float((dose.max(0).values-dose.min(0).values).max()))
    return torch.stack(output), int(allowed.sum()), mismatch


@torch.no_grad()
def evaluate(rt, route, *, smoke=False):
    root = Path(rt['run_root'])
    device = 'cuda:0'
    src, cache, contract, reference, tracker = source_control(rt, route, device)
    parent = root.parent / PARENT
    if read(parent/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Require the completed, frozen paired source run')
    saved = torch.load(parent/f'{route}-model.pt', map_location=device, weights_only=True)
    if saved['proof'] != proof(contract, route) or set(saved['models']) != set(ARMS):
        raise ValueError('Expected the individually frozen source noise models')
    models = [Denoise().to(device).eval().requires_grad_(False) for _ in ARMS]
    for model, key in zip(models, ARMS, strict=True):
        model.load_state_dict(saved['models'][key], strict=True)
    iq, rx, ids, _ = source_role(src, contract, 'V')
    selected = selection(rx, 3 if smoke else EVAL_PER_RX)
    rows = list(zip(iq[selected], rx[selected], ids[selected], strict=True))
    registered = np.load(cache/'V.npy', mmap_mode='r', allow_pickle=False) if route=='full' else None
    views = (VIEWS[0], VIEWS[-1]) if smoke else VIEWS
    pred, metrics = {}, {}
    if not smoke:
        f = recognizer(device).eval().requires_grad_(False)
        weights = complete_f(root.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
        f.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for view in views:
            wave, noise = (np.stack(r) for r in zip(*pool.map(partial(replay, route, view), rows,
                                                              chunksize=8), strict=True))
            if registered is not None and not np.array_equal(wave, registered[VIEWS.index(view), selected]):
                raise ValueError('Source V used different full practical IQ')
            active, max_gap, parts = 0, 0., []
            for b, n in zip(tensor(wave, device).split(16), tensor(noise, device).split(16), strict=True):
                values, count, gap = readouts(b, n, reference, tracker, models)
                if not torch.isfinite(values).all() or gap > 1e-6:
                    raise ValueError('Nonfinite source readout or unmatched source correction amplitude')
                if view == 'clean' and (count != 0 or not torch.equal(values[0], values[2])):
                    raise ValueError('Invalid unchanged clean waveform')
                active += count
                max_gap = max(max_gap, gap)
                if not smoke:
                    logits = f(values.flatten(0, 1))[0]
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite frozen Tri-set logits')
                    parts.append(logits.argmax(1).reshape(len(MODES), len(b)).cpu().tolist())
            metrics[view] = dict(active=active, max_equal_dose_error=max_gap)
            if not smoke:
                pred[view] = np.concatenate(parts, axis=1)
            print('DENOISE_COMPONENTS', route, view, len(selected), active, max_gap, flush=True)
    if smoke:
        if not metrics[VIEWS[-1]]['active']:
            raise ValueError('Source replay did not exercise any practical correction')
        write(root/f'smoke-{route}.json', dict(status='PASS', n=len(selected), route=route,
              modes=list(MODES), metrics=metrics, F_reads=0, TX_reads=0, target_reads=0))
    else:
        summarize(root, route, src, selected, ids, pred, metrics, MODES)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root/'state.json'
    try:
        for phase in ('smoke', 'score'):
            write(state, dict(status=phase.upper()), exclusive=False)
            schedule(rt, [(phase, route) for route in ROUTES], partial(start, module=rt['runner']))
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    cfg = read(context)
    rt = {k: cfg[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project/'runs'/RESIDUAL_RUN
    sub = read(parent_submit)
    if sub['archive_transfer'] != 'VERIFIED' or PurePosixPath(sub['run_root']) != parent:
        raise ValueError('Mismatched frozen source dataset')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project/'runs'/RUN), release_root=str(project/'releases'/f'{RUN}-release'),
              source_root=str(project/'releases'/f'{DATA_RUN}-release'/'data'/'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'score'):
        entry = commands.add_parser(action)
        entry.add_argument('runtime', type=Path)
        if action != 'dispatch':
            entry.add_argument('route', choices=ROUTES)
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
            evaluate(runtime, opts['route'], smoke=action == 'smoke')
