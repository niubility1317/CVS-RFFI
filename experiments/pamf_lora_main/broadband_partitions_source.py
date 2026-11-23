"""Find the learned spectral neighborhoods responsible for LoRa DURA errors."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .broadband_partitions import LEARNED, SWAPS, deploy_vectors, source_swaps
from .broadband_vector import target_vectors
from .broadband_vector_source import RECIPE, RUN as PARENT, panel as old_panel
from .complex_budget import chirp_rms
from .complex_vector_source import load_models, publish, score
from .data import SOURCE, VIEWS
from .denoise_source import EVAL_PER_RX, normalized, replay
from .dura_repair import complex_iq, real_iq
from .full_packet_views import source_role
from .joint_unroll_source import coordinates, source_control
from .packet_runner import schedule
from .refine_runner import start
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-broadband-partitions-source-m2-r01'
DRAW = 2026112777
MATCHED = tuple(f'matched_{k}' for k in SWAPS)
MODES = ('direct', 'reference', *LEARNED, *(f'known_{k}' for k in SWAPS), *MATCHED, 'known_noise')


def panel(rx, count=EVAL_PER_RX):
    excluded = old_panel(rx)
    rng = np.random.default_rng(DRAW)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == name), excluded),
                                             count, replace=False)) for name in SOURCE])


@torch.no_grad()
def readouts(received, noise, baseline, tracker, models, *, diagnostic=False):
    ref, active = baseline(received)
    out = {k: ref.clone() for k in MODES[2:]}
    gap = 0.
    if active.any():
        observed = received[active]
        mature = ref[active]
        work, rotation, q, rms = coordinates(tracker, observed)
        basis = rotation*rms
        base = complex_iq(mature)/basis
        n, b = (m(work, base, q) for m in models)
        known = target_vectors(base, work-complex_iq(noise[active])/basis)['broad']
        replacements, gap = source_swaps(b, known)
        options = dict(deploy_vectors(n, b), **replacements)
        for key, delta in options.items():
            out[key][active] = mature + real_iq(torch.fft.ifft(delta, norm='ortho')*basis)
        out['known_noise'][active] = normalized(observed-noise[active], observed)
        nominal = chirp_rms(complex_iq(out['broad'][active]-mature)/basis)
        for key in MATCHED:
            delivered = chirp_rms(complex_iq(out[key][active]-mature)/basis)
            gap = max(gap, float((delivered-nominal).abs().max()))
        if gap > 1e-6:
            raise ValueError('Delivery altered the matched correction RMS')
    result = torch.stack((received, ref, *out.values()))
    count = int(active.sum())
    return (result, count, gap) if diagnostic else (result, count)


@torch.no_grad()
def smoke(rt, route):
    root = Path(rt['run_root'])
    device = 'cuda:0'
    source, cache, contract, baseline, tracker = source_control(rt, route, device)
    parent = root.parent/PARENT
    if read(parent/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Broadband parent must be fully frozen and scored')
    models = load_models(parent, route, contract, RECIPE, device)
    raw, rx, ids, _ = source_role(source, contract, 'V')
    chosen = panel(rx, 3)
    items = list(zip(raw[chosen], rx[chosen], ids[chosen], strict=True))
    actual = np.load(cache/'V.npy', mmap_mode='r', allow_pickle=False) if cache is not None else None
    out = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for view in (VIEWS[0], VIEWS[-1]):
            wave, eps = (np.stack(v) for v in zip(*pool.map(partial(replay, route, view),
                                                              items, chunksize=3), strict=True))
            if actual is not None and not np.array_equal(wave, actual[VIEWS.index(view), chosen]):
                raise ValueError('Unmatched source practical observation')
            result, active, phase_gap = readouts(tensor(wave, device), tensor(eps, device),
                                                 baseline, tracker, models, diagnostic=True)
            if not torch.isfinite(result).all() or (view=='clean' and active) or (view!='clean' and not active):
                raise ValueError('Invalid clean or practical smoke waveform')
            out[view] = dict(active=active, max_delivered_dose_error=phase_gap)
            print('BROADBAND_PARTITIONS_SOURCE', route, view, active, phase_gap, flush=True)
    write(root/f'smoke-{route}.json', dict(status='PASS', route=route, n=len(chosen), F_reads=0,
          TX_reads=0, target_reads=0, metrics=out))


def dispatch(rt):
    # The common trainer's scheduler is inappropriate: this is frozen readout only.
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


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'score'):
        program = commands.add_parser(action)
        program.add_argument('runtime', type=Path)
        if action != 'dispatch':
            program.add_argument('route', choices=ROUTES)
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args, run=RUN, runner=__spec__.name)
    else:
        rt = read(args['runtime'])
        if action == 'dispatch':
            dispatch(rt)
        else:
            activate(rt)
            if action == 'smoke':
                smoke(rt, args['route'])
            else:
                score(rt, args['route'], recipe=RECIPE, panel=panel, modes=MODES,
                      render=readouts, checkpoint_run=PARENT)
