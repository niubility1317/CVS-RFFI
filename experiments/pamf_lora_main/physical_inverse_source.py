"""Measure exact source-operator inversion before changing DURA training."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .data import SOURCE, VIEWS
from .dura_repair import no_tone, real_iq
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_supervised import construct as recognizer
from .joint_unroll_source import coordinates, source_control, source_validation
from .packet_physics import aligned_view, chirp_windows
from .packet_runner import schedule
from .physical_inverse import full_replay, residual_replay
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import RUN as RESIDUAL_RUN, SEED
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-exact-action-source-m2-r01'
DRAW = 2026112607
PER_RX = 60
MODES = ('direct', 'reference', 'known_denoise', 'known_action', 'known_both',
         'projected_action', 'projected_both', 'paired_clean')


def select_panel(rx, count):
    previous = source_validation(rx)
    rng = np.random.default_rng(DRAW)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == key), previous),
                                             count, replace=False)) for key in SOURCE])


def simulate(route, scene, row):
    iq, rx, pid = row
    if scene == 'clean':
        return np.broadcast_to(chirp_windows(iq), (4, 6, 2, 256)), 0.
    replay = full_replay if route == 'full' else residual_replay
    wave, error = replay(iq, scene, rx, pid)
    if route == 'residual':
        expected = aligned_view(iq, scene, rx, pid, 'source-V', processing_route='residual')
        if not np.array_equal(wave[0], expected):
            raise ValueError('Source snapshot inverse does not use the registered observation')
    return wave, error


@torch.no_grad()
def readouts(wave, truth, baseline, tracker):
    received, denoised, inverse, both = wave
    reference, allowed = baseline(received)
    if not allowed.any():
        return torch.stack((received, reference, received, received, received, received, received, truth)), 0
    obs = received[allowed]
    original, rotation, _, rms = coordinates(tracker, obs)
    projected = []
    for estimate in (inverse, both):
        wanted = coordinates(tracker, estimate[allowed])[0]
        delta = no_tone(wanted - original)
        delta = delta / (1. + delta.abs().square().mean(-1, keepdim=True) / .15**2).sqrt()
        out = received.clone()
        out[allowed] = obs + real_iq(delta*rotation*rms)
        projected.append(out)
    practical = [torch.where(allowed[:, None, None, None], value, received)
                 for value in (denoised, inverse, both)]
    return torch.stack((received, reference, *practical, *projected, truth)), int(allowed.sum())


def summarize(root, route, source, ids, chosen, preds, metrics):
    write(root / f'{route}-predictions.json',
          dict(ids=ids[chosen].tolist(), rows={k: v.tolist() for k, v in preds.items()}))
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        if not np.array_equal(metadata['ids'][chosen], ids[chosen]):
            raise ValueError('Source V scoring changed the held-out row order')
        truth = metadata['y'][chosen]
    hits = {view: row == truth for view, row in preds.items()}
    for view, row in hits.items():
        baseline = row[1]
        metrics[view].update(correct=row.sum(1).tolist(),
            repairs=(row & ~baseline).sum(1).tolist(), harms=(~row & baseline).sum(1).tolist(),
            by_rx=row.reshape(len(MODES), len(SOURCE), PER_RX).sum(-1).tolist())
    six = np.stack([hits[k] for k in VIEWS[1:]]).sum(0)
    rng = np.random.default_rng(DRAW+1)
    bootstrap = np.concatenate([rng.integers(j*PER_RX, (j+1)*PER_RX, (1600, PER_RX))
                               for j in range(len(SOURCE))], axis=-1)
    ci = {k: [np.quantile(delta[bootstrap].sum(1), (.025, .975)).tolist() for delta in h-h[1]]
          for k, h in (('six', six), ('low', hits[VIEWS[-1]].astype(int)))}
    write(root / f'{route}-source-scores.json', dict(status='SOURCE_SCORED', route=route,
          n=len(chosen), modes=list(MODES), metrics=metrics, six=six.sum(1).tolist(),
          paired_ci_vs_reference=ci, trained_G_updates=0, target_reads=0, truth_after_predictions=True))


@torch.no_grad()
def evaluate(rt, route, *, smoke=False):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    source, cache, contract, baseline, tracker = source_control(rt, route, device)
    raw, rx, ids, _ = source_role(source, contract, 'V')
    chosen = select_panel(rx, 2 if smoke else PER_RX)
    rows = list(zip(raw[chosen], rx[chosen], ids[chosen], strict=True))
    clean = np.stack([chirp_windows(row[0]) for row in rows])
    reference = np.load(cache / 'V.npy', allow_pickle=False, mmap_mode='r') if cache is not None else None
    scenes = (VIEWS[1], VIEWS[-1]) if smoke else VIEWS
    if not smoke:
        f = recognizer(device).eval().requires_grad_(False)
        f_file = complete_f(root.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
        f.load_state_dict(torch.load(f_file, map_location=device, weights_only=True), strict=True)
    metrics, preds = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene in scenes:
            replay = zip(*pool.map(partial(simulate, route, scene), rows, chunksize=2), strict=True)
            stack, errors = replay
            captures = np.stack(stack, axis=1)
            if (not np.isfinite(captures).all() or
                    (reference is not None and not np.array_equal(captures[0], reference[VIEWS.index(scene), chosen]))):
                raise ValueError('Expected finite inverse arms on the registered physical observations')
            errors = np.asarray(errors)
            count, parts = 0, []
            for batch, truth in zip(tensor(captures, device).split(20, dim=1),
                                    tensor(clean, device).split(20), strict=True):
                wave, active = readouts(batch, truth, baseline, tracker)
                count += active
                if not torch.isfinite(wave).all():
                    raise ValueError('Nonfinite physical inverse output')
                if not smoke:
                    logits = f(wave.flatten(0, 1))[0]
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite frozen Tri-set logits')
                    parts.append(logits.argmax(1).reshape(len(MODES), len(truth)).cpu().tolist())
            metrics[scene] = dict(active=count, replay_relative_error=errors.max().item())
            if not smoke:
                preds[scene] = np.concatenate(parts, axis=1)
            print('EXACT_SOURCE_INVERSE', route, scene, len(chosen), count, errors.max(), flush=True)
    if smoke:
        write(root / f'smoke-{route}.json', dict(status='PASS', route=route, n=len(chosen),
              metrics=metrics, F_reads=0, TX_reads=0, target_reads=0))
    else:
        summarize(root, route, source, ids, chosen, preds, metrics)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
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
    parent = project / 'runs' / RESIDUAL_RUN
    sub = read(parent_submit)
    if sub['archive_transfer'] != 'VERIFIED' or PurePosixPath(sub['run_root']) != parent:
        raise ValueError('Matching source-only LoRa contract required')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


def source_trace_implementation():
    # The frozen F/G source releases stay fixed. Use this release's channel
    # module for the added trace fields, then verify received IQ against cache.
    code = Path(__file__).resolve().parents[2] / 'code'
    sys.path.insert(0, str(code))
    try:
        from leo_practical import channel
    finally:
        sys.path.pop(0)
    if Path(channel.__file__).resolve().parent.parent != code:
        raise ValueError('The exact operator requires the registered trace extension')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for command in ('dispatch', 'smoke', 'score'):
        entry = commands.add_parser(command)
        entry.add_argument('runtime', type=Path)
        if command != 'dispatch':
            entry.add_argument('route', choices=ROUTES)
    opts = vars(parser.parse_args())
    command = opts.pop('command')
    if command == 'publish':
        publish(**opts)
    else:
        rt = read(opts['runtime'])
        if command == 'dispatch':
            dispatch(rt)
        else:
            activate(rt)
            source_trace_implementation()
            evaluate(rt, opts['route'], smoke=command == 'smoke')
