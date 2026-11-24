"""Attribute shared LoRa source failures using isolated noise and action controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .channel_blind_source import panel
from .chirp_spectrum_source import freeze, read_frozen
from .complex_vector_source import load_reader, publish
from .data import SOURCE, VIEWS
from .denoise_source import summarize
from .frame_marginal_source import MODES as HIST_MODES, RUN as HISTORY
from .full_packet_views import source_contract, source_role
from .joint_unroll_source import source_control
from .packet_physics import aligned_view, chirp_samples, chirp_windows, pack_windows
from .packet_runner import schedule
from .physical_inverse import full_action, noise_doses, residual_actions, solve_observations
from .physical_inverse_source import source_trace_implementation
from .refine_runner import start
from .registered_inverse import projected_operator, sampling_matrix
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-common-failure-physics-source-m2-r01'
DOSES = (0., .8, 1.)
ARMS = ('noise80', 'noise100', 'action', 'noise80_action', 'noise100_action')
MODES = ('direct', 'reference', *ARMS)
PAIRS = (('noise80', 'direct'), ('noise100', 'direct'), ('action', 'direct'),
         ('noise80_action', 'noise80'), ('noise100_action', 'noise100'),
         ('noise80_action', 'action'), ('noise100_action', 'action'))
CONTROLS = tuple(HIST_MODES.index(name) for name in ('direct', 'reference', 'rank_half', 'bounded_half'))


def isolated_inverse(operator, observed, signal, noise):
    # Use the unedited observation's noise level in all three solves.
    damping = np.sqrt(3 * np.mean(np.abs(noise)**2))
    return solve_observations(operator, noise_doses(observed, signal, DOSES), damping)


def full_controls(iq, scene, rx, pid):
    action, source, observed, signal, noise, gain, offset, replay_error = full_action(iq, scene, rx, pid)
    sample = sampling_matrix(len(observed), offset)
    native = chirp_samples(observed, offset)
    clear, error = sample @ signal, sample @ (action(source) - signal)
    mismatch = np.linalg.norm(error) / max(np.linalg.norm(clear), 1e-12)
    if not np.isfinite(mismatch) or mismatch > 2e-9:
        raise ValueError('Projected action diverges on the registered Full windows')
    answers = isolated_inverse(projected_operator(action, sample),
                               native.ravel().astype(np.complex128)/gain, clear, sample @ noise)
    direct = [chirp_samples(value, offset) for value in noise_doses(observed, gain*signal, DOSES)]
    recovered = [chirp_samples(x) for x in answers]
    return np.stack([pack_windows(v) for v in (*direct, *recovered)]), max(replay_error, mismatch)


def residual_controls(iq, scene, rx, pid):
    cases, errors = [], []
    for action, observed, signal, noise, gain, replay_error in residual_actions(iq, scene, rx, pid):
        answers = isolated_inverse(action.operator(), observed.astype(np.complex128)/gain, signal, noise)
        direct = noise_doses(observed, gain*signal, DOSES)
        cases.append((*direct, *(x[32:288] for x in answers)))
        errors.append(replay_error)
    return np.stack([pack_windows(np.stack(v)) for v in zip(*cases, strict=True)]), max(errors)


def diagnostic_capture(route, scene, item):
    iq, rx, pid = item
    if scene == 'clean':
        return np.broadcast_to(chirp_windows(iq), (1+len(ARMS), 6, 2, 256)), 0.
    if route == 'full':
        return full_controls(iq, scene, rx, pid)
    waves, error = residual_controls(iq, scene, rx, pid)
    expected = aligned_view(iq, scene, rx, pid, 'source-V', processing_route='residual')
    if not np.array_equal(waves[0], expected):
        raise ValueError('Residual replay differs from the registered input')
    return waves, error


@torch.no_grad()
def evaluate(rt, route, *, smoke=False):
    started = perf_counter()
    root = Path(rt['run_root'])
    device = 'cuda:0'
    source, cache, contract, baseline, _ = source_control(rt, route, device)
    reader = load_reader(root, contract, device)
    raw, rx, ids, _ = source_role(source, contract, 'U_s' if smoke else 'V')
    chosen = (np.array([np.flatnonzero(rx == key)[0] for key in SOURCE]) if smoke else panel(rx))
    jobs = list(zip(raw[chosen], rx[chosen], ids[chosen], strict=True))
    cached = np.load(cache/'V.npy', mmap_mode='r', allow_pickle=False) if not smoke and cache else None
    predictions, metrics = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene in ('clean', VIEWS[-1]) if smoke else VIEWS:
            values, errors = zip(*pool.map(partial(diagnostic_capture, route, scene), jobs, chunksize=3), strict=True)
            captures = np.stack(values, axis=1)
            if not np.isfinite(captures).all() or not np.isfinite(errors).all():
                raise FloatingPointError('Nonfinite source counterfactual')
            if cached is not None and not np.array_equal(captures[0], cached[VIEWS.index(scene), chosen]):
                raise ValueError('Full replay differs from the registered input')
            if scene == 'clean' and not np.all(captures[1:] == captures[:1]):
                raise ValueError('Diagnostic clean changed')
            if smoke and scene != 'clean' and any(np.array_equal(captures[0], x) for x in captures[1:]):
                raise ValueError('Source U smoke did not exercise every diagnostic arm')
            parts, active = [], 0
            for block in tensor(captures, device).split(16, dim=1):
                reference, allowed = baseline(block[0])
                inputs = torch.cat((block[:1], reference.unsqueeze(0), block[1:]))
                logits = reader(inputs.flatten(0, 1))[0]
                if logits.shape != (inputs.shape[0]*inputs.shape[1], 10) or not logits.isfinite().all():
                    raise ValueError('Invalid frozen classifier output')
                parts.append(logits.argmax(1).reshape(len(MODES), -1).cpu().numpy())
                active += int(allowed.sum())
            predictions[scene] = np.concatenate(parts, axis=1)
            metrics[scene] = dict(max_operator_replay_error=float(max(errors)), reference_active=active)
            print('LORA_COMMON_FAILURE_CAPTURE', route, scene, len(chosen), active, flush=True)

    cost = dict(worker_seconds=perf_counter()-started, gpu=torch.cuda.get_device_name(), cpu_workers=8)
    if smoke:
        write(root / f'smoke-{route}.json', dict(status='PASS', source_U=len(chosen), target_reads=0,
              TX_reads=0, metrics=metrics, **cost))
    else:
        freeze(root, route, source, chosen, ids, predictions, metrics, MODES)
        write(root / f'{route}-cost.json', cost)


def cohort_counts(arms, eligible, sections):
    n = int(eligible.sum())
    correct = (arms & eligible).sum(axis=(-2, -1))
    by_rx = (arms & eligible).reshape(len(arms), eligible.shape[0], len(SOURCE), sections).sum(axis=(1,3))
    noise, action, joint = (arms[ARMS.index(key)] for key in ('noise100', 'action', 'noise100_action'))
    return dict(n=n, correct=dict(zip(ARMS, correct.tolist(), strict=True)), by_rx=by_rx.tolist(),
                noise_only=int((eligible & noise & ~action).sum()),
                action_only=int((eligible & action & ~noise).sum()),
                combined_only=int((eligible & joint & ~noise & ~action).sum()),
                neither_nor_joint=int((eligible & ~joint & ~noise & ~action).sum()))


def cohort_statistics(frozen, historical, labels, sections):
    views = VIEWS[1:]
    clean = frozen['full'][1]['clean'][0] == labels
    failures, results = {}, {}
    for route in ROUTES:
        rows, old = frozen[route][1], historical[route][1]
        for view in VIEWS:
            for idx, name in enumerate(MODES[:2]):
                if not np.array_equal(rows[view][idx], old[view][HIST_MODES.index(name)]):
                    raise ValueError(f'Historical {route} {view} {name} predictions changed')
        previous = np.stack([old[v][list(CONTROLS)] == labels for v in views], axis=1)
        failures[route] = ~previous.any(axis=0) & clean
    shared = failures['full'] & failures['residual']
    for route in ROUTES:
        rows = frozen[route][1]
        correct = np.stack([rows[v][2:] == labels for v in views], axis=1)
        local = failures[route]
        results[route] = dict(
            local_six=cohort_counts(correct, local, sections),
            shared_six=cohort_counts(correct, shared, sections),
            local_by_view={v: cohort_counts(correct[:, i:i+1], local[i:i+1], sections)
                           for i, v in enumerate(views)},
            shared_by_view={v: cohort_counts(correct[:, i:i+1], shared[i:i+1], sections)
                            for i, v in enumerate(views)})
    return results


def score(rt):
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    history = root.parent/HISTORY
    if read(history/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Need the prior matched source predictions')
    frozen = {r: read_frozen(root, r, MODES) for r in ROUTES}
    old = {r: read_frozen(history, r, HIST_MODES) for r in ROUTES}
    physical_order = frozen['full'][0]
    if any(bundle[0] != physical_order for bundle in (*frozen.values(), *old.values())):
        raise ValueError('Source counterfactuals require identical physical IDs and order')
    ids = np.asarray(source_contract(source)['role_ids']['V'])
    lookup = {identity: i for i, identity in enumerate(ids)}
    chosen = np.asarray([lookup[key] for key in physical_order])
    with np.load(source/'V_metadata.npz', allow_pickle=False) as metadata:
        if not np.array_equal(panel(np.asarray(SOURCE)[metadata['rx_i']]), chosen):
            raise ValueError('Unexpected registered source selection')
        labels = metadata['y'][chosen]
    results = cohort_statistics(frozen, old, labels, len(chosen)//len(SOURCE))
    write(root/'common-failure-scores.json', dict(status='SOURCE_DIAGNOSTIC_SCORED',
          modes=ARMS, n=len(chosen), routes=results, target_reads=0, training_updates=0,
          truth_after_predictions=True, history=HISTORY))
    for route, (_, rows, metrics) in frozen.items():
        summarize(root, route, source, chosen, ids, rows, metrics, MODES,
                  predictions_written=True, comparisons=PAIRS, per_rx=len(chosen)//len(SOURCE))
    print('LORA_COMMON_FAILURE_SOURCE_SCORED', flush=True)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    launch = partial(start, module=rt['runner'])
    try:
        for phase in ('smoke', 'evaluate'):
            schedule(rt, [(phase, route) for route in ROUTES], launch)
        if launch(rt, 'score', 'source').wait():
            raise RuntimeError('Independent LoRa common failure score failed')
        write(root/'state.json', dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root/'state.json', dict(status='FAILED', error=str(exc)))
        raise


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest='action', required=True)
    release = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        release.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'score'):
        sub = actions.add_parser(action)
        sub.add_argument('runtime', type=Path)
        if action != 'dispatch':
            sub.add_argument('row', choices=('source',) if action == 'score' else ROUTES)
    options = vars(cli.parse_args())
    action = options.pop('action')
    if action == 'publish':
        return publish(**options, run=RUN, runner=__spec__.name)
    runtime = read(options['runtime'])
    if action == 'dispatch':
        return dispatch(runtime)
    if action == 'score':
        return score(runtime)
    activate(runtime)
    source_trace_implementation()
    return evaluate(runtime, options['row'], smoke=action == 'smoke')


if __name__ == '__main__':
    main()
