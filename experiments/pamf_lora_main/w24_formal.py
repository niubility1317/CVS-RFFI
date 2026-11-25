"""Blind three-seed, 30,000-packet LoRa Full and native Residual evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import monotonic, sleep

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_repair import complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import native, source_contract, target_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_native_full import (
    BATCH, DATA_RUN, LIMIT, METHOD, SEED, SOURCE, TARGET, VIEWS, W24_COMMIT, W24_RUN,
    channel_job, classify_aperture, load_restorers, recover,
)
from experiments.pamf_lora_main.w24_residual_pilot import (
    R69_GATE, R69_TRAIN, ROUTE, r69_baseline, receive, restore_r69,
)
from experiments.pamf_lora_main.w24_sat_aug import AUG_WEIGHT, ELIGIBLE, EPOCHS, RUNS
from experiments.pamf_lora_main.w24_wide_transfer import (
    TRAIN_RUN, load_restorer, received_aperture, training_runtime,
)


RUN = '20261125-phase1-lora-w24-formal-both-routes-m3-r01'
RUN_UNGATED = '20261125-phase1-lora-w24-formal-ungated-s2026092702-r01'
SEEDS = (2026092701, SEED, 2026092703)
FULL = 'full_native'
ROUTES = (FULL, ROUTE)
WIDE = 'wide_dura_half'
WIDE_UNGATED = 'wide_dura_ungated_half'
MODES = {
    FULL: ('direct', 'dura_half', 'complex_response_half', WIDE),
    ROUTE: ('direct', 'r69_gated_half', WIDE),
}
RX_PER_JOB = 5
SHARDS = range((len(TARGET) + RX_PER_JOB - 1) // RX_PER_JOB)
ORIGINAL_AUG_COMMIT = '502629361a6aafa564a94947f3116e82f58c3692'
REPLICATION_COMMIT = 'abd0f4332aba04ac4cbca42cb5a3449193a9bbf2'


def cases():
    # Interleave routes to make progress on both when only a few cards are free.
    return {f'{route}-{i}': route for i in SHARDS for route in ROUTES}


def evaluation_seeds(ungated):
    return (SEED,) if ungated else SEEDS


def evaluation_modes(route, ungated):
    names = MODES[route]
    return (*names[:-1], WIDE_UNGATED) if ungated else names


def receipt(case, ungated=False):
    route = cases()[case]
    i = int(case.rsplit('-', 1)[1])
    proof = dict(status='PREDICTIONS_FROZEN', processing_route=route,
                 seeds=list(evaluation_seeds(ungated)), views=list(VIEWS),
                 modes=list(evaluation_modes(route, ungated)), per_receiver=LIMIT,
                 receivers=list(TARGET[i*RX_PER_JOB:(i+1)*RX_PER_JOB]), target_truth_read=False)
    if ungated:
        proof['restoration_input_policy'] = 'all_packets'
    return proof


def load_heads(rt, contract, device):
    from experiments.pamf_lora_main.tri_clean95 import NATIVE_WIDE_ROWS, clean_model

    project = Path(rt['project'])
    heads, epochs = [], {}
    for seed in evaluation_seeds(rt.get('ungated', False)):
        name = RUNS[seed]
        run = project / 'runs' / name
        archived = read(project / 'releases' / f'{name}-release' / 'runtime.json')
        expected = ORIGINAL_AUG_COMMIT if seed == SEED else REPLICATION_COMMIT
        if (archived['run_id'] != name or archived['source_root'] != rt['source_root']
                or archived['code_commit'] != expected or Path(archived['run_root']) != run):
            raise ValueError(f'Mismatched augmented W24 source release for {seed}')
        proof = read(run / 'F' / 'complete.json')
        epoch = proof['frozen']['epoch']
        if (proof['status'] != 'SOURCE_FROZEN' or proof['method'] != name or proof['contract'] != contract
                or proof['seed'] != seed or proof['w24_recipe_commit'] != W24_COMMIT
                or proof['epochs'] != EPOCHS or proof['steps'] != 9856 or proof['satellite_batches'] != 9856
                or proof['initialization'] != 'random_seed' or proof['clean_only']
                or proof['satellite_source_role'] != 'L_s' or proof['satellite_weight'] != AUG_WEIGHT
                or epoch not in ELIGIBLE or (seed == SEED and epoch != 104)
                or proof['target_reads'] or proof['U_truth_reads']):
            raise ValueError(f'Mismatched frozen augmented W24 checkpoint for {seed}')
        f = clean_model(dict(scratch=True), run, contract, NATIVE_WIDE_ROWS[METHOD], device)
        f.load_state_dict(torch.load(run / 'F' / 'best.pt', map_location=device, weights_only=True), strict=True)
        heads.append(f.eval().requires_grad_(False))
        epochs[str(seed)] = epoch
    return heads, epochs


def full_restore(baseline, device, packets):
    return recover(packets, baseline, device)[1:]


def residual_restore(baseline, device, packets):
    return (restore_r69(packets, *baseline, device)[0],)


def bindings(rt, routes, device):
    activate(rt)
    contract = source_contract(Path(rt['source_root']))
    heads, epochs = load_heads(rt, contract, device)
    archive, parent = training_runtime(rt)
    wide, tracker, threshold, frozen = load_restorer(archive, parent, contract, device)
    # Keep the original response and R69 loader imports after the archived G.
    readers = {}
    if FULL in routes:
        base = load_restorers(contract, Path(rt['run_root']).parent, device)
        readers[FULL] = partial(full_restore, base, device)
    if ROUTE in routes:
        base = r69_baseline(rt, contract, device)
        readers[ROUTE] = partial(residual_restore, base, device)
    gate_key = 'archived_source_threshold' if rt.get('ungated', False) else 'threshold'
    return heads, wide, tracker, threshold, readers, dict(epochs=epochs, **{gate_key: threshold},
             source_frozen_G=TRAIN_RUN, steps=frozen['steps'], parameters=frozen['parameters'])


def wide_delta(g, tracker, observed):
    y, phase, quality, rms = coordinates(tracker, observed[..., :384])
    stages, *_ = g.propose(y, quality, rms, complex_iq(observed[..., 384:]))
    return real_iq((stages[-1] - y) * phase * rms)


@torch.no_grad()
def batch_labels(heads, g, tracker, threshold, baseline, packets, device, *, ungated=False):
    original = received_aperture(packets, device)
    direct = np.stack([classify_aperture(f, original) for f in heads])
    variants, activity = [], []
    for restored in baseline(packets):
        changed = np.any(restored != packets, axis=1)
        values = direct.copy()
        if changed.any():
            aperture = received_aperture(restored[changed], device)
            values[:, changed] = np.stack([classify_aperture(f, aperture) for f in heads])
        variants.append(values)
        activity.append(int(changed.sum()))
    if ungated:
        wide = original.clone()
        wide[..., :384] += .5 * wide_delta(g, tracker, original)
        values = np.stack([classify_aperture(f, wide) for f in heads])
        count = len(original)
    else:
        wide, active = g.restore(tracker, original, threshold)
        changed = np.asarray(active.tolist(), dtype=bool)
        values = direct.copy()
        if changed.any():
            values[:, changed] = np.stack([classify_aperture(f, wide[active]) for f in heads])
        count = int(changed.sum())
    return np.stack((direct, *variants, values), axis=1), np.asarray([*activity, count], dtype=int)


@torch.no_grad()
def smoke(rt):
    device = 'cuda:0'
    ungated = rt.get('ungated', False)
    seeds = evaluation_seeds(ungated)
    heads, g, track, gate, baseline, proof = bindings(rt, ROUTES, device)
    source = Path(rt['source_root'])
    u = native(source, 'U_s', source_contract(source)['counts']['U_s'])[:4]
    clean = []
    for route, receiver in ((FULL, channel_job), (ROUTE, receive)):
        batch = partial(batch_labels, heads, g, track, gate, baseline[route], device=device, ungated=ungated)
        c, clean_activity = batch(u)
        y = np.stack([receiver(VIEWS[-1], SOURCE[0], pair) for pair in enumerate(u)])
        disturbed, active = batch(y)
        coverage = (clean_activity[-1] == active[-1] == len(u) if ungated
                    else clean_activity[-1] < len(u) and active[-1] > 0)
        if (disturbed.shape != (len(seeds), len(evaluation_modes(route, ungated)), len(u)) or not coverage):
            raise ValueError(f'Real frozen checkpoint source smoke failed for {route}')
        clean.append(c[:, 0])
    if not np.array_equal(*clean):
        raise ValueError('Shared clean received IQ must give identical direct outputs')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', target_reads=0,
          seeds=list(seeds), ungated=ungated, **proof))
    print('W24_FORMAL_SOURCE_SMOKE', proof['epochs'], flush=True)


def predict(rt, case):
    device = 'cuda:0'
    ungated = rt.get('ungated', False)
    root = Path(rt['run_root'])
    if read(root / 'smoke.json')['status'] != 'PASS':
        raise ValueError('Source-only W24 smoke required before target')
    expected = receipt(case, ungated)
    route = expected['processing_route']
    heads, g, track, gate, baseline, _ = bindings(rt, (route,), device)
    batch = partial(batch_labels, heads, g, track, gate, baseline[route], device=device, ungated=ungated)
    receiver = channel_job if route == FULL else receive
    data = Path(rt['native_target_root'])
    target_contract(data)
    out = root / 'predictions' / case
    out.mkdir()
    activity = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for rx in expected['receivers']:
            raw = native(data, rx, LIMIT)
            predictions = {}
            for scene in VIEWS:
                if scene == 'clean':
                    frames = raw
                else:
                    received = pool.map(partial(receiver, scene, rx), enumerate(raw), chunksize=16)
                    frames = np.stack(list(received))
                labels = []
                active = np.zeros(len(MODES[route]) - 1, dtype=int)
                for i in range(0, LIMIT, BATCH):
                    prediction, changed = batch(frames[i:i+BATCH])
                    labels.append(prediction)
                    active += changed
                predictions[scene] = np.concatenate(labels, axis=2)
                activity[f'{rx}/{scene}'] = active.tolist()
                print('W24_FORMAL_PREDICTED', case, rx, scene, active.tolist(), flush=True)
            np.savez(out / f'{rx}.npz', **predictions)
    write(out / 'activity.json', activity)
    write(out / 'complete.json', expected)


def dispatch(runtime):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    (root / 'predictions').mkdir()
    state = root / 'state.json'

    def status(**fields):
        write(state, fields, exclusive=False)

    try:
        if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Original frozen W24 model implementation required')
        runs = Path(rt['project']) / 'runs'
        seeds = evaluation_seeds(rt.get('ungated', False))
        status(status='WAITING_FOR_FROZEN_READERS', seeds=list(seeds))
        deadline = monotonic() + 7200
        while True:
            missing = [runs / RUNS[s] for s in seeds if not (runs / RUNS[s] / 'F' / 'complete.json').exists()]
            if not missing:
                break
            if monotonic() >= deadline or any(read(p / 'state.json')['status'] == 'FAILED' for p in missing):
                raise ValueError('Augmented W24 readers did not reach all source-frozen checkpoints')
            sleep(30)

        status(status='WAIT_GPU')
        deadline = monotonic() + 1800
        allowed = rt['allowed_gpus']
        while not (cards := available(allowed, max_compute_processes=rt['max_compute_processes'])):
            if monotonic() >= deadline:
                raise ValueError('GPU wait expired')
            sleep(20)
        script = str(Path(__file__).resolve())

        def start(_plan, phase, name, gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(reference))
            with (root / 'logs' / f'{name}.log').open('x', encoding='utf-8') as log:
                job = subprocess.Popen([rt['python'], '-u', script, phase, str(runtime), name],
                                       cwd=reference, env=env, stdout=log, stderr=subprocess.STDOUT)
            print('W24_FORMAL_STARTED', phase, name, job.pid, gpu, flush=True)
            return job

        gpu = next(i for i in allowed if i in cards)
        status(status='SOURCE_SMOKE')
        if start(rt, 'smoke', 'source-smoke', gpu).wait():
            raise ValueError('Formal real checkpoint source smoke failed')
        status(status='TARGET_PREDICTING')
        schedule(rt, [('predict', case) for case in cases()], start)
        for case in cases():
            if read(root / 'predictions' / case / 'complete.json') != receipt(case, rt.get('ungated', False)):
                raise ValueError(f'Incomplete frozen prediction shard {case}')
        status(status='PREDICTIONS_FROZEN', cases=list(cases()), target_truth_read=False)
    except Exception as exc:
        status(status='FAILED', error=str(exc))
        raise


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('dispatch').add_argument('runtime', type=Path)
    for phase in ('smoke', 'predict'):
        cmd = actions.add_parser(phase)
        cmd.add_argument('runtime', type=Path)
        cmd.add_argument('case')
    publish = actions.add_parser('publish')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        publish.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k: ctx[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project / 'runs', project / 'releases'
        data = releases / f'{DATA_RUN}-release' / 'data'
        run = RUN_UNGATED
        rt.update(run_id=run, code_commit=release_commit(args.repo), runner=__spec__.name, ungated=True,
                  release_root=str(releases / f'{run}-release'), run_root=str(runs / run),
                  source_root=str(data / 'source'), native_target_root=str(data / 'target'),
                  w24_release_root=str(releases / f'{W24_RUN}-release'),
                  training_release_root=str(releases / f'{TRAIN_RUN}-release'),
                  allowed_gpus=[0, 5, 3, 4, 2, 7, 1, 6], cpu_workers=8, max_workers=2, max_compute_processes=8)
        rt['external_releases'] = [rt[k] for k in ('source_root', 'native_target_root', 'w24_release_root',
                                                  'training_release_root')]
        rt['external_releases'] += [str(runs / x) for x in (RUNS[SEED], R69_TRAIN, R69_GATE)]
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'smoke':
        smoke(read(args.runtime))
    else:
        predict(read(args.runtime), args.case)


if __name__ == '__main__':
    main()
