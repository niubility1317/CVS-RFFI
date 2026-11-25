"""Complete the WiSig forced-restoration matrix with actual clean recovery."""
import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import monotonic, sleep

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.runner import available


RUN = '20261125-phase1-wisig-dura-ungated-clean-m2-r01'
FORCED = '20261125-phase1-wisig-dura-force-main-m2-r01'
FORCED_COMMIT = '5b78d952419d5ac5a2538115c97489ff767a2423'
REPLAY = '20261125-phase1-wisig-r69-fresh-main-m3-r01'
SEEDS = (2026092701, 2026092702, 2026092703)
ROUTES = ('full', 'residual')
SPLITS = ('cross_rx', 'cross_both')
COUNTS = {'cross_rx': 126000, 'cross_both': 42000}
ARMS = ('trained_estimated', 'trained_blind')
METHODS = ('raw', 'R69', *ARMS)


def reference_runtime(rt):
    project = Path(rt['project'])
    reference = project / 'releases' / f'{FORCED}-release'
    if Path(rt['reference']) != reference:
        raise ValueError('Expected the registered forced recovery release')
    c = read(reference / 'runtime.json')
    if (c['run_id'], c['code_commit'], c['project'], Path(c['run_root'])) != (
            FORCED, FORCED_COMMIT, rt['project'], project / 'runs' / FORCED):
        raise ValueError('Unexpected forced recovery reference')
    return c


def classify_restored(restore, classify, model, iq):
    recovered, active = restore(model, None, iq, None, force=True)
    if recovered.shape != iq.shape or len(active) != len(iq) or not bool(active.all()):
        raise ValueError('Clean recovery did not process the entire batch')
    return classify(recovered)


def merge_clean(replay, clean):
    if not np.array_equal(replay['raw'][:, 0], clean['raw']):
        raise ValueError('New clean predictions must preserve raw parity')
    merged = dict(replay)
    for arm in ARMS:
        merged[arm] = replay[arm].copy()
        merged[arm][:, 0] = clean[arm]
    return merged


def binding(c):
    from experiments.pamf_repl_view_shuffle.runner import activate
    from experiments.pamf_unrolled_restore.wisig_dura_estimated import frozen_fit
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import readers

    activate(c)
    original, heads = readers(c)
    if tuple(heads) != SEEDS:
        raise ValueError('Expected all three frozen readers')
    models = {arm: frozen_fit(Path(c['frozen_root']), arm.removeprefix('trained_'))[0] for arm in ARMS}
    return original, heads, models


@torch.no_grad()
def smoke(rt):
    from experiments.pamf_unrolled_restore.wisig_dura_stress import classify
    from experiments.pamf_unrolled_restore.wisig_dura_stress_fit import restore
    from experiments.pamf_unrolled_restore.wisig_lora_source_fit import source_iq

    c = reference_runtime(rt)
    _, heads, models = binding(c)
    clean, _, calibration = source_iq(Path(c['source_root']))
    iq = tensor(clean[calibration[:12]], 'cuda:0')

    def readout(restored):
        if not torch.isfinite(restored).all():
            raise ValueError('Non-finite ungated clean IQ in source smoke')
        return classify(heads, restored)

    for model in models.values():
        classify_restored(restore, readout, model, iq)
    write(Path(rt['run_root']) / 'smoke.json',
          dict(status='PASS', clean_per_arm=len(iq), arms=list(ARMS), seeds=list(SEEDS), target_reads=0))
    print('WISIG_UNGATED_CLEAN_SOURCE_PASS', len(iq), flush=True)


def clean_receipt(split):
    return dict(status='PREDICTIONS_FROZEN', count=COUNTS[split], split=split, seeds=list(SEEDS),
                active={arm: COUNTS[split] for arm in ARMS}, same_clean_iq=True, truth_read=False)


@torch.no_grad()
def predict(rt, split):
    from functools import partial
    from experiments.cric_dura_comparison.evaluation import handle
    from experiments.pamf_unrolled_restore.wisig_dura_stress import classify
    from experiments.pamf_unrolled_restore.wisig_dura_stress_fit import restore
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import BATCH, view_root

    root = Path(rt['run_root'])
    if read(root / 'smoke.json')['status'] != 'PASS':
        raise ValueError('Registered source-only smoke required')
    c = reference_runtime(rt)
    originals, heads, models = binding(c)
    first = originals[SEEDS[0]]
    views, index = {}, None
    for route in ROUTES:
        path = view_root(c, first, route, split)
        if any(view_root(c, x, route, split) != path or x['classes'] != first['classes']
               for x in originals.values()):
            raise ValueError('Frozen readers disagree on Main split order')
        m, ids, positions = handle(path, first['classes'])
        if len(ids) != COUNTS[split] or m['channel'] != f'{route}/post_sync/noeq':
            raise ValueError('Unexpected Main split clean reference')
        if index is not None and not np.array_equal(index, ids):
            raise ValueError('Full and Residual clean IDs differ')
        index = ids
        data = np.load(m['clean_ref'], mmap_mode='r', allow_pickle=False)
        covered = len(data) == len(ids) if positions is None else 0 <= positions.min() <= positions.max() < len(data)
        if not covered or data.shape[1:] != (2, 256) or data.dtype != np.float32:
            raise ValueError('Unexpected clean samples or layout')
        views[route] = (data, positions)
    output = {k: np.empty((len(SEEDS), len(index)), dtype=np.uint8) for k in ('raw', *ARMS)}
    readout = partial(classify, heads)
    for start in range(0, len(index), BATCH):
        selection = slice(start, start+BATCH)
        full, raw = (data[selection if pos is None else pos[selection]] for data, pos in views.values())
        if not np.array_equal(full, raw):
            raise ValueError('Full and Residual clean received IQ differ')
        iq = tensor(full, 'cuda:0')
        output['raw'][:, selection] = readout(iq)
        for arm, model in models.items():
            output[arm][:, selection] = classify_restored(restore, readout, model, iq)
        if (start//BATCH) % 128 == 0:
            print('WISIG_UNGATED_CLEAN_PROGRESS', split, start+len(full), flush=True)
    folder = root / 'predictions' / split
    folder.mkdir()
    with (folder / 'decisions.npz').open('xb') as stream:
        np.savez(stream, ids=index, **output)
    write(folder / 'complete.json', clean_receipt(split))


def score(rt):
    from experiments.cric_dura_comparison.evaluation import handle
    from experiments.pamf_unrolled_restore.wisig_dura_stress_score import REPLICATES, cells, closed_predictions
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import VIEWS, historical, view_root

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    c = reference_runtime(rt)
    root = Path(rt['run_root'])
    if read(root / 'state.json')['status'] != 'PREDICTIONS_FROZEN':
        raise ValueError('Expected all clean and satellite predictions to be frozen')
    project = Path(rt['project'])
    forced = Path(c['run_root'])
    replay = project / 'runs' / REPLAY
    if (read(forced/'state.json')['status'], read(replay/'state.json')['status']) != ('ANALYZED', 'ANALYZED'):
        raise ValueError('Expected independently closed forced and fresh R69 reference runs')
    original = historical(Path(c['recognizer_run']), 'predict', SEEDS[0])
    matrix, physical = {}, {}
    for split in SPLITS:
        folder = root/'predictions'/split
        if read(folder/'complete.json') != clean_receipt(split):
            raise ValueError('Missing full coverage clean prediction')
        with np.load(folder/'decisions.npz', allow_pickle=False) as pack:
            if set(pack.files) != {'ids', 'raw', *ARMS}:
                raise ValueError('Unexpected clean prediction keys')
            physical[split] = pack['ids']
            clean = {k:pack[k] for k in ('raw', *ARMS)}
        if any(x.shape != (len(SEEDS), COUNTS[split]) or x.dtype != np.uint8 or x.max() > 5
               for x in clean.values()):
            raise ValueError('Unexpected clean prediction rows or classes')
        for route in ROUTES:
            manifest, ids, _ = handle(view_root(c, original, route, split), original['classes'])
            if (len(ids) != COUNTS[split] or not np.array_equal(physical[split], ids)
                    or manifest['channel'] != f'{route}/post_sync/noeq'):
                raise ValueError('Clean predictions must match the registered Main split order')
            previous, _ = closed_predictions(forced/'predictions'/f'default-{route}-{split}', ids,
                   'default', route, split, METHODS, force_practical=True)
            current, _ = closed_predictions(replay/'predictions'/f'{route}-{split}', ids,
                   'default', route, split, ('raw', 'R69'))
            if any(not np.array_equal(previous[m], current[m]) for m in ('raw','R69')):
                raise ValueError('Forced run baselines differ from the fresh R69 replay')
            matrix[route, split] = merge_clean(previous, clean)
    if np.intersect1d(*(physical[x] for x in SPLITS)).size:
        raise ValueError('WiSig Main split union must be disjoint')
    truth = historical(Path(c['recognizer_run']), 'score', SEEDS[0])
    if truth['classes'] != original['classes']:
        raise ValueError('Truth and reader classes differ')
    y, rx = {}, {}
    for split, ids in physical.items():
        truth_map = read(truth['tests'][split]['truth'])
        rows = [truth_map[str(ident)] for ident in ids]
        y[split] = np.asarray([row['label'] for row in rows], dtype=np.int64)
        rx[split] = np.asarray([row['receiver'] for row in rows])
        if not ((0 <= y[split]) & (y[split] < len(original['classes']))).all():
            raise ValueError('Truth class index outside registered order')
    rng = np.random.default_rng(20261125)
    results, acceptance = {}, {}
    for route in ROUTES:
        results[route] = {}
        for split in (*SPLITS, 'combined'):
            groups = SPLITS if split == 'combined' else (split,)
            combine = lambda values: np.concatenate([values[s] for s in groups])
            predictions = {m: np.concatenate([matrix[route,s][m] for s in groups], axis=2) for m in METHODS}
            results[route][split] = cells(predictions, combine(y), combine(rx), rng)
        rows = results[route]['combined']['absolute']
        acceptance[route] = {m: dict(
            six_over_r69=sum(rows[m][v]['correct'] for v in VIEWS[1:]) > sum(rows['R69'][v]['correct'] for v in VIEWS[1:]),
            clean_at_least_r69=rows[m]['clean']['correct'] >= rows['R69']['clean']['correct'],
            clean_held_by_seed=all(a['correct'] >= b['correct'] for a,b in
                                 zip(rows[m]['clean']['by_seed'], rows['R69']['clean']['by_seed'], strict=True)))
            for m in ARMS}
    write(root/'scores.json', dict(status='TARGET_SCORED', truth_last=True, seeds=list(SEEDS),
          parent_satellite=FORCED, matched_r69=REPLAY, input_policy='all_clean_and_practical',
          bootstrap_replicates=REPLICATES, results=results, acceptance=acceptance))


def dispatch(runtime):
    rt = read(runtime)
    c = reference_runtime(rt)
    root, reference = Path(rt['run_root']), Path(rt['reference'])
    root.mkdir()
    (root/'logs').mkdir()
    (root/'predictions').mkdir()
    state = root/'state.json'
    script = str(Path(__file__).resolve())

    def launch(_plan, phase, name, gpu=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   PYTHONPATH=str(reference), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        with (root/'logs'/f'{name}.log').open('x', encoding='utf-8') as log:
            job = subprocess.Popen([rt['python'], '-u', script, phase, str(runtime), name],
                                   cwd=reference, env=env, stdout=log, stderr=subprocess.STDOUT)
        print('WISIG_UNGATED_CLEAN_STARTED', phase, name, job.pid, gpu, flush=True)
        return job

    try:
        cmd = ['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free','--format=csv,noheader,nounits']
        devices = [tuple(map(int, row.split(','))) for row in subprocess.check_output(cmd, text=True).splitlines()]
        rt['allowed_gpus'] = [i for _,_,i in sorted((util, -free, i) for i,util,free in devices)]
        write(state, dict(status='WAIT_GPU'))
        deadline = monotonic()+1800
        while not (cards := available(rt['allowed_gpus'], rt['max_compute_processes'])):
            if monotonic() >= deadline:
                raise ValueError('WiSig clean smoke GPU wait expired')
            sleep(20)
        write(state, dict(status='SOURCE_SMOKE'), exclusive=False)
        gpu = next(i for i in rt['allowed_gpus'] if i in cards)
        if launch(rt, 'smoke', 'source-smoke', gpu).wait():
            raise ValueError('Ungated clean source smoke failed')
        write(state, dict(status='TARGET_PREDICTING'), exclusive=False)
        schedule(rt, [('predict', split) for split in SPLITS], launch)
        write(state, dict(status='WAITING_FOR_SATELLITE_PREDICTIONS'), exclusive=False)
        deadline = monotonic()+4*3600
        while (status := read(Path(c['run_root'])/'state.json')['status']) != 'ANALYZED':
            if status == 'FAILED' or monotonic() >= deadline:
                raise ValueError('Forced satellite reference did not complete')
            sleep(35)
        write(state, dict(status='PREDICTIONS_FROZEN'), exclusive=False)
        if launch(rt, 'score', 'independent-score').wait():
            raise ValueError('Independent ungated Main split scorer failed')
        write(state, dict(status='ANALYZED'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('dispatch').add_argument('runtime', type=Path)
    for phase in ('smoke', 'predict', 'score'):
        cmd = commands.add_parser(phase)
        cmd.add_argument('runtime', type=Path)
        cmd.add_argument('case')
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        reference = str(project/'releases'/f'{FORCED}-release')
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo), reference=reference,
                 release_root=str(project/'releases'/f'{RUN}-release'), run_root=str(project/'runs'/RUN),
                 external_releases=[reference, str(project/'runs'/FORCED), str(project/'runs'/REPLAY)],
                 max_workers=2, max_compute_processes=8)
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'predict':
        predict(read(args.runtime), args.case)
    else:
        {'smoke': smoke, 'score': score}[args.action](read(args.runtime))


if __name__ == '__main__':
    main()
