"""Replay the original WiSig R69 and pair fresh decisions with fixed candidates."""
import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import monotonic, perf_counter, sleep

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_unrolled_restore

    pamf_unrolled_restore.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.runner import available


RUN = '20261125-phase1-wisig-r69-fresh-main-m3-r01'
SPECTRAL_MAIN = '20261125-phase1-wisig-spectral-precision-main-m3-r01'
REPLAY_COMMIT = '5690abc1c29cc7f28b0db893f56344d85820b5d1'
SPECTRAL_COMMIT = '69ece5f94b5a6d1490116eabae4e9a1371eb97d3'
PARENT = '20261125-phase1-wisig-dura-estimated-main-m2-r01'
PARENT_COMMIT = '400260cbc1f9562c79f9f2f044dac4e6aa47bafc'
SEEDS = (2026092701, 2026092702, 2026092703)
FRESH = ('raw', 'R69')
CANDIDATE = ('trained_estimated', 'trained_blind')
ROUTES = ('full', 'residual')
SPLITS = ('cross_rx', 'cross_both')
COUNTS = {'cross_rx': 126000, 'cross_both': 42000}
CASES = {f'{route}-{split}': (route, split) for route in ROUTES for split in SPLITS}


def parent_runtime(rt):
    if rt.get('spectral_main'):
        from experiments.pamf_unrolled_restore.wisig_forced_clean import reference_runtime

        return reference_runtime(dict(rt, reference=rt['parent_release']))
    parent = Path(rt['parent_release'])
    c = read(parent / 'runtime.json')
    project = Path(rt['project'])
    if (c['run_id'] != PARENT or c['code_commit'] != PARENT_COMMIT or c['project'] != rt['project']
            or Path(c['run_root']) != project / 'runs' / PARENT
            or parent != project / 'releases' / f'{PARENT}-release'):
        raise ValueError('Exact archived WiSig candidate plan required')
    return c


def qualified_spectral(rt):
    from experiments.pamf_unrolled_restore.wisig_spectral_precision_source import ARMS, MODES, VIEWS, RUN as SOURCE_RUN

    project = Path(rt['project'])
    root = project / 'runs' / SOURCE_RUN
    launch = read(project / 'releases' / f'{SOURCE_RUN}-release' / 'runtime.json')
    if (launch['run_id'] != SOURCE_RUN or launch['code_commit'] != SPECTRAL_COMMIT
            or Path(launch['run_root']) != root or launch['reference'] != rt['native_code']
            or read(root / 'state.json')['status'] != 'SOURCE_SCORED'):
        raise ValueError('The registered spectral-precision source run is still pending')
    frozen = read(root / 'predictions' / 'complete.json')
    score = read(root / 'source-scores.json')
    views = frozen['views']
    if (frozen['status'] != 'FROZEN' or frozen['count'] != 1800 or frozen['modes'] != list(MODES)
            or views != list(VIEWS) or frozen['target_reads'] or frozen['source_truth_reads']
            or set(frozen['summaries']) != {f'{route}_{v}' for route in ROUTES for v in views}
            or any(row['count'] != 1800 or row['active'] != dict.fromkeys(ARMS, 1800)
                   for row in frozen['summaries'].values())
            or (score['status'], score['truth_last'], score['target_reads'], score['reader_seed'])
            != ('SOURCE_SCORED', True, 0, SEEDS[0]) or score['modes'] != list(MODES) or score['views'] != views):
        raise ValueError('Incomplete frozen dual-route spectral source evidence')
    passed = dict.fromkeys(ARMS, True)
    for route in ROUTES:
        result = score['scores'][route]['all']
        correct = np.asarray(result['correct'])
        if (result['count'] != 1800 or correct.shape != (7, len(MODES)) or correct.dtype.kind not in 'iu'
                or not ((0 <= correct) & (correct <= 1800)).all()):
            raise ValueError(f'Invalid single-seed source correctness matrix for {route}')
        r69 = correct[:, MODES.index('r69')]
        for arm in ARMS:
            row = correct[:, MODES.index(f'{arm}_final')]
            passed[arm] &= bool(row[0] >= r69[0] and row[1:].sum() > r69[1:].sum())
    return dict(source_run=SOURCE_RUN, arms=[arm for arm in ARMS if passed[arm]])


def replay_root(rt):
    project = Path(rt['project'])
    root = project / 'runs' / RUN
    archived = read(project / 'releases' / f'{RUN}-release' / 'runtime.json')
    if (archived['run_id'] != RUN or archived['code_commit'] != REPLAY_COMMIT or Path(archived['run_root']) != root
            or read(root / 'state.json')['status'] != 'ANALYZED'):
        raise ValueError('Expected the independent frozen raw and R69 replay')
    return root / 'predictions'


def spectral_models(rt):
    from experiments import pamf_unrolled_restore
    from experiments.pamf_unrolled_restore.wisig_spectral_precision_source import models_from

    selection = read(Path(rt['run_root']) / 'source-qualification.json')
    path = str(Path(rt['native_code']) / 'experiments' / 'pamf_unrolled_restore')
    pamf_unrolled_restore.__path__.append(path)
    try:
        root = Path(rt['project']) / 'runs' / selection['source_run']
        models = dict(zip(selection['arms'], (model for model, _ in models_from({'run_root': root}, selection['arms'])),
                          strict=True))
    finally:
        pamf_unrolled_restore.__path__.remove(path)
    return models


def restore_spectral(model, iq):
    from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex, as_iq

    stages, _ = model.stages(model.analyze(as_complex(iq)))
    restored = as_iq(stages[-1])
    if restored.shape != iq.shape or not torch.isfinite(restored).all():
        raise ValueError('Spectral precision recovery must process every received window')
    return restored


def matching_raw(parent, fresh):
    if not np.array_equal(parent['raw'], fresh['raw']):
        raise ValueError('Rerun raw predictions differ from the frozen paired raw predictions')
    return int(np.count_nonzero(parent['R69'] != fresh['R69']))


@torch.no_grad()
def smoke(rt):
    from experiments.cric_dura_comparison.io import tensor
    from experiments.pamf_repl_view_shuffle.runner import activate
    from experiments.pamf_unrolled_restore.wisig_dura_stress import classify
    from experiments.pamf_unrolled_restore.wisig_dura_stress_data import read_plan
    from experiments.pamf_unrolled_restore.wisig_lora_source_fit import source_iq
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import readers, single_packet_median
    from experiments.pamf_unrolled_restore.wisig_short_training_source import original_r69

    c = parent_runtime(rt)
    activate(c)
    _, heads = readers(c)
    restore = original_r69(Path(c['parent_root']))
    source = Path(c['source_root'])
    clean, ids, cal = source_iq(source)
    folder, quantile = read_plan(source, 1)
    stress_ids = ids[quantile[ids] >= .575][:8]
    if len(stress_ids) != 8:
        raise ValueError('Registered Full source views required for smoke')
    raw = tensor(clean[cal[:8]], 'cuda:0')
    distorted = tensor(np.load(folder / 'full.npy', mmap_mode='r', allow_pickle=False)[stress_ids], 'cuda:0')
    models = spectral_models(rt) if rt.get('spectral_main') else {}
    for pack in (raw, distorted):
        classify(heads, pack)
        classify(heads, restore(pack)[0])
        for model in models.values():
            classify(heads, restore_spectral(model, pack))
    latency = single_packet_median(lambda: restore(distorted[:1]))
    candidate = {name: dict(parameters=sum(p.numel() for p in model.parameters()),
                            active_B1_ms=single_packet_median(lambda: restore_spectral(model, distorted[:1])))
                 for name, model in models.items()}
    proof = dict(status='PASS', source_rows=16, target_reads=0, source_classifier_seeds=list(SEEDS),
                 r69_parameters=266951, r69_active_B1_ms=latency, device=torch.cuda.get_device_name())
    if candidate:
        proof['candidates'] = candidate
    write(Path(rt['run_root']) / 'smoke.json', proof)
    print('WISIG_R69_RECHECK_SOURCE_PASS', latency, flush=True)


@torch.no_grad()
def predict(rt, case):
    from experiments.cric_dura_comparison.evaluation import handle
    from experiments.cric_dura_comparison.io import tensor
    from experiments.pamf_repl_view_shuffle.runner import activate
    from experiments.pamf_unrolled_restore.wisig_dura_stress import classify
    from experiments.pamf_unrolled_restore.wisig_dura_stress_score import closed_predictions
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import BATCH, VIEWS, readers, view_root
    from experiments.pamf_unrolled_restore.wisig_short_training_source import original_r69

    root = Path(rt['run_root'])
    if read(root / 'smoke.json')['status'] != 'PASS':
        raise ValueError('Source-only original R69 smoke required')
    c = parent_runtime(rt)
    activate(c)
    route, split = CASES[case]
    originals, heads = readers(c)
    first = originals[SEEDS[0]]
    path = view_root(c, first, route, split)
    if tuple(heads) != SEEDS or any(view_root(c, obj, route, split) != path
                                 or obj['classes'] != first['classes'] for obj in originals.values()):
        raise ValueError('Frozen Main split readers disagree on sample or class order')
    manifest, ids, positions = handle(path, first['classes'])
    if len(ids) != COUNTS[split] or manifest['channel'] != f'{route}/post_sync/noeq':
        raise ValueError('Wrong paired Main split observation manifest')
    spectral = rt.get('spectral_main', False)
    models = {}
    if spectral:
        models = spectral_models(rt)
        previous, _ = closed_predictions(replay_root(rt) / case, ids, 'default', route, split, FRESH)
    else:
        candidate = Path(c['run_root']) / 'predictions' / f'default-{case}'
        previous, _ = closed_predictions(candidate, ids, 'default', route, split, (*FRESH, *CANDIDATE))
    restore = original_r69(Path(c['parent_root']))
    output, activity = {}, {}
    started = perf_counter()
    for j, view in enumerate(VIEWS):
        file = Path(manifest['clean_ref']) if view == 'clean' else Path(manifest['observations_root']) / f'{view}.npy'
        data = np.load(file, mmap_mode='r', allow_pickle=False)
        matched = len(data) == len(ids) if positions is None else 0 <= positions.min() <= positions.max() < len(data)
        if not matched or data.shape[1:] != (2, 256) or data.dtype != np.float32:
            raise ValueError('Main split received IQ shape/order differs')
        values = {m: np.empty((len(SEEDS), len(ids)), dtype=np.uint8) for m in (*FRESH, *models)}
        active = dict.fromkeys(models, 0)
        for start in range(0, len(ids), BATCH):
            ix = slice(start, min(start + BATCH, len(ids)))
            iq = tensor(data[ix if positions is None else positions[ix]], 'cuda:0')
            raw = np.asarray(classify(heads, iq), dtype=np.uint8)
            if not np.array_equal(raw, previous['raw'][:, j, ix]):
                raise ValueError(f'Raw replay parity differs for {case}/{view} at row {start}')
            values['raw'][:, ix] = raw
            values['R69'][:, ix] = classify(heads, restore(iq)[0])
            for name, model in models.items():
                values[name][:, ix] = classify(heads, restore_spectral(model, iq))
                active[name] += len(iq)
        if spectral:
            activity[view] = active
        output.update({f'{m}__{view}': array for m, array in values.items()})
        print('WISIG_R69_RECHECK_VIEW', route, split, view, len(ids), flush=True)
    folder = root / 'predictions' / case
    folder.mkdir()
    with (folder / 'decisions.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **output)
    write(folder / 'complete.json', dict(status='PREDICTIONS_FROZEN', profile='default', route=route, split=split,
          count=len(ids), seeds=list(SEEDS), views=list(VIEWS), truth_read=False,
          active=activity, seconds=perf_counter()-started))


def require_spectral_coverage(activity, views, arms, count):
    if set(activity) != set(views) or any(row != dict.fromkeys(arms, count) for row in activity.values()):
        raise ValueError('All clean and practical packets must actually run each selected recovery model')


def score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    from experiments.cric_dura_comparison.evaluation import handle
    from experiments.pamf_unrolled_restore.wisig_dura_stress_score import REPLICATES, cells, closed_predictions
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import VIEWS, historical, view_root

    c = parent_runtime(rt)
    root, parent = Path(rt['run_root']), Path(c['run_root'])
    spectral = rt.get('spectral_main', False)
    selection = read(root / 'source-qualification.json') if spectral else None
    arms = tuple(selection['arms']) if spectral else CANDIDATE
    if not arms:
        raise ValueError('No registered recovery arm qualified for the Main split')
    frozen = read(root / 'state.json')
    if (frozen['status'] != 'PREDICTIONS_FROZEN' or frozen['cases'] != list(CASES)
            or frozen['truth_read']):
        raise ValueError('All rerun predictions must be frozen before scoring')
    original = historical(Path(c['recognizer_run']), 'predict', SEEDS[0])
    ids_by_split, matrix, drift, activity = {}, {}, {}, {}
    for case, (route, split) in CASES.items():
        manifest, ids, _ = handle(view_root(c, original, route, split), original['classes'])
        if len(ids) != COUNTS[split] or manifest['channel'] != f'{route}/post_sync/noeq':
            raise ValueError('Mismatched registered Main split')
        if split in ids_by_split and not np.array_equal(ids_by_split[split], ids):
            raise ValueError('Main physical order differs across paired routes')
        ids_by_split[split] = ids
        fresh, live_activity = closed_predictions(root / 'predictions' / case, ids, 'default', route, split,
                                                    (*FRESH, *arms) if spectral else FRESH)
        if spectral:
            previous, _ = closed_predictions(replay_root(rt) / case, ids, 'default', route, split, FRESH)
            require_spectral_coverage(live_activity, VIEWS, arms, len(ids))
            matrix[route, split] = fresh
            activity[case] = live_activity
        else:
            previous, activities = closed_predictions(parent / 'predictions' / f'default-{case}', ids, 'default',
                                                       route, split, (*FRESH, *arms))
            matrix[route, split] = {**fresh, **{name: previous[name] for name in arms}}
            activity[case] = activities
        drift[case] = matching_raw(previous, fresh)
    if np.intersect1d(*(ids_by_split[s] for s in SPLITS)).size:
        raise ValueError('Main split union must have 168000 disjoint physical packets')
    for split in SPLITS:
        for name in ('raw', *arms) if spectral else ('raw',):
            if not np.array_equal(matrix['full', split][name][:, 0], matrix['residual', split][name][:, 0]):
                raise ValueError(f'Both routes must retain the same clean decisions for {name}')

    truth = historical(Path(c['recognizer_run']), 'score', SEEDS[0])
    if truth['classes'] != original['classes']:
        raise ValueError('Unexpected frozen-recognizer class order in truth')
    labels, receivers = {}, {}
    for split in SPLITS:
        metadata = read(truth['tests'][split]['truth'])
        rows = [metadata[str(ident)] for ident in ids_by_split[split]]
        labels[split] = np.asarray([r['label'] for r in rows], dtype=np.int64)
        receivers[split] = np.asarray([r['receiver'] for r in rows])
        if not ((0 <= labels[split]) & (labels[split] < len(original['classes']))).all():
            raise ValueError('Truth class order differs from frozen readers')
    rng = np.random.default_rng(20261125)
    result, checks = {}, {}
    for route in ROUTES:
        result[route] = {}
        for part in (*SPLITS, 'combined'):
            parts = SPLITS if part == 'combined' else (part,)
            predictions = {m: np.concatenate([matrix[route, s][m] for s in parts], axis=2)
                           for m in (*FRESH, *arms)}
            y = np.concatenate([labels[s] for s in parts])
            rx = np.concatenate([receivers[s] for s in parts])
            result[route][part] = cells(predictions, y, rx, rng)
        all_rows = result[route]['combined']['absolute']
        r69 = all_rows['R69']
        checks[route] = {}
        for name in arms:
            row = all_rows[name]
            checks[route][name] = dict(
                six_over_r69=sum(row[v]['correct'] for v in VIEWS[1:]) > sum(r69[v]['correct'] for v in VIEWS[1:]),
                clean_held_by_seed=all(a['correct'] >= b['correct'] for a,b in
                                       zip(row['clean']['by_seed'], r69['clean']['by_seed'], strict=True)))
    write(root / 'scores.json', dict(status='TARGET_SCORED', truth_last=True,
          candidate_parent=selection['source_run'] if spectral else PARENT, seeds=list(SEEDS),
          bootstrap_replicates=REPLICATES, raw_parity='exact', r69_replay_drift=drift,
          results=result, acceptance=checks, activation=activity))


def dispatch(runtime):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['parent_release'])
    parent_runtime(rt)
    root.mkdir()
    (root / 'logs').mkdir()
    (root / 'predictions').mkdir()
    state = root / 'state.json'
    try:
        if rt.get('spectral_main'):
            selection = qualified_spectral(rt)
            write(root / 'source-qualification.json', selection)
            if not selection['arms']:
                write(state, dict(status='SOURCE_REJECTED', target_reads=0))
                return
            replay_root(rt)
        fields = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free', '--format=csv,noheader,nounits']
        inventory = (tuple(map(int, r.split(','))) for r in subprocess.check_output(fields, text=True).splitlines())
        rt['allowed_gpus'] = [idx for _, _, idx in sorted((load, -free, idx) for idx, load, free in inventory)]
        write(state, dict(status='WAIT_GPU'))
        deadline = monotonic()+1800
        while not (cards := available(rt['allowed_gpus'], rt['max_compute_processes'])):
            if monotonic() >= deadline:
                raise ValueError('WiSig R69 source smoke GPU wait expired')
            sleep(20)
        script = str(Path(__file__).resolve())

        def start(_plan, phase, name, gpu=None):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(reference))
            with (root / 'logs' / f'{name}.log').open('x', encoding='utf-8') as log:
                p = subprocess.Popen([rt['python'], '-u', script, phase, str(runtime), name],
                                     cwd=reference, env=env, stdout=log, stderr=subprocess.STDOUT)
            print('WISIG_R69_RECHECK_STARTED', phase, name, p.pid, gpu, flush=True)
            return p

        gpu = next(i for i in rt['allowed_gpus'] if i in cards)
        write(state, dict(status='SOURCE_SMOKE'), exclusive=False)
        if start(rt, 'smoke', 'source-smoke', gpu).wait():
            raise ValueError('Actual original R69 no-query smoke failed')
        write(state, dict(status='TARGET_PREDICTING'), exclusive=False)
        schedule(rt, [('predict', name) for name in CASES], start)
        for name in CASES:
            if read(root/'predictions'/name/'complete.json')['status'] != 'PREDICTIONS_FROZEN':
                raise ValueError(f'Incomplete R69 shard {name}')
        write(state, dict(status='PREDICTIONS_FROZEN', cases=list(CASES), truth_read=False), exclusive=False)
        if start(rt, 'score', 'independent-score').wait():
            raise ValueError('Independent paired Main split scorer failed')
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
    pub.add_argument('--spectral-main', action='store_true')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        c = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(c['project'])
        parent_name, name = PARENT, RUN
        if args.spectral_main:
            from experiments.pamf_unrolled_restore.wisig_forced_clean import FORCED
            from experiments.pamf_unrolled_restore.wisig_spectral_precision_source import REFERENCE, RUN as SOURCE_RUN

            parent_name, name = FORCED, SPECTRAL_MAIN
            c['native_code'] = str(project / 'releases' / f'{REFERENCE}-release')
            dependencies = [c['native_code'], str(project / 'runs' / SOURCE_RUN), str(project / 'runs' / RUN)]
        else:
            dependencies = []
        parent = str(project / 'releases' / f'{parent_name}-release')
        c.update(run_id=name, runner=__spec__.name, code_commit=release_commit(args.repo),
                 release_root=str(project/'releases'/f'{name}-release'), run_root=str(project/'runs'/name),
                 spectral_main=args.spectral_main, parent_release=parent,
                 external_releases=[parent, str(project/'runs'/parent_name), *dependencies],
                 max_workers=2, max_compute_processes=8)
        release(args.repo, args.output, c, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'predict':
        predict(read(args.runtime), args.case)
    else:
        {'smoke': smoke, 'score': score}[args.action](read(args.runtime))


if __name__ == '__main__':
    main()
