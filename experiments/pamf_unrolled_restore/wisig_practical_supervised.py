"""Release and independently score matched WiSig Full known-noise-supervised recovery."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_unrolled_restore

    pamf_unrolled_restore.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_unrolled_restore.wisig_forced_clean import COUNTS, FORCED, REPLAY, SEEDS, SPLITS, reference_runtime
from experiments.pamf_unrolled_restore.wisig_practical_supervised_fit import ARMS, fit, load_models, make_bank, validate_bank


RUN = '20261127-phase1-wisig-practical-known-noise-full-s2026112795-r03'
BANK_RUN = '20261127-phase1-wisig-practical-known-noise-full-s2026112795-r01'
FIT_RUN = '20261127-phase1-wisig-practical-known-noise-full-s2026112795-r02'
METHODS = ('raw', 'R69', *ARMS)


def bind_readers(c):
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    from experiments.pamf_repl_view_shuffle.runner import activate
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import F_RUN, historical, load_reader

    activate(c)
    numerical_policy()
    originals, heads = {}, {}
    for seed in SEEDS:
        cfg = historical(Path(c['recognizer_run']), 'predict', seed)
        head, evidence = load_reader(cfg['source_output'], read(cfg['source_contract']), 'cpu')
        if (evidence['run_id'], evidence['row_id'], evidence['model_seed']) != (F_RUN, cfg['row_id'], seed):
            raise ValueError('Frozen source head or WiSig seed changed')
        originals[seed], heads[seed] = cfg, head.eval().requires_grad_(False)
    return originals, heads


def restored_iq(model, iq):
    recovered = model(iq)
    if recovered.shape != iq.shape:
        raise ValueError('Unexpected frozen recoverer output shape')
    return recovered


def source_bank(rt):
    bank = Path(rt['source_bank'])
    if bank != Path(rt['project'])/'runs'/BANK_RUN/'source-bank':
        raise ValueError('Expected the registered r01 source pairing artifact')
    validate_bank(bank)
    return bank


def frozen_fit_root(rt):
    root = Path(rt['fit_root'])
    if root != Path(rt['project'])/'runs'/FIT_RUN:
        raise ValueError('Expected the registered r02 frozen models')
    return root


@torch.no_grad()
def smoke(rt):
    from experiments.pamf_unrolled_restore.wisig_dura_stress import classify

    root = Path(rt['run_root'])
    torch.set_num_threads(2)
    models = load_models(frozen_fit_root(rt), 'cpu')
    _, heads = bind_readers(reference_runtime(rt))
    with np.load(source_bank(rt)/'held.npz', allow_pickle=False) as src:
        panels = [tensor(src[name][:8], 'cpu') for name in ('clean', 'observed')]
    timings = {}
    for arm, model in models.items():
        for iq in panels:
            restored = restored_iq(model, iq)
            if not torch.isfinite(restored).all() or np.asarray(classify(heads, restored)).shape != (len(SEEDS), 8):
                raise ValueError('Failed frozen source-only reader integration')
        latency = []
        for _ in range(16):
            start = perf_counter()
            classify(heads, restored_iq(model, panels[1][:1]))
            latency.append((perf_counter()-start)*1000)
        timings[arm] = float(np.median(latency[4:]))
    write(root/'smoke.json', dict(status='PASS', arms=list(ARMS), seeds=list(SEEDS), source_only=True,
          target_reads=0, tx_reads=0, device='cpu', threads=2, batch=1,
          restoration_and_three_readers_median_ms=timings))
    print('WISIG_KNOWN_NOISE_FROZEN_SOURCE_SMOKE_PASS', timings, flush=True)


def split_data(c, original, split):
    from experiments.cric_dura_comparison.evaluation import handle
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import view_root

    path = view_root(c, original, 'full', split)
    manifest, ids, positions = handle(path, original['classes'])
    if len(ids) != COUNTS[split] or manifest['channel'] != 'full/post_sync/noeq' or not manifest['receiver_disjoint']:
        raise ValueError('Wrong official Full Main split')
    return manifest, ids, positions


def receipt(views, split):
    return dict(status='PREDICTIONS_FROZEN', profile='default', route='full', split=split,
                count=COUNTS[split], seeds=list(SEEDS), views=list(views), truth_read=False,
                active={view: {arm: COUNTS[split] for arm in ARMS} for view in views},
                inputs='received_iq_only')


@torch.no_grad()
def predict(rt, split):
    from experiments.pamf_unrolled_restore.wisig_dura_stress import classify
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import BATCH, VIEWS, view_root

    root = Path(rt['run_root'])
    if read(root/'smoke.json')['status'] != 'PASS':
        raise ValueError('Expected the source-only frozen IQ smoke')
    c = reference_runtime(rt)
    torch.set_num_threads(2)
    models = load_models(frozen_fit_root(rt), 'cpu')
    originals, heads = bind_readers(c)
    first = originals[SEEDS[0]]
    if any(view_root(c, z, 'full', split) != view_root(c, first, 'full', split)
           or z['classes'] != first['classes'] for z in originals.values()):
        raise ValueError('Frozen readers disagree on the registered input layout')
    manifest, ids, positions = split_data(c, first, split)
    decisions = {}
    for view in VIEWS:
        path = manifest['clean_ref'] if view == 'clean' else Path(manifest['observations_root'])/f'{view}.npy'
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        covers = len(data) == len(ids) if positions is None else 0 <= positions.min() <= positions.max() < len(data)
        if not covers or data.shape[1:] != (2, 256) or data.dtype != np.float32:
            raise ValueError('Unexpected registered Full IQ coverage or layout')
        arrays = {arm: np.empty((len(SEEDS), len(ids)), dtype=np.uint8) for arm in ARMS}
        for start in range(0, len(ids), BATCH):
            order = slice(start, start+BATCH)
            iq = tensor(data[order if positions is None else positions[order]], 'cpu')
            for arm, model in models.items():
                arrays[arm][:, order] = classify(heads, restored_iq(model, iq))
            if (start//BATCH) % 128 == 0:
                print('WISIG_KNOWN_NOISE_FULL_PROGRESS', split, view, min(start+BATCH, len(ids)), flush=True)
        decisions.update({f'{arm}__{view}': a for arm, a in arrays.items()})
        print('WISIG_KNOWN_NOISE_FULL_VIEW_CLOSED', split, view, len(ids), flush=True)
    folder = root/'predictions'/split
    folder.mkdir()
    with (folder/'decisions.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **decisions)
    write(folder/'complete.json', receipt(VIEWS, split))


def extra_metrics(predictions, labels, receivers, views):
    def metrics(value, take):
        per_view = (value[:, :, take] == labels[take]).sum(axis=(0, 2))
        count = int(take.sum()) * len(SEEDS)
        return dict(clean_correct=int(per_view[0]), clean_count=count,
                    practical_correct=int(per_view[1:].sum()), practical_count=count*(len(views)-1),
                    low_urban_correct=int(per_view[-1]), low_urban_count=count)

    result = {}
    for method, value in predictions.items():
        result[method] = dict(metrics(value, np.ones(len(labels), dtype=bool)),
                             receiver={str(r): metrics(value, receivers==r) for r in np.unique(receivers)},
                             tx={str(y): metrics(value, labels==y) for y in np.unique(labels)})
    return result


def score(rt):
    from experiments.pamf_unrolled_restore.wisig_dura_stress_score import REPLICATES, cells, closed_predictions
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import VIEWS, historical

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Truth-side scoring requires the independent CPU worker')
    root = Path(rt['run_root'])
    if read(root/'state.json')['status'] != 'PREDICTIONS_FROZEN':
        raise ValueError('All candidate Full predictions must close before scoring')
    c = reference_runtime(rt)
    replay = Path(rt['project'])/'runs'/REPLAY
    if read(replay/'state.json')['status'] != 'ANALYZED':
        raise ValueError('Expected the independently closed matched raw and R69 references')
    first = historical(Path(c['recognizer_run']), 'predict', SEEDS[0])
    matrix, physical = {}, {}
    for split in SPLITS:
        _, ids, _ = split_data(c, first, split)
        folder = root/'predictions'/split
        if read(folder/'complete.json') != receipt(VIEWS, split):
            raise ValueError('Incomplete official candidate shard')
        candidates, _ = closed_predictions(folder, ids, 'default', 'full', split, ARMS)
        old, _ = closed_predictions(replay/'predictions'/f'full-{split}', ids,
                                    'default', 'full', split, ('raw', 'R69'))
        matrix[split] = dict(old, **candidates)
        physical[split] = ids
    if np.intersect1d(*(physical[s] for s in SPLITS)).size:
        raise ValueError('Official Full physical partitions overlap')

    truth = historical(Path(c['recognizer_run']), 'score', SEEDS[0])
    if truth['classes'] != first['classes']:
        raise ValueError('Truth and frozen reader classes differ')
    labels, receivers = {}, {}
    for split, ids in physical.items():
        mapping = read(truth['tests'][split]['truth'])
        rows = [mapping[str(ident)] for ident in ids]
        labels[split] = np.asarray([row['label'] for row in rows], dtype=np.int64)
        receivers[split] = np.asarray([row['receiver'] for row in rows])
        if not ((0 <= labels[split]) & (labels[split] < len(first['classes']))).all():
            raise ValueError('Truth class outside the frozen class order')
    rng = np.random.default_rng(20261127)
    scores = {}
    for split in (*SPLITS, 'combined'):
        roles = SPLITS if split == 'combined' else (split,)
        pred = {m: np.concatenate([matrix[s][m] for s in roles], axis=-1) for m in METHODS}
        y, rx = (np.concatenate([group[s] for s in roles]) for group in (labels, receivers))
        scores[split] = dict(cells(pred, y, rx, rng), metrics=extra_metrics(pred, y, rx, VIEWS))
        print('WISIG_KNOWN_NOISE_FULL_SCORED', split, len(y), flush=True)
    write(root/'scores.json', dict(status='TARGET_SCORED', truth_last=True, route='full', profile='default',
          seeds=list(SEEDS), views=list(VIEWS), matched_r69=REPLAY, bootstrap_replicates=REPLICATES, results=scores))


def dispatch(runtime):
    rt = read(runtime)
    reference_runtime(rt)
    root = Path(rt['run_root'])
    root.mkdir()
    for name in ('logs', 'predictions'):
        (root/name).mkdir()
    state = root/'state.json'
    entry = str(Path(__file__).resolve())
    reference = Path(rt['reference'])

    def launch(phase, name):
        env = dict(os.environ, PYTHONPATH=str(reference), CUDA_VISIBLE_DEVICES='',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        with (root/'logs'/f'{phase}-{name}.log').open('x', encoding='utf-8') as log:
            job = subprocess.Popen([rt['python'], '-u', entry, phase, str(runtime), name], cwd=reference,
                                   env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print('WISIG_KNOWN_NOISE_STARTED', phase, name, job.pid, 'cpu', flush=True)
        return job

    try:
        write(state, dict(status='SOURCE_SMOKE', device='cpu'))
        if launch('smoke', 'frozen').wait():
            raise RuntimeError('Frozen WiSig reader CPU source smoke failed')
        write(state, dict(status='TARGET_PREDICTING', device='cpu'), exclusive=False)
        workers = {split: launch('predict', split) for split in SPLITS}
        exit_codes = {split: job.wait() for split, job in workers.items()}
        if any(exit_codes.values()):
            raise RuntimeError(f'WiSig Full CPU prediction failed: {exit_codes}')
        write(state, dict(status='PREDICTIONS_FROZEN'), exclusive=False)
        if launch('score', 'independent').wait():
            raise RuntimeError('Independent WiSig Full scorer exited unsuccessfully')
        write(state, dict(status='ANALYZED'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('dispatch').add_argument('runtime', type=Path)
    for phase in ('bank', 'fit', 'smoke', 'predict', 'score'):
        item = actions.add_parser(phase)
        item.add_argument('runtime', type=Path)
        item.add_argument('case')
    pub = actions.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        context = read(args.context)
        rt = {k: context[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        releases, runs = project/'releases', project/'runs'
        reference = releases/f'{FORCED}-release'
        bank = runs/BANK_RUN/'source-bank'
        fit_root = runs/FIT_RUN
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN),
                  reference=str(reference), source_bank=str(bank), fit_root=str(fit_root),
                  external_releases=list(map(str, (reference, bank, fit_root, runs/FORCED, runs/REPLAY))))
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    else:
        rt = read(args.runtime)
        if args.action == 'bank':
            sys.path.insert(0, str(Path(rt['teacher_release'])/'code'))
            make_bank(rt)
        elif args.action == 'fit':
            fit(Path(rt['run_root']), Path(rt['bank']), args.case)
        elif args.action == 'predict':
            predict(rt, args.case)
        else:
            {'smoke': smoke, 'score': score}[args.action](rt)


if __name__ == '__main__':
    main()
