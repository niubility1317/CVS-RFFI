"""Frozen baseline predictions, separated from independent truth-side scoring."""
from pathlib import Path
import os
import time

import numpy as np

from experiments.cric_dura_comparison.evaluation import reject_truth
from experiments.cric_dura_comparison.io import read, tensor, write

RUN = '20261110-evaluation-trisig-dura-f0-s2026092702-03-m2-r01'
ORIGINAL_RUN = '20261027-phase1-pamf-repl-shuffled-views-m16-r02'
SEEDS = (2026092702, 2026092703)
TASKS = ('cross_rx', 'cross_both')
COUNTS = {'cross_rx': 126000, 'cross_both': 42000}
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
DURA_CHECKPOINT = 'runs/20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt'


def plan():
    return dict(reuse_seed=2026092701, new_seeds=list(SEEDS), views=list(VIEWS),
                tasks=list(TASKS), restorer=DURA_CHECKPOINT)


def predict_config(project, seed):
    if seed not in SEEDS:
        raise ValueError('Only the two missing matched baseline seeds')
    project = str(project).rstrip('/')
    row = f'main_split-repl-shuffled_time-3view-s{seed}'
    parent = f'{project}/runs/{ORIGINAL_RUN}'
    return dict(run_id=RUN, row_id=row, model_seed=seed, source_run=ORIGINAL_RUN,
                source_output=f'{parent}/F/{row}/source',
                original_predict_config=f'{parent}/configs/predict__{row}.json',
                raw_predictions=f'{parent}/readouts/{row}/predictions',
                cache=f'{project}/runs/20261023-phase1-practical-residual-control-r03/restored_IQ/F0-DURA-LU',
                restorer=DURA_CHECKPOINT, views=list(VIEWS), tasks=list(TASKS),
                output_root=f'{project}/runs/{RUN}/predictions/{seed}')


def validate_prediction_config(c):
    reject_truth(c)
    seed = c.get('model_seed')
    if seed not in SEEDS or c.get('source_run') != ORIGINAL_RUN:
        raise ValueError('Original three-view frozen source required')
    row = f'main_split-repl-shuffled_time-3view-s{seed}'
    if (c.get('run_id') != RUN or c.get('row_id') != row or
            not c.get('source_output', '').endswith('/' + ORIGINAL_RUN + '/F/' + row + '/source') or
            c.get('views') != list(VIEWS) or c.get('tasks') != list(TASKS) or
            c.get('restorer') != DURA_CHECKPOINT):
        raise ValueError('Matched seeds, seven scenes, two populations and fixed DURA required')


def check_ids(expected, raw, restored):
    if (expected.ndim != 1 or not len(expected) or
            not np.array_equal(expected, raw) or not np.array_equal(expected, restored)):
        raise ValueError('Raw and DURA physical IDs must match exactly')
    return len(expected)


def check_prediction_barrier(root):
    root = Path(root)
    for seed in SEEDS:
        try:
            d = read(root / str(seed) / 'complete.json')
        except FileNotFoundError as exc:
            raise ValueError('Both seed predictions required before truth') from exc
        if (d.get('status') != 'ALL_PREDICTIONS_COMPLETE' or d.get('model_seed') != seed or
                d.get('truth_read') is not False or d.get('views') != list(VIEWS) or
                d.get('tests') != list(TASKS) or d.get('readouts') != ['raw', 'DURA_v1']):
            raise ValueError('Both seed predictions required before truth')


def load_frozen(c, device='cuda:0'):
    validate_prediction_config(c)
    import torch
    from experiments.pamf_repl_view_shuffle.runner import activate
    original = read(c['original_predict_config'])
    reject_truth(original)
    if (original['run_id'], original['row_id'], original['model_seed']) != (ORIGINAL_RUN, c['row_id'], c['model_seed']):
        raise ValueError('Original predictor metadata differs')
    c['frozen_sources'] = original['frozen_sources']
    activate(c)
    from experiments.pamf_repl_view_shuffle.source import load_reader
    # Original loading checks own source ancestry, source contract and architecture.
    model, source = load_reader(c['source_output'], read(original['source_contract']), device)
    if source['model_seed'] != c['model_seed']:
        raise ValueError('Wrong original frozen seed')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_iq = np.load(Path(source['source_cache']) / 'L_s_x.npy', mmap_mode='r', allow_pickle=False)[:2]
    with torch.inference_mode():
        logits = model.classify_features(model.features(tensor(source_iq, device)))
    if logits.shape != (2, 6) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Real source-only frozen-checkpoint smoke failed')
    return model, original


def smoke(project):
    model, _ = load_frozen(predict_config(project, SEEDS[0]))
    del model


def predict(c, device='cuda:0'):
    import torch
    from experiments.cric_dura_comparison.evaluation import handle
    model, original = load_frozen(c, device)
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    cache = Path(c['cache'])
    header = read(cache / 'complete.json')
    if (header.get('status') != 'DURA_IQ_CACHE_COMPLETE' or header.get('truth_read') is not False or
            header.get('protocol_id') != 'F0' or header.get('readout') != 'DURA-LU' or
            header.get('F_reads') != 0 or header.get('model_updates') != 0 or
            header.get('views') != list(VIEWS) or
            not header.get('G', {}).get('checkpoint', '').endswith('/' + DURA_CHECKPOINT)):
        raise ValueError('Fixed DURA v1 source cache required')
    if read(header['G']['source_contract']) != read(original['source_contract']):
        raise ValueError('DURA and original F source contracts differ')
    raw_root = Path(c['raw_predictions'])
    raw_done = read(raw_root / 'complete.json')
    if (raw_done.get('status') != 'ALL_PREDICTIONS_COMPLETE' or raw_done.get('truth_read') is not False or
            raw_done.get('run_id') != ORIGINAL_RUN or raw_done.get('row_id') != c['row_id']):
        raise ValueError('Original frozen raw predictions required')
    start = time.monotonic()
    for task in TASKS:
        _, ids, _ = handle(original['tests'][task]['views_root'], original['classes'])
        if len(ids) != COUNTS[task]:
            raise ValueError('Full original population required')
        with np.load(raw_root / task / 'predictions.npz', allow_pickle=False) as raw, \
                np.load(cache / task / 'index.npz', allow_pickle=False) as restored_index:
            check_ids(ids, raw['ids'], restored_index['ids'])
            values = {'raw__' + view: raw['raw__' + view].copy() for view in VIEWS}
        for view in VIEWS:
            arr = np.load(cache / task / (view + '.npy'), mmap_mode='r', allow_pickle=False)
            if arr.shape != (len(ids), 2, 256) or arr.dtype != np.float32:
                raise ValueError('DURA cache shape or precision differs')
            result = []
            with torch.inference_mode():
                for begin in range(0, len(ids), 128):
                    x = tensor(np.ascontiguousarray(arr[begin:begin+128]), device)
                    logits = model.classify_features(model.features(x))
                    if logits.shape != (len(x), 6) or not bool(torch.isfinite(logits).all()):
                        raise ValueError('Invalid frozen original default head')
                    result.extend(logits.argmax(1).cpu().tolist())
            values['DURA_v1__' + view] = np.asarray(result, dtype=np.int64)
            print('BASELINE_FROZEN_PREDICTION', c['model_seed'], task, view, len(ids), flush=True)
        target = out / task; target.mkdir()
        np.savez(target / 'predictions.npz', ids=ids, **values)
    write(out / 'complete.json', dict(status='ALL_PREDICTIONS_COMPLETE', run_id=RUN, row_id=c['row_id'],
          model_seed=c['model_seed'], tests=list(TASKS), views=list(VIEWS), readouts=['raw', 'DURA_v1'],
          truth_read=False, query_fit=False, seconds=time.monotonic()-start))


def score(project):
    """Run a full two-seed preflight before either independent scorer reads truth."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent truth-side scorer must be CPU-only')
    from experiments.cric_dura_comparison import evaluation as old
    project = Path(project)
    root = project / 'runs' / RUN
    check_prediction_barrier(root / 'predictions')
    configs = []
    for seed in SEEDS:
        row = f'main_split-repl-shuffled_time-3view-s{seed}'
        c = read(project / 'runs' / ORIGINAL_RUN / 'configs' / ('score__' + row + '.json'))
        if (c['run_id'], c['row_id'], c['model_seed']) != (ORIGINAL_RUN, row, seed):
            raise ValueError('Original score contract differs')
        c.update(run_id=RUN, prediction_output=str(root / 'predictions' / str(seed)),
                 output_root=str(root / 'scores' / str(seed)), readouts=['raw', 'DURA_v1'])
        configs.append(c)
    original_tasks = old.tasks
    old.tasks = lambda _: list(TASKS)
    try:
        for c in configs:
            old.preflight(c)
        for c in configs:
            old.score(c)
    finally:
        old.tasks = original_tasks
