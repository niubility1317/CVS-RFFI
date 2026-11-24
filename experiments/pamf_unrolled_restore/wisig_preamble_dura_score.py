"""Independent truth-last scorer for the registered paired WiSig Main split replay."""

import os
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.evaluation import handle
from experiments.cric_dura_comparison.io import read, write
from tools.wisig_paired_main_uncertainty import REPLICATES, summarize
from .wisig_preamble_dura_target import (
    BASELINE, COUNTS, F_RUN, MODES, ORIGIN, ROUTES, ROWS, SEEDS, TASKS, VIEWS, historical, view_root,
)


def read_decisions(root, baseline, route, split, ids):
    folder = root/route/split
    closure = read(folder/'complete.json')
    expected = dict(status='PREDICTIONS_FROZEN', route=route, split=split, source_run=ORIGIN,
                    count=len(ids), seeds=list(SEEDS), modes=list(MODES), views=list(VIEWS), truth_read=False)
    if any(closure.get(k) != value for k, value in expected.items()):
        raise ValueError('Expected the complete exact Main split prediction shard')
    keys = {f'{mode}__{view}' for mode in MODES for view in VIEWS}
    with np.load(folder/'decisions.npz', allow_pickle=False) as pack:
        if set(pack.files) != keys | {'ids'} or not np.array_equal(pack['ids'], ids):
            raise ValueError('Target decisions or physical order differ from the registered view')
        values = {mode: np.stack([pack[f'{mode}__{view}'] for view in VIEWS], axis=1) for mode in MODES}
    if any(matrix.shape != (len(SEEDS), len(VIEWS), len(ids)) or matrix.dtype != np.uint8 or matrix.max() > 5
           for matrix in values.values()):
        raise ValueError('Target class decisions do not match the registered matrix')
    for index, seed in enumerate(SEEDS):
        old_root = baseline/route/str(seed)/'predictions'
        old_meta = read(old_root/'complete.json')
        if (old_meta['run_id'] != BASELINE or old_meta['row_id'] != f'{route}-s{seed}'
                or old_meta['status'] != 'ALL_PREDICTIONS_COMPLETE' or old_meta['truth_read']
                or old_meta['query_fit'] or old_meta['readouts'] != list(MODES[:2])):
            raise ValueError('Baseline run or truth-free status changed')
        with np.load(old_root/split/'predictions.npz', allow_pickle=False) as old:
            if not np.array_equal(old['ids'], ids) or any(
                not np.array_equal(values[mode][index, ordinal], old[f'{mode}__{view}'])
                for mode in MODES[:2] for ordinal, view in enumerate(VIEWS)
            ):
                raise ValueError(f'Archived and replayed direct/R69 differ for {route} {split} seed {seed}')
    return values, closure['active']


def score_partition(decisions, y, rx, rng):
    hits = {mode: values == y for mode, values in decisions.items()}

    def paired(select):
        return {base: summarize(hits[base][select], hits[MODES[-1]][select], rx, VIEWS, rng,
                                 labels=(base, MODES[-1])) for base in MODES[:2]}

    return dict(physical_packets=len(y), combined_seeds=paired(slice(None)),
                by_seed={str(seed): paired(slice(index, index+1)) for index, seed in enumerate(SEEDS)})


def acceptance(summary):
    result = {}
    for route in ROUTES:
        views = summary[route]['combined']['combined_seeds']['R69']
        six = views['six_practical']['overall']['net']
        clean = views['clean']['overall']['net']
        result[route] = dict(six_scene_net=six, clean_net=clean, passed=six > 0 and clean >= 0)
    return dict(passed=all(row['passed'] for row in result.values()), by_route=result)


def score(rt):
    root = Path(rt['run_root'])
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or read(root/'state.json')['status'] != 'PREDICTIONS_FROZEN'):
        raise ValueError('Truth remains closed until all four blind shards freeze')
    baseline = Path(rt['baseline_root'])
    old_summary = read(baseline/'summary.json')
    if (read(baseline/'state.json')['status'] != 'ANALYZED' or old_summary['frozen_reader_run'] != F_RUN
            or old_summary['truth_last'] is not True):
        raise ValueError('Expected the previously verified R69 Main split reference')
    smoke = read(root/'smoke.json')
    if smoke['status'] != 'PASS' or smoke['seeds'] != list(SEEDS) or smoke['target_reads'] != 0:
        raise ValueError('Same reader real-checkpoint smoke is required')
    reader_root = Path(rt['recognizer_run'])
    blind = historical(reader_root, 'predict', SEEDS[0])
    physical, decisions, activity = {}, {}, {}
    for route, split in ROWS.values():
        manifest, ids, _ = handle(view_root(rt, blind, route, split), blind['classes'])
        if len(ids) != COUNTS[split] or manifest['channel'] != f'{route}/post_sync/noeq':
            raise ValueError('Target route manifest changed before independent scoring')
        if split in physical and not np.array_equal(ids, physical[split]):
            raise ValueError('Full and residual must use the same physical order')
        physical[split] = ids
        decisions[route, split], activity[route, split] = read_decisions(root, baseline, route, split, ids)
    if len(set(np.concatenate([physical[split] for split in TASKS]))) != sum(COUNTS.values()):
        raise ValueError('The 168,000-packet combined split must be a disjoint physical union')

    truth = historical(reader_root, 'score', SEEDS[0])
    if truth['classes'] != blind['classes']:
        raise ValueError('The truth-side class ordering differs from the frozen reader')
    labels, receivers = {}, {}
    for split in TASKS:
        metadata = read(truth['tests'][split]['truth'])
        rows = [metadata[str(key)] for key in physical[split]]
        labels[split] = np.asarray([row['label'] for row in rows], dtype=np.int64)
        receivers[split] = np.asarray([row['receiver'] for row in rows])
        if labels[split].min() < 0 or labels[split].max() >= len(blind['classes']):
            raise ValueError('Truth class ordering changed')
    rng = np.random.default_rng(2026112408)
    summary = {}
    for route in ROUTES:
        summary[route] = {split: score_partition(decisions[route, split], labels[split], receivers[split], rng)
                          for split in TASKS}
        merged = {mode: np.concatenate([decisions[route, split][mode] for split in TASKS], axis=2) for mode in MODES}
        summary[route]['combined'] = score_partition(merged, np.concatenate([labels[split] for split in TASKS]),
                                                     np.concatenate([receivers[split] for split in TASKS]), rng)
    write(root/'scores.json', dict(status='TARGET_SCORED', truth_last=True, source_feedback=False, modes=list(MODES),
          seeds=list(SEEDS), bootstrap_replicates=REPLICATES, combined_physical_count=sum(COUNTS.values()),
          activations={f'{route}-{split}': values for (route, split), values in activity.items()},
          summary=summary, formal_acceptance=acceptance(summary)))
