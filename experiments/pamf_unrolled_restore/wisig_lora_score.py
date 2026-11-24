"""Independent scorer: all candidate prediction shards must close first."""

import json
import os
from pathlib import Path

import numpy as np


def read_candidate(root, arm, mode, route, split, ids, seeds, views):
    folder = Path(root) / mode / arm / route / split
    closure = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
    required = dict(status='PREDICTIONS_FROZEN', arm=arm, mode=mode, route=route, split=split,
                    count=len(ids), seeds=list(seeds), views=list(views), truth_read=False)
    if any(closure.get(key) != value for key, value in required.items()):
        raise ValueError('Candidate predictions are incomplete or do not match the selected split')
    with np.load(folder / 'decisions.npz', allow_pickle=False) as data:
        if set(data.files) != {'ids', *views} or not np.array_equal(data['ids'], ids):
            raise ValueError('Candidate physical IDs or views do not match')
        decisions = np.stack([data[view] for view in views], axis=1)
    if decisions.shape != (len(seeds), len(views), len(ids)) or decisions.dtype != np.uint8 or decisions.max() > 5:
        raise ValueError('Invalid WiSig class decisions')
    return decisions, closure['active']


def score(rt):
    from experiments.cric_dura_comparison.evaluation import handle
    from experiments.pamf_unrolled_restore.wisig_preamble_dura_target import (
        COUNTS, ROWS, SEEDS, TASKS, VIEWS, historical, view_root,
    )
    from tools.wisig_paired_main_uncertainty import REPLICATES, summarize
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Use the independent CPU scorer')
    root = Path(rt['run_root'])
    reader_root = Path(rt['recognizer_run'])
    blind = historical(reader_root, 'predict', SEEDS[0])
    baseline_root = Path(rt['baseline_replay'])
    physical, predictions, activities = {}, {}, {}
    methods = ('frozen_expanded', 'frozen_guard', 'retrained_expanded', 'retrained_guard')
    for route, split in ROWS.values():
        manifest, ids, _ = handle(view_root(rt, blind, route, split), blind['classes'])
        if len(ids) != COUNTS[split] or manifest['channel'] != f'{route}/post_sync/noeq':
            raise ValueError('Incorrect target physical data')
        if split in physical and not np.array_equal(physical[split], ids):
            raise ValueError('Routes do not have the same physical order')
        physical[split] = ids
        baseline = baseline_root / route / split
        closure = json.loads((baseline / 'complete.json').read_text(encoding='utf-8'))
        expected = dict(status='PREDICTIONS_FROZEN', route=route, split=split, count=len(ids),
                        seeds=list(SEEDS), views=list(VIEWS), truth_read=False)
        if any(closure.get(key) != value for key, value in expected.items()):
            raise ValueError('Expected the frozen raw/R69 reference')
        with np.load(baseline / 'decisions.npz', allow_pickle=False) as old:
            if not np.array_equal(old['ids'], ids):
                raise ValueError('Baseline packet order differs')
            values = {name: np.stack([old[f'{name}__{view}'] for view in VIEWS], axis=1)
                      for name in ('raw', 'R69')}
        if any(value.shape != (len(SEEDS), len(VIEWS), len(ids)) or value.dtype != np.uint8
               or value.max() > 5 for value in values.values()):
            raise ValueError('Baseline decision matrix differs from the frozen reader')
        for mode in ('frozen', 'retrained'):
            for arm in ('expanded', 'guard'):
                name = f'{mode}_{arm}'
                values[name], activities[f'{name}/{route}/{split}'] = read_candidate(
                    root, arm, mode, route, split, ids, SEEDS, VIEWS)
        predictions[route, split] = values
    if len(set(np.concatenate([physical[split] for split in TASKS]))) != sum(COUNTS.values()):
        raise ValueError('Target splits must be physically disjoint')

    # All four candidates across all four shards are frozen before truth opens.
    truth = historical(reader_root, 'score', SEEDS[0])
    if truth['classes'] != blind['classes']:
        raise ValueError('Truth class order differs from the frozen reader')
    labels, receivers = {}, {}
    for split in TASKS:
        metadata = json.loads(Path(truth['tests'][split]['truth']).read_text(encoding='utf-8'))
        rows = [metadata[str(key)] for key in physical[split]]
        labels[split] = np.asarray([row['label'] for row in rows], dtype=np.int64)
        receivers[split] = np.asarray([row['receiver'] for row in rows])
    rng = np.random.default_rng(2026112408)
    result = {}
    for route in ('full', 'residual'):
        result[route] = {}
        for partition in (*TASKS, 'combined'):
            parts = TASKS if partition == 'combined' else (partition,)
            y = np.concatenate([labels[split] for split in parts])
            rx = np.concatenate([receivers[split] for split in parts])
            choices = {name: np.concatenate([predictions[route, split][name] for split in parts], axis=2)
                       for name in ('raw', 'R69', *methods)}
            correct = {name: value == y for name, value in choices.items()}
            result[route][partition] = {name: summarize(correct['R69'], correct[name], rx, VIEWS, rng,
                                                        labels=('R69', name)) for name in methods}
    with (root / 'scores.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(status='TARGET_SCORED', truth_last=True, simulator_parameters_read=False,
                       bootstrap_replicates=REPLICATES, results=result, activation=activities), stream)
