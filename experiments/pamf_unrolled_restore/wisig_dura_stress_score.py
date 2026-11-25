"""Open WiSig truth only after both profiles and both routes are frozen."""

import os
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.evaluation import handle
from experiments.cric_dura_comparison.io import read, write
from tools.wisig_paired_main_uncertainty import REPLICATES, summarize
from .wisig_dura_stress import CASES, COUNTS, METHODS, SEEDS, TASKS, VIEWS, view_root
from .wisig_dura_stress_data import PROFILES, ROUTES
from .wisig_preamble_dura_target import historical


def closed_predictions(root, ids, profile, route, split, methods=METHODS, *, force_practical=False,
                       clean_passthrough=True):
    marker = read(root / 'complete.json')
    expected = dict(status='PREDICTIONS_FROZEN', profile=profile, route=route, split=split,
                    count=len(ids), seeds=list(SEEDS), views=list(VIEWS), truth_read=False)
    if any(marker[key] != value for key, value in expected.items()):
        raise ValueError('The full blind prediction shard is required')
    if force_practical:
        trained = set(methods)-{'raw','R69'}
        active = marker['active']
        if (marker.get('force_practical') is not True or marker.get('clean_passthrough', True) != clean_passthrough
                or set(active)!=set(VIEWS)) or any(
                set(active[view])!=trained or any(n!=(0 if view=='clean' and clean_passthrough else len(ids)) for n in active[view].values())
                for view in VIEWS):
            raise ValueError('The forced test does not cover every packet in the required views')
    with np.load(root / 'decisions.npz', allow_pickle=False) as contents:
        expected_keys = {'ids', *(f'{name}__{view}' for name in methods for view in VIEWS)}
        if set(contents.files) != expected_keys or not np.array_equal(ids, contents['ids']):
            raise ValueError('Prediction views, modes or physical IDs differ')
        decisions = {name: np.stack([contents[f'{name}__{view}'] for view in VIEWS], axis=1)
                     for name in methods}
    if any(value.shape != (len(SEEDS), len(VIEWS), len(ids)) or value.dtype != np.uint8 or value.max() > 5
           for value in decisions.values()):
        raise ValueError('Predicted split, seeds or class codes differ')
    if force_practical and clean_passthrough and any(not np.array_equal(decisions[arm][:,0],decisions['raw'][:,0]) for arm in trained):
        raise ValueError('Clean bypass decisions differ from raw')
    return decisions, marker['active']


def macro_f1(value, labels):
    matrix = np.bincount(6*labels + value, minlength=36).reshape(6, 6)
    denom = matrix.sum(0) + matrix.sum(1)
    return float(np.divide(2*matrix.diagonal(), denom, out=np.zeros(6), where=denom > 0).mean())


def cells(predictions, labels, receivers, rng):
    correct = {name: value == labels for name, value in predictions.items()}
    absolute = {}
    for name, value in predictions.items():
        absolute[name] = {}
        for axis, view in enumerate(VIEWS):
            by_seed = [dict(seed=seed, correct=int((row[axis] == labels).sum()), count=len(labels),
                            macro_f1=macro_f1(row[axis], labels)) for seed, row in zip(SEEDS, value)]
            absolute[name][view] = dict(correct=int(correct[name][:, axis].sum()),
                                        count=len(labels)*len(SEEDS), by_seed=by_seed)
    paired = {f'{first}_to_{last}': summarize(correct[first], correct[last], receivers, VIEWS, rng,
                                               labels=(first, last)) for first, last in combinations(predictions,2)}
    return dict(absolute=absolute, paired=paired)


def score(rt, *, methods=METHODS, force_practical=False, clean_passthrough=True):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Use the separate CPU scoring process')
    root = Path(rt['run_root'])
    readers = Path(rt['recognizer_run'])
    original = historical(readers, 'predict', SEEDS[0])
    physical, predictions, activations = {}, {}, {}
    for case, (profile, route, split) in CASES.items():
        manifest, ids, _ = handle(view_root(rt, original, route, split), original['classes'])
        if len(ids) != COUNTS[split] or manifest['channel'] != f'{route}/post_sync/noeq':
            raise ValueError('Wrong Main split IQ set')
        if split in physical and not np.array_equal(physical[split], ids):
            raise ValueError('Profiles and routes must share the same physical order')
        physical[split] = ids
        predictions[profile, route, split], activations[case] = closed_predictions(
            root / 'predictions' / case, ids, profile, route, split, methods, force_practical=force_practical,
            clean_passthrough=clean_passthrough)
    if np.intersect1d(*(physical[split] for split in TASKS)).size:
        raise ValueError('Main split populations must be disjoint')
    for route in ROUTES:
        for split in TASKS:
            for method in methods:
                if not np.array_equal(predictions['default', route, split][method][:, 0],
                                      predictions['strong', route, split][method][:, 0]):
                    raise ValueError('The clean decision reference changed between profile replays')

    truth = historical(readers, 'score', SEEDS[0])
    if truth['classes'] != original['classes']:
        raise ValueError('Truth and frozen reader class orders differ')
    labels, receivers = {}, {}
    for split in TASKS:
        metadata = read(truth['tests'][split]['truth'])
        matching = [metadata[str(ident)] for ident in physical[split]]
        labels[split] = np.asarray([entry['label'] for entry in matching], dtype=np.int64)
        receivers[split] = np.asarray([entry['receiver'] for entry in matching])
    rng = np.random.default_rng(20261125)
    result = {}
    for profile in PROFILES:
        result[profile] = {}
        for route in ROUTES:
            result[profile][route] = {}
            for partition in (*TASKS, 'combined'):
                parts = TASKS if partition == 'combined' else (partition,)
                p = {name: np.concatenate([predictions[profile, route, split][name] for split in parts], axis=2)
                     for name in methods}
                y = np.concatenate([labels[split] for split in parts])
                rx = np.concatenate([receivers[split] for split in parts])
                result[profile][route][partition] = cells(p, y, rx, rng)
    write(root / 'scores.json', dict(status='TARGET_SCORED', truth_last=True,
          bootstrap_replicates=REPLICATES, seeds=list(SEEDS), results=result, activation=activations,
          force_practical=force_practical, clean_passthrough=clean_passthrough))
