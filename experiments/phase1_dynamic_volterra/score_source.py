"""Truth-last source scoring for the independent dynamic Volterra model."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
SCHEMA = 'phase1_dynamic_volterra_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def count_pair(target, parent, candidate, mask):
    a, b = parent == target, candidate == target
    return dict(count=int(mask.sum()), baseline_correct=int((mask & a).sum()),
                candidate_correct=int((mask & b).sum()),
                repairs=int((mask & ~a & b).sum()), damages=int((mask & a & ~b).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.out.exists():
        raise ValueError('Independent CPU scorer and unique score output required')
    manifest = read(args.source/'prediction_complete.json')
    expected = dict(status='SOURCE_PREDICTIONS_FIXED', commit=args.commit, count=672, fit_ids=8192,
                    F_frozen=True, blind_frozen=True, gate_frozen=True, DURA_loaded=False,
                    TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates_after_prediction=0)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Complete fixed source predictions are required before identity truth')
    with np.load(args.source/'source_predictions.npz', allow_pickle=False) as z:
        ids, split, rx, scene, values = (z[key].copy() for key in ('indices', 'split', 'rx', 'scene', 'values'))
    if (ids.shape != (672,) or len(np.unique(ids)) != 672 or values.shape != (672, 9)
            or not np.isfinite(values).all() or not np.isin(values[:, 2], [0., 1.]).all()
            or not np.isin(values[:, [1, 3, 5, 7]], np.arange(6)).all()
            or np.any(values[:, [4, 6, 8]] < 0)):
        raise ValueError('Complete finite source prediction arrays are required')
    import torch
    state = torch.load(args.source/'checkpoint.pt', map_location='cpu', weights_only=False)
    if (state.get('schema') != SCHEMA or state.get('commit') != args.commit
            or state.get('epochs') != 5 or len(state.get('fit_ids', [])) != 8192
            or np.intersect1d(ids, state['fit_ids']).size):
        raise ValueError('Fixed source checkpoint and disjoint holdout IDs are required')
    cache = args.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache'
    physical_ids = np.load(cache/'train/physical_ids.npy', allow_pickle=False)
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    all_rx = np.load(cache/'train/rx.npy', allow_pickle=False)
    all_day = np.load(cache/'train/day.npy', allow_pickle=False)
    with np.load(cache/'epochs/19/groups.npz', allow_pickle=False) as z:
        all_scene = z['scene'].copy()
    if (np.any((ids < 0) | (ids >= len(roles))) or not np.array_equal(all_rx[ids], rx)
            or not np.array_equal(all_scene[ids], scene) or not np.all(all_day[ids] == 3)
            or not np.array_equal(np.isin(rx, [4, 6]), split == 'RX4_6')
            or not np.array_equal(rx == 8, split == 'RX8')):
        raise ValueError('Physical IDs, receiver groups, dates and scenes must match')
    if any(np.count_nonzero((split == group) & (scene == view)) != 48
           for group in ('RX4_6', 'RX8') for view in VIEWS):
        raise ValueError('Exactly 48 source holdout examples per receiver/scene cell are required')
    reference = args.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01'
    ref_manifest = read(reference/'prediction_complete.json')
    if (ref_manifest.get('status') != 'SOURCE_PREDICTIONS_FIXED' or ref_manifest.get('count') != 11761
            or ref_manifest.get('source_only') is not True
            or ref_manifest.get('DURA_comparator_only') is not True
            or any(ref_manifest.get(key) != 0 for key in ('TX_truth_reads', 'U_s_truth_reads', 'V_reads', 'test_reads'))):
        raise ValueError('Fixed source DURA comparator predictions are required')
    with np.load(reference/'source_predictions.npz', allow_pickle=False) as z:
        ref_ids, ref_rx, ref_scene, ref_values = (z[key].copy() for key in ('indices', 'rx', 'scene', 'values'))
    if (ref_ids.shape != (11761,) or len(np.unique(ref_ids)) != 11761 or ref_values.shape != (11761, 15)
            or not np.isfinite(ref_values).all() or not np.isin(ref_values[:, [2, 5, 7]], np.arange(6)).all()
            or np.any((ref_values[:, 3] < 0) | (ref_values[:, 3] > 1))):
        raise ValueError('Complete unique source comparator records are required')
    order = np.argsort(ref_ids)
    positions = np.searchsorted(ref_ids[order], ids)
    if np.any(positions >= len(order)) or not np.array_equal(ref_ids[order[positions]], ids):
        raise ValueError('Each candidate source ID needs a fixed DURA comparator')
    selected = order[positions]
    old = ref_values[selected]
    parent_expected = old[:, 5]
    if (not np.array_equal(ref_rx[selected], rx) or not np.array_equal(ref_scene[selected], scene)
            or not np.array_equal(old[:, 5], values[:, 3]) or not np.array_equal(parent_expected, values[:, 5])
            or not np.array_equal(old[:, 3] > .8, values[:, 2] == 1)):
        raise ValueError('Common raw predictions, input parent and gate must match')
    known, unlabeled = roles[ids] == 'L_s', roles[ids] == 'U_s'
    if not np.all(known | unlabeled):
        raise ValueError('Only Phase1 source roles are authorized')
    complete = read(args.root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/complete.json')
    contract = read(Path(complete['config']['source_contract']))
    metadata = Path(complete['config']['source_cache'])/'L_s_metadata.npz'
    # This is the first identity-truth access and occurs after fixed prediction checks.
    with np.load(metadata, allow_pickle=False) as z:
        keys, labels = z['ids'].copy(), z['y'].copy()
    if (len(keys) != 6300 or labels.shape != (6300,) or keys.tolist() != contract['role_ids']['L_s']
            or len(set(map(str, keys))) != 6300 or np.any((labels < 0) | (labels >= 6))):
        raise ValueError('Exact permitted L_s identity metadata is required')
    lookup = {str(key): int(value) for key, value in zip(keys, labels)}
    if (not all(str(key) in lookup for key in physical_ids[ids[known]])
            or not all(str(key) not in lookup for key in physical_ids[ids[unlabeled]])):
        raise ValueError('L_s and U_s physical role binding failed')
    truth = np.full(len(ids), -1, dtype=np.int64)
    truth[known] = [lookup[str(key)] for key in physical_ids[ids[known]]]
    pseudo = values[:, 1].astype(np.int64)
    trusted = values[:, 0] >= .95
    parent = values[:, 5].astype(np.int64)
    candidate = values[:, 7].astype(np.int64)
    dura = old[:, 2].astype(np.int64)
    results = {}
    for group in ('RX4_6', 'RX8'):
        results[group] = {}
        for view in (*VIEWS, 'all_impaired'):
            mask = (split == group) & ((scene != 'clean') if view == 'all_impaired' else (scene == view))
            if not mask.any():
                raise ValueError('Every receiver/scene cell must be present')
            results[group][view] = dict(count=int(mask.sum()), gate_active=int((mask & (values[:, 2] == 1)).sum()),
                L_s=count_pair(truth, parent, candidate, mask & known),
                pseudo=count_pair(pseudo, parent, candidate, mask & trusted),
                L_s_vs_DURA=count_pair(truth, dura, candidate, mask & known),
                pseudo_vs_DURA=count_pair(pseudo, dura, candidate, mask & trusted),
                parent_aligned_iq=float(values[mask, 6].mean()), candidate_aligned_iq=float(values[mask, 8].mean()))
    result = dict(status='SOURCE_SCORE_COMPLETE', commit=args.commit, fixed_predictions_before_truth=True,
                  L_s_scored=int(known.sum()), U_s_truth_reads=0, V_reads=0, test_reads=0, results=results,
                  caveat='Source records have been explored; clean F pseudo targets are not identity truth.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], L_s_scored=result['L_s_scored'],
                          impaired={key: results[key]['all_impaired'] for key in results})), flush=True)


if __name__ == '__main__':
    main()
