"""Connect only permitted source L_s labels after all predictions are fixed."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def score_cell(target, baseline, candidate, mask):
    good_base = baseline == target
    good_new = candidate == target
    return dict(count=int(mask.sum()), dura_correct=int((mask & good_base).sum()),
                candidate_correct=int((mask & good_new).sum()),
                repairs=int((mask & ~good_base & good_new).sum()),
                damages=int((mask & good_base & ~good_new).sum()))


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def validate_manifest(manifest, commit):
    expected = dict(status='SOURCE_PREDICTIONS_FIXED', commit=commit, count=672, fit_ids=8192,
                    F_frozen=True, DURA_frozen=True, gate_frozen=True, posterior_frozen=True,
                    TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0,
                    model_updates_after_prediction=0)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Complete frozen source predictions are required before identity truth')
    kind = manifest.get('posterior_kind')
    if kind not in ('mean', 'nonlocal'):
        raise ValueError('A registered source posterior kind is required')
    return kind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.out.exists():
        raise ValueError('Independent CPU scorer and new score output are required')
    manifest = read(args.source/'prediction_complete.json')
    kind = validate_manifest(manifest, args.commit)
    with np.load(args.source/'source_predictions.npz', allow_pickle=False) as z:
        ids, split, rx, scene, values = (z[k].copy() for k in ('indices', 'split', 'rx', 'scene', 'values'))
    if (ids.shape != (672,) or len(np.unique(ids)) != len(ids) or values.shape != (672, 9)
            or not np.isfinite(values).all() or not np.isin(values[:, 2], [0., 1.]).all()
            or not np.isin(values[:, [1, 3, 5, 7]], np.arange(6)).all()
            or np.any(values[:, [4, 6, 8]] < 0)):
        raise ValueError('Complete, finite, valid source predictions are required')
    import torch
    checkpoint = torch.load(args.source/'checkpoint.pt', map_location='cpu', weights_only=False)
    if (checkpoint.get('schema') != 'phase1_dura_posterior_source_v1' or checkpoint.get('commit') != args.commit
            or checkpoint.get('epochs') != 5 or checkpoint.get('posterior_kind') != kind
            or checkpoint.get('upstream_L_s_fit_labels_used') != manifest.get('upstream_L_s_fit_labels_used')
            or len(checkpoint.get('fit_ids', [])) != 8192
            or np.intersect1d(ids, checkpoint['fit_ids']).size):
        raise ValueError('Fixed checkpoint and disjoint source fit records are required')
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
        raise ValueError('Physical source records, scenes and receiver splits must match')
    if any(np.count_nonzero((split == group) & (scene == view)) != 48
           for group in ('RX4_6', 'RX8') for view in VIEWS):
        raise ValueError('Exactly 48 held-out source records per receiver group and scene are required')
    known = roles[ids] == 'L_s'
    unlabeled = roles[ids] == 'U_s'
    if not np.all(known | unlabeled):
        raise ValueError('Only Phase1 L_s and U_s source records are allowed')
    complete = read(args.root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/complete.json')
    contract = read(Path(complete['config']['source_contract']))
    meta = Path(complete['config']['source_cache'])/'L_s_metadata.npz'
    # Identity metadata is opened only after fixed predictions and all checks above.
    with np.load(meta, allow_pickle=False) as z:
        label_ids, label_values = z['ids'].copy(), z['y'].copy()
    if (len(label_ids) != 6300 or label_values.shape != (6300,) or label_ids.tolist() != contract['role_ids']['L_s']
            or len(set(map(str, label_ids))) != 6300 or np.any((label_values < 0) | (label_values >= 6))):
        raise ValueError('Exact authorized L_s identity metadata is required')
    lookup = {str(key): int(value) for key, value in zip(label_ids, label_values)}
    if (not all(str(i) in lookup for i in physical_ids[ids[known]])
            or not all(str(i) not in lookup for i in physical_ids[ids[unlabeled]])):
        raise ValueError('Source role and physical identity binding is invalid')
    truth = np.full(len(ids), -1, dtype=np.int64)
    truth[known] = [lookup[str(i)] for i in physical_ids[ids[known]]]
    pseudo = values[:, 1].astype(np.int64)
    trusted = values[:, 0] >= 0.95
    dura = values[:, 5].astype(np.int64)
    candidate = values[:, 7].astype(np.int64)
    result = {}
    for group in ('RX4_6', 'RX8'):
        result[group] = {}
        for view in (*VIEWS, 'all_impaired'):
            mask = (split == group) & ((scene != 'clean') if view == 'all_impaired' else (scene == view))
            if not mask.any():
                raise ValueError('Each source receiver and scene must have predictions')
            result[group][view] = dict(count=int(mask.sum()), gate_active=int((mask & (values[:, 2] == 1)).sum()),
                L_s=score_cell(truth, dura, candidate, mask & known),
                pseudo=score_cell(pseudo, dura, candidate, mask & trusted),
                dura_iq_error=float(values[mask, 6].mean()), candidate_iq_error=float(values[mask, 8].mean()))
    report = dict(status='SOURCE_SCORE_COMPLETE', commit=args.commit, fixed_predictions_before_truth=True,
                  posterior_kind=kind,
                  upstream_L_s_fit_labels_used=manifest['upstream_L_s_fit_labels_used'],
                  L_s_scored=int(known.sum()), U_s_truth_reads=0, V_reads=0, test_reads=0, results=result,
                  caveat='Source records previously explored; F pseudo targets are not identity truth.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], L_s_scored=report['L_s_scored'],
                          impaired={g: result[g]['all_impaired'] for g in result})), flush=True)


if __name__ == '__main__':
    main()
