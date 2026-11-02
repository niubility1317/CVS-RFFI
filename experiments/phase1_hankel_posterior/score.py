"""Truth-last source scores for fixed Hankel and corresponding DURA predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

REFERENCE = '20261102-phase1-dura-posterior-internal-source-s2026092701-r01'
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def count_cell(target, baseline, candidate, mask):
    base = baseline == target
    new = candidate == target
    return dict(count=int(mask.sum()), baseline_correct=int((mask & base).sum()),
                candidate_correct=int((mask & new).sum()),
                repairs=int((mask & ~base & new).sum()), damages=int((mask & base & ~new).sum()))


def validate_fixed(metadata, commit):
    expected = dict(status='SOURCE_PREDICTIONS_FIXED', commit=commit, count=672, source_only=True,
                    DURA_loaded=False, F_frozen=True, TX_truth_reads=0, U_s_truth_reads=0,
                    V_reads=0, test_reads=0, model_updates=0)
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('Fixed source predictions are required before truth access')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--commit', required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source score is required')
    metadata = json.loads((args.source/'prediction_complete.json').read_text(encoding='utf8'))
    validate_fixed(metadata, args.commit)
    with np.load(args.source/'source_predictions.npz', allow_pickle=False) as z:
        ids, split, rx, scene, values = (z[k].copy() for k in ('indices', 'split', 'rx', 'scene', 'values'))
    if (ids.shape != (672,) or len(np.unique(ids)) != 672 or values.shape != (672, 9)
            or not np.isfinite(values).all() or not np.isin(values[:, [1, 3, 5, 7]], np.arange(6)).all()):
        raise ValueError('Complete fixed finite source predictions are required')
    reference = args.root/'runs'/REFERENCE
    fixed = json.loads((reference/'prediction_complete.json').read_text(encoding='utf8'))
    if fixed.get('status') != 'SOURCE_PREDICTIONS_FIXED' or fixed.get('TX_labels_read') != 0:
        raise ValueError('Fixed label-hidden DURA reference is required')
    with np.load(reference/'source_predictions.npz', allow_pickle=False) as z:
        if not all(np.array_equal(v, z[k]) for k, v in (('indices', ids), ('split', split), ('rx', rx), ('scene', scene))):
            raise ValueError('DURA reference must use the same source rows')
        dura = z['values'][:, 5].astype(np.int64)
        prior_raw = z['values'][:, 3].astype(np.int64)
    cache = args.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train'
    physical = np.load(cache/'physical_ids.npy', allow_pickle=False)
    roles = np.load(cache/'roles.npy', allow_pickle=False)
    known, unlabeled = roles[ids] == 'L_s', roles[ids] == 'U_s'
    if not np.all(known | unlabeled):
        raise ValueError('Only authorized source roles are allowed')
    completed = json.loads((args.root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/complete.json').read_text())
    contract = json.loads(Path(completed['config']['source_contract']).read_text())
    with np.load(Path(completed['config']['source_cache'])/'L_s_metadata.npz', allow_pickle=False) as z:
        keys, labels = z['ids'].copy(), z['y'].copy()
    if (len(keys) != 6300 or keys.tolist() != contract['role_ids']['L_s']
            or len(set(map(str, keys))) != 6300 or np.any((labels < 0) | (labels >= 6))):
        raise ValueError('Exact source identity mapping is required')
    lookup = {str(k): int(v) for k, v in zip(keys, labels)}
    if (not all(str(k) in lookup for k in physical[ids[known]])
            or not all(str(k) not in lookup for k in physical[ids[unlabeled]])):
        raise ValueError('Source physical IDs and label roles must agree')
    truth = np.full(len(ids), -1, dtype=np.int64)
    truth[known] = [lookup[str(k)] for k in physical[ids[known]]]
    pseudo = values[:, 1].astype(np.int64)
    trusted = values[:, 0] >= .95
    predictions = dict(raw=values[:, 3].astype(np.int64), physical=values[:, 5].astype(np.int64),
                       hankel=values[:, 7].astype(np.int64), dura=dura)
    result = {}
    for group in ('RX4_6', 'RX8'):
        result[group] = {}
        for view in (*VIEWS, 'all_impaired'):
            mask = (split == group) & ((scene != 'clean') if view == 'all_impaired' else (scene == view))
            if not mask.any():
                raise ValueError('Each source receiver and view requires observations')
            cell = dict(count=int(mask.sum()), gate_active=int((mask & (values[:, 2] == 1)).sum()),
                        raw_cpu_gpu_mismatches=int((mask & (predictions['raw'] != prior_raw)).sum()),
                        iq_error={name: float(values[mask, col].mean())
                                  for name, col in (('raw', 4), ('physical', 6), ('hankel', 8))})
            for key, select, target in (('L_s', mask & known, truth), ('pseudo', mask & trusted, pseudo)):
                cell[key] = {name: count_cell(target, predictions['physical'], prediction, select)
                             for name, prediction in predictions.items()}
            result[group][view] = cell
    report = dict(status='SOURCE_SCORE_COMPLETE', fixed_predictions_before_truth=True,
                  commit=args.commit, L_s_scored=int(known.sum()), U_s_truth_reads=0,
                  V_reads=0, test_reads=0, results=result,
                  caveat='Source records previously explored; pseudo classes are not TX truth; source F predictions can differ by hardware.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=report['status'], L_s_scored=report['L_s_scored'],
                          impaired={g: result[g]['all_impaired'] for g in result})), flush=True)


if __name__ == '__main__':
    main()
