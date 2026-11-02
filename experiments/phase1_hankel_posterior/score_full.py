"""Truth-last full-source scores for the fixed Hankel replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.phase1_hankel_posterior.score import count_cell, VIEWS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--commit', required=True)
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError('Unique full-source score is required')
    m = json.loads((args.source/'prediction_complete.json').read_text())
    if (m.get('status') != 'SOURCE_PREDICTIONS_FIXED' or m.get('commit') != args.commit
            or m.get('count') != 11761 or m.get('pilot_replayed') != 672
            or m.get('DURA_loaded') is not False
            or any(m.get(k) != 0 for k in ('TX_truth_reads', 'U_s_truth_reads', 'V_reads', 'test_reads', 'model_updates'))):
        raise ValueError('Complete fixed source replay is required before truth access')
    with np.load(args.source/'source_predictions.npz', allow_pickle=False) as z:
        ids, rx, scene, old, new = (z[k].copy() for k in ('indices', 'rx', 'scene', 'prior_values', 'hankel_values'))
    if (ids.shape != (11761,) or len(np.unique(ids)) != 11761 or old.shape != (11761, 15)
            or new.shape != (11761, 2) or not np.isfinite(old).all() or not np.isfinite(new).all()
            or not np.isin(new[:, 0], np.arange(6)).all() or np.any(new[:, 1] < 0)):
        raise ValueError('Finite full source predictions are required')
    cache = args.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train'
    physical = np.load(cache/'physical_ids.npy', allow_pickle=False)
    roles = np.load(cache/'roles.npy', allow_pickle=False)
    known, unlabeled = roles[ids] == 'L_s', roles[ids] == 'U_s'
    if not np.all(known | unlabeled): raise ValueError('Only source L_s and U_s records are allowed')
    complete = json.loads((args.root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/complete.json').read_text())
    contract = json.loads(Path(complete['config']['source_contract']).read_text())
    with np.load(Path(complete['config']['source_cache'])/'L_s_metadata.npz', allow_pickle=False) as z:
        keys, labels = z['ids'].copy(), z['y'].copy()
    if (len(keys) != 6300 or labels.shape != (6300,) or keys.tolist() != contract['role_ids']['L_s']
            or len(set(map(str, keys))) != 6300 or np.any((labels < 0) | (labels >= 6))):
        raise ValueError('Exact legal source identity mapping is required')
    lookup = {str(k): int(y) for k, y in zip(keys, labels)}
    if (not all(str(x) in lookup for x in physical[ids[known]])
            or not all(str(x) not in lookup for x in physical[ids[unlabeled]])):
        raise ValueError('Source physical identities must match roles')
    truth = np.full(len(ids), -1, dtype=np.int64)
    truth[known] = [lookup[str(x)] for x in physical[ids[known]]]
    pseudo = old[:, 0].astype(np.int64); trusted = old[:, 1] >= .95
    pred = dict(raw=old[:, 5].astype(np.int64), physical=old[:, 7].astype(np.int64),
                hankel=new[:, 0].astype(np.int64), dura=old[:, 2].astype(np.int64))
    iq = dict(raw=old[:, 6], physical=old[:, 8], hankel=new[:, 1], dura=old[:, 4])
    results = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        results[group] = {}
        for view in (*VIEWS, 'all_impaired'):
            mask = np.isin(rx, receivers) & ((scene != 'clean') if view == 'all_impaired' else (scene == view))
            if not mask.any(): raise ValueError('Every source group and view is required')
            cell = dict(count=int(mask.sum()), gate_active=int((mask & (old[:, 3] > .8)).sum()),
                        iq_error={key: float(val[mask].mean()) for key, val in iq.items()})
            for name, select, target in (('L_s', mask & known, truth), ('pseudo', mask & trusted, pseudo)):
                cell[name] = {key: count_cell(target, pred['physical'], val, select) for key, val in pred.items()}
            results[group][view] = cell
    report = dict(status='SOURCE_FULL_SCORE_COMPLETE', commit=args.commit, fixed_predictions_before_truth=True,
                  L_s_scored=int(known.sum()), U_s_truth_reads=0, V_reads=0, test_reads=0, results=results,
                  caveat='Previously explored source records; pseudo labels are not true identity; no ground claim.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf8') as f: json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(status=report['status'], L_s_scored=report['L_s_scored'],
                          impaired={g: results[g]['all_impaired'] for g in results})), flush=True)


if __name__ == '__main__': main()
