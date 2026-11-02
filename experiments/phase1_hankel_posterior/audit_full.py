"""Independently recompute full-source results using sorted IDs and set counts."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--score', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError('Unique full-source audit is required')
    m = json.loads((args.source/'prediction_complete.json').read_text())
    score = json.loads(args.score.read_text())
    if (m.get('status') != 'SOURCE_PREDICTIONS_FIXED' or m.get('count') != 11761
            or m.get('pilot_replayed') != 672 or score.get('status') != 'SOURCE_FULL_SCORE_COMPLETE'
            or score.get('fixed_predictions_before_truth') is not True or score.get('commit') != m.get('commit')
            or any(m.get(k) != 0 for k in ('TX_truth_reads', 'U_s_truth_reads', 'V_reads', 'test_reads', 'model_updates'))):
        raise ValueError('Fixed source predictions and truth-last score are required')
    with np.load(args.source/'source_predictions.npz', allow_pickle=False) as z:
        ids, rx, scene, old, new = (z[k].copy() for k in ('indices', 'rx', 'scene', 'prior_values', 'hankel_values'))
    prior = args.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01/source_predictions.npz'
    with np.load(prior, allow_pickle=False) as z:
        if (not np.array_equal(ids, z['indices']) or not np.array_equal(rx, z['rx'])
                or not np.array_equal(scene, z['scene']) or not np.array_equal(old, z['values'])):
            raise ValueError('Fixed source reference differs')
    cache = args.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train'
    physical = np.load(cache/'physical_ids.npy', allow_pickle=False).astype(str)
    roles = np.load(cache/'roles.npy', allow_pickle=False)
    complete = json.loads((args.root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/complete.json').read_text())
    with np.load(Path(complete['config']['source_cache'])/'L_s_metadata.npz', allow_pickle=False) as z:
        keys, labels = z['ids'].astype(str), z['y'].copy()
    order = np.argsort(keys); keys = keys[order]; pos = np.searchsorted(keys, physical[ids])
    hit = (pos < len(keys)) & (keys[np.minimum(pos, len(keys)-1)] == physical[ids])
    if not np.array_equal(hit, roles[ids] == 'L_s') or int(hit.sum()) != score['L_s_scored']:
        raise ValueError('Independent source identity join failed')
    truth = np.full(len(ids), -1, dtype=np.int64); truth[hit] = labels[order[pos[hit]]]
    pred = dict(raw=old[:, 5].astype(int), physical=old[:, 7].astype(int),
                hankel=new[:, 0].astype(int), dura=old[:, 2].astype(int))
    iq = dict(raw=old[:, 6], physical=old[:, 8], hankel=new[:, 1], dura=old[:, 4])
    cells = fields = 0
    for group, views in score['results'].items():
        for view, cell in views.items():
            allowed = (4, 6) if group == 'RX4_6' else (8,)
            rows = {i for i in range(len(ids)) if rx[i] in allowed and
                    (scene[i] != 'clean' if view == 'all_impaired' else scene[i] == view)}
            if cell['count'] != len(rows) or cell['gate_active'] != sum(old[i, 3] > .8 for i in rows):
                raise ValueError('Independent source group counts differ')
            fields += 2
            for name, values in iq.items():
                avg = sum(float(values[i]) for i in rows)/len(rows)
                if not math.isclose(avg, cell['iq_error'][name], rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError('Independent IQ error differs')
                fields += 1
            for label, valid, target in (('L_s', hit, truth), ('pseudo', old[:, 1] >= .95, old[:, 0])):
                subset = {i for i in rows if valid[i]}
                baseline = {i for i in subset if pred['physical'][i] == target[i]}
                for name, values in pred.items():
                    good = {i for i in subset if values[i] == target[i]}
                    result = cell[label][name]
                    if [result[k] for k in ('count', 'baseline_correct', 'candidate_correct', 'repairs', 'damages')] != [len(subset), len(baseline), len(good), len(good-baseline), len(baseline-good)]:
                        raise ValueError('Independent paired source counts differ')
                    fields += 5
            cells += 1
    if cells != 16 or fields != 736: raise ValueError('Complete source audit is required')
    result = dict(status='PASS', records=len(ids), L_s_scored=int(hit.sum()), cells=cells,
                  fields_recomputed=fields, method='sorted physical-ID join and independent set counting',
                  U_s_truth_reads=0, V_reads=0, test_reads=0)
    with args.out.open('x', encoding='utf8') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
