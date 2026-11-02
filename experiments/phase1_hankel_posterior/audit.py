"""Independently recompute fixed-source Hankel results via sorted identity joins."""

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
    if args.out.exists(): raise FileExistsError('Unique source audit is required')
    m = json.loads((args.source/'prediction_complete.json').read_text())
    score = json.loads(args.score.read_text())
    if (m.get('status') != 'SOURCE_PREDICTIONS_FIXED' or score.get('status') != 'SOURCE_SCORE_COMPLETE'
            or score.get('fixed_predictions_before_truth') is not True or score.get('commit') != m.get('commit')
            or any(m.get(k) != 0 for k in ('TX_truth_reads', 'U_s_truth_reads', 'V_reads', 'test_reads', 'model_updates'))):
        raise ValueError('Fixed source prediction and truth-last score are required')
    with np.load(args.source/'source_predictions.npz', allow_pickle=False) as z:
        ids, split, scene, values = (z[k].copy() for k in ('indices', 'split', 'scene', 'values'))
    reference = args.root/'runs/20261102-phase1-dura-posterior-internal-source-s2026092701-r01/source_predictions.npz'
    with np.load(reference, allow_pickle=False) as z:
        if not np.array_equal(ids, z['indices']): raise ValueError('Source row mismatch')
        dura, old_raw = z['values'][:, 5].astype(int), z['values'][:, 3].astype(int)
    cache = args.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train'
    physical = np.load(cache/'physical_ids.npy', allow_pickle=False).astype(str)
    roles = np.load(cache/'roles.npy', allow_pickle=False)
    complete = json.loads((args.root/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/complete.json').read_text())
    with np.load(Path(complete['config']['source_cache'])/'L_s_metadata.npz', allow_pickle=False) as z:
        keys, labels = z['ids'].astype(str), z['y'].copy()
    order = np.argsort(keys); keys = keys[order]; positions = np.searchsorted(keys, physical[ids])
    hit = (positions < len(keys)) & (keys[np.minimum(positions, len(keys)-1)] == physical[ids])
    if not np.array_equal(hit, roles[ids] == 'L_s') or int(hit.sum()) != score['L_s_scored']:
        raise ValueError('Independent source identity join failed')
    truth = np.full(len(ids), -1, dtype=np.int64); truth[hit] = labels[order[positions[hit]]]
    preds = {'raw': values[:, 3].astype(int), 'physical': values[:, 5].astype(int),
             'hankel': values[:, 7].astype(int), 'dura': dura}
    cells = fields = 0
    for group, views in score['results'].items():
        for view, cell in views.items():
            ids_cell = {i for i in range(len(ids)) if split[i] == group and
                        (scene[i] != 'clean' if view == 'all_impaired' else scene[i] == view)}
            if (cell['count'] != len(ids_cell) or cell['gate_active'] != sum(values[i, 2] == 1 for i in ids_cell)
                    or cell['raw_cpu_gpu_mismatches'] != sum(preds['raw'][i] != old_raw[i] for i in ids_cell)):
                raise ValueError('Independent source counts differ')
            fields += 3
            for name, col in (('raw', 4), ('physical', 6), ('hankel', 8)):
                mean = sum(float(values[i, col]) for i in ids_cell)/len(ids_cell)
                if not math.isclose(mean, cell['iq_error'][name], rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError('Independent source IQ error differs')
                fields += 1
            for name, valid, target in (('L_s', hit, truth), ('pseudo', values[:, 0] >= .95, values[:, 1])):
                rows = {i for i in ids_cell if valid[i]}
                baseline = {i for i in rows if preds['physical'][i] == target[i]}
                for key, pred in preds.items():
                    correct = {i for i in rows if pred[i] == target[i]}
                    actual = cell[name][key]
                    if [actual[k] for k in ('count', 'baseline_correct', 'candidate_correct', 'repairs', 'damages')] != [len(rows), len(baseline), len(correct), len(correct-baseline), len(baseline-correct)]:
                        raise ValueError('Independent paired source counts differ')
                    fields += 5
            cells += 1
    if cells != 16 or fields != 736: raise ValueError('Complete source audit is required')
    result = dict(status='PASS', records=len(ids), L_s_scored=int(hit.sum()), cells=cells, fields_recomputed=fields,
                  method='sorted physical-ID join and independent set counting', U_s_truth_reads=0, V_reads=0, test_reads=0)
    with args.out.open('x', encoding='utf8') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
