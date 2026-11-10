"""Check exact F rows and packet-aligned raw decision reuse without truth."""
import argparse
import json
from pathlib import Path

import numpy as np

ORIGINAL = '20261027-phase1-pamf-repl-shuffled-views-m16-r02'
SUPPLEMENT = '20261110-evaluation-trisig-dura-f0-s2026092702-03-m2-r01'
OLD = '20261101-evaluation-P3Restore-TF-TriSIGSSL-main-s2026092701-r03'
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
TASKS = ('cross_rx', 'cross_both')
COUNTS = {'cross_rx':126000, 'cross_both':42000}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    project = parser.parse_args().project
    cache = project/'runs/20261023-phase1-practical-residual-control-r03/restored_IQ/F0-DURA-LU'
    header = read(cache/'complete.json')
    if (header['G']['kind'] != 'original_F0_LU' or header['G']['roles'] != 'LU' or
            not header['G']['checkpoint'].endswith('20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt')):
        raise ValueError('Wrong DURA cache')
    results = []
    for seed in (2026092701, 2026092702, 2026092703):
        row = f'main_split-repl-shuffled_time-3view-s{seed}'
        root = project/'runs'/ORIGINAL
        c = read(root/'configs'/f'predict__{row}.json')
        done = read(root/'F'/row/'source/complete.json')
        if (c['run_id'],c['row_id'],c['model_seed']) != (ORIGINAL,row,seed):
            raise ValueError('Original predictor identity differs')
        if (done['status'] != 'SOURCE_FROZEN' or done['epoch'] != 50 or done['updates'] != 11100 or
                done['config']['model_seed'] != seed or done['config']['view_comparison']['view_count'] != 3 or
                done['config']['repl_row']['structure'] != 'SHUFFLE_TIME' or done['target_access'] or done['U_truth_read']):
            raise ValueError('Wrong original source model')
        matches = {}
        for task in TASKS:
            with np.load(root/'readouts'/row/'predictions'/task/'predictions.npz',allow_pickle=False) as original, \
                 np.load(cache/task/'index.npz',allow_pickle=False) as cached:
                if len(original['ids']) != COUNTS[task] or not np.array_equal(original['ids'],cached['ids']):
                    raise ValueError('DURA cached IDs differ from original source predictions')
                other = (project/'runs'/OLD/'prediction'/task/'predictions.npz' if seed == 2026092701 else
                         project/'runs'/SUPPLEMENT/'predictions'/str(seed)/task/'predictions.npz')
                with np.load(other,allow_pickle=False) as supplement:
                    if not np.array_equal(original['ids'], supplement['ids']):
                        raise ValueError('Baseline prediction IDs differ')
                    mismatches = {view:int(np.count_nonzero(original['raw__'+view] != supplement['raw__'+view])) for view in VIEWS}
                    if any(mismatches.values()):
                        raise ValueError('Baseline raw predictions differ from original frozen predictions')
            matches[task] = dict(count=COUNTS[task], raw_decision_mismatches=mismatches)
        results.append(dict(seed=seed, original_run=ORIGINAL, original_row=row,
                            source_frozen_epoch=50, source_updates=11100,
                            original_fixed_shuffle=True, populations=matches))
    print(json.dumps(dict(status='VERIFIED', rows=results, truth_reads=0, model_updates=0,
                          note='Matches same F rows and physical IDs; seed01 DURA was evaluated live, 02/03 use fixed DURA cache.'),indent=2))


if __name__ == '__main__':
    main()
