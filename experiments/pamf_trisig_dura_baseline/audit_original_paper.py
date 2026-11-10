"""Compare completed paper and supplementary decisions without accessing truth."""
import argparse
import json
from pathlib import Path
import numpy as np

PAPER='20261027-phase1-pamf-paper-completion-m119-r03'
ORIGINAL='20261027-phase1-pamf-repl-shuffled-views-m16-r02'
SUPPLEMENT='20261110-evaluation-trisig-dura-f0-s2026092702-03-m2-r01'
OLD='20261101-evaluation-P3Restore-TF-TriSIGSSL-main-s2026092701-r03'
TASKS=('cross_rx','cross_both')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',required=True,type=Path)
    runs=parser.parse_args().project/'runs';sources=read(runs/PAPER/'sources.json');rows=[]
    for seed in (2026092701,2026092702,2026092703):
        row=f'main_split-factorial-ssl-full-permuted-s{seed}'
        entry=sources[row]
        expected=runs/ORIGINAL/'F'/f'main_split-repl-shuffled_time-3view-s{seed}'/'source'
        if Path(entry['source_output'])!=expected or entry['reused'] is not True:
            raise ValueError('Paper baseline is not original Tri-SIG-TIME row')
        paper=Path(entry['predictions']);scores=Path(entry['scores'])
        if read(paper/'complete.json')['status']!='ALL_PREDICTIONS_COMPLETE' or read(scores/'complete.json')['status']!='SCORED_COMPLETE':
            raise ValueError('Paper prediction/score not complete')
        alternate=(runs/OLD/'prediction' if seed==2026092701 else runs/SUPPLEMENT/'predictions'/str(seed))
        per_task={}
        for task in TASKS:
            with np.load(paper/task/'predictions.npz',allow_pickle=False) as a, np.load(alternate/task/'predictions.npz',allow_pickle=False) as b:
                if not np.array_equal(a['ids'],b['ids']):raise ValueError('Physical IDs differ')
                per_task[task]={route:{view:int(np.count_nonzero(a[f'{route}__{view}']!=b[f'{other}__{view}']))
                                       for view in VIEWS} for route,other in (('raw','raw'),('DURA-LU','DURA_v1'))}
        rows.append(dict(seed=seed,paper_row=row,source_reused_original=True,mismatches=per_task))
    print(json.dumps(dict(status='VERIFIED',rows=rows,truth_reads=0,model_updates=0),indent=2))


if __name__=='__main__':main()
