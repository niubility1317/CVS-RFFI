"""Reconcile matched LoRa DURA scores against the full three-seed goal."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from .data import TARGET, VIEWS
from .full_packet_data import LIMIT


SEEDS=(2026092701,2026092702,2026092703)
SCENES=VIEWS[1:]
N=len(TARGET)*LIMIT
WISIG_RECOVERY=(.784277-.3736)/(.84677976-.3736)


def score_rows(root, modes):
    score=read(root/'scores.json')
    if score['status']!='SCORED_COMPLETE' or score['truth_last'] is not True:
        raise ValueError(f'Independent scoring is incomplete in {root}')
    rows={}
    for row in score['by_seed']:
        mode=row['mode']
        if mode not in modes:
            continue
        key=(row['seed'],row['view'],mode)
        if (key in rows or row['n']!=N or row['view'] not in VIEWS
                or set(row['by_rx'])!=set(TARGET)):
            raise ValueError(f'Invalid full LoRa score cell {key}')
        counts={r:round(row['by_rx'][r]*LIMIT) for r in TARGET}
        if any(not 0<=n<=LIMIT for n in counts.values()) or sum(counts.values())!=round(row['accuracy']*N):
            raise ValueError(f'Receiver denominators do not reconcile for {key}')
        rows[key]=dict(correct=sum(counts.values()),by_rx=counts,
                       gained=row['gained_direct'],lost=row['lost_direct'])
    return rows


def verify_direct(candidate,reference,seed):
    for view in VIEWS:
        for rx in TARGET:
            name=f'{rx}__{view}__direct.npy'
            a=np.load(candidate/'predictions'/str(seed)/name,allow_pickle=False)
            b=np.load(reference/'predictions'/str(seed)/name,allow_pickle=False)
            if a.shape!=(LIMIT,) or not np.array_equal(a,b):
                raise ValueError(f'Direct Tri-set predictions differ for {(seed,view,rx)}')


def aggregate(rows,n):
    clean=rows['clean']['correct']
    return dict(clean_correct=clean,clean=clean/n,
                practical=sum(rows[v]['correct'] for v in SCENES)/(n*len(SCENES)),
                low_urban=rows['practical_low_urban']['correct']/n)


def compare(reference,candidates,mode):
    baseline=score_rows(reference,{'direct','coherent_half'})
    by_seed={}
    for candidate in candidates:
        cells=score_rows(candidate,{'direct',mode})
        for seed in sorted({key[0] for key in cells}):
            if seed not in SEEDS or seed in by_seed:
                raise ValueError(f'Unexpected or duplicate seed {seed}')
            verify_direct(candidate,reference,seed)
            rows={}
            for view in VIEWS:
                direct=baseline[seed,view,'direct']
                half=baseline[seed,view,'coherent_half']
                recovered=cells[seed,view,mode]
                if cells[seed,view,'direct']!=direct:
                    raise ValueError(f'Direct score differs for {(seed,view)}')
                if recovered['correct']-direct['correct']!=recovered['gained']-recovered['lost']:
                    raise ValueError(f'Paired correct transitions differ for {(seed,view)}')
                rows[view]=dict(direct=direct,baseline_half=half,candidate=recovered)
            by_seed[seed]=rows
    if set(by_seed)!=set(SEEDS):
        raise ValueError(f'Need all three matched seeds, found {sorted(by_seed)}')

    modes=('direct','baseline_half','candidate')
    pooled={}
    for mode_name in modes:
        pooled[mode_name]={v:{
            'correct':sum(by_seed[s][v][mode_name]['correct'] for s in SEEDS),
            'by_rx':{r:sum(by_seed[s][v][mode_name]['by_rx'][r] for s in SEEDS) for r in TARGET},
        } for v in VIEWS}
    readings={m:aggregate(rows,N*len(SEEDS)) for m,rows in pooled.items()}
    base,new=readings['direct'],readings['candidate']
    gap=new['clean']-new['practical']
    closed=(new['practical']-base['practical'])/(base['clean']-base['practical'])
    checks=dict(practical_at_least_57=new['practical']>=.57,low_urban_at_least_43=new['low_urban']>=.43,
                keeps_direct_clean_count=new['clean_correct']>=base['clean_correct'],
                clean_gap_within_three_points=abs(gap)<=.03,
                reaches_wisig_reference_fraction=closed>=WISIG_RECOVERY)
    return dict(mode=mode,checks=checks,clean_gap=gap,gap_closed=closed,wisig_reference=WISIG_RECOVERY,
                pooled=readings,pooled_by_scene_receiver=pooled,
                seed_summary={s:{m:aggregate({v:by_seed[s][v][m] for v in VIEWS},N) for m in modes}
                              for s in SEEDS},
                by_seed_scene_receiver=by_seed)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,action='append',required=True)
    p.add_argument('--mode',required=True)
    a=p.parse_args()
    if a.mode=='direct':
        p.error('Choose a restoration mode')
    print(json.dumps(compare(a.baseline,a.candidate,a.mode),allow_nan=False,indent=2))
