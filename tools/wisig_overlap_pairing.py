"""Reconcile the frozen source scores and quantify two WiSig swap interactions."""

import json
import os
import sys
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_unrolled_restore import wisig_r69_stage_isolation as stage_study
from experiments.pamf_unrolled_restore.wisig_overlap_isolation import MODES, ROUTES, RUN, VIEWS
from experiments.pamf_unrolled_restore.wisig_short_training_source import handles


CONTRASTS = {
    'boundary_overlap': ('r69', 'paired_align', 'r69_edges', 'r69_overlap'),
    'overlap_amplitude_phase': ('r69', 'r69_edges', 'r69_overlap_phase', 'r69_overlap_amplitude'),
}


def bootstrap(delta, draw):
    endpoints = {'six':delta[1:].sum(0), **dict(zip(VIEWS, delta, strict=True))}
    return {name:dict(net=int(rows.sum()), ci=np.quantile(rows[draw].sum(1), (.025,.975)).tolist())
            for name,rows in endpoints.items()}


def score(runtime):
    root = Path(runtime['run_root'])
    stages = runtime['run_id']==stage_study.RUN
    modes = stage_study.MODES if stages else MODES
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or runtime['run_id'] not in (RUN,stage_study.RUN):
        raise ValueError('WiSig overlap verification requires its own CPU source workflow')
    _, _, source, _, ids, rx, positions = handles(runtime)
    marker = read(root/'predictions/complete.json')
    result = read(root/'source-scores.json')
    if (read(root/'state.json')['status'] != 'SOURCE_SCORED' or marker['status'] != 'FROZEN'
            or marker['target_reads'] or marker['source_truth_reads'] or not result['truth_last']
            or result['target_reads'] or result['modes'] != list(modes) or marker['modes']!=list(modes)):
        raise ValueError('Both prediction matrices and truth-last scores must already be complete')
    decisions = {}
    for route in ROUTES:
        rows = []
        for view in VIEWS:
            with np.load(root/f'predictions/{route}_{view}.npz',allow_pickle=False) as values:
                grid = values['decision']
                if not np.array_equal(values['ids'],ids) or grid.shape != (len(modes),len(ids)):
                    raise ValueError('Expected original physical source rows')
                rows.append(grid)
        decisions[route] = np.stack(rows)
    labels = np.load(source/'V'/'labels.npy',allow_pickle=False)[positions]
    groups = {str(key):np.flatnonzero(rx==key) for key in np.unique(rx)}
    rng = np.random.default_rng(2026112331)
    draw = np.concatenate([group[rng.integers(len(group),size=(1600,len(group)))] for group in groups.values()], axis=1)
    evidence = {}
    for route, rows in decisions.items():
        hits = (rows==labels).astype(np.int16)
        for name, group in {'all': np.arange(len(ids)), **groups}.items():
            if hits[:,:,group].sum(-1).tolist() != result['scores'][route][name]['correct']:
                raise ValueError('Independent correct counts differ from recorded source scores')
        evidence[route] = {}
        if stages:
            for proposed,reference in stage_study.COMPARISONS:
                a,b = (hits[:,modes.index(mode)] for mode in (proposed,reference))
                delta = a-b
                evidence[route][f'{proposed}_vs_{reference}'] = dict(
                    paired=bootstrap(delta,draw),
                    repaired=(delta>0).sum(1).tolist(),damaged=(delta<0).sum(1).tolist(),
                    by_rx={rx:delta[:,indices].sum(1).tolist() for rx,indices in groups.items()})
        else:
            for name, contrast in CONTRASTS.items():
                a,b,c,d = (hits[:,modes.index(mode)] for mode in contrast)
                delta = a + b - c - d
                evidence[route][name] = dict(contrasts=contrast, paired=bootstrap(delta,draw),
                      six_by_rx={rx:int(delta[1:,indices].sum()) for rx,indices in groups.items()})
    return dict(status='VERIFIED', reader_seed=result['reader_seed'], target_reads=0, rows=len(ids), routes=evidence)


if __name__ == '__main__':
    print(json.dumps(score(read(Path(sys.argv[1]))), allow_nan=False))
