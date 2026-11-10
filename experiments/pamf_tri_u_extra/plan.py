from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read
from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_repl_view_shuffle import plan as tri

RUN='20261111-phase1-pamf-tri-u-beta0-extra-f0-m3-r01'
SEEDS=(2026092704,2026092705,2026092706)
RECIPE=dict(beta=0.,l_weight=0.,u_weight=.05)


def candidate_rows():
    return [dict(row_id=f'tri-u-beta0-s{seed}',model_seed=seed,views=list(tri.VIEWS)) for seed in SEEDS]


def paper_row(seed):
    if seed not in SEEDS:raise ValueError('Registered extra seed required')
    r=deepcopy(next(r for r in paper.make_plan()['rows']
        if r['row_id']=='main_split-factorial-ssl-full-permuted-s2026092701'))
    r.update(row_id=f'main_split-factorial-ssl-full-permuted-s{seed}',model_seed=seed,
             reuse_key=f'current/3/{seed}')
    return r


def configuration(context,row,project,commit,*,run_id=RUN):
    if row not in candidate_rows() or run_id!=RUN:raise ValueError('Only the three registered extra seeds are allowed')
    seed=row['model_seed'];original=f'main_split-repl-shuffled_time-3view-s{seed}'
    prefix=Path(project)/'runs'/tri.RUN/'F'/original/'source'/'segment20'
    c=dict(read(prefix/'complete.json')['config']);tri.validate_config(c)
    for k in ('parent_source','segment_epochs','epoch_offset','step_offset'):c.pop(k,None)
    c.update(run_id=RUN,row_id=row['row_id'],paper_row=paper_row(seed),code_commit=commit,
             algorithm_id='pamf_tri_u_beta0_extra_v1',shared_prefix=str(prefix),physical_relation=dict(RECIPE),
             output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    if read(c['source_contract'])!=read(context['handles']['F0']['source_contract']):
        raise ValueError('Original extra-seed prefix and F0 source contract differ')
    return c
