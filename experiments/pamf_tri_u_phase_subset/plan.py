from pathlib import Path
from experiments.pamf_tri_u_phase import plan as phase
RUN='20261111-phase1-pamf-tri-u-phase-subset-f0-m6-r01'
SEEDS=phase.SEEDS
ARMS={'tri-phase-mag':('frequency','pa'),'tri-phase-complement':('time','path','reference')}


def candidate_rows():
    return [dict(row_id=f'{arm}-s{seed}',model_seed=seed,views=list(r['views']),arm=arm)
            for seed in SEEDS for arm in ARMS for r in phase.candidate_rows() if r['model_seed']==seed]


def configuration(context,row,project,commit,*,run_id=RUN):
    if row not in candidate_rows() or run_id!=RUN:raise ValueError('Only six registered phase subset rows allowed')
    base=next(r for r in phase.candidate_rows() if r['model_seed']==row['model_seed'])
    c=phase.configuration(context,base,project,commit)
    c.update(run_id=RUN,row_id=row['row_id'],algorithm_id=row['arm'],phase_subset=list(ARMS[row['arm']]),
             output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    return c
