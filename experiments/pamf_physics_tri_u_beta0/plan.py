from pathlib import Path
from experiments.pamf_physics_tri_u import plan as physical

RUN='20261110-phase1-pamf-tri-u-beta0-f0-m3-r01'
SEEDS=physical.SEEDS
RECIPE=dict(beta=0., l_weight=0., u_weight=.05)


def candidate_rows():
    return [dict(row_id=f'tri-u-beta0-s{r["model_seed"]}',model_seed=r['model_seed'],views=list(r['views']))
            for r in physical.candidate_rows()]


def configuration(context,row,project,commit,*,run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only three registered beta-zero U relation rows are allowed')
    base=next(r for r in physical.candidate_rows() if r['model_seed']==row['model_seed'])
    c=physical.configuration(context,base,project,commit)
    c.update(run_id=RUN,row_id=row['row_id'],algorithm_id='pamf_tri_u_beta0_v1',physical_relation=dict(RECIPE),
             output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    return c
