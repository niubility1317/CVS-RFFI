from pathlib import Path
from experiments.pamf_physics_tri import plan as original

RUN = '20261110-phase1-pamf-tri-u-physical-f0-m3-r01'
SEEDS = original.SEEDS
RECIPE = dict(beta=.5, l_weight=0., u_weight=.05)


def candidate_rows():
    return [dict(row_id=f'tri-u-physical-s{r["model_seed"]}', model_seed=r['model_seed'], views=list(r['views']))
            for r in original.candidate_rows()]


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only the registered three source-U physical rows are allowed')
    base = next(r for r in original.candidate_rows() if r['model_seed'] == row['model_seed'])
    c = original.configuration(context, base, project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_u_physical_v1', physical_relation=dict(RECIPE),
             output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    return c
