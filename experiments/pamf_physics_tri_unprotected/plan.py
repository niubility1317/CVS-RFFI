"""Fixed source-only continuation with the original SSL objective intact."""
from pathlib import Path

from experiments.pamf_physics_tri import plan as protected

RUN = '20261110-phase1-pamf-tri-unprotected-f0-m3-r01'
SEEDS = protected.SEEDS


def candidate_rows():
    return [dict(row_id=f'tri-unprotected-s{r["model_seed"]}', model_seed=r['model_seed'],
                 views=list(r['views'])) for r in protected.candidate_rows()]


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only the registered three unprotected SSL continuations are allowed')
    original = next(r for r in protected.candidate_rows() if r['model_seed'] == row['model_seed'])
    c = protected.configuration(context, original, project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_unprotected_v1',
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    return c
