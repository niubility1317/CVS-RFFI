from pathlib import Path
from experiments.pamf_tri_u_extra import plan as extra
from experiments.pamf_tri_u_phase.plan import RECIPE as PHASE_RECIPE
RUN='20261111-phase1-pamf-tri-phase-complement-extra-f0-m3-r01'
SEEDS=extra.SEEDS
SUBSET=('time','path','reference')
ARM='tri-phase-complement'


def candidate_rows():
    return [dict(row_id=f'{ARM}-s{r["model_seed"]}',model_seed=r['model_seed'],views=list(r['views']))
            for r in extra.candidate_rows()]


def configuration(context,row,project,commit,*,run_id=RUN):
    if row not in candidate_rows() or run_id!=RUN:raise ValueError('Only three registered phase extra seeds allowed')
    base=next(r for r in extra.candidate_rows() if r['model_seed']==row['model_seed'])
    c=extra.configuration(context,base,project,commit);c.pop('physical_relation')
    c.update(run_id=RUN,row_id=row['row_id'],algorithm_id=ARM,phase_relation=dict(PHASE_RECIPE),
             phase_subset=list(SUBSET),output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    return c
