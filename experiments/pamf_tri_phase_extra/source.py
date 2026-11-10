from pathlib import Path
from experiments.pamf_tri_u_extra import source as extra
from experiments.pamf_tri_u_extra.plan import RUN as EXTRA_RUN,RECIPE as EXTRA_RECIPE
from experiments.pamf_tri_u_phase import source as phase
from experiments.pamf_tri_u_phase_subset.scope import installed
from experiments.pamf_tri_u_phase.plan import RECIPE as PHASE_RECIPE
from .plan import RUN,SEEDS,SUBSET,ARM


def parent(c):
    seed,root=c['model_seed'],Path(c['output_root'])
    if (seed not in SEEDS or c['run_id']!=RUN or c['row_id']!=f'{ARM}-s{seed}' or c.get('algorithm_id')!=ARM
        or root.parts[-5:]!=('runs',RUN,'F',c['row_id'],'source') or c.get('phase_subset')!=list(SUBSET)
        or c.get('phase_relation')!=PHASE_RECIPE or c.get('physical_relation')):
        raise ValueError('Unregistered extra-seed phase subset continuation')
    row=f'tri-u-beta0-s{seed}'
    proxy=dict(c,run_id=EXTRA_RUN,row_id=row,algorithm_id='pamf_tri_u_beta0_extra_v1',
               physical_relation=dict(EXTRA_RECIPE),output_root=str(root.parents[3]/EXTRA_RUN/'F'/row/'source'))
    return extra.parent(proxy)


def train(c):
    old=phase.parent,phase.installed;phase.parent,phase.installed=parent,installed
    try:return phase.train(c)
    finally:phase.parent,phase.installed=old
