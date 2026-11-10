from pathlib import Path
from experiments.pamf_tri_u_phase import source as phase
from experiments.pamf_tri_u_phase.plan import RUN as PHASE_RUN
from .plan import RUN,SEEDS,ARMS
from .scope import installed

_phase_parent=phase.parent


def parent(c):
    seed,root=c['model_seed'],Path(c['output_root']);arm=c.get('algorithm_id')
    if (seed not in SEEDS or arm not in ARMS or c['run_id']!=RUN or c['row_id']!=f'{arm}-s{seed}'
        or root.parts[-5:]!=('runs',RUN,'F',c['row_id'],'source') or c.get('phase_subset')!=list(ARMS[arm])):
        raise ValueError('Unregistered phase subset continuation')
    row=f'tri-u-phase-s{seed}'
    proxy=dict(c,run_id=PHASE_RUN,row_id=row,algorithm_id='pamf_tri_u_phase_v1',
               output_root=str(root.parents[3]/PHASE_RUN/'F'/row/'source'))
    return _phase_parent(proxy)


def train(c):
    old=phase.parent,phase.installed;phase.parent,phase.installed=parent,installed
    try:return phase.train(c)
    finally:phase.parent,phase.installed=old
