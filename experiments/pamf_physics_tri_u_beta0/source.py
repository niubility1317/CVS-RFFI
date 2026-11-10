from pathlib import Path
from experiments.pamf_physics_tri_u import source as physical
from experiments.pamf_physics_tri_u.plan import RUN as PHYSICAL_RUN, RECIPE as PHYSICAL_RECIPE
from .plan import RUN, SEEDS, RECIPE

_original_parent=physical.parent


def parent(c):
    seed,root=c['model_seed'],Path(c['output_root'])
    if (seed not in SEEDS or c['run_id']!=RUN or c['row_id']!=f'tri-u-beta0-s{seed}'
        or root.parts[-5:]!=('runs',RUN,'F',c['row_id'],'source')
        or c.get('algorithm_id')!='pamf_tri_u_beta0_v1' or c.get('physical_relation')!=RECIPE):
        raise ValueError('Unregistered no-direction-discount U continuation')
    row=f'tri-u-physical-s{seed}'
    proxy=dict(c,run_id=PHYSICAL_RUN,row_id=row,algorithm_id='pamf_tri_u_physical_v1',
               physical_relation=dict(PHYSICAL_RECIPE),output_root=str(root.parents[3]/PHYSICAL_RUN/'F'/row/'source'))
    return _original_parent(proxy)


def train(c):
    previous=physical.parent
    physical.parent=parent
    try:
        return physical.train(c)
    finally:
        physical.parent=previous
