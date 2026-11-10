from pathlib import Path
from experiments.pamf_branch_training.source import checked_e20_parent
from experiments.pamf_repl_view_shuffle import plan as tri
from experiments.pamf_physics_tri_u import source as physical
from .plan import RUN,SEEDS,RECIPE,paper_row


def parent(c):
    seed,root=c['model_seed'],Path(c['output_root'])
    original=f'main_split-repl-shuffled_time-3view-s{seed}'
    if (seed not in SEEDS or c['run_id']!=RUN or c['row_id']!=f'tri-u-beta0-s{seed}'
        or root.parts[-5:]!=('runs',RUN,'F',c['row_id'],'source')
        or c.get('algorithm_id')!='pamf_tri_u_beta0_extra_v1' or c.get('physical_relation')!=RECIPE
        or c.get('paper_row')!=paper_row(seed)):
        raise ValueError('Unregistered extra-seed continuation')
    expected=root.parents[3]/tri.RUN/'F'/original/'source'/'segment20'
    if Path(c['shared_prefix'])!=expected:raise ValueError('Exact same-seed original E20 required')
    proxy=dict(c,run_id=tri.RUN,row_id=original);proxy.pop('shared_prefix');tri.validate_config(proxy)
    if any(c.get(k) for k in ('target_inputs','initial_checkpoint','parent_source')):
        raise ValueError('Only source-prefix continuation allowed')
    evidence=checked_e20_parent(c);pc=evidence['config']
    keys=('view_comparison','repl_row','source_cache','source_rxs','source_days')
    if (pc['row_id']!=original or pc['run_id']!=tri.RUN or any(pc[k]!=c[k] for k in keys)
        or any(pc[k]!=c.get(k) for k in pc if k.startswith('hard_pl'))):
        raise ValueError('Original same-seed graph or source settings differ')
    return evidence


def train(c):
    previous=physical.parent;physical.parent=parent
    try:return physical.train(c)
    finally:physical.parent=previous
