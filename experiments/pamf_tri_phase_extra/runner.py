import argparse
from functools import partial
from experiments.cric_dura_comparison.io import read
from experiments.pamf_tri_u_phase import runner as phase
from experiments.pamf_tri_u_phase_subset.scope import installed
from experiments.pamf_physics_branch import runner as evaluation
from experiments.pamf_physics_route import runner as dispatching
from experiments.pamf_repl_view_shuffle.source import seed_scope
from .plan import candidate_rows,configuration
from .source import parent,train
from .prepare import prepare


def smoke(rt):
    old=phase.candidate_rows,phase.configuration,phase.parent,phase.installed
    phase.candidate_rows,phase.configuration,phase.parent,phase.installed=candidate_rows,configuration,parent,installed
    try:return phase.smoke(rt)
    finally:phase.candidate_rows,phase.configuration,phase.parent,phase.installed=old


def worker(c,kind):
    if kind=='score':return evaluation.worker(c,kind)
    if kind=='predict':
        with seed_scope():return evaluation.worker(c,kind)
    if kind not in ('train','smoke'):raise ValueError('Only source and practical evaluation stages allowed')
    from experiments.pamf_repl_completion.runner import activate
    activate(c);return train(c) if kind=='train' else smoke(c)


def dispatch(rt):
    launch=partial(dispatching.launch,module='experiments.pamf_tri_phase_extra.runner')
    dispatching.dispatch(rt,prepare_run=prepare,launch_worker=launch)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker'])
    p.add_argument('--kind');p.add_argument('--config',required=True);a=p.parse_args()
    dispatch(read(a.config)) if a.action=='dispatch' else worker(read(a.config),a.kind)
