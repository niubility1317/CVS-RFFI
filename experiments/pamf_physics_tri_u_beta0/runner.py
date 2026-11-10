import argparse
from functools import partial
from experiments.cric_dura_comparison.io import read
from experiments.pamf_physics_tri_u import runner as physical
from experiments.pamf_physics_branch import runner as evaluation
from experiments.pamf_physics_route import prepare as routes, runner as dispatching
from .plan import candidate_rows,configuration
from .source import parent,train


def smoke(rt):
    old=physical.candidate_rows,physical.configuration,physical.parent
    physical.candidate_rows,physical.configuration,physical.parent=candidate_rows,configuration,parent
    try:
        return physical.smoke(rt)
    finally:
        physical.candidate_rows,physical.configuration,physical.parent=old


def worker(c,kind):
    if kind in ('predict','score'):
        return evaluation.worker(c,kind)
    if kind not in ('train','smoke'):
        raise ValueError('Only source and practical evaluation stages are allowed')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    return train(c) if kind=='train' else smoke(c)


def dispatch(rt):
    prepare=partial(routes.prepare,rows_for_run=lambda _:candidate_rows(),configure=configuration,
                    source_diagnostic=False,tri_dura=True)
    launch=partial(dispatching.launch,module='experiments.pamf_physics_tri_u_beta0.runner')
    dispatching.dispatch(rt,prepare_run=prepare,launch_worker=launch)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker'])
    p.add_argument('--kind');p.add_argument('--config',required=True);a=p.parse_args()
    dispatch(read(a.config)) if a.action=='dispatch' else worker(read(a.config),a.kind)
