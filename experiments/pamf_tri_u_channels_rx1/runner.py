import argparse
from functools import partial
from experiments.cric_dura_comparison.io import read
from experiments.pamf_physics_branch import runner as evaluation
from experiments.pamf_physics_route import runner as dispatching
from experiments.pamf_repl_view_shuffle import source as shuffle_source
from experiments.pamf_tri_u_phase import runner as phase
from .plan import candidate_rows,configuration
from .prepare import prepare
from .source import parent,installed,train


def smoke(rt):
    def graph(net,cfg):
        ref=net.id_backbone.encoder;views=ref.sequence_pool.views;order=[0,5,3,6,2,1,4,7]
        if len(views)!=3 or any(v.window_order.tolist()!=order for v in views):raise ValueError('Wrong Tri order')
        if sum(p.numel() for p in ref.parameters())!=232993:raise ValueError('Wrong full network')
        return dict(order=order,views=3,parameters=232993)
    old=phase.candidate_rows,phase.configuration,phase.parent,phase.installed,shuffle_source.assert_structure
    phase.candidate_rows,phase.configuration,phase.parent,phase.installed=candidate_rows,configuration,parent,installed
    shuffle_source.assert_structure=graph
    try:return phase.smoke(rt)
    finally:phase.candidate_rows,phase.configuration,phase.parent,phase.installed,shuffle_source.assert_structure=old


def worker(c,kind):
    if kind in ('predict','score'):return evaluation.worker(c,kind)
    if kind not in ('train','smoke'):raise ValueError('Only registered stages allowed')
    from experiments.pamf_repl_completion.runner import activate
    activate(c);return train(c) if kind=='train' else smoke(c)


def dispatch(rt):
    launch=partial(dispatching.launch,module='experiments.pamf_tri_u_channels_rx1.runner')
    dispatching.dispatch(rt,prepare_run=prepare,launch_worker=launch)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker'])
    p.add_argument('--kind');p.add_argument('--config',required=True);a=p.parse_args()
    dispatch(read(a.config)) if a.action=='dispatch' else worker(read(a.config),a.kind)
