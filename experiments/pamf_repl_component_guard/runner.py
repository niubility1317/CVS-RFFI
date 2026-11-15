"""Independent additional run; no state mutation of existing owners."""
import argparse
from functools import partial
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_repl_weight import runner as prior
from . import plan
from .objective import installed


def prepare(rt):
    old=prior.plan;prior.plan=plan
    try:return prior.prepare(rt)
    finally:prior.plan=old


def smoke(rt):
    from experiments.pamf_physics_tri.runner import smoke as checks
    def check_u(state):
        if not state['accepted'] or state['U_truth_read'] or state['U_gradient_guard']['calls']!=1:
            raise ValueError('Original accepted-U and guard must run')
        if sum(state['U_pseudo_observation']['accepted_by_pseudo_class'].values())!=state['accepted']:
            raise ValueError('Accepted histogram mismatch')
        return dict(accepted=state['accepted'],guard=state['U_gradient_guard'])
    values=checks(rt,rows=plan.candidate_rows,configure=plan.configuration,check_parent=plan.parent,
        scope=installed,check_u=check_u,graph_check=plan.check_graph,segment_output=True,write_result=False)
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',checks=values,real_checkpoint=True,
        optimizer_updates=0,query_reads=0,U_truth_reads=0))


def worker(c,kind):
    if kind=='score':
        from experiments.pamf_paper_completion.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind=='smoke':return smoke(c)
    if kind=='train':
        from experiments.pamf_physics_tri.source import train_trajectory
        return train_trajectory(c,plan.parent(c),installed,ssl=True,physical_L=False)
    if kind=='predict':
        from experiments.pamf_physics_tri.development_evaluation import predict
        return predict(c)
    raise ValueError('Unknown stage')


def analyze(rt):
    old=prior.plan;prior.plan=plan
    try:return prior.analyze(rt)
    finally:prior.plan=old


def dispatch(rt):
    from experiments.pamf_branch_training import runner as queue,prepare as preparation
    from experiments.pamf_physics_branch.runner import admits
    from experiments.pamf_physics_route.runner import launch
    old=preparation.prepare,queue.admits,queue.launch,queue.analyze
    preparation.prepare=prepare
    queue.admits=lambda gpu,active,mib:admits(gpu,active,mib,max_total=5)
    queue.launch=partial(launch,module='experiments.pamf_repl_component_guard.runner')
    queue.analyze=analyze
    try:return queue.dispatch(dict(rt,prioritize_prediction=True))
    finally:preparation.prepare,queue.admits,queue.launch,queue.analyze=old


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
