"""Run fixed FP32 reference repair comparisons on idle GPUs only."""
import argparse
from functools import partial
from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_step_guard import runner as original
from .plan import scoped
from .objective import installed


def dispatch(runtime):
    from experiments.pamf_branch_training import runner as queue,prepare as preparation
    from experiments.pamf_physics_branch.runner import admits
    from experiments.pamf_physics_route.runner import launch
    old=preparation.prepare,queue.admits,queue.launch,queue.analyze
    preparation.prepare=original.prepare
    queue.admits=lambda gpu,active,mib:admits(gpu,active,mib,max_total=1)
    queue.launch=partial(launch,module='experiments.pamf_repl_reference_repair.runner')
    queue.analyze=original.analyze
    try:return queue.dispatch(dict(runtime,prioritize_prediction=True))
    finally:preparation.prepare,queue.admits,queue.launch,queue.analyze=old


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker'])
    p.add_argument('--kind');p.add_argument('--config',required=True);a=p.parse_args();c=read(a.config)
    with scoped():
        previous=original.installed;original.installed=installed
        try:dispatch(c) if a.action=='dispatch' else original.worker(c,a.kind)
        finally:original.installed=previous
