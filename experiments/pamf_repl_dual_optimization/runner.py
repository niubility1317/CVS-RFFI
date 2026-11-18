"""Run new fixed matrix through the previously validated diagnostic chain."""
import argparse
from functools import partial
from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_component_guard import plan as base_plan, runner as base
from experiments.pamf_repl_step_guard import runner as scored
from experiments.pamf_repl_priority import runner as priority
from experiments.pamf_repl_uaux_soft.objective import installed
from . import plan

def dispatch(rt, *, worker_module='experiments.pamf_repl_dual_optimization.runner',
             max_total=5, reserve_mib=12288):
    from experiments.pamf_branch_training import runner as queue, prepare as preparation
    from experiments.pamf_physics_branch.runner import admits as physical_admits
    from experiments.pamf_physics_route.runner import launch
    def admits(gpu, active, mib):
        return physical_admits(gpu, active, max(mib, reserve_mib), max_total=max_total)
    old = preparation.prepare, queue.admits, queue.launch, queue.analyze
    preparation.prepare, queue.admits = scored.prepare, admits
    queue.launch = partial(launch, module=worker_module)
    queue.analyze = base.analyze
    try:
        return queue.dispatch(dict(rt, prioritize_prediction=True))
    finally:
        preparation.prepare, queue.admits, queue.launch, queue.analyze = old

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['dispatch', 'worker'])
    p.add_argument('--kind'); p.add_argument('--config', required=True)
    args = p.parse_args(); c = read(args.config)
    with plan.scoped():
        old = base.installed, scored.plan, priority.installed
        base.installed, scored.plan, priority.installed = installed, base_plan, installed
        try:
            dispatch(c) if args.action == 'dispatch' else priority.worker(c, args.kind)
        finally:
            base.installed, scored.plan, priority.installed = old
