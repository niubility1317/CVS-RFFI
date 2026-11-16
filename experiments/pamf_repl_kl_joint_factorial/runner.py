"""Run the fixed KL-by-joint attribution matrix with post-freeze U scoring."""
import argparse
from functools import partial
from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_component_guard import runner as base
from experiments.pamf_repl_step_guard import runner as scored
from .plan import scoped
from .objective import installed


def dispatch(runtime):
    from experiments.pamf_branch_training import runner as queue, prepare as preparation
    from experiments.pamf_physics_branch.runner import admits
    from experiments.pamf_physics_route.runner import launch
    old = preparation.prepare, queue.admits, queue.launch, queue.analyze
    preparation.prepare = scored.prepare
    queue.admits = lambda gpu, active, mib: admits(gpu, active, mib, max_total=3)
    queue.launch = partial(launch, module='experiments.pamf_repl_kl_joint_factorial.runner')
    queue.analyze = base.analyze
    try:
        return queue.dispatch(dict(runtime, prioritize_prediction=True))
    finally:
        preparation.prepare, queue.admits, queue.launch, queue.analyze = old


def worker(config, kind):
    if kind == 'score' and config.get('diagnostic_type') == 'U_truth_after_freeze':
        from experiments.pamf_repl_step_guard.u_scoring import score
        return score(config)
    return base.worker(config, kind)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch','worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    with scoped():
        previous = base.installed, scored.plan
        base.installed, scored.plan = installed, base.plan
        try:
            dispatch(config) if args.action == 'dispatch' else worker(config,args.kind)
        finally:
            base.installed, scored.plan = previous
