"""Three matched supervised physical runs and seven-scene frozen evaluation."""
import argparse
from functools import partial

from experiments.cric_dura_comparison.io import read
from experiments.pamf_physics_branch import runner as physical
from experiments.pamf_physics_route import prepare as routes, runner as dispatching
from experiments.pamf_physics_tri.runner import smoke
from .plan import candidate_rows, configuration
from .source import installed, parent, train


def worker(c, kind):
    if kind in ('predict', 'score'):
        return physical.worker(c, kind)
    if kind not in ('train', 'smoke'):
        raise ValueError('Only L-only source and practical evaluation stages are allowed')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == 'train':
        return train(c)
    return smoke(c, rows=candidate_rows, configure=configuration, check_parent=parent, scope=installed, ssl=False)


def dispatch(rt):
    prepare = partial(routes.prepare, rows_for_run=lambda _: candidate_rows(), configure=configuration,
                      source_diagnostic=False, tri_dura=True)
    launch = partial(dispatching.launch, module='experiments.pamf_physics_sup.runner')
    dispatching.dispatch(rt, prepare_run=prepare, launch_worker=launch)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dispatch', 'worker'])
    p.add_argument('--kind')
    p.add_argument('--config', required=True)
    a = p.parse_args()
    dispatch(read(a.config)) if a.action == 'dispatch' else worker(read(a.config), a.kind)
