"""Three-seed training, seven-scene frozen prediction and truth-last scoring."""
import argparse
from functools import partial

from experiments.cric_dura_comparison.io import read
from experiments.pamf_physics_tri import runner as tri
from experiments.pamf_physics_branch import runner as physical
from experiments.pamf_physics_route import prepare as routes, runner as dispatching
from .plan import candidate_rows, configuration
from .source import parent, train
from .scope import installed


def smoke(rt):
    old = tri.candidate_rows, tri.configuration, tri.parent, tri.installed
    tri.candidate_rows, tri.configuration, tri.parent, tri.installed = candidate_rows, configuration, parent, installed
    try:
        return tri.smoke(rt)
    finally:
        tri.candidate_rows, tri.configuration, tri.parent, tri.installed = old


def worker(c, kind):
    if kind in ('predict', 'score'):
        return physical.worker(c, kind)
    if kind not in ('train', 'smoke'):
        raise ValueError('Only source and practical evaluation stages are allowed')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    return train(c) if kind == 'train' else smoke(c)


def dispatch(rt):
    prepare = partial(routes.prepare, rows_for_run=lambda _: candidate_rows(), configure=configuration,
                      source_diagnostic=False, tri_dura=True)
    launch = partial(dispatching.launch, module='experiments.pamf_physics_tri_unprotected.runner')
    dispatching.dispatch(rt, prepare_run=prepare, launch_worker=launch)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dispatch', 'worker'])
    p.add_argument('--kind')
    p.add_argument('--config', required=True)
    a = p.parse_args()
    dispatch(read(a.config)) if a.action == 'dispatch' else worker(read(a.config), a.kind)
