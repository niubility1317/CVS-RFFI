"""Fixed semi-supervised six-row run, followed by truth-last practical scoring."""
import argparse
from functools import partial

from experiments.cric_dura_comparison.io import read
from experiments.pamf_physics_route import prepare as routes
from experiments.pamf_physics_route import runner as dispatching
from .pair_plan import candidate_rows, configuration, parent
from .pairing import installed
from .runner import smoke, worker as existing_worker
from .source import train_trajectory


def worker(c, kind):
    if kind in ('predict', 'score'):
        return existing_worker(c, kind)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == 'train':
        return train_trajectory(c, parent(c), installed, ssl=True, physical_L=False)
    if kind == 'smoke':
        return smoke(c, rows=candidate_rows, configure=configuration, check_parent=parent, scope=installed,
                     check_pairing=True)
    raise ValueError('Only source and clean/practical evaluation stages are allowed')


def dispatch(rt):
    prepare = partial(routes.prepare, rows_for_run=lambda _: candidate_rows(), configure=configuration,
                      source_diagnostic=False, tri_dura=True)
    launch = partial(dispatching.launch, module='experiments.pamf_physics_tri.pair_runner')
    dispatching.dispatch(rt, prepare_run=prepare, launch_worker=launch)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dispatch', 'worker'])
    p.add_argument('--kind')
    p.add_argument('--config', required=True)
    a = p.parse_args()
    dispatch(read(a.config)) if a.action == 'dispatch' else worker(read(a.config), a.kind)
