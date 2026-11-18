"""Launch the view variants through the established clean-only workflow."""
import argparse

from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_component_guard import plan as base_plan, runner as base
from experiments.pamf_repl_dual_optimization.runner import dispatch
from experiments.pamf_repl_priority import runner as priority
from experiments.pamf_repl_step_guard import runner as scored
from . import plan
from .objective import installed


def main(matrix=plan, worker_module='experiments.pamf_repl_view_dose.runner',
         *, objective=installed, max_total=5, reserve_mib=12288, prepare_jobs=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    with matrix.scoped():
        original = base.installed, scored.plan, priority.installed
        base.installed, scored.plan, priority.installed = objective, base_plan, objective
        try:
            if args.action == 'dispatch':
                dispatch(config, worker_module=worker_module, max_total=max_total,
                         reserve_mib=reserve_mib, prepare_jobs=prepare_jobs)
            else:
                priority.worker(config, args.kind)
        finally:
            base.installed, scored.plan, priority.installed = original


if __name__ == '__main__':
    main()
