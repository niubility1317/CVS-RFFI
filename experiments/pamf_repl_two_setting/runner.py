import argparse
from functools import partial

from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_component_guard import plan as base_plan, runner as base
from experiments.pamf_repl_priority import runner as priority
from experiments.pamf_repl_source_adapt import transfer
from experiments.pamf_repl_step_guard.runner import add_u_scoring
from . import plan
from .objective import installed


def prepare(rt):
    rows = plan.candidate_rows()
    return add_u_scoring(rt, rows, transfer.prepare_jobs(rt, rows, base_plan.configuration))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    with plan.scoped():
        old = base.installed, priority.installed
        base.installed = priority.installed = installed
        try:
            if args.action == 'dispatch':
                analyze = partial(transfer.analyze, source_decision_run=None,
                                  interpretation='ten-class factorial development and six-source fixed held-seed verification')
                transfer.dispatch(config, prepare_fn=prepare, analyze_fn=analyze,
                                  module='experiments.pamf_repl_two_setting.runner')
            else:
                priority.worker(config, args.kind)
        finally:
            base.installed, priority.installed = old
