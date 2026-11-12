"""Independent six-source-receiver source-development temperature sweep."""
import argparse
from contextlib import contextmanager

from experiments.cric_dura_comparison.io import read
from . import source_tuning_plan as plan

RUN = '20261112-phase1-pamf-prototype-m6-source-s2026092701-r01'


def rows():
    return [dict(row_id=f'm6-prototype-temp-{name}-s2026092701',
                 setting='source_rx_6', protocol_id='m6-natural', model_seed=2026092701,
                 temperature=value)
            for name, value in (('005', .05), ('010', .1), ('020', .2), ('040', .4), ('080', .8))]


@contextmanager
def configured():
    old = plan.RUN, plan.candidate_rows
    plan.RUN, plan.candidate_rows = RUN, rows
    try:
        yield
    finally:
        plan.RUN, plan.candidate_rows = old


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with configured():
        from . import source_tuning_runner as runner
        config = read(args.config)
        runner.dispatch(config) if args.action == 'dispatch' else runner.worker(config, args.kind)
