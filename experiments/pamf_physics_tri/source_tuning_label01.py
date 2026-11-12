"""Independent 1%-labeled source-development temperature pilot."""
import argparse
from contextlib import contextmanager

from experiments.cric_dura_comparison.io import read
from . import source_tuning_plan as plan

RUN = '20261112-phase1-pamf-prototype-label01-source-s2026092701-r01'


def rows():
    return [dict(row_id=f'label01-prototype-temp-{name}-s2026092701',
                 setting='labeled_1pct', protocol_id='label01', model_seed=2026092701,
                 temperature=value)
            for name, value in (('010', .1), ('020', .2), ('040', .4))]


@contextmanager
def configured():
    old = plan.RUN, plan.candidate_rows
    plan.RUN, plan.candidate_rows = RUN, rows
    try:
        yield
    finally:
        plan.RUN, plan.candidate_rows = old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with configured():
        from . import source_tuning_runner as runner
        cfg = read(args.config)
        runner.dispatch(cfg) if args.action == 'dispatch' else runner.worker(cfg, args.kind)


if __name__ == '__main__':
    main()
