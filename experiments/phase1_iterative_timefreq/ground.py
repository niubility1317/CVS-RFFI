"""Load the fixed alignment-trained memory model for Phase1 inference."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_iterative_timefreq.model import IterativeTimeFreq

SOURCE = '20261103-phase1-iterative-timefreq-source-s2026092701-r01'


def main():
    old = base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX
    base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = IterativeTimeFreq, SOURCE, 'phase1_iterative_timefreq_v1', 'ITERATIVE_TIMEFREQ'
    try: base.main()
    finally: base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = old


if __name__ == '__main__': main()
