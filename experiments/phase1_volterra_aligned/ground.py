"""Load the fixed alignment-trained memory model for Phase1 inference."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_dynamic_volterra.model import DynamicVolterra

SOURCE = '20261102-phase1-volterra-aligned-source-s2026092701-r01'


def main():
    old = base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX
    base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = DynamicVolterra, SOURCE, 'phase1_volterra_aligned_v1', 'VOLTERRA_ALIGNED'
    try: base.main()
    finally: base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = old


if __name__ == '__main__': main()
