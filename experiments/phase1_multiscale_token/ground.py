"""Load the fixed alignment-trained memory model for Phase1 inference."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_multiscale_token.model import MultiScaleToken

SOURCE = '20261103-phase1-multiscale-token-source-s2026092701-r01'


def main():
    old = base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX
    base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = MultiScaleToken, SOURCE, 'phase1_multiscale_token_v1', 'MULTISCALE_TOKEN'
    try: base.main()
    finally: base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = old


if __name__ == '__main__': main()
