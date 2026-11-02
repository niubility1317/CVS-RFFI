"""Load the fixed latent-consistency model for single-record inference."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_invariant_latent.model import InvariantLatentRecovery

SOURCE = '20261103-phase1-invariant-latent-source-s2026092701-r01'


def main():
    old = base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX
    base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = (
        InvariantLatentRecovery, SOURCE, 'phase1_invariant_latent_v1', 'INVARIANT_LATENT')
    try:
        base.main()
    finally:
        base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = old


if __name__ == '__main__':
    main()
