"""Load the fixed spectral model for source replay and ground inference."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_spectral_coupling.model import SpectralCoupling

SOURCE = '20261102-phase1-spectral-coupling-source-s2026092701-r01'


def main():
    old = base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX
    base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = SpectralCoupling, SOURCE, 'phase1_spectral_coupling_v1', 'SPECTRAL_COUPLING'
    try: base.main()
    finally: base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = old


if __name__ == '__main__': main()
