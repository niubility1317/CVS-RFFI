"""Load the fixed margin-trained spectral model for Phase1 inference."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_spectral_coupling.model import SpectralCoupling

SOURCE = '20261102-phase1-spectral-margin-source-s2026092701-r01'


def main():
    old = base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX
    base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = SpectralCoupling, SOURCE, 'phase1_spectral_margin_v1', 'SPECTRAL_MARGIN'
    try: base.main()
    finally: base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = old


if __name__ == '__main__': main()
