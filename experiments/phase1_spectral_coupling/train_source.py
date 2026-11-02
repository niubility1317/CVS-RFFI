"""Use the source-only direct-IQ trainer with the spectral coupling model."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_spectral_coupling.model import SpectralCoupling


def main():
    old = base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX
    base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX = SpectralCoupling, 'phase1_spectral_coupling_v1', 'SPECTRAL_COUPLING'
    try: base.main()
    finally: base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX = old


if __name__ == '__main__': main()
