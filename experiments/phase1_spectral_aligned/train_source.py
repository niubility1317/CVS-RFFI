"""Train the dynamic memory network with aligned-IQ and reader guidance."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_spectral_coupling.model import SpectralCoupling
from experiments.phase1_spectral_aligned.objective import source_loss


def main():
    old = base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss
    base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss = (
        SpectralCoupling, 'phase1_spectral_aligned_v1', 'SPECTRAL_ALIGNED', source_loss)
    try: base.main()
    finally: base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss = old


if __name__ == '__main__': main()
