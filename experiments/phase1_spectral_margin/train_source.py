"""Train the spectral model with clean-reader distribution and margin guidance."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_spectral_coupling.model import SpectralCoupling
from experiments.phase1_spectral_margin.objective import source_loss


def main():
    old = base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss
    base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss = (
        SpectralCoupling, 'phase1_spectral_margin_v1', 'SPECTRAL_MARGIN', source_loss)
    try: base.main()
    finally: base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss = old


if __name__ == '__main__': main()
