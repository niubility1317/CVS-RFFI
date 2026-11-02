"""Train a spectral model using paired source-channel views."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_spectral_coupling.model import SpectralCoupling
from experiments.phase1_spectral_crossview.objective import source_loss


def main():
    old = base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss, base.PAIRED_VIEWS
    base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss, base.PAIRED_VIEWS = (
        SpectralCoupling, 'phase1_spectral_crossview_v1', 'SPECTRAL_CROSSVIEW', source_loss, True)
    try: base.main()
    finally: base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss, base.PAIRED_VIEWS = old


if __name__ == '__main__': main()
