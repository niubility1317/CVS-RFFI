"""Load the fixed DURA-anchored spectral coupling correction for Phase1 inference."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_spectral_coupling.model import SpectralCoupling
from experiments.phase1_dura_volterra.pipeline import load_components, predict_batch

SOURCE = '20261102-phase1-dura-spectral-residual-source-s2026092701-r01'


def main():
    names = ('DynamicVolterra', 'SOURCE', 'SCHEMA', 'LOG_PREFIX', 'USES_DURA', 'load_components', 'predict_batch')
    old = tuple(getattr(base, name) for name in names)
    values = (SpectralCoupling, SOURCE, 'phase1_dura_spectral_residual_v1', 'DURA_SPECTRAL_RESIDUAL', True, load_components, predict_batch)
    for name, value in zip(names, values): setattr(base, name, value)
    try: base.main()
    finally:
        for name, value in zip(names, old): setattr(base, name, value)


if __name__ == '__main__': main()
