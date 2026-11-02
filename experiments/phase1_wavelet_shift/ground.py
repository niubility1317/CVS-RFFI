"""Load the fixed alignment-trained memory model for Phase1 inference."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_wavelet_shift.model import ShiftAugmentedWavelet

SOURCE = '20261103-phase1-wavelet-shift-source-s2026092701-r01'


def main():
    old = base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX
    base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = ShiftAugmentedWavelet, SOURCE, 'phase1_wavelet_shift_v1', 'WAVELET_SHIFT'
    try: base.main()
    finally: base.DynamicVolterra, base.SOURCE, base.SCHEMA, base.LOG_PREFIX = old


if __name__ == '__main__': main()
