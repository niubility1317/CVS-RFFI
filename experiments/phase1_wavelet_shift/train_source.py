"""Train the dynamic memory network with aligned-IQ and reader guidance."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_wavelet_shift.model import ShiftAugmentedWavelet
from experiments.phase1_volterra_aligned.objective import source_loss


def main():
    old = base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss
    base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss = (
        ShiftAugmentedWavelet, 'phase1_wavelet_shift_v1', 'WAVELET_SHIFT', source_loss)
    try: base.main()
    finally: base.DynamicVolterra, base.SCHEMA, base.LOG_PREFIX, base.source_loss = old


if __name__ == '__main__': main()
