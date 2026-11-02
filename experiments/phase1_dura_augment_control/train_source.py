"""Unaugmented matched control for source-channel augmentation."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_context.model import ContextRecovery
from experiments.phase1_dura_volterra.pipeline import load_components, input_noise, predict_batch
from experiments.phase1_volterra_aligned.objective import source_loss


def main():
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_loss', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
    old = tuple(getattr(base, name) for name in names)
    values = (ContextRecovery, 'phase1_dura_augment_control_v1', 'DURA_AUGMENT_CONTROL',
              source_loss, True, load_components, input_noise, predict_batch, lambda dura, eye: (dura,))
    for name, value in zip(names, values):
        setattr(base, name, value)
    try:
        base.main()
    finally:
        for name, value in zip(names, old):
            setattr(base, name, value)


if __name__ == '__main__':
    main()
