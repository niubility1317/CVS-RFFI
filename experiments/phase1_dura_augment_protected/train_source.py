"""Train a DURA correction using additional source-only channels."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_augment_protected.augmentation import SourceChannelAugment
from experiments.phase1_dura_context.model import ContextRecovery
from experiments.phase1_dura_volterra.pipeline import load_components, input_noise, predict_batch
from experiments.phase1_volterra_aligned.objective import source_loss
from experiments.phase1_dura_safe_memory.objective import protected_source_objective


def main():
    augment = SourceChannelAugment()
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_loss', 'source_objective', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components',
             'prepare_training_input')
    old = tuple(getattr(base, name) for name in names)
    values = (ContextRecovery, 'phase1_dura_augment_protected_v1', 'DURA_AUGMENT_PROTECTED', source_loss, protected_source_objective, True,
              load_components, input_noise, predict_batch, lambda dura, eye: (dura,), augment)
    for name, value in zip(names, values):
        setattr(base, name, value)
    try:
        base.main()
        if augment.total == 0 or augment.augmented == 0:
            raise ValueError('Source augmentation must be exercised')
        print('DURA_AUGMENT_PROTECTED_SOURCE_COUNTS', augment.total, augment.augmented, flush=True)
    finally:
        for name, value in zip(names, old):
            setattr(base, name, value)


if __name__ == '__main__':
    main()
