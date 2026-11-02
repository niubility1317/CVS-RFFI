"""Train source-only three-wave fusion with frozen reader conditioning."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_tripleblend.model import make_model as DynamicVolterra
from experiments.phase1_dura_feedback.pipeline import load_components, input_noise, predict_batch
from experiments.phase1_dura_safe_memory.objective import protected_source_objective


def triple_source_objective(reader, output, packed, clean, clean_features, groups):
    if packed.shape != (len(output), 3, 256):
        raise ValueError('Matching DURA anchor and same-record reader condition required')
    return protected_source_objective(reader, output, packed[:, :2], clean, clean_features, groups)


def main():
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components',
             'extra_frozen_model_components')
    old = tuple(getattr(base, name) for name in names)
    new = (DynamicVolterra, 'phase1_dura_tripleblend_v1', 'DURA_TRIPLEBLEND',
           triple_source_objective, True, load_components, input_noise, predict_batch,
           lambda dura, condition_reader: (dura, condition_reader), lambda model: (model.expert,))
    for name, value in zip(names, new): setattr(base, name, value)
    try: base.main()
    finally:
        for name, value in zip(names, old): setattr(base, name, value)


if __name__ == '__main__': main()
