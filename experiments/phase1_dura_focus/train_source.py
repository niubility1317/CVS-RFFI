"""Train an independently initialized DURA memory correction with source protection."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_context.model import ContextRecovery as DynamicVolterra
from experiments.phase1_dura_focus.objective import SOURCE_WEIGHT_COUNTS, focused_source_objective
from experiments.phase1_dura_volterra.pipeline import load_components, input_noise, predict_batch


def main():
    SOURCE_WEIGHT_COUNTS[:] = [0, 0]
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
    old = tuple(getattr(base, name) for name in names)
    values = (DynamicVolterra, 'phase1_dura_focus_v1', 'DURA_FOCUS',
              focused_source_objective, True, load_components, input_noise, predict_batch,
              lambda dura, eye: (dura,))
    for name, value in zip(names, values): setattr(base, name, value)
    try: base.main()
    finally:
        for name, value in zip(names, old): setattr(base, name, value)
        print("DURA_FOCUS_WEIGHT_COUNTS", *SOURCE_WEIGHT_COUNTS, flush=True)


if __name__ == '__main__': main()
