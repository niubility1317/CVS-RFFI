"""Train cross-view recovery with output-layer gradient balancing."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_context.model import ContextRecovery as DynamicVolterra
from experiments.phase1_dura_safe_memory.objective import protected_source_objective
from experiments.phase1_dura_crosscycle.objective import CYCLE_COUNTS, crossview_objective
from experiments.phase1_dura_gradcycle.objective import GRAD_COUNTS, balanced_objective
from experiments.phase1_dura_volterra.pipeline import load_components, input_noise, predict_batch


def main():
    CYCLE_COUNTS[:] = [0, 0.0]
    GRAD_COUNTS[:] = [0, 0, 0.0, 0]
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective',
             'auxiliary_source_objective', 'combine_source_objectives', 'PAIRED_VIEWS', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
    old = tuple(getattr(base, name) for name in names)
    values = (DynamicVolterra, 'phase1_dura_gradcycle_v1', 'DURA_GRADCYCLE',
              protected_source_objective, crossview_objective, balanced_objective, True, True,
              load_components, input_noise, predict_batch, lambda dura, eye: (dura,))
    for name, value in zip(names, values): setattr(base, name, value)
    try: base.main()
    finally:
        for name, value in zip(names, old): setattr(base, name, value)
        print('DURA_GRADCYCLE_TERM', *CYCLE_COUNTS, flush=True)
        print('DURA_GRADCYCLE_GRAD_COUNTS', *GRAD_COUNTS, flush=True)


if __name__ == '__main__': main()
