"""Train DURA-anchored context recovery using paired source views."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_context.model import ContextRecovery as DynamicVolterra
from experiments.phase1_dura_safe_memory.objective import protected_source_objective
from experiments.phase1_dura_crosscycle.objective import CYCLE_COUNTS, crossview_objective
from experiments.phase1_dura_volterra.pipeline import load_components, input_noise, predict_batch


def main():
    CYCLE_COUNTS[:] = [0, 0.0]
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective',
             'auxiliary_source_objective', 'PAIRED_VIEWS', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
    old = tuple(getattr(base, name) for name in names)
    values = (DynamicVolterra, 'phase1_dura_crosscycle_v1', 'DURA_CROSSCYCLE',
              protected_source_objective, crossview_objective, True, True,
              load_components, input_noise, predict_batch, lambda dura, eye: (dura,))
    for name, value in zip(names, values): setattr(base, name, value)
    try: base.main()
    finally:
        for name, value in zip(names, old): setattr(base, name, value)
        print("DURA_CROSSCYCLE_TERM", *CYCLE_COUNTS, flush=True)


if __name__ == '__main__': main()
