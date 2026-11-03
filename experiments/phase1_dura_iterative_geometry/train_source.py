"""DURA-anchored iterative restoration with source recognition protection."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_iterative_timefreq.model import IterativeDura as DynamicVolterra
from experiments.phase1_dura_iterative_geometry.objective import geometry_protected_objective
from experiments.phase1_dura_volterra.pipeline import load_components, input_noise, predict_batch


def main():
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
    old = tuple(getattr(base, name) for name in names)
    values = (DynamicVolterra, 'phase1_dura_iterative_geometry_v1', 'DURA_ITERATIVE_GEOMETRY',
              geometry_protected_objective, True, load_components, input_noise, predict_batch,
              lambda dura, eye: (dura,))
    for name, value in zip(names, values):
        setattr(base, name, value)
    try:
        base.main()
    finally:
        for name, value in zip(names, old):
            setattr(base, name, value)


if __name__ == '__main__':
    main()
