"""Train a DURA-anchored flow with reader guidance and source velocity matching."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_flow.model import DuraFlow, flow_matching_loss
from experiments.phase1_dura_safe_memory.objective import protected_source_objective
from experiments.phase1_dura_volterra.pipeline import load_components, input_noise, predict_batch

_current = [None]


def make_model():
    _current[0] = DuraFlow()
    return _current[0]


def flow_training_objective(reader, output, anchor, clean, clean_features, groups, received, log_variance):
    if _current[0] is None:
        raise RuntimeError('Current flow model required')
    protected, terms = protected_source_objective(reader, output, anchor, clean, clean_features, groups)
    velocity = flow_matching_loss(_current[0], received, anchor, clean, log_variance, groups)
    return protected + 0.25 * velocity, dict(terms, flow_matching=velocity)


def main():
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'training_source_objective', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
    old = tuple(getattr(base, name) for name in names)
    new = (make_model, 'phase1_dura_flow_v1', 'DURA_FLOW', flow_training_objective, True,
           load_components, input_noise, predict_batch, lambda dura, eye: (dura,))
    for name, value in zip(names, new): setattr(base, name, value)
    try: base.main()
    finally:
        for name, value in zip(names, old): setattr(base, name, value)
        _current[0] = None


if __name__ == '__main__': main()
