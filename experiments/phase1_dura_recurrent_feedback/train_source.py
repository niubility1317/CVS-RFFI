"""Train shared recurrent IQ refinement on legal source supervision."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_recurrent_feedback.model import RecurrentFeedbackRecovery
from experiments.phase1_dura_feedback.pipeline import load_components, input_noise, predict_batch
from experiments.phase1_dura_feedback.train_source import feedback_source_objective


def main():
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_objective', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
    old = tuple(getattr(base, name) for name in names)
    values = (RecurrentFeedbackRecovery, 'phase1_dura_recurrent_feedback_v1', 'DURA_RECURRENT_FEEDBACK',
              feedback_source_objective, True, load_components, input_noise, predict_batch,
              lambda dura, reader: (dura, reader))
    for name, value in zip(names, values):
        setattr(base, name, value)
    try:
        base.main()
    finally:
        for name, value in zip(names, old):
            setattr(base, name, value)


if __name__ == '__main__':
    main()
