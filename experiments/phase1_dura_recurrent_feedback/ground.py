"""Load the fixed DURA-anchored memory correction for Phase1 inference."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_dura_recurrent_feedback.model import RecurrentFeedbackRecovery as DynamicVolterra
from experiments.phase1_dura_feedback.pipeline import load_components, predict_batch

SOURCE = '20261103-phase1-dura-recurrent-feedback-source-s2026092701-r01'


def main():
    names = ('DynamicVolterra', 'SOURCE', 'SCHEMA', 'LOG_PREFIX', 'USES_DURA', 'load_components', 'predict_batch')
    old = tuple(getattr(base, name) for name in names)
    values = (DynamicVolterra, SOURCE, 'phase1_dura_recurrent_feedback_v1', 'DURA_RECURRENT_FEEDBACK', True, load_components, predict_batch)
    for name, value in zip(names, values): setattr(base, name, value)
    try: base.main()
    finally:
        for name, value in zip(names, old): setattr(base, name, value)


if __name__ == '__main__': main()
