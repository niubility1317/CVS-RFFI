"""Load the fixed single-record conditioned three-wave fusion."""
from experiments.phase1_dynamic_volterra import ground as base
from experiments.phase1_dura_tripleblend.model import make_model as DynamicVolterra
from experiments.phase1_dura_feedback.pipeline import load_components, predict_batch

SOURCE = '20261103-phase1-dura-tripleblend-source-s2026092701-r01'


def main():
    names = ('DynamicVolterra', 'SOURCE', 'SCHEMA', 'LOG_PREFIX', 'USES_DURA', 'load_components', 'predict_batch')
    old = tuple(getattr(base, name) for name in names)
    values = (DynamicVolterra, SOURCE, 'phase1_dura_tripleblend_v1', 'DURA_TRIPLEBLEND', True, load_components, predict_batch)
    for name, value in zip(names, values): setattr(base, name, value)
    try: base.main()
    finally:
        for name, value in zip(names, old): setattr(base, name, value)


if __name__ == '__main__': main()
