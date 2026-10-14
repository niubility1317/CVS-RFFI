from types import SimpleNamespace
from . import design as d, model as api
from experiments.cvs_signature_stats50.feature_check import run as shared


def run(seed=None, only_row=None, *, design=None, build_fn=None):
    d = design or globals()['d']
    build_fn = build_fn or api.build
    value = shared(seed, only_row, design=d, build_fn=build_fn)
    if only_row is None and value['status'] == 'VERIFIED':
        shadow = SimpleNamespace(**vars(d)); shadow.PRIMARY = 'MATCHED_RANDOM'
        other = shared(seed, only_row, design=shadow, build_fn=build_fn)
        if other['status'] != 'VERIFIED' or other['models'] != value['models']:
            raise ValueError('Matched-random feature comparison incomplete')
        value['paired'] += other['paired']
    return value
