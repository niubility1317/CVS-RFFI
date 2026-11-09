"""Freeze declared predictions first; independent CPU truth-last scoring."""
from contextlib import contextmanager
from .plan import TASKS
from .model import load_reader, ce_scope


@contextmanager
def selected_native(row):
    from experiments.pamf_single_view_paper import evaluation as native
    old=(native.TASKS,native.READOUTS,native.load_reader)
    native.TASKS,native.READOUTS,native.load_reader=TASKS,tuple(row['readouts']),load_reader
    try: yield native
    finally: native.TASKS,native.READOUTS,native.load_reader=old


def predict(c):
    if c['paper_row']['family']=='architecture':
        from experiments.standard_ce_baselines.predict import predict as native
        with ce_scope(): return native(c)
    with selected_native(c['paper_row']) as native:
        return native.predict(c)


def score(c):
    if any(c.get(k) for k in ('source_output','source_cache','restored_caches','frozen_sources','dataset')):
        raise ValueError('Independent truth-side scoring only')
    if c['paper_row']['family']=='architecture':
        from experiments.pamf_paper_completion.evaluation import score_ce
        return score_ce(c)
    with selected_native(c['paper_row']) as native:
        return native.score(c)
