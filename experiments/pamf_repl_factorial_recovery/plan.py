"""Reproduce only the four unfinished six-receiver factorial cells."""
from contextlib import contextmanager
from experiments.pamf_repl_component_guard import plan as base

RUN = '20261117-phase1-pamf-repl-factorial-recovery-clean-m4-r01'
PREVIOUS_RUN = '20261116-phase1-pamf-repl-attribution-factorial-clean-m16-r01'
SEED = base.SEED
SETTINGS = {'source_rx_6': 'm6-natural'}
RECIPES = {f'w3_s{s}_e{e}': {'weight': 3., 'shallow_factor': float(s),
                               'extrap_coefficient': .25 * e}
           for s in (1, 0) for e in (1, 0)}


def candidate_rows():
    return [dict(row_id=f'source_rx_6-repl-{name}-s{SEED}', setting='source_rx_6',
                 protocol_id='m6-natural', model_seed=SEED, recipe=name, views=['clean'])
            for name in RECIPES]


@contextmanager
def scoped():
    old = base.RUN, base.RECIPES, base.SETTINGS, base.candidate_rows, base.configuration
    original_configuration = base.configuration
    def configuration(ctx, row, project, commit, *, run_id=RUN):
        return original_configuration(ctx, row, project, commit, run_id=run_id)
    base.RUN, base.RECIPES, base.SETTINGS, base.candidate_rows, base.configuration = (
        RUN, RECIPES, SETTINGS, candidate_rows, configuration)
    try:
        yield
    finally:
        base.RUN, base.RECIPES, base.SETTINGS, base.candidate_rows, base.configuration = old
