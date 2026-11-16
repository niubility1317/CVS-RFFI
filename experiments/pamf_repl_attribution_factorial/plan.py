"""Independent three-factor full matrix for source-led attribution."""
from contextlib import contextmanager
from experiments.pamf_repl_component_guard import plan as original

RUN = '20261116-phase1-pamf-repl-attribution-factorial-clean-m16-r01'
RECIPES = {f'w{int(w)}_s{int(s)}_e{int(e > 0)}':
           {'weight': w, 'shallow_factor': s, 'extrap_coefficient': e}
           for w in (6., 3.) for s in (1., 0.) for e in (.25, 0.)}


@contextmanager
def scoped():
    old = original.RUN, original.RECIPES, original.configuration
    native_configuration = original.configuration
    def configuration(ctx, row, project, commit, *, run_id=RUN):
        return native_configuration(ctx, row, project, commit, run_id=run_id)
    original.RUN, original.RECIPES = RUN, RECIPES
    original.configuration = configuration
    try:
        yield
    finally:
        original.RUN, original.RECIPES, original.configuration = old


def candidate_rows():
    with scoped():
        return original.candidate_rows()
