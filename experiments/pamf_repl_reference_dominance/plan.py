"""Fixed U-weight by reference-dominance matrix."""
from contextlib import contextmanager
from experiments.pamf_repl_step_factor import plan as factor
from experiments.pamf_repl_step_guard import plan as original

RUN = '20261116-phase1-pamf-repl-reference-dominance-clean-m8-r01'
RECIPES = {f'ref_w{int(w)}_c{int(100*c)}': {'weight':w, 'reference_cap':c}
           for w in (6.,3.) for c in (.25,.5)}


@contextmanager
def scoped():
    old = factor.RUN, factor.RECIPES
    factor.RUN, factor.RECIPES = RUN, RECIPES
    try:
        with factor.scoped():
            yield
    finally:
        factor.RUN, factor.RECIPES = old


def candidate_rows():
    with scoped():
        return original.candidate_rows()
