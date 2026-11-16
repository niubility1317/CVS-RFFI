"""Fixed two-anchor, two-weight exploratory matrix."""
from contextlib import contextmanager
from experiments.pamf_repl_step_factor import plan as factor
from experiments.pamf_repl_step_guard import plan as original

RUN = '20261116-phase1-pamf-repl-dual-ce-clean-m4-r01'
RECIPES = {'dual_ce6': {'weight': 6.0}, 'dual_ce3': {'weight': 3.0}}


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
