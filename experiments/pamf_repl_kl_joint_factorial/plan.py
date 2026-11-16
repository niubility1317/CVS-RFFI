"""Preserve original E20 contracts for a KL-by-joint full matrix."""
from contextlib import contextmanager
from experiments.pamf_repl_attribution_factorial import plan as factorial
from experiments.pamf_repl_component_guard import plan as original

RUN = '20261116-phase1-pamf-repl-kl-joint-factorial-clean-m16-r01'
RECIPES = {f'w{int(w)}_j{int(j>0)}_k{int(k)}':
           {'weight':w, 'joint_weight':j, 'kl_factor':k}
           for w in (6.,3.) for j in (.05,0.) for k in (1.,0.)}


@contextmanager
def scoped():
    old = factorial.RUN, factorial.RECIPES
    factorial.RUN, factorial.RECIPES = RUN, RECIPES
    try:
        with factorial.scoped():
            yield
    finally:
        factorial.RUN, factorial.RECIPES = old


def candidate_rows():
    with scoped():
        return original.candidate_rows()
