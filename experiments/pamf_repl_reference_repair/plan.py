"""Fixed .25-cap comparison against the original full-L reference."""
from contextlib import contextmanager
from experiments.pamf_repl_reference_dominance import plan as reference
from experiments.pamf_repl_step_guard import plan as original

RUN='20261116-phase1-pamf-repl-reference-repair-clean-m4-r01'
RECIPES={f'repair_w{int(w)}_c25':{'weight':w,'reference_cap':.25} for w in (6.,3.)}


@contextmanager
def scoped():
    old=reference.RUN,reference.RECIPES;reference.RUN,reference.RECIPES=RUN,RECIPES
    try:
        with reference.scoped():yield
    finally:reference.RUN,reference.RECIPES=old


def candidate_rows():
    with scoped():return original.candidate_rows()
