"""Ordinary CE reference matched to the fixed full-L reference matrix."""
from contextlib import contextmanager
from experiments.pamf_repl_reference_dominance import plan as reference
from experiments.pamf_repl_step_guard import plan as original

RUN='20261116-phase1-pamf-repl-ce-reference-clean-m8-r01'
RECIPES={f'ceref_w{int(w)}_c{int(100*c)}':{'weight':w,'reference_cap':c}
         for w in (6.,3.) for c in (.25,.5)}


@contextmanager
def scoped():
    old=reference.RUN,reference.RECIPES
    reference.RUN,reference.RECIPES=RUN,RECIPES
    try:
        with reference.scoped():
            yield
    finally:
        reference.RUN,reference.RECIPES=old


def candidate_rows():
    with scoped():return original.candidate_rows()
