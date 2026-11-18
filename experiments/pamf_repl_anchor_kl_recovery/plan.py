"""Identical registered anchor KL rows in an independent recovery run."""
from contextlib import contextmanager

from experiments.pamf_repl_anchor_kl import plan as prior

PREVIOUS_RUN = prior.RUN
RUN = PREVIOUS_RUN.removesuffix('r01') + 'r02'
SETTINGS, SEED, GPUS = prior.SETTINGS, prior.SEED, prior.GPUS
RECIPES = prior.RECIPES
candidate_rows = prior.candidate_rows
MAX_GPU_PROCESSES = 2


@contextmanager
def scoped():
    original = prior.RUN
    prior.RUN = RUN
    try:
        with prior.scoped():
            yield
    finally:
        prior.RUN = original
