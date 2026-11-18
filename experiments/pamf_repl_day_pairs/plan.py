"""Matched date-aware positive-target probes from the strongest fixed recipes."""
from contextlib import contextmanager

from experiments.pamf_repl_adaptive_kl.plan import ANCHORS, GPUS, SEED, SETTINGS, fixed
from experiments.pamf_repl_priority import plan as priority


RUN = '20261119-phase1-pamf-repl-daypairs-clean-m14-r01'
PROFILES = {
    'joint25': ('joint', .25),
    'joint50': ('joint', .50),
    'joint75': ('joint', .75),
    'same_date50': ('same_date', .50),
    'coverage50': ('coverage', .50),
    'coverage75': ('coverage', .75),
    'date_balance50': ('date_balance', .50),
}
RECIPES = {
    f'{setting}_{name}': dict(fixed.RECIPES[anchor], day_rule=rule, day_strength=strength)
    for setting, (anchor, _) in ANCHORS.items()
    for name, (rule, strength) in PROFILES.items()
}


def candidate_rows():
    return [dict(row_id=f"{recipe['setting']}-repl-{name}-s{SEED}", setting=recipe['setting'],
                 protocol_id=SETTINGS[recipe['setting']], model_seed=SEED, recipe=name, views=['clean'])
            for name, recipe in RECIPES.items()]


@contextmanager
def scoped():
    old = priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows
    priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows = RUN, RECIPES, SETTINGS, candidate_rows
    try:
        with priority.scoped():
            yield
    finally:
        priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows = old
