"""Ten-class donor-by-dose matrix with matched original donor controls."""
from contextlib import contextmanager

from experiments.pamf_repl_adaptive_kl import plan as adaptive
from experiments.pamf_repl_priority import plan as priority


RUN = '20261120-phase1-pamf-repl-styledose-clean-m9-r01'
GPUS = tuple(range(7))
SEED = adaptive.SEED
SETTINGS = {'ten_class': adaptive.SETTINGS['ten_class']}
ANCHOR = adaptive.RECIPES['ten_class_vote_hard1']
RECIPES = {
    f'style_{rule}_{dose}': dict(ANCHOR, date_donor=rule, weight=weight)
    for rule in ('original', 'same_day', 'cross_day')
    for dose, weight in (('u0375', .375), ('u075', .75), ('u15', 1.5))
}


def candidate_rows():
    return [dict(row_id=f'ten_class-repl-{name}-s{SEED}', setting='ten_class',
                 protocol_id=SETTINGS['ten_class'], model_seed=SEED, recipe=name, views=['clean'])
            for name in RECIPES]


@contextmanager
def scoped():
    old = priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows
    priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows = RUN, RECIPES, SETTINGS, candidate_rows
    try:
        with priority.scoped():
            yield
    finally:
        priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows = old
