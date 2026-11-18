"""Seven registered KL routes by two doses in both source settings."""
from contextlib import contextmanager

from experiments.pamf_repl_priority import plan as priority

RUN = '20261118-phase1-pamf-repl-anchorkl-clean-m28-r01'
SEED, SETTINGS = priority.SEED, priority.SETTINGS
GPUS = tuple(range(7))
ROUTES = {
    'cross_symmetric': ('cross', 'ordinary', 1.),
    'raw_train_to_ordinary': ('raw_train', 'ordinary', 1.),
    'eval_to_ordinary': ('raw_eval', 'ordinary', 1.),
    'eval_to_raw': ('raw_eval', 'raw', 1.),
    'eval_to_both': ('raw_eval', 'both', 1.),
    'eval_t2_to_ordinary': ('raw_eval', 'ordinary', 2.),
    'eval_t2_to_both': ('raw_eval', 'both', 2.),
}
RECIPES = {}
for setting, prefix in (('ten_class', 'ten'), ('source_rx_6', 'six')):
    ten = setting == 'ten_class'
    for route, (teacher, students, temperature) in ROUTES.items():
        for level, multiplier in (('half', .5), ('full', 1.)):
            RECIPES[f'{prefix}_{route}_{level}'] = dict(
                setting=setting, weight=.75, shallow_factor=1. if ten else 0.,
                joint_weight=0. if ten else .05, kl_factor=(1. if ten else 1.5) * multiplier,
                extrap_coefficient=0., reference='shallow' if ten else 'pair_kl',
                mode='soft', alpha=0., beta=0., kl_teacher=teacher, kl_students=students,
                kl_temperature=temperature)


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
