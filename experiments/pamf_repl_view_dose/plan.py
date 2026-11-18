"""Fixed source-only continuation matrix for two independent view contrasts."""
from contextlib import contextmanager

from experiments.pamf_repl_priority import plan as priority


RUN = '20261118-phase1-pamf-repl-viewdose-clean-m12-r01'
SEED, SETTINGS = priority.SEED, priority.SETTINGS
ROUTES = {'ordinary': (1., 0., 0., 0.), 'raw': (0., 0., 0., 1.),
          'ordinary_raw': (.5, 0., 0., .5)}
ORIGINAL_KL = (.4, .2, 0., .2)
RECIPES = {}
for setting, prefix in (('ten_class', 'ten'), ('source_rx_6', 'six')):
    ten = setting == 'ten_class'
    common = dict(setting=setting, weight=.75, shallow_factor=1. if ten else 0.,
                  joint_weight=0. if ten else .05, kl_factor=1. if ten else 1.5,
                  extrap_coefficient=0., reference='shallow' if ten else 'pair_kl',
                  mode='soft', alpha=0., beta=0.)
    for route, weights in ROUTES.items():
        RECIPES[f'{prefix}_ce_{route}'] = dict(common, ce_views=list(weights), kl_views=list(ORIGINAL_KL))
        RECIPES[f'{prefix}_kl_{route}'] = dict(common, ce_views=[0.] * 4,
                                              kl_views=[.8 * w for w in weights])


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
