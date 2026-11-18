"""Cross the base CE dose with auxiliary presence in each source setting."""
from contextlib import contextmanager

from experiments.pamf_repl_priority import plan as priority

RUN = '20261118-phase1-pamf-repl-lossdose-clean-m16-r01'
SEED, SETTINGS = priority.SEED, priority.SETTINGS
RECIPES = {}
for setting, prefix in (('ten_class', 'ten'), ('source_rx_6', 'six')):
    ten_class = setting == 'ten_class'
    for auxiliary in (1., 0.):
        for ce in (1., .5, .25, 0.):
            dose = str(ce).replace('.', 'p')
            RECIPES[f'{prefix}_aux{int(auxiliary)}_ce{dose}'] = dict(
                setting=setting, weight=.75, shallow_factor=1. if ten_class else 0.,
                joint_weight=0. if ten_class else .05, kl_factor=1. if ten_class else 1.5,
                extrap_coefficient=0., reference='shallow' if ten_class else 'pair_kl',
                mode='soft', alpha=0., beta=0., ce_dose=ce, auxiliary_dose=auxiliary)


def candidate_rows():
    return [dict(row_id=f"{config['setting']}-repl-{name}-s{SEED}", setting=config['setting'],
                 protocol_id=SETTINGS[config['setting']], model_seed=SEED, recipe=name, views=['clean'])
            for name, config in RECIPES.items()]


@contextmanager
def scoped():
    original = priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows
    priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows = RUN, RECIPES, SETTINGS, candidate_rows
    try:
        with priority.scoped():
            yield
    finally:
        priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows = original
