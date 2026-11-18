"""Axis-matched source-only continuation with one frozen anchor per setting."""
from contextlib import contextmanager

from experiments.pamf_repl_priority import plan as priority


RUN = '20261118-phase1-pamf-repl-auxviewdose-clean-m16-r01'
SEED, SETTINGS = priority.SEED, priority.SETTINGS
FAMILIES = {
    'ten_class': dict(prefix='ten', auxiliary='shallow_factor', aux_values=(.5, 1., 2.),
                      kl_values=(.5, 1., 1.5), anchor=(1., 1.), kl_views=[.8, 0., 0., 0.]),
    'source_rx_6': dict(prefix='six', auxiliary='joint_weight', aux_values=(0., .05, .10),
                        kl_values=(.75, 1.5, 2.25), anchor=(.05, 1.5), kl_views=[0., 0., 0., .8]),
}
RECIPES = {}
for setting, axes in FAMILIES.items():
    for auxiliary in axes['aux_values']:
        for kl in axes['kl_values']:
            if (auxiliary, kl) == axes['anchor']:
                continue
            name = f"{axes['prefix']}_aux{auxiliary:g}_kl{kl:g}".replace('.', 'p')
            recipe = dict(setting=setting, weight=.75, shallow_factor=0., joint_weight=0.,
                          kl_factor=kl, extrap_coefficient=0., ce_views=[0.] * 4,
                          kl_views=axes['kl_views'].copy(),
                          reference='shallow' if setting == 'ten_class' else 'pair_kl',
                          mode='soft', alpha=0., beta=0.)
            recipe[axes['auxiliary']] = auxiliary
            RECIPES[name] = recipe


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
