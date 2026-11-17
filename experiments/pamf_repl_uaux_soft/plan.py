"""Fixed matched-reference, label-conflict and loss-ablation matrix."""
from contextlib import contextmanager
from experiments.pamf_repl_priority import plan as priority

RUN = '20261117-phase1-pamf-repl-uaux-soft-clean-m16-r01'
SEED, SETTINGS = priority.SEED, priority.SETTINGS
RECIPES = {}
for setting in ('ten_class', 'source_rx_6'):
    ten = setting == 'ten_class'
    common = dict(setting=setting, weight=.75, shallow_factor=1. if ten else 0.,
                  extrap_coefficient=0., joint_weight=0. if ten else .05,
                  kl_factor=1. if ten else 1.5, reference='shallow' if ten else 'pair_kl')
    prefix = 'ten' if ten else 'six'
    for alpha in (0., .25, .5, 1.):
        RECIPES[f'{prefix}_alpha{str(alpha).replace(".", "p")}'] = dict(common, mode='soft', alpha=alpha, beta=0.)
    for beta in (.5, 1.):
        RECIPES[f'{prefix}_beta{str(beta).replace(".", "p")}'] = dict(common, mode='soft', alpha=0., beta=beta)
RECIPES['ten_shallow0'] = dict(RECIPES['ten_alpha0p0'], shallow_factor=0.)
RECIPES['ten_kl0'] = dict(RECIPES['ten_alpha0p0'], kl_factor=0.)
RECIPES['six_kl0_native'] = dict(RECIPES['six_alpha0p0'], kl_factor=0., mode='native')
RECIPES['six_kl2_native'] = dict(RECIPES['six_alpha0p0'], kl_factor=2., mode='native')


def candidate_rows():
    return [dict(row_id=f"{r['setting']}-repl-{name}-s{SEED}", setting=r['setting'],
                 protocol_id=SETTINGS[r['setting']], model_seed=SEED, recipe=name, views=['clean'])
            for name, r in RECIPES.items()]


@contextmanager
def scoped():
    old = priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows
    priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows = RUN, RECIPES, SETTINGS, candidate_rows
    try:
        with priority.scoped():
            yield
    finally:
        priority.RUN, priority.RECIPES, priority.SETTINGS, priority.candidate_rows = old
