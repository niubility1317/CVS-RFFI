"""Fixed setting-specific follow-up using existing soft auxiliary projections."""
from contextlib import contextmanager
from experiments.pamf_repl_priority import plan as priority

RUN = '20261118-phase1-pamf-repl-dual-clean-m16-r01'
SEED, SETTINGS = priority.SEED, priority.SETTINGS
RECIPES = {}

def add(name, setting, *, alpha, beta, weight=.75, kl=None):
    ten = setting == 'ten_class'
    RECIPES[name] = dict(setting=setting, weight=weight, shallow_factor=1. if ten else 0.,
        extrap_coefficient=0., joint_weight=0. if ten else .05,
        kl_factor=(1. if ten else 1.5) if kl is None else kl,
        reference='shallow' if ten else 'pair_kl', mode='soft', alpha=alpha, beta=beta)

for beta in (.25, .5, 1.):
    add(f'ten_a05_b{str(beta).replace(".", "p")}', 'ten_class', alpha=.5, beta=beta)
add('ten_a1_b05', 'ten_class', alpha=1., beta=.5)
for weight in (.5, 1.):
    add(f'ten_a05_b05_w{str(weight).replace(".", "p")}', 'ten_class', alpha=.5, beta=.5, weight=weight)
for kl in (.5, 1.5):
    add(f'ten_a05_b05_kl{str(kl).replace(".", "p")}', 'ten_class', alpha=.5, beta=.5, kl=kl)

for alpha in (.25, .5, 1.):
    add(f'six_a{str(alpha).replace(".", "p")}_b1', 'source_rx_6', alpha=alpha, beta=1.)
add('six_a05_b05', 'source_rx_6', alpha=.5, beta=.5)
for kl in (1., 2.):
    add(f'six_b1_kl{str(kl).replace(".", "p")}', 'source_rx_6', alpha=0., beta=1., kl=kl)
for weight in (.5, 1.):
    add(f'six_b1_w{str(weight).replace(".", "p")}', 'source_rx_6', alpha=0., beta=1., weight=weight)

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
