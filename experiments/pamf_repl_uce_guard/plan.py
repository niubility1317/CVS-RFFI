"""Matched CE-projection strengths with unchanged source contracts."""
from contextlib import contextmanager
from experiments.pamf_repl_priority import plan as priority

RUN='20261118-phase1-pamf-repl-uceguard-clean-m16-r01'
SEED, SETTINGS = priority.SEED, priority.SETTINGS
RECIPES={}
def add(name, setting, *, alpha, beta, gamma, weight=.75):
    ten=setting=='ten_class'
    RECIPES[name]=dict(setting=setting,weight=weight,shallow_factor=1. if ten else 0.,
        extrap_coefficient=0.,joint_weight=0. if ten else .05,kl_factor=1. if ten else 1.5,
        reference='shallow' if ten else 'pair_kl',mode='soft',alpha=alpha,beta=beta,gamma=gamma)
for setting,pre in (('ten_class','ten'),('source_rx_6','six')):
    ten=setting=='ten_class'
    for name,alpha,beta in (('raw',0.,0.),('aux',.5,.5) if ten else ('aux',0.,1.)):
        for gamma in (.25,.5,1.):
            add(f'{pre}_{name}_g{str(gamma).replace(".","p")}',setting,alpha=alpha,beta=beta,gamma=gamma)
    for weight in (.5,1.):
        add(f'{pre}_aux_g05_w{str(weight).replace(".","p")}',setting,
            alpha=.5 if ten else 0.,beta=.5 if ten else 1.,gamma=.5,weight=weight)
def candidate_rows():
    return [dict(row_id=f"{r['setting']}-repl-{name}-s{SEED}",setting=r['setting'],
                 protocol_id=SETTINGS[r['setting']],model_seed=SEED,recipe=name,views=['clean'])
            for name,r in RECIPES.items()]
@contextmanager
def scoped():
    old=priority.RUN,priority.RECIPES,priority.SETTINGS,priority.candidate_rows
    priority.RUN,priority.RECIPES,priority.SETTINGS,priority.candidate_rows=RUN,RECIPES,SETTINGS,candidate_rows
    try:
        with priority.scoped(): yield
    finally:
        priority.RUN,priority.RECIPES,priority.SETTINGS,priority.candidate_rows=old
