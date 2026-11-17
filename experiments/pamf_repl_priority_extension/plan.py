"""Four distinct six-source low-weight KL factor cells."""
from contextlib import contextmanager
from experiments.pamf_repl_priority import plan as priority

RUN='20261117-phase1-pamf-repl-lowweight-kl-clean-m4-r01'
SEED=priority.SEED
SETTINGS={'source_rx_6':'m6-natural'}
RECIPES={f'six_w0p75_k{str(kl).replace(".","p")}_{mode}':
         dict(setting='source_rx_6',weight=.75,shallow_factor=0.,extrap_coefficient=0.,
              joint_weight=.05,kl_factor=kl,reference='pair_kl',mode=mode)
         for kl in (1.,1.5) for mode in ('native','kl')}


def candidate_rows():
    return [dict(row_id=f'source_rx_6-repl-{name}-s{SEED}',setting='source_rx_6',
                 protocol_id='m6-natural',model_seed=SEED,recipe=name,views=['clean']) for name in RECIPES]


@contextmanager
def scoped():
    old=priority.RUN,priority.RECIPES,priority.SETTINGS,priority.candidate_rows
    priority.RUN,priority.RECIPES,priority.SETTINGS,priority.candidate_rows=RUN,RECIPES,SETTINGS,candidate_rows
    try:
        with priority.scoped():yield
    finally:
        priority.RUN,priority.RECIPES,priority.SETTINGS,priority.candidate_rows=old
