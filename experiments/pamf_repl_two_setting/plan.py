from contextlib import contextmanager
from itertools import product

from experiments.pamf_repl_adaptive_kl import plan as adaptive
from experiments.pamf_repl_priority import plan as priority


RUN = '20261119-phase1-pamf-repl-twosetting-clean-m12-r01'
GPUS = adaptive.GPUS
SETTINGS = adaptive.SETTINGS
EXPLORE_SEED = 2026092701
HELD_SEEDS = (2026092702, 2026092703)
TEN_ANCHOR = adaptive.fixed.RECIPES['ten_aux2_kl1p5']
SIX_ANCHOR = adaptive.fixed.RECIPES['six_aux0p1_kl1p5']
RECIPES = {
    f'ten_ce{ce}_gap{gap}_rx{rx}': dict(TEN_ANCHOR, raw_ce_dose=.25 * ce, adaptive_kl_view='ordinary',
        adaptive_signal='view_gap', adaptive_strength=.5 * gap, rx_balance=bool(rx))
    for ce, gap, rx in product((0, 1), repeat=3)
}
RECIPES.update(six_fixed=SIX_ANCHOR.copy(), six_same_date=dict(SIX_ANCHOR, day_rule='same_date', day_strength=.5))


def candidate_rows():
    return [dict(row_id=f"{recipe['setting']}-repl-{name}-s{seed}", setting=recipe['setting'],
                 protocol_id=SETTINGS[recipe['setting']], model_seed=seed, recipe=name,
                 views=['clean'], reuses_original=False)
            for name, recipe in RECIPES.items()
            for seed in ((EXPLORE_SEED,) if recipe['setting'] == 'ten_class' else HELD_SEEDS)]


@contextmanager
def scoped():
    original = priority.RUN, priority.SETTINGS, priority.RECIPES, priority.candidate_rows
    priority.RUN, priority.SETTINGS, priority.RECIPES, priority.candidate_rows = RUN, SETTINGS, RECIPES, candidate_rows
    try:
        with priority.scoped():
            yield
    finally:
        priority.RUN, priority.SETTINGS, priority.RECIPES, priority.candidate_rows = original
