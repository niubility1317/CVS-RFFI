"""Hold the intervention to accepted-sample weights on the two fixed anchors."""
from contextlib import contextmanager

from experiments.pamf_repl_aux_view_dose import plan as fixed
from experiments.pamf_repl_priority import plan as priority


RUN = '20261119-phase1-pamf-repl-adaptivekl-clean-m14-r01'
GPUS = tuple(range(7))
SEED, SETTINGS = fixed.SEED, fixed.SETTINGS
ANCHORS = {'ten_class': ('ten_aux2_kl1p5', 'ordinary'), 'source_rx_6': ('six_aux0p1_kl1p5', 'raw')}
PROFILES = {
    'vote_good05': ('vote', .5),
    'vote_good1': ('vote', 1.),
    'vote_hard05': ('vote', -.5),
    'vote_hard1': ('vote', -1.),
    'gap_low05': ('view_gap', -.5),
    'gap_high05': ('view_gap', .5),
    'rx_balance05': ('rx_balance', .5),
}
RECIPES = {
    f'{setting}_{name}': dict(fixed.RECIPES[anchor], adaptive_kl_view=view,
                             adaptive_signal=signal, adaptive_strength=strength)
    for setting, (anchor, view) in ANCHORS.items()
    for name, (signal, strength) in PROFILES.items()
}


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
