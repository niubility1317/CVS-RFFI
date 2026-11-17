"""Fixed ten-class shallow and six-receiver KL factorial matrices."""
from contextlib import contextmanager
from experiments.pamf_repl_component_guard import plan as base

RUN = '20261117-phase1-pamf-repl-priority-clean-m20-r01'
SEED = base.SEED
SETTINGS = {'ten_class': 'ManyRx', 'source_rx_6': 'm6-natural'}
RECIPES = {}
for weight in (.75, 3.):
    for factor in (1., 2.):
        for mode in ('raw', 'shallow'):
            RECIPES[f'ten_w{str(weight).replace(".", "p")}_s{int(factor)}_{mode}'] = dict(
                setting='ten_class', weight=weight, shallow_factor=factor, extrap_coefficient=0.,
                joint_weight=0., kl_factor=1., reference='shallow', mode=mode)
for weight in (3., 6.):
    for kl in (1., 1.5, 2.):
        for mode in ('native', 'kl'):
            RECIPES[f'six_w{int(weight)}_k{str(kl).replace(".", "p")}_{mode}'] = dict(
                setting='source_rx_6', weight=weight, shallow_factor=0., extrap_coefficient=0.,
                joint_weight=.05, kl_factor=kl, reference='pair_kl', mode=mode)


def candidate_rows():
    return [dict(row_id=f"{r['setting']}-repl-{name}-s{SEED}", setting=r['setting'],
                 protocol_id=SETTINGS[r['setting']], model_seed=SEED, recipe=name, views=['clean'])
            for name, r in RECIPES.items()]


@contextmanager
def scoped():
    old = base.RUN, base.RECIPES, base.SETTINGS, base.candidate_rows, base.configuration
    original = base.configuration
    def configuration(ctx, row, project, commit, *, run_id=RUN):
        return original(ctx, row, project, commit, run_id=run_id)
    base.RUN, base.RECIPES, base.SETTINGS, base.candidate_rows, base.configuration = (
        RUN, RECIPES, SETTINGS, candidate_rows, configuration)
    try:
        yield
    finally:
        base.RUN, base.RECIPES, base.SETTINGS, base.candidate_rows, base.configuration = old
