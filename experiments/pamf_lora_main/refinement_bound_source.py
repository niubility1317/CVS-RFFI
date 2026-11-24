"""Compare training with a physical refinement cap to an identical post-hoc cap."""

import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read

from .action_support_source import independent_score
from .antithetic_detail_source import evaluate, reserve, smoke, timing
from .channel_blind_source import fit
from .complex_vector_source import publish
from .gated_slope_source import dispatch
from .lag_profile_source import LAG_WEIGHT, RECIPES as CONTROL_RECIPES, RUN as CONTROL_RUN
from .private_direction_risk_source import WEIGHT
from .refinement_bound import bound_refinement
from .receiver_physics_source import calculation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-bounded-refinement-source-m3-r01'
NEW_ARM = 'dose_bound'
CONTROL = CONTROL_RECIPES['local_lag']
RECIPES = {
    'local_lag': CONTROL,
    'local_lag_postcap': replace(CONTROL, stage_transform=bound_refinement),
    NEW_ARM: replace(CONTROL, name=NEW_ARM, stage_transform=bound_refinement),
}
REUSED = {'local_lag': CONTROL_RUN}
MODES = ('direct', 'reference', 'rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse', 'half', 'full')))
PAIRS = (('dose_bound_half', 'local_lag_postcap_half'), ('dose_bound_half', 'local_lag_half'),
         ('local_lag_postcap_half', 'local_lag_half'), ('dose_bound_half', 'rank_parent_half'),
         ('dose_bound_coarse', 'local_lag_coarse'), ('dose_bound_full', 'local_lag_postcap_full'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        stage = actions.add_parser(action)
        stage.add_argument('runtime', type=Path)
        if action != 'dispatch':
            stage.add_argument('row', choices=(NEW_ARM,) if action == 'fit' else
                               (*ROUTES, 'reserve') if action == 'evaluate' else ('source',))
    args = vars(parser.parse_args())
    phase = args.pop('action')
    if phase == 'publish':
        return publish(**args, run=RUN, runner=__spec__.name)
    rt = read(args['runtime'])
    if phase == 'dispatch':
        return dispatch(rt, fit_arms=(NEW_ARM,))
    if phase == 'score':
        return independent_score(rt, modes=MODES, comparisons=PAIRS)
    activate(rt)
    if phase in ('smoke', 'fit'):
        calculate = calculation('cuda:0', private_risk_weight=WEIGHT, lag_weight=LAG_WEIGHT)
    if phase == 'smoke':
        return smoke(rt, recipes=RECIPES, modes=MODES, split_risk=(False, False, False),
                     calculations={NEW_ARM: calculate}, reused=REUSED)
    if phase == 'fit':
        return fit(rt, recipe=RECIPES[NEW_ARM], receiver_batches=True, calculate=calculate)
    if phase == 'timing':
        return timing(rt, recipes=RECIPES, reused=REUSED)
    if args['row'] == 'reserve':
        return reserve(rt, recipes=RECIPES, modes=MODES, reused=REUSED)
    return evaluate(rt, args['row'], recipes=RECIPES, modes=MODES, reused=REUSED)


if __name__ == '__main__':
    main()
