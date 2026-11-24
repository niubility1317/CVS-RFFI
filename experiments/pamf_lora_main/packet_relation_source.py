"""Train LoRa recovery to retain complex spectral relations to other clean source packets."""

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
from .private_direction_risk_source import RECIPES as CONTROL_RECIPES, RUN as CONTROL_RUN, WEIGHT
from .receiver_physics_source import calculation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-packet-relation-source-m3-r01'
NEW_ARM = 'packet_relation'
RELATION_WEIGHT = .30
RECIPES = {
    'direction_risk': CONTROL_RECIPES['direction_risk'],
    NEW_ARM: replace(CONTROL_RECIPES['direction_risk'], name=NEW_ARM),
}
REUSED = {'direction_risk': CONTROL_RUN}
MODES = ('direct', 'reference', 'rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse', 'half', 'full')))
PAIRS = tuple((f'{NEW_ARM}_{stage}', f'direction_risk_{stage}') for stage in ('half', 'coarse', 'full'))
PAIRS += (('packet_relation_half', 'rank_parent_half'), ('direction_risk_half', 'rank_parent_half'))


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
        calculate = calculation('cuda:0', private_risk_weight=WEIGHT, relation_weight=RELATION_WEIGHT)
    if phase == 'smoke':
        return smoke(rt, recipes=RECIPES, modes=MODES, split_risk=(False, False),
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
