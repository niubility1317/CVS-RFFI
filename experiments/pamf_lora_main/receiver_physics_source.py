"""Train a generic receiver-physics LoRa arm against a frozen matched control."""

import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read

from .action_support_source import independent_score
from .antithetic_detail import antithetic_inputs, objective
from .antithetic_detail_source import RECIPES as CONTROL_RECIPES, RUN as CONTROL_RUN
from .antithetic_detail_source import evaluate, reserve, smoke, timing
from .channel_blind_source import fit
from .complex_vector_source import publish
from .gated_slope_source import dispatch
from .receiver_physics import apply_generic_physics
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-generic-receiver-physics-source-m3-r01'
NEW_ARM = 'receiver_physics'
RECIPES = {
    'pointwise': CONTROL_RECIPES['pointwise'],
    NEW_ARM: replace(CONTROL_RECIPES['pointwise'], name=NEW_ARM),
}
REUSED = {'pointwise': CONTROL_RUN}
MODES = ('direct', 'reference', 'rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse', 'half', 'full')))
PAIRS = tuple((f'{NEW_ARM}_{stage}', f'pointwise_{stage}') for stage in ('half', 'coarse', 'full'))
PAIRS += (('receiver_physics_half', 'rank_parent_half'), ('pointwise_half', 'rank_parent_half'))


def calculation(device, *, private_risk_weight=0., relation_weight=0.):
    training_rng = torch.Generator(device=device).manual_seed(2026113181)
    held_rng = torch.Generator(device=device).manual_seed(2026113182)

    def build_inputs(signal, change, perturb, rng, *, with_identity):
        rng_physics = training_rng if with_identity else held_rng
        transform = partial(apply_generic_physics, generator=rng_physics)
        return antithetic_inputs(signal, change, perturb, rng,
                                 with_identity=with_identity, transform=transform)

    return partial(objective, split_risk=False, build_inputs=build_inputs,
                   private_risk_weight=private_risk_weight, relation_weight=relation_weight)


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
    if phase == 'smoke':
        return smoke(rt, recipes=RECIPES, modes=MODES, split_risk=(False, False),
                     calculations={NEW_ARM: calculation('cuda:0')}, reused=REUSED)
    if phase == 'fit':
        return fit(rt, recipe=RECIPES[NEW_ARM], receiver_batches=True, calculate=calculation('cuda:0'))
    if phase == 'timing':
        return timing(rt, recipes=RECIPES, reused=REUSED)
    if args['row'] == 'reserve':
        return reserve(rt, recipes=RECIPES, modes=MODES, reused=REUSED)
    return evaluate(rt, args['row'], recipes=RECIPES, modes=MODES, reused=REUSED)


if __name__ == '__main__':
    main()
