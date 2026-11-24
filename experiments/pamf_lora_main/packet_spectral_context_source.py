"""Train matched LoRa spectral attention arms with legal source IQ."""

import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read

from .action_support_source import independent_score
from .antithetic_detail import objective
from .antithetic_detail_source import evaluate, reserve, smoke, timing
from .channel_blind_source import FitRecipe, fit
from .complex_vector_source import publish
from .factorized_chirps_source import calibrate
from .gated_slope_source import dispatch
from .packet_spectral_context import PacketSpectralContext
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, WIDTHS


RUN = '20261124-phase1-lora-packet-spectral-context-source-m4-r01'
RECIPES = {
    arm: FitRecipe(arm, STEPS, partial(PacketSpectralContext, WIDTHS, packet=scope),
                   colored_observation, PacketSpectralContext.forward, initialize=calibrate)
    for arm, scope in (('local_context', False), ('packet_context', True))
}
MODES = ('direct', 'reference', 'rank_parent_half',
         *(f'{arm}_{stage}' for arm in RECIPES for stage in ('coarse', 'half', 'full')))
PAIRS = tuple((f'packet_context_{stage}', f'local_context_{stage}') for stage in ('half', 'coarse', 'full'))
PAIRS += (('packet_context_half', 'rank_parent_half'), ('local_context_half', 'rank_parent_half'))


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
            stage.add_argument('row', choices=RECIPES if action == 'fit' else
                               (*ROUTES, 'reserve') if action == 'evaluate' else ('source',))
    args = vars(parser.parse_args())
    phase = args.pop('action')
    if phase == 'publish':
        return publish(**args, run=RUN, runner=__spec__.name)
    rt = read(args['runtime'])
    if phase == 'dispatch':
        return dispatch(rt, fit_arms=RECIPES)
    if phase == 'score':
        return independent_score(rt, modes=MODES, comparisons=PAIRS)
    activate(rt)
    if phase == 'smoke':
        return smoke(rt, recipes=RECIPES, modes=MODES, split_risk=(False, False))
    if phase == 'fit':
        recipe = RECIPES[args['row']]
        return fit(rt, recipe=recipe, receiver_batches=True, calculate=partial(objective, split_risk=False))
    if phase == 'timing':
        return timing(rt, recipes=RECIPES)
    if args['row'] == 'reserve':
        return reserve(rt, recipes=RECIPES, modes=MODES)
    return evaluate(rt, args['row'], recipes=RECIPES, modes=MODES)


if __name__ == '__main__':
    main()
