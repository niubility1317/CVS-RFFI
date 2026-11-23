"""Train and score matched pointwise and actual-source-pair LoRa recovery."""

import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read

from .channel_blind_source import FitRecipe, fit
from .chirp_spectrum_source import dispatch, evaluate, score_frozen
from .complex_vector_source import publish
from .pilot_joint import PilotJoint
from .real_pair import objective, paired_view
from .rf_noise_transport_source import checkpoint_smoke, render, timing
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import STEPS, TAIL_WEIGHT, WIDTHS


RUN = '20261124-phase1-lora-real-source-pair-contrast-m4-r01'
PAIR_WEIGHT = .40
MODES = ('direct', 'reference', 'point_coarse', 'point', 'point_half',
         'pair_coarse', 'pair', 'pair_half')
PAIRS = (('pair_coarse', 'point_coarse'), ('pair', 'point'), ('pair_half', 'point_half'))
RECIPES = {
    name: FitRecipe(name, STEPS, partial(PilotJoint, equalize=False, widths=WIDTHS, companion='mirror'),
                    paired_view, PilotJoint.forward)
    for name in ('point', 'pair')
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for action in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        stage = commands.add_parser(action)
        stage.add_argument('runtime', type=Path)
        if action != 'dispatch':
            names = RECIPES if action in ('smoke', 'fit') else ROUTES if action == 'evaluate' else ('source',)
            stage.add_argument('name', choices=names)
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        publish(**options, run=RUN, runner=__spec__.name)
        return
    rt = read(options['runtime'])
    if action == 'dispatch':
        dispatch(rt, RECIPES, initial_stages=(('checkpoint_smoke', ('source',)),))
    elif action == 'score':
        score_frozen(rt, modes=MODES, comparisons=PAIRS)
    else:
        activate(rt)
        if action == 'checkpoint_smoke':
            checkpoint_smoke(rt, RECIPES, MODES)
        elif action == 'timing':
            timing(rt, RECIPES['pair'])
        elif action == 'evaluate':
            evaluate(rt, options['name'], recipes=RECIPES, modes=MODES, renderer=render)
        else:
            choice = options['name']
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[choice], receiver_batches=True,
                calculate=partial(objective, pair_weight=PAIR_WEIGHT if choice == 'pair' else 0.,
                                  tail_weight=TAIL_WEIGHT))


if __name__ == '__main__':
    main()
