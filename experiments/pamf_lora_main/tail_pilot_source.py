"""Matched training for LoRa average versus upper-tail complex waveform loss."""

import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from .channel_blind_source import FitRecipe, fit
from .chirp_spectrum_source import dispatch, evaluate, score_frozen
from .complex_vector_source import publish
from .natural_detail import natural_objective, paired_echo_inputs
from .pilot_joint import PilotJoint
from .rf_noise_transport import colored_observation
from .rf_noise_transport_source import checkpoint_smoke, render, timing
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-tail-pilot-source-m4-r01'
STEPS = 3600
WIDTHS = (16, 20, 28, 40)
TAIL_WEIGHT = .45
MODES = ('direct', 'reference', 'average_coarse', 'average', 'average_half',
         'tail_coarse', 'tail', 'tail_half')
PAIRS = (('tail_coarse', 'average_coarse'), ('tail', 'average'), ('tail_half', 'average_half'))
RECIPES = {name: FitRecipe(name, STEPS, partial(PilotJoint, equalize=False, widths=WIDTHS),
                           colored_observation, PilotJoint.forward)
           for name in ('average', 'tail')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + name, type=Path, required=True)
    for stage in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        cmd = commands.add_parser(stage)
        cmd.add_argument('runtime', type=Path)
        if stage != 'dispatch':
            names = RECIPES if stage in ('smoke', 'fit') else ROUTES if stage == 'evaluate' else ('source',)
            cmd.add_argument('name', choices=names)
    args = vars(parser.parse_args())
    stage = args.pop('action')
    if stage == 'publish':
        publish(**args, run=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if stage == 'dispatch':
        dispatch(rt, RECIPES, initial_stages=(('checkpoint_smoke', ('source',)),))
    elif stage == 'score':
        score_frozen(rt, modes=MODES, comparisons=PAIRS)
    else:
        activate(rt)
        if stage == 'checkpoint_smoke':
            checkpoint_smoke(rt, RECIPES, MODES)
        elif stage == 'evaluate':
            evaluate(rt, args['name'], recipes=RECIPES, modes=MODES, renderer=render)
        elif stage == 'timing':
            timing(rt, RECIPES['tail'])
        else:
            name = args['name']
            fit(rt, smoke=stage == 'smoke', recipe=RECIPES[name], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35, make_pair=paired_echo_inputs,
                                  tail_weight=TAIL_WEIGHT if name == 'tail' else 0.))


if __name__ == '__main__':
    main()
