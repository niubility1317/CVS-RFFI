"""Measure LoRa weak differential complex gain with a matched source ablation."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read

from . import chirp_spectrum_source as pipeline
from .channel_blind import generic_observation
from .channel_blind_source import FitRecipe, fit
from .complex_vector_source import publish
from .natural_detail import natural_objective, paired_echo_inputs
from .pilot_joint import PilotJoint
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-complex-transport-source-m4-r01'
STEPS = 1200
WEIGHTS = {'waveform_transport': 0., 'complex_transport': .35}
MODES = ('direct', 'reference', 'waveform_transport', 'waveform_transport_half',
         'complex_transport', 'complex_transport_half')


class CarrierPreservingPrior(PilotJoint):
    def __init__(self):
        super().__init__(equalize=False)

    def forward(self, observed, quality, template=None):
        return super().forward(observed, quality, template)


RECIPES = {
    name: FitRecipe(name, STEPS, CarrierPreservingPrior, generic_observation, CarrierPreservingPrior.forward)
    for name in WEIGHTS
}


def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference, _ = baseline(iq)
    (ordinary, projected, projected_half), active = pipeline.recover(iq, tracker, models, threshold, half=True)
    ordinary_half = iq + .5 * (ordinary - iq)
    return torch.stack((iq, reference, ordinary, ordinary_half, projected, projected_half)), active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + key, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        cmd = commands.add_parser(action)
        cmd.add_argument('runtime', type=Path)
        if action != 'dispatch':
            cmd.add_argument('name', choices=RECIPES if action in ('smoke', 'fit') else
                             ROUTES if action == 'evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args, run=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action == 'dispatch':
        pipeline.dispatch(rt, RECIPES)
    elif action == 'score':
        pipeline.score_frozen(rt, MODES, comparisons=(('complex_transport', 'waveform_transport'),
                                                        ('complex_transport_half', 'waveform_transport_half'),
                                                        ('complex_transport_half', 'direct')))
    else:
        activate(rt)
        if action == 'evaluate':
            pipeline.evaluate(rt, args['name'], RECIPES, MODES, renderer=render)
        elif action == 'timing':
            pipeline.measure(rt, RECIPES['complex_transport'])
        else:
            name = args['name']
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[name], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35, make_pair=paired_echo_inputs,
                                  projection_weight=WEIGHTS[name]))


if __name__ == '__main__':
    main()
