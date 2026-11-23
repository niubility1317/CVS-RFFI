"""Measure LoRa weak differential complex gain with a matched source ablation."""

import argparse
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write

from . import chirp_spectrum_source as pipeline
from .channel_blind import generic_observation
from .channel_blind_source import FitRecipe, fit, setup
from .complex_vector_source import load_reader, publish
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import source_control
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


@torch.no_grad()
def checkpoint_smoke(rt):
    root = Path(rt['run_root'])
    device = 'cuda:0'
    (_, calibration, _), contract = setup(rt, device, smoke=True)
    reader = load_reader(root, contract, device)
    generator = torch.Generator(device=device).manual_seed(2026112302)
    x = real_iq(generic_observation(complex_iq(calibration[:4]), generator))
    triggered = {}
    for route in ROUTES:
        _, _, bound, baseline, _ = source_control(rt, route, device)
        if bound != contract:
            raise ValueError('No-query smoke found a source contract mismatch')
        recovered, active = baseline(x)
        logits = reader(torch.cat((x, recovered)))[0]
        if logits.shape != (8, 10) or not torch.isfinite(logits).all():
            raise ValueError('No-query frozen checkpoint smoke failed')
        triggered[route] = int(active.sum())
    write(root / 'checkpoint_smoke.json', dict(status='PASS', n=4, triggered=triggered,
          target_reads=0, source_truth_reads=0))
    print('LORA_SOURCE_CHECKPOINT_SMOKE_PASS', triggered, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + key, type=Path, required=True)
    for action in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
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
        pipeline.dispatch(rt, RECIPES, initial_stages=(('checkpoint_smoke', ('source',)),))
    elif action == 'score':
        pipeline.score_frozen(rt, MODES, comparisons=(('complex_transport', 'waveform_transport'),
                                                        ('complex_transport_half', 'waveform_transport_half'),
                                                        ('complex_transport_half', 'direct')))
    else:
        activate(rt)
        if action == 'evaluate':
            pipeline.evaluate(rt, args['name'], RECIPES, MODES, renderer=render)
        elif action == 'timing':
            pipeline.measure(rt, RECIPES['complex_transport'], verify_reader=False)
        elif action == 'checkpoint_smoke':
            checkpoint_smoke(rt)
        else:
            name = args['name']
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[name], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35, make_pair=paired_echo_inputs,
                                  projection_weight=WEIGHTS[name]))


if __name__ == '__main__':
    main()
