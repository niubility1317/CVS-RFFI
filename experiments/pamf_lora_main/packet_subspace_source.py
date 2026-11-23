"""Compare LoRa near-carrier recovery with and without a packet-axis factorization."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read

from .band_phase_source import panel as phase_panel
from .complex_vector_source import VectorRecipe, dispatch, fit, publish, score
from .data import SOURCE
from .denoise_source import EVAL_PER_RX
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .packet_subspace import PacketSubspace, component_risk, components, target_vectors
from .physical_inverse_source import source_trace_implementation
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_signal_probe_source import panel as weak_panel


RUN = '20261123-phase1-lora-dura-packet-subspace-source-m2-r01'
ARMS = ('unrestricted', 'factorized')
RECIPE = VectorRecipe(2026112869, 2200, 1000, 80, ARMS, lambda a: PacketSubspace(factorized=a==ARMS[1]),
                      target_vectors, lambda y, t, _: component_risk(y, t))
MODES = ('direct', 'reference', 'unrestricted', 'factorized', 'factorized_half',
         'learned_shared', 'learned_deviation', 'known_shared', 'known_deviation', 'known_both')


def panel(rx):
    exclude = np.union1d(phase_panel(rx), weak_panel(rx))
    rng = np.random.default_rng(RECIPE.seed+1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == key), exclude),
                                             EVAL_PER_RX, replace=False)) for key in SOURCE])


@torch.no_grad()
def render(iq, noise, baseline, tracker, models):
    reference, active = baseline(iq)
    output = [reference.clone() for _ in MODES[2:]]
    if active.any():
        observation, rotation, q, rms = coordinates(tracker, iq[active])
        basis = rotation*rms
        current = complex_iq(reference[active])/basis
        free, split = (m(observation, current, q) for m in models)
        signal = observation - complex_iq(noise[active])/basis
        truth = target_vectors(current, signal)[ARMS[0]]
        directions = (free, split, .5*split, *components(split), *components(truth), truth)
        for restored, delta in zip(output, directions, strict=True):
            restored[active] = reference[active]+real_iq(torch.fft.ifft(delta, norm='ortho')*basis)
    return torch.stack((iq, reference, *output)), int(active.sum())


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'score'):
        entry = commands.add_parser(action)
        entry.add_argument('runtime', type=Path)
        if action != 'dispatch':
            entry.add_argument('route', choices=ROUTES)
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args, run=RUN, runner=__spec__.name)
    else:
        runtime = read(args['runtime'])
        if action == 'dispatch':
            dispatch(runtime)
        else:
            activate(runtime)
            route = args['route']
            if action == 'score':
                score(runtime, route, recipe=RECIPE, panel=panel, modes=MODES, render=render)
            else:
                source_trace_implementation()
                fit(runtime, route, recipe=RECIPE, smoke=action=='smoke')
