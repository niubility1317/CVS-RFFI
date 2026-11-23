"""Train LoRa DURA to correct the noncarrier spectrum at a larger dose."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read

from .broadband_vector import BroadbandVector, band_risk, target_vectors
from .complex_budget_source import panel as budget_panel
from .complex_vector_source import VectorRecipe, dispatch, fit, publish, result_panel, score
from .data import SOURCE
from .denoise_source import EVAL_PER_RX, normalized
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .physical_inverse_source import source_trace_implementation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-broadband-vector-source-m2-r01'
RECIPE = VectorRecipe(2026112758, 1800, 700, 80, ('narrow', 'broad'),
                      lambda key: BroadbandVector(wide=key=='broad'), target_vectors,
                      lambda value, target, key: band_risk(value, target, key=='broad'))
MODES = ('direct', 'reference', 'narrow_vector', 'broad_vector', 'broad_75', 'broad_half',
         'ideal_narrow', 'ideal_broad', 'known_noise')


def panel(rx):
    excluded = np.union1d(result_panel(rx), budget_panel(rx, EVAL_PER_RX))
    rng = np.random.default_rng(RECIPE.seed+1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == key), excluded),
                                             EVAL_PER_RX, replace=False)) for key in SOURCE])


def readouts(received, noise, baseline, tracker, models):
    mature, active = baseline(received)
    output = [mature.clone() for _ in MODES[2:]]
    if active.any():
        y = received[active]
        obs, rotation, q, rms = coordinates(tracker, y)
        basis = rotation*rms
        b = complex_iq(mature[active])/basis
        oracle = target_vectors(b, obs - complex_iq(noise[active])/basis)
        narrow, broad = (m(obs, b, q) for m in models)
        changes = (narrow, broad, .75*broad, .5*broad, oracle['narrow'], oracle['broad'])
        for row, delta in zip(output[:-1], changes, strict=True):
            row[active] = mature[active] + real_iq(torch.fft.ifft(delta, norm='ortho')*basis)
        output[-1][active] = normalized(y-noise[active], y)
    return torch.stack((received, mature, *output)), int(active.sum())


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
    options = vars(cli.parse_args())
    action = options.pop('action')
    if action == 'publish':
        publish(**options, run=RUN, runner=__spec__.name)
    else:
        runtime = read(options['runtime'])
        if action == 'dispatch':
            dispatch(runtime)
        else:
            activate(runtime)
            if action == 'score':
                score(runtime, options['route'], recipe=RECIPE, panel=panel, modes=MODES, render=readouts)
            else:
                source_trace_implementation()
                fit(runtime, options['route'], smoke=action == 'smoke', recipe=RECIPE)
