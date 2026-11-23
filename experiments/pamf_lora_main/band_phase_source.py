"""Fit the LoRa DURA phase transport head on source re-noising pairs."""

import argparse
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read

from .band_phase import CalibratedPacketNear, phase_risk
from .broadband_partitions_source import panel as partition_panel
from .complex_vector_source import dispatch, publish, score
from .data import SOURCE
from .denoise_source import EVAL_PER_RX
from .joint_unroll_source import source_control
from .near_spectrum import near_error, same_signal
from .near_spectrum_source import broadband, fit, panel as near_panel, readouts
from .packet_near_source import panel as packet_panel
from .physical_inverse_source import source_trace_implementation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-band-phase-source-m2-r01'
PHASE_WEIGHT = .08
RECIPE = SimpleNamespace(seed=2026112817, steps=1900, train_per_rx=900, held_per_rx=80,
                         arms=('waveform', 'phase'), model=lambda _: CalibratedPacketNear())
MODES = ('direct', 'reference', 'frozen_broad', 'waveform_band_phase', 'supervised_band_phase',
         'supervised_band_phase_half', 'known_matched_near', 'known_noise')


def panel(rx):
    used = np.unique(np.concatenate([f(rx) for f in (partition_panel, near_panel, packet_panel)]))
    rng = np.random.default_rng(RECIPE.seed+1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == key), used),
                                             EVAL_PER_RX, replace=False)) for key in SOURCE])


def risk(model, arrays, indices):
    y, b, q, signal, common, prior = (x[indices] for x in arrays)
    prediction, phase = model(*(v.flatten(0, 1) for v in (y, b, q, prior)), phase_readout=True)
    prediction = prediction.reshape_as(prior)
    phase_by_band = phase_risk(phase, signal.flatten(0, 1))
    summary = torch.stack((near_error(prediction, b, signal),
                           same_signal(prediction, b, signal, common),
                           phase_by_band.mean()))
    return torch.cat((summary, phase_by_band))


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
        rt = read(args['runtime'])
        if action == 'dispatch':
            dispatch(rt)
        else:
            activate(rt)
            route = args['route']
            if action == 'score':
                binding = source_control(rt, route, 'cuda:0')
                wide = broadband(Path(rt['run_root']), route, binding[2], 'cuda:0')
                score(rt, route, recipe=RECIPE, modes=MODES, panel=panel,
                      render=partial(readouts, wide=wide), binding=binding)
            else:
                source_trace_implementation()
                fit(rt, route, smoke=action=='smoke', recipe=RECIPE, pair_weights=(0., 0.),
                    risk=risk, auxiliary_weights=(0., PHASE_WEIGHT))
