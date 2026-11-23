"""Measure how denoising quality affects LoRa DURA's physical inverse."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read

from .band_phase_source import panel as phase_panel
from .data import SOURCE
from .packet_physics import aligned_view, chirp_windows
from .physical_inverse import full_replay, residual_replay
from .physical_inverse_source import dispatch, evaluate, publish, select_panel, source_trace_implementation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-inverse-noise-dose-source-m2-r01'
REMOVED = (0., .50, .80, 1.)
PER_RX = 60
MODES = ('direct', 'reference', 'removed_50', 'removed_80', 'removed_100',
         'inverse_0', 'inverse_50', 'inverse_80', 'inverse_100', 'paired_clean')


def panel(rx, count):
    used = np.union1d(select_panel(rx, 60), phase_panel(rx))
    rng = np.random.default_rng(2026112829)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == key), used),
                                             count, replace=False)) for key in SOURCE])


def simulate(route, scene, row):
    iq, rx, pid = row
    if scene == 'clean':
        return np.broadcast_to(chirp_windows(iq), (2*len(REMOVED), 6, 2, 256)), 0.
    replay = full_replay if route == 'full' else residual_replay
    wave, error = replay(iq, scene, rx, pid, removed=REMOVED)
    if route == 'residual':
        expected = aligned_view(iq, scene, rx, pid, 'source-V', processing_route='residual')
        if not np.array_equal(wave[0], expected):
            raise ValueError('Noise-dose evaluation changed the registered snapshot')
    return wave, error


@torch.no_grad()
def readouts(wave, clean, baseline, tracker):
    received = wave[0]
    reference, active = baseline(received)
    mask = active[None, :, None, None, None]
    candidates = torch.where(mask, wave[1:], received)
    return torch.cat((received[None], reference[None], candidates, clean[None])), int(active.sum())


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+option, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'score'):
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
            source_trace_implementation()
            evaluate(rt, args['route'], smoke=action=='smoke', per_rx=PER_RX,
                     sampler=panel, simulator=simulate, renderer=readouts, modes=MODES)
