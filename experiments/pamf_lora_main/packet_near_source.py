"""Train a phase-aware LoRa near-carrier refiner on all six observed chirps."""

import argparse
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.cric_dura_comparison.io import read

from .broadband_partitions_source import panel as partition_panel
from .complex_vector_source import dispatch, publish, score
from .data import SOURCE
from .denoise_source import EVAL_PER_RX
from .joint_unroll_source import source_control
from .near_spectrum_source import broadband, fit, panel as old_near_panel, readouts
from .packet_near import PacketNear
from .physical_inverse_source import source_trace_implementation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-dura-packet-near-source-m2-r01'
RECIPE = SimpleNamespace(seed=2026112799, steps=1700, train_per_rx=800, held_per_rx=80,
                         arms=('individual', 'packet'), model=lambda x: PacketNear(packet=x=='packet'))
MODES = ('direct', 'reference', 'frozen_broad', 'individual_context', 'packet_context',
         'packet_context_half', 'known_matched_near', 'known_noise')


def panel(rx):
    previous = np.union1d(partition_panel(rx), old_near_panel(rx))
    rng = np.random.default_rng(RECIPE.seed+1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == key), previous),
                                             EVAL_PER_RX, replace=False)) for key in SOURCE])


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for x in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--'+x, type=Path, required=True)
    for x in ('dispatch', 'smoke', 'fit', 'score'):
        entry = commands.add_parser(x)
        entry.add_argument('runtime', type=Path)
        if x != 'dispatch':
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
                context = source_control(rt, route, 'cuda:0')
                parent = broadband(Path(rt['run_root']), route, context[2], 'cuda:0')
                score(rt, route, recipe=RECIPE, panel=panel, modes=MODES,
                      render=partial(readouts, wide=parent), binding=context)
            else:
                source_trace_implementation()
                fit(rt, route, smoke=action=='smoke', recipe=RECIPE, pair_weights=(0., 0.))
