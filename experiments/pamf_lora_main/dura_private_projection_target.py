"""Test frozen DURA near-band protection on both blind LoRa target routes."""
import argparse
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write
from .channel_blind import corruption_score
from .channel_blind_source import setup
from .complex_vector_source import load_reader
from .dura_private_projection_source import MODES as SOURCE_MODES, RUN as SOURCE_RUN, bind, readouts
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .fixed_mid_target import dispatch, predict, publish, source_smoke_metadata
from .joint_unroll_source import coordinates
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-dura-private-projection-target-m2-r01'
MODES = ('wide_half', 'wide_quarter', 'equal_near_edit', 'radial_observed')
PRIMARY = 'radial_observed'


def frozen_runtime(rt, contract, device):
    root = Path(rt['run_root'])
    model, threshold = bind(root, contract, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    reader = load_reader(root, contract, device)
    return tracker, model, threshold, reader


def selected_readouts(iq, tracker, model, threshold):
    waves, count = readouts(iq, tracker, model, threshold)
    return tuple(waves[SOURCE_MODES.index(k) - 2] for k in MODES), count


@torch.no_grad()
def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    tracker, model, threshold, reader = frozen_runtime(rt, contract, device)
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupted = real_iq(colored_observation(complex_iq(cal), rng))
    active = corruption_score(coordinates(tracker, corrupted)[0]) > threshold
    iq = torch.cat((cal[:2], corrupted[active][:2]))
    waves, count = selected_readouts(iq, tracker, model, threshold)
    values = torch.stack(waves)
    scores = reader(values.flatten(0,1))[0]
    if (count != 2 or not torch.equal(values[:,:2], iq[:2].expand_as(values[:,:2]))
            or torch.equal(values[:,2:], iq[2:].expand_as(values[:,2:]))
            or not values.isfinite().all() or not scores.isfinite().all()
            or scores.shape != (len(MODES)*len(iq),10)):
        raise ValueError('Real no-query DURA target smoke failed')
    write(root/'smoke.json', dict(source_smoke_metadata(SOURCE_RUN, MODES), active=count,
                                 clean_bypassed=2))
    print('LORA_DURA_PROJECTION_TARGET_SOURCE_SMOKE_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    release = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config','source-submit'):
        release.add_argument('--'+key, type=Path, required=True)
    score = actions.add_parser('score')
    for key in ('submit','ssh-config','dataset','output','full-control','residual-control'):
        score.add_argument('--'+key, type=Path, required=True)
    for action in ('dispatch','smoke','predict'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        if action != 'dispatch':
            worker.add_argument('route', choices=ROUTES if action == 'predict' else ('source',))
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        return publish(**options, run=RUN, source_run=SOURCE_RUN, runner=__spec__.name)
    if action == 'score':
        from .fixed_mid_target_score import collect
        return collect(**options, run=RUN, modes=MODES, primary=PRIMARY)
    rt = read(options['runtime'])
    if action == 'dispatch':
        return dispatch(rt)
    activate(rt)
    if action == 'smoke':
        return smoke(rt)
    return predict(rt, options['route'], modes=MODES, source_run=SOURCE_RUN,
                   prepare=frozen_runtime, evaluate=selected_readouts)


if __name__ == '__main__':
    main()
