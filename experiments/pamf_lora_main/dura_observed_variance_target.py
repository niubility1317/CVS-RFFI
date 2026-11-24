"""Test the frozen receive-variance DURA at two preset doses on both LoRa routes."""
import argparse
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write
from .antithetic_detail_source import load_arm
from .carrier_coupled import packet_frame
from .channel_blind import corruption_score
from .channel_blind_source import setup
from .complex_vector_source import load_reader
from .dura_observed_variance_source import ARM, RECIPES, RUN as SOURCE_RUN
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .fixed_mid_target import dispatch, predict, publish, source_smoke_metadata
from .joint_unroll_source import coordinates
from .relative_carrier import mix
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-dura-observed-variance-target-m2-r01'
DOSES = (.5, .25)
MODES = ('observed_variance_half', 'observed_variance_quarter')
PRIMARY = MODES[0]
MATCHED_PAIRS = dict(zip(MODES, ('wide_half', 'wide_quarter'), strict=True))


def frozen_runtime(rt, contract, device):
    root = Path(rt['run_root'])
    model, threshold = load_arm(root, RECIPES[ARM], contract, device, {ARM: SOURCE_RUN})
    tracker = PacketRestorer('consensus').to(device).eval()
    return tracker, model, threshold, load_reader(root, contract, device)


@torch.no_grad()
def readouts(iq, tracker, model, threshold):
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    count = int(active.sum())
    if not count:
        return (iq,) * len(MODES), 0
    observed, rotation, quality, rms = packet_frame(tuple(value[active] for value in frame))
    _, restored = model(observed, quality, rotation)
    waves = []
    for dose in DOSES:
        wave = iq.clone()
        wave[active] += real_iq((mix(observed, restored, dose) - observed) * rotation * rms)
        waves.append(wave)
    return tuple(waves), count


@torch.no_grad()
def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    tracker, model, threshold, reader = frozen_runtime(rt, contract, device)
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupt = real_iq(colored_observation(complex_iq(cal), rng))
    active = corruption_score(coordinates(tracker, corrupt)[0]) > threshold
    iq = torch.cat((cal[:2], corrupt[active][:2]))
    waves, count = readouts(iq, tracker, model, threshold)
    values = torch.stack(waves)
    scores = reader(values.flatten(0,1))[0]
    if (count != 2 or scores.shape != (len(MODES)*len(iq), 10)
            or not values.isfinite().all() or not scores.isfinite().all()
            or not torch.equal(values[:,:2], iq[:2].expand_as(values[:,:2]))
            or torch.equal(values[0,2:], values[1,2:])
            or any(torch.equal(wave[2:], iq[2:]) for wave in waves)):
        raise ValueError('Frozen observed-variance DURA no-query smoke failed')
    write(root/'smoke.json', dict(source_smoke_metadata(SOURCE_RUN, MODES), active=count,
                                 clean_bypassed=2))
    print('LORA_DURA_OBSERVED_VARIANCE_TARGET_SMOKE_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    release = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config','source-submit'):
        release.add_argument('--'+key, type=Path, required=True)
    score = actions.add_parser('score')
    for key in ('submit','ssh-config','dataset','output','full-control','residual-control','width-control'):
        score.add_argument('--'+key, type=Path, required=True)
    for action in ('dispatch','smoke','predict'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        if action != 'dispatch':
            worker.add_argument('route', choices=ROUTES if action == 'predict' else ('source',))
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        return publish(**options, run=RUN, source_run=SOURCE_RUN, runner=__spec__.name,
                       allowed_gpus=(7,3,6,2,0,4,1,5))
    if action == 'score':
        from .dura_private_projection_target import MODES as WIDTH_MODES
        from .fixed_mid_target_score import collect
        historical = (options.pop('width_control'), WIDTH_MODES)
        return collect(**options, run=RUN, modes=MODES, primary=PRIMARY,
                       previous=historical, additional_pairs=MATCHED_PAIRS)
    rt = read(options['runtime'])
    if action == 'dispatch':
        return dispatch(rt)
    activate(rt)
    if action == 'smoke':
        return smoke(rt)
    return predict(rt, options['route'], modes=MODES, source_run=SOURCE_RUN,
                   prepare=frozen_runtime, evaluate=readouts)


if __name__ == '__main__':
    main()
