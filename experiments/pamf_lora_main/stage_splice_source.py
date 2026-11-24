"""Evaluate all four frozen LoRa coarse and refinement pairings on source IQ."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.func import functional_call

from experiments.cric_dura_comparison.io import read, write

from .action_support_source import independent_score, predict_clean_reserve
from .antithetic_detail_source import bind_models
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .gated_slope_source import restore
from .joint_unroll_source import coordinates, source_control
from .lag_profile_source import RECIPES, RUN as LAG_RUN
from .private_direction_risk_source import RUN as DIRECTION_RUN
from .private_transport_source import dispatch
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .stage_splice import read_stages, refinement_state


RUN = '20261124-phase1-lora-local-lag-stage-factorial-source-m2-r01'
MODES = ('direct', 'reference', 'rank_half', 'direction_coarse', 'lag_coarse',
         'direction_direction', 'lag_lag', 'direction_lag', 'lag_direction')
PAIRS = (('lag_coarse', 'direction_coarse'),
         ('direction_lag', 'direction_direction'), ('lag_lag', 'lag_direction'),
         ('lag_direction', 'direction_direction'), ('lag_lag', 'direction_lag'),
         ('lag_direction', 'lag_lag'), ('lag_direction', 'rank_half'))
REUSED = {'direction_risk': DIRECTION_RUN, 'local_lag': LAG_RUN}


def bind(root, contract, device):
    rank, stages, threshold = bind_models(root, contract, device, recipes=RECIPES, reused=REUSED)
    return (rank, *stages), threshold


@torch.no_grad()
def readouts(iq, tracker, models, threshold):
    rank, *stages = models
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    n = int(active.sum())
    if not n:
        return (iq,) * (len(MODES) - 2), n
    selected = tuple(v[active] for v in frame)
    y, rotation, quality, rms = selected
    rank_half, = restore(iq, active, rank, selected, (.5,))
    result = [rank_half]
    for wave in read_stages(y, rotation, quality, stages):
        value = iq.clone()
        value[active] += real_iq((wave - y) * rotation * rms)
        result.append(value)
    return tuple(result), n


@torch.no_grad()
def render(iq, _, baseline, tracker, models, *, threshold):
    reference, _ = baseline(iq)
    waves, count = readouts(iq, tracker, models, threshold)
    return torch.stack((iq, reference, *waves)), count


def evaluate(rt, route):
    device = 'cuda:0'
    binding = source_control(rt, route, device)
    models, threshold = bind(Path(rt['run_root']), binding[2], device)
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=threshold),
          binding=binding, frozen_models=models, require_clean_bypass=False, replay_fn=receive_only,
          finalize=freeze, reader_chunk=144)


def reserve(rt):
    def build(root, contract, device, baseline, tracker):
        models, threshold = bind(root, contract, device)
        return lambda iq: render(iq, None, baseline, tracker, models, threshold=threshold)
    predict_clean_reserve(rt, modes=MODES, bind=build)


@torch.no_grad()
def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    _, _, verified, baseline, tracker = source_control(rt, 'full', device)
    if verified != contract:
        raise ValueError('Source U and the frozen reader use different IQ contracts')
    models, threshold = bind(root, contract, device)
    _, direction, lag = models
    buffers = lag.clean_detail.state_dict()
    for name, value in direction.clean_detail.state_dict().items():
        torch.testing.assert_close(value, buffers[name], atol=2e-7, rtol=1e-5)
    rng = torch.Generator(device=device).manual_seed(2026113627)
    noisy = real_iq(colored_observation(complex_iq(cal), rng))
    active = corruption_score(coordinates(tracker, noisy)[0]) > threshold
    disturbed = noisy[active][:2]
    samples = torch.cat((cal[:2], disturbed))
    waves, count = render(samples, None, baseline, tracker, models, threshold=threshold)
    logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    if (count != 2 or logits.shape != (len(MODES)*len(samples), 10) or not logits.isfinite().all()
            or not torch.equal(waves[2:, :2], samples[:2].expand_as(waves[2:, :2]))
            or torch.equal(waves[-2, 2:], waves[-4, 2:]) or torch.equal(waves[-1, 2:], waves[-3, 2:])):
        raise ValueError('Real checkpoint coarse/refinement swap smoke failed')
    write(root / 'smoke.json', dict(status='PASS', active=count, clean_bypassed=2,
                                  prior_buffers_equal=True, target_reads=0))
    print('LORA_FROZEN_STAGE_SPLICE_REAL_CHECKPOINT_PASS', flush=True)


@torch.no_grad()
def timing(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    (_, direction, lag), threshold = bind(root, contract, device)
    donor = refinement_state(direction)
    tracker = PacketRestorer('consensus').to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026113627)
    noisy = real_iq(colored_observation(complex_iq(cal), rng))
    pick = corruption_score(coordinates(tracker, noisy)[0]).argmax().view(1)
    result = {}
    for name, iq in (('clean', cal[:1]), ('generic', noisy[pick])):
        durations = []
        for _ in range(32):
            torch.cuda.synchronize()
            started = perf_counter()
            y, rot, q, rms = coordinates(tracker, iq)
            active = corruption_score(y) > threshold
            if active.any():
                output = functional_call(lag, donor, (y[active], q[active], rot[active]))[1]
                processed = iq.clone()
                processed[active] += real_iq((torch.lerp(y[active], output, .5) - y[active])
                                             * rot[active] * rms[active])
            torch.cuda.synchronize()
            durations.append(1000*(perf_counter()-started))
        result[name] = dict(active=int(active.sum()), median_ms=float(np.median(durations[8:])))
    write(root / 'b1.json', dict(gpu=torch.cuda.get_device_name(), n_measurements=24, timing=result,
          restoration_weights=sum(p.numel() for p in lag.parameters()),
          scope='one lag-coarse/direction-refinement path excluding DURA and F'))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest='action', required=True)
    release = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        release.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'timing', 'evaluate', 'score'):
        parser = actions.add_parser(action)
        parser.add_argument('runtime', type=Path)
        if action != 'dispatch':
            parser.add_argument('row', choices=(*ROUTES, 'reserve') if action == 'evaluate' else ('source',))
    opts = vars(cli.parse_args())
    stage = opts.pop('action')
    if stage == 'publish':
        return publish(**opts, run=RUN, runner=__spec__.name)
    rt = read(opts['runtime'])
    if stage == 'dispatch':
        return dispatch(rt)
    if stage == 'score':
        return independent_score(rt, modes=MODES, comparisons=PAIRS)
    activate(rt)
    if stage == 'smoke':
        return smoke(rt)
    if stage == 'timing':
        return timing(rt)
    if opts['row'] == 'reserve':
        return reserve(rt)
    return evaluate(rt, opts['row'])


if __name__ == '__main__':
    main()
