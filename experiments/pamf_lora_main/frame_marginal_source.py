"""Score synchronous uncertainty averaging for the frozen bounded LoRa recovery."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .action_support_source import independent_score, predict_clean_reserve
from .antithetic_detail_source import bind_models, read_stages
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .gated_slope_source import restore
from .joint_unroll_source import coordinates, source_control
from .frame_marginal import marginal_readouts
from .refinement_bound_source import RECIPES as FROZEN, RUN as BOUND_RUN
from .private_transport_source import dispatch
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-frame-marginal-source-m2-r01'
MODES = ('direct', 'reference', 'rank_half', 'bounded_half', 'frame_marginal', 'frame_equal_norm')
PAIRS = (('frame_marginal', 'bounded_half'), ('frame_equal_norm', 'bounded_half'),
         ('frame_equal_norm', 'frame_marginal'), ('frame_marginal', 'rank_half'))
RECIPES = {'dose_bound': FROZEN['dose_bound']}
REUSED = {'dose_bound': BOUND_RUN}


def bind(root, contract, device):
    rank, stages, threshold = bind_models(root, contract, device, recipes=RECIPES, reused=REUSED)
    return (rank, *stages), threshold


@torch.no_grad()
def readouts(iq, tracker, models, threshold, *, spreads=None):
    rank, bounded = models
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    n = int(active.sum())
    if not n:
        return (iq,) * (len(MODES) - 2), n
    selected = tuple(v[active] for v in frame)
    y, rotation, quality, rms = selected
    rank_half, = restore(iq, active, rank, selected, (.5,))
    result = [rank_half]
    edits, uncertainty = marginal_readouts(y, rotation, quality, rms, bounded, RECIPES['dose_bound'])
    if spreads is not None:
        spreads.append(torch.cat(uncertainty, dim=-1).abs().cpu())
    for delta in edits:
        value = iq.clone()
        value[active] += .5 * real_iq(delta * rotation * rms)
        result.append(value)
    return tuple(result), n


@torch.no_grad()
def render(iq, _, baseline, tracker, models, *, threshold, spreads=None):
    reference, _ = baseline(iq)
    waves, count = readouts(iq, tracker, models, threshold, spreads=spreads)
    return torch.stack((iq, reference, *waves)), count


def evaluate(rt, route):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    binding = source_control(rt, route, device)
    models, threshold = bind(root, binding[2], device)
    spreads = []
    score(rt, route, panel=panel, modes=MODES,
          render=partial(render, threshold=threshold, spreads=spreads), binding=binding,
          frozen_models=models, require_clean_bypass=False, replay_fn=receive_only,
          finalize=freeze, reader_chunk=144)
    values = torch.cat(spreads).flatten(0, 1)
    write(root / f'{route}-spreads.json', dict(n_chirps=len(values), units='rad',
          labels=['phase', 'slope_from_midpoint_to_edge'],
          quantiles=torch.quantile(values, torch.tensor([.5, .90, .99]), dim=0).tolist()))


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
    rng = torch.Generator(device=device).manual_seed(2026113627)
    noisy = real_iq(colored_observation(complex_iq(cal), rng))
    active = corruption_score(coordinates(tracker, noisy)[0]) > threshold
    disturbed = noisy[active][:2]
    samples = torch.cat((cal[:2], disturbed))
    waves, count = render(samples, None, baseline, tracker, models, threshold=threshold)
    logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    _, model = models
    y, rot, q, rms = coordinates(tracker, disturbed)
    recipe = RECIPES['dose_bound']
    _, ref = read_stages(model, y, q, rot, rms, recipe)
    frozen = disturbed + .5 * real_iq((ref - y) * rot * rms)
    torch.testing.assert_close(waves[3, 2:], frozen, atol=2e-6, rtol=2e-5)
    if (count != 2 or logits.shape != (len(MODES)*len(samples), 10) or not logits.isfinite().all()
            or not torch.equal(waves[2:, :2], samples[:2].expand_as(waves[2:, :2]))
            or torch.equal(waves[4, 2:], frozen)):
        raise ValueError('Real checkpoint frame marginalization smoke failed')
    write(root / 'smoke.json', dict(status='PASS', active=count, clean_bypassed=2, target_reads=0))
    print('LORA_FRAME_MARGINAL_REAL_CHECKPOINT_PASS', flush=True)



@torch.no_grad()
def timing(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    (_, model), threshold = bind(root, contract, device)
    recipe = RECIPES['dose_bound']
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
                y, rot, q, rms = (v[active] for v in (y, rot, q, rms))
                (_, edit, _), _ = marginal_readouts(y, rot, q, rms, model, recipe)
                processed = iq.clone()
                processed[active] += .5 * real_iq(edit * rot * rms)
            torch.cuda.synchronize()
            durations.append(1000*(perf_counter()-started))
        result[name] = dict(active=int(active.sum()), median_ms=float(np.median(durations[8:])))
    write(root / 'b1.json', dict(gpu=torch.cuda.get_device_name(), n_measurements=24, timing=result,
          restoration_weights=sum(p.numel() for p in model.parameters()),
          scope='one frozen arm in three plausible rotations; excludes DURA and F'))


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
