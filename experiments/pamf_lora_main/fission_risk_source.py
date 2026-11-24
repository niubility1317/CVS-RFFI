"""Freeze and score one-packet coupled-bootstrap recovery mixtures on source LoRa."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .action_support_source import independent_score, predict_clean_reserve
from .antithetic_detail_source import bind_models
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .fission_risk import PROBE_SEED, mix_readouts
from .joint_unroll_source import coordinates, source_control
from .private_direction_risk_source import RECIPES, RUN as DIRECTION_RUN
from .private_transport_source import dispatch
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-fission-risk-source-m2-r01'
MODES = ('direct', 'reference', 'rank_half', 'direction_half', 'fixed_mid', 'split_global', 'split_bands')
PAIRS = tuple(('split_bands', control) for control in ('rank_half', 'direction_half', 'fixed_mid', 'split_global'))
RECIPE = {'direction_risk': RECIPES['direction_risk']}
REUSED = {'direction_risk': DIRECTION_RUN}


def bind(root, contract, device):
    parent, (direction,), threshold = bind_models(root, contract, device, recipes=RECIPE, reused=REUSED)
    return (parent, direction), threshold


@torch.no_grad()
def readouts(iq, tracker, controls, threshold, *, doses=None):
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    n = int(active.sum())
    if not n:
        return (iq,) * (len(MODES) - 2), n
    y, rotation, quality, rms = (v[active] for v in frame)
    edits, weights = mix_readouts(y, rotation, quality, rms, controls)
    if doses is not None:
        doses.append(torch.cat(weights, dim=-1).cpu())
    result = []
    for edit in edits:
        wave = iq.clone()
        wave[active] += real_iq(edit * rotation * rms)
        result.append(wave)
    return tuple(result), n


@torch.no_grad()
def render(iq, _, baseline, tracker, models, *, threshold, doses=None):
    reference, _ = baseline(iq)
    waves, n = readouts(iq, tracker, models, threshold, doses=doses)
    return torch.stack((iq, reference, *waves)), n


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    models, threshold = bind(root, binding[2], device)
    doses = []
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=threshold, doses=doses),
          binding=binding, frozen_models=models, require_clean_bypass=False, replay_fn=receive_only,
          finalize=freeze, reader_chunk=144)
    values = torch.cat(doses)
    write(root / f'{route}-doses.json', dict(active=len(values), probe_seed=PROBE_SEED,
          bands=['whole', '1_to_8', '9_to_32', '33_to_128'],
          quantiles=torch.quantile(values, torch.tensor([.1, .5, .9]), dim=0).tolist(),
          frac_zero=(values==0).float().mean(0).tolist(), frac_one=(values==1).float().mean(0).tolist()))


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
        raise ValueError('Source IQ and frozen classifier contracts differ')
    models, threshold = bind(root, contract, device)
    rng = torch.Generator(device=device).manual_seed(2026113627)
    noisy = real_iq(colored_observation(complex_iq(cal), rng))
    active = corruption_score(coordinates(tracker, noisy)[0]) > threshold
    disturbed = noisy[active][:2]
    y, rotation, quality, rms = coordinates(tracker, disturbed)
    result, doses = mix_readouts(y, rotation, quality, rms, models)
    for got, model in zip(result[:2], models, strict=True):
        expected = torch.lerp(y, model(y, quality, rotation)[-1], .5) - y
        torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)
    iq = torch.cat((cal[:2], disturbed))
    waves, n = render(iq, None, baseline, tracker, models, threshold=threshold)
    logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    if (n != 2 or logits.shape != (len(MODES)*len(iq), 10) or not logits.isfinite().all()
            or not torch.equal(waves[2:, :2], iq[:2].expand_as(waves[2:, :2]))
            or not all(v.isfinite().all() and ((v >= 0) & (v <= 1)).all() for v in doses)
            or torch.equal(result[-1], result[2])):
        raise ValueError('Real checkpoint fission smoke failed')
    write(root / 'smoke.json', dict(status='PASS', active=n, clean_bypassed=2, target_reads=0))
    print('LORA_FISSION_RISK_REAL_CHECKPOINT_PASS', flush=True)


@torch.no_grad()
def timing(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    models, threshold = bind(root, contract, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026113627)
    noisy = real_iq(colored_observation(complex_iq(cal), rng))
    pick = corruption_score(coordinates(tracker, noisy)[0]).argmax().view(1)
    result = {}
    for name, iq in (('clean', cal[:1]), ('generic', noisy[pick])):
        durations = []
        for _ in range(32):
            torch.cuda.synchronize()
            begin = perf_counter()
            _, n = readouts(iq, tracker, models, threshold)
            torch.cuda.synchronize()
            durations.append(1000*(perf_counter()-begin))
        result[name] = dict(active=n, median_ms=float(np.median(durations[8:])))
    write(root / 'b1.json', dict(gpu=torch.cuda.get_device_name(), n_measurements=24, timing=result,
          restoration_weights=sum(p.numel() for m in models for p in m.parameters()),
          scope='original and two antithetic views per model, excluding DURA and F'))


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
    options = vars(cli.parse_args())
    stage = options.pop('action')
    if stage == 'publish':
        return publish(**options, run=RUN, runner=__spec__.name)
    rt = read(options['runtime'])
    if stage == 'dispatch':
        return dispatch(rt)
    if stage == 'score':
        return independent_score(rt, modes=MODES, comparisons=PAIRS)
    activate(rt)
    if stage == 'smoke':
        return smoke(rt)
    if stage == 'timing':
        return timing(rt)
    if options['row'] == 'reserve':
        return reserve(rt)
    return evaluate(rt, options['row'])


if __name__ == '__main__':
    main()
