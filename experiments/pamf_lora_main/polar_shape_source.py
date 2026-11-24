"""Isolate the amplitude and phase edits of one frozen source LoRa recovery."""
import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .action_support_source import independent_score, predict_clean_reserve
from .antithetic_detail_source import load_arm, read_stages
from .carrier_coupled import packet_frame
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze
from .complex_vector_source import load_reader, publish, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .independent_shape_source import NEW_ARM, RECIPES, RUN as SHAPE_RUN
from .joint_unroll_source import coordinates, source_control
from .private_transport_source import dispatch
from .relative_carrier import mix
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-polar-shape-source-m2-r01'
MODES = ('direct', 'reference', 'own_shape_coarse', 'own_shape_half',
         'phase_coarse', 'amplitude_coarse', 'phase_half', 'amplitude_half')
PAIRS = (('own_shape_coarse', 'direct'), ('own_shape_half', 'direct'),
         ('phase_coarse', 'own_shape_coarse'), ('amplitude_coarse', 'own_shape_coarse'),
         ('phase_half', 'own_shape_half'), ('amplitude_half', 'own_shape_half'),
         ('phase_half', 'amplitude_half'))
RECIPE = RECIPES[NEW_ARM]


def polar_parts(observed, proposal):
    observed_abs, proposal_abs = observed.abs(), proposal.abs()
    observed_nonzero, proposal_nonzero = observed_abs > 0, proposal_abs > 0
    observed_unit = observed / torch.where(observed_nonzero, observed_abs, 1.)
    proposal_unit = proposal / torch.where(proposal_nonzero, proposal_abs, 1.)
    phase_only = torch.where(proposal_nonzero, observed_abs * proposal_unit, observed)
    amplitude_only = torch.where(observed_nonzero, proposal_abs * observed_unit, proposal)
    return phase_only, amplitude_only


def bind(root, contract, device):
    return load_arm(root, RECIPE, contract, device, {NEW_ARM: SHAPE_RUN})


@torch.no_grad()
def readouts(iq, tracker, model, threshold):
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    count = int(active.sum())
    if not count:
        return (iq,) * (len(MODES) - 2), 0
    y, rotation, quality, rms = packet_frame(tuple(value[active] for value in frame))
    coarse, fine = read_stages(model, y, quality, rotation, rms, RECIPE)
    half = mix(y, fine, .5)
    proposed = (coarse, half, *polar_parts(y, coarse), *polar_parts(y, half))
    result = []
    for wave in proposed:
        value = iq.clone()
        value[active] += real_iq((wave - y) * rotation * rms)
        result.append(value)
    return tuple(result), count


@torch.no_grad()
def render(iq, _unused, baseline, tracker, model, *, threshold):
    reference, _ = baseline(iq)
    waves, count = readouts(iq, tracker, model, threshold)
    return torch.stack((iq, reference, *waves)), count


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    model, threshold = bind(root, binding[2], device)
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=threshold),
          binding=binding, frozen_models=model, require_clean_bypass=False,
          replay_fn=receive_only, finalize=freeze, reader_chunk=144)


def reserve(rt):
    def build(root, contract, device, baseline, tracker):
        model, threshold = bind(root, contract, device)
        return lambda iq: render(iq, None, baseline, tracker, model, threshold=threshold)
    predict_clean_reserve(rt, modes=MODES, bind=build)


@torch.no_grad()
def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    _, _, matched, baseline, tracker = source_control(rt, 'full', device)
    if matched != contract:
        raise ValueError('Source IQ and frozen recognition input contracts differ')
    model, threshold = bind(root, contract, device)
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupted = real_iq(colored_observation(complex_iq(cal), rng))
    active = corruption_score(coordinates(tracker, corrupted)[0]) > threshold
    iq = torch.cat((cal[:2], corrupted[active][:2]))
    waves, count = render(iq, None, baseline, tracker, model, threshold=threshold)
    logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    if (count != 2 or logits.shape != (len(MODES) * len(iq), 10)
            or not logits.isfinite().all()
            or not torch.equal(waves[2:, :2], iq[:2].expand_as(waves[2:, :2]))):
        raise ValueError('Polar readouts failed real source smoke')
    for joint, phase, amplitude in ((2,4,5),(3,6,7)):
        raw, combined, angle, size = (complex_iq(waves[index, 2:]) for index in (0,joint,phase,amplitude))
        torch.testing.assert_close(angle.abs(), raw.abs(), atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(size.abs(), combined.abs(), atol=3e-6, rtol=2e-5)
        if torch.equal(angle, raw) or torch.equal(size, raw):
            raise ValueError('Expected both polar edits to be exercised')
    write(root/'smoke.json', dict(status='PASS', source_U=len(iq), active=count, clean_bypassed=2, target_reads=0))
    print('LORA_POLAR_SHAPE_REAL_CHECKPOINT_PASS', flush=True)


@torch.no_grad()
def timing(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    model, threshold = bind(root, contract, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupt = real_iq(colored_observation(complex_iq(cal), rng))
    pick = corruption_score(coordinates(tracker, corrupt)[0]).argmax().view(1)
    results = {}
    for label, iq in (('clean', cal[:1]), ('generic', corrupt[pick])):
        durations = []
        for _ in range(32):
            torch.cuda.synchronize()
            start = perf_counter()
            _, active = readouts(iq, tracker, model, threshold)
            torch.cuda.synchronize()
            durations.append(1000 * (perf_counter() - start))
        results[label] = dict(active=active, median_ms=float(np.median(durations[8:])))
    write(root/'b1.json', dict(gpu=torch.cuda.get_device_name(), n_measurements=24,
          parameters=sum(p.numel() for p in model.parameters()), timing=results,
          scope='one frozen restoration forward and six waveform readouts, excludes DURA and F'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest='action', required=True)
    release = steps.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        release.add_argument('--'+name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'timing', 'evaluate', 'score'):
        step = steps.add_parser(action)
        step.add_argument('runtime', type=Path)
        if action != 'dispatch':
            step.add_argument('row', choices=(*ROUTES, 'reserve') if action == 'evaluate' else ('source',))
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        return publish(**options, run=RUN, runner=__spec__.name)
    rt = read(options['runtime'])
    if action == 'dispatch':
        return dispatch(rt)
    if action == 'score':
        return independent_score(rt, modes=MODES, comparisons=PAIRS)
    activate(rt)
    if action == 'smoke':
        return smoke(rt)
    if action == 'timing':
        return timing(rt)
    if options['row'] == 'reserve':
        return reserve(rt)
    return evaluate(rt, options['row'])


if __name__ == '__main__':
    main()
