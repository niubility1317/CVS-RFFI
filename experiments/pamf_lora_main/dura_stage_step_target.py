"""Gate DURA's inter-stage displacement using the fitted receive action."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .antithetic_detail_source import load_arm
from .carrier_coupled import packet_frame
from .channel_blind import corruption_score, zero_carrier
from .channel_blind_source import setup
from .complex_vector_source import load_reader
from .dura_prior_width_source import RECIPES, RUN as WIDTH_RUN
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .fixed_mid_target import dispatch, predict, publish, source_smoke_metadata
from .joint_unroll_source import coordinates
from .relative_carrier import match_packet_power, mix
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-dura-stage-step-target-m2-r01'
MODES = ('first_quarter', 'second_quarter', 'fixed_mid_quarter', 'broadband_step', 'noncarrier_step')
PRIMARY = 'noncarrier_step'
RECIPE = RECIPES['wide']
DOSE = .25


def weighted_inner(left, right, weight):
    return (weight * (left.conj() * right).real).sum(dim=(1, 2), keepdim=True)


def accepted_step(y, first, second, first_observation, second_observation, observe, weight, *, noncarrier):
    """Minimize along the raw chord, then safeguard the emitted power-matched waveform."""
    residual = y - first_observation
    change = second_observation - first_observation
    if noncarrier:
        residual = zero_carrier(residual)
        change = zero_carrier(change)
    denominator = weighted_inner(change, change, weight)
    alpha = (weighted_inner(residual, change, weight) /
             torch.where(denominator > 0, denominator, 1.)).clamp(0., 1.)
    interior = match_packet_power(first + alpha * (second - first), first)
    proposal = torch.where(alpha == 1, second, interior)
    new_residual = y - observe(proposal)
    if noncarrier:
        new_residual = zero_carrier(new_residual)
    before = weighted_inner(residual, residual, weight)
    after = weighted_inner(new_residual, new_residual, weight)
    keep = (alpha > 0) & (after < before)
    return (torch.where(keep, proposal, first), torch.where(keep, alpha, 0.),
            before, torch.where(keep, after, before))


def stage_reads(y, rotation, model):
    native = real_iq(y * rotation).flatten(0, 1)
    stages, variance, action, _ = model.restorer.stage_outputs(native)

    def to_frame(wave):
        framed = complex_iq(wave.unflatten(0, y.shape[:2])) * rotation.conj()
        return match_packet_power(framed, y)

    first, second = (mix(y, to_frame(stage), DOSE) for stage in stages)
    midpoint = mix(first, second, .5)
    variance = variance.reshape_as(y.real)
    weight = variance.amin(dim=(1, 2), keepdim=True) / variance

    def observe(wave):
        return action((wave * rotation).flatten(0,1)).reshape_as(y) * rotation.conj()

    first_observation, second_observation = observe(first), observe(second)
    results = [accepted_step(y, first, second, first_observation, second_observation, observe, weight,
                             noncarrier=flag) for flag in (False, True)]
    waves = (first, second, midpoint, *(entry[0] for entry in results))
    return waves, tuple(entry[1:] for entry in results)


def bind(root, contract, device):
    return load_arm(root, RECIPE, contract, device, {'wide': WIDTH_RUN})


def frozen_runtime(rt, contract, device):
    root = Path(rt['run_root'])
    model, threshold = bind(root, contract, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    reader = load_reader(root, contract, device)
    return tracker, model, threshold, reader


@torch.no_grad()
def readouts(iq, tracker, model, threshold):
    frame = coordinates(tracker, iq)
    active = corruption_score(frame[0]) > threshold
    count = int(active.sum())
    if not count:
        return (iq,) * len(MODES), 0
    y, rotation, _, rms = packet_frame(tuple(v[active] for v in frame))
    waves, _ = stage_reads(y, rotation, model)
    rows = []
    for wave in waves:
        current = iq.clone()
        current[active] += real_iq((wave - y) * rotation * rms)
        rows.append(current)
    return tuple(rows), count


@torch.no_grad()
def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    tracker, model, threshold, reader = frozen_runtime(rt, contract, device)
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupt = real_iq(colored_observation(complex_iq(cal), rng))
    active = corruption_score(coordinates(tracker, corrupt)[0]) > threshold
    disturbed = corrupt[active][:2]
    iq = torch.cat((cal[:2], disturbed))
    waves, count = readouts(iq, tracker, model, threshold)
    values = torch.stack(waves)
    scores = reader(values.flatten(0, 1))[0]
    y, rotation, _, _ = packet_frame(coordinates(tracker, disturbed))
    stages, checks = stage_reads(y, rotation, model)
    expected = mix(y, model(y, None, rotation)[-1], DOSE)
    torch.testing.assert_close(stages[1], expected, atol=0., rtol=0.)
    if (count != 2 or scores.shape != (len(MODES)*len(iq), 10)
            or not values.isfinite().all() or not scores.isfinite().all()
            or not torch.equal(values[:, :2], iq[:2].expand_as(values[:, :2]))
            or torch.equal(values[0,2:], values[1,2:])
            or any(not ((after <= before) & alpha.isfinite()).all() for alpha,before,after in checks)):
        raise ValueError('Real frozen DURA stage-step smoke failed')
    record = dict(source_smoke_metadata(WIDTH_RUN, MODES), active=count, clean_bypassed=2,
                  unchanged_second_stage=True)
    write(root/'smoke.json', record)
    print('LORA_DURA_STAGE_STEP_REAL_CHECKPOINT_PASS', flush=True)


@torch.no_grad()
def timing(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    model, threshold = bind(root, contract, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026114451)
    corrupt = real_iq(colored_observation(complex_iq(cal), rng))
    pick = corruption_score(coordinates(tracker, corrupt)[0]).argmax().view(1)
    data = {}
    for label, example in (('clean', cal[:1]), ('generic', corrupt[pick])):
        times = []
        for _ in range(32):
            torch.cuda.synchronize()
            start = perf_counter()
            _, active = readouts(example, tracker, model, threshold)
            torch.cuda.synchronize()
            times.append((perf_counter() - start) * 1000)
        data[label] = dict(active=active, median_ms=float(np.median(times[8:])))
    write(root/'b1.json', dict(gpu=torch.cuda.get_device_name(), n_measurements=24,
          parameters=sum(p.numel() for p in model.parameters()), timing=data,
          scope='one frozen DURA with both consistency tests and all staged readouts, excludes classifier'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    release = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config','source-submit'):
        release.add_argument('--'+key, type=Path, required=True)
    scorer = actions.add_parser('score')
    for key in ('submit','ssh-config','dataset','output','full-control','residual-control'):
        scorer.add_argument('--'+key, type=Path, required=True)
    for action in ('dispatch','smoke','predict'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        if action != 'dispatch':
            worker.add_argument('route', choices=ROUTES if action == 'predict' else ('source',))
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        return publish(**options, run=RUN, source_run=WIDTH_RUN, runner=__spec__.name)
    if action == 'score':
        from .fixed_mid_target_score import collect
        return collect(**options, run=RUN, modes=MODES, primary=PRIMARY)
    rt = read(options['runtime'])
    if action == 'dispatch':
        return dispatch(rt)
    activate(rt)
    if action == 'smoke':
        smoke(rt)
        return timing(rt)
    return predict(rt, options['route'], modes=MODES, source_run=WIDTH_RUN,
                   prepare=frozen_runtime, evaluate=readouts)


if __name__ == '__main__':
    main()
