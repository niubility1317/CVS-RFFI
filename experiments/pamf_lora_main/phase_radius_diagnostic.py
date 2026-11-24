"""Test observable LoRa envelope recovery on a fixed phase-restored waveform."""

import argparse
from functools import partial
from math import log
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from . import phase_dose_bursts_source as parent
from .channel_blind import corruption_score
from .channel_blind_source import panel, setup
from .complex_vector_source import load_reader, score as predict
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze, publish, read_frozen, score_frozen
from .noise_floor_diagnostic import ROUTES, dispatch
from .noise_split_diagnostic import replay_trace
from .phase_dose_source import recover
from .reference_cancel import generic_observation
from .runner import activate
from .spectral_block_source import load_models


RUN = '20261124-diagnostic-lora-phase-radius-source-m2-r01'
CONTEXT = ('direct', 'reference', 'local_full')
RECEIVER = ('power_radius', 'fold_radius', 'power_joint', 'fold_joint', 'power_equal', 'fold_equal')
ISOLATED = ('exact_radius_equal', 'exact_joint_equal', 'exact_radius_all', 'exact_joint_all')
MODES = (*CONTEXT, *RECEIVER, *ISOLATED)
COMPARISONS = (('fold_equal', 'power_equal'), ('exact_joint_equal', 'fold_equal'),
               ('exact_joint_equal', 'power_equal'), ('exact_joint_all', 'exact_radius_all'),
               *((mode, base) for mode in (*RECEIVER, *ISOLATED)
                 for base in ('direct', 'reference', 'local_full')))
FIELDS = ('active', 'outer_power', 'fold_coherence', 'power_radius_rms', 'fold_radius_rms',
          'equal_radius_rms', 'exact_equal_radius_rms', 'exact_all_radius_rms',
          'power_total_rms', 'fold_total_rms')
DEVICE = 'cuda:0'


def radial_ratio(amplitude, corrected):
    total = amplitude.square().sum((-2, -1)).clamp_min(1e-12)
    return ((amplitude - corrected).square().sum((-2, -1)) / total).sqrt()


def radius_path(amplitude, proposal, ratio):
    """Match a bounded physical edit while preserving each original chirp's RMS."""
    proposal = torch.where(amplitude > 1e-10, proposal, amplitude)
    target_energy = amplitude.norm(dim=-1, keepdim=True)
    bound = torch.as_tensor(ratio, device=amplitude.device, dtype=amplitude.dtype)
    bound = bound.expand_as(amplitude[:, :1, :1])
    target_square = (target_energy.square().sum(dim=1, keepdim=True) * bound.square())

    def develop(scale):
        mixed = torch.lerp(amplitude, proposal, scale)
        norm = mixed.norm(dim=-1, keepdim=True)
        return torch.where(norm > 1e-9, mixed * (target_energy / norm.clamp_min(1e-9)), amplitude)

    high = torch.ones_like(bound)
    needs_cap = (develop(high) - amplitude).square().sum((-2, -1), keepdim=True) > target_square
    low = torch.zeros_like(high)
    for _ in range(18):
        mid = (low + high) * .5
        allowed = (develop(mid) - amplitude).square().sum((-2, -1), keepdim=True) <= target_square
        low, high = torch.where(allowed, mid, low), torch.where(allowed, high, mid)
    scale = torch.where(needs_cap, low, 1.)
    return torch.where((bound > 0.) & (scale > 0.), develop(scale), amplitude)


def radius_proposals(measured, quality, level):
    magnitude = measured.abs()
    distance = torch.fft.fftfreq(measured.shape[-1], device=measured.device).abs()
    outer = torch.fft.fft(measured, norm='ortho')[..., distance > .25]
    variance = outer.abs().square().median(-1, keepdim=True).values / log(2.)
    power = magnitude.square()
    corrected = torch.maximum(power - variance, .25 * power).sqrt()
    weights = quality.clamp_min(.01).square()

    def fold(shifts):
        numerator = sum((measured * weights).roll(i, 1) for i in shifts)
        denominator = sum(weights.roll(i, 1) for i in shifts)
        return numerator / denominator.clamp_min(1e-8)

    left, right = fold((1, 3, 5)), fold((2, 4))
    joint = (left * right.conj()).real
    positive = joint.clamp_min(0.)
    coherence = (positive / (left.abs() * right.abs()).clamp_min(1e-8)).clamp_max(1.)
    support = .5 * coherence * positive / (.10 + positive + variance)
    envelope = torch.lerp(magnitude, positive.sqrt(), support)
    return corrected * level, envelope * level, variance, coherence


def replace_radius(wave, amplitude, new):
    # At zero amplitude retain the observed IQ. No phase is observed there.
    return torch.where((amplitude > 1e-10) & (new != amplitude),
                       wave * (new / amplitude.clamp_min(1e-10)), wave)


def received(iq, tracker, model, threshold):
    phase_iq, _, _ = recover(iq, tracker, (model,), threshold)
    phased = complex_iq(phase_iq[1])
    raw = complex_iq(iq)
    y, rotation, quality, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    outputs = raw.expand(len(RECEIVER), *raw.shape).clone()
    matching = iq.new_zeros((len(iq), 1, 1))
    evidence = iq.new_zeros((len(iq), 6))
    if active.any():
        index = (y[active], quality[active], rms[active])
        power, fold, variance, coherence = radius_proposals(*index)
        sample, phase = raw[active], phased[active]
        amplitude = sample.abs()
        power_radius, fold_radius = (radius_path(amplitude, prop, .06) for prop in (power, fold))
        ratios = [radial_ratio(amplitude, prop) for prop in (power_radius, fold_radius)]
        budget = torch.minimum(*ratios)[:, None, None]
        equal_power = radius_path(amplitude, power, budget)
        equal_fold = radius_path(amplitude, fold, budget)
        power_joint = replace_radius(phase, amplitude, power_radius)
        fold_joint = replace_radius(phase, amplitude, fold_radius)
        outputs[:, active] = torch.stack((replace_radius(sample, amplitude, power_radius),
                                          replace_radius(sample, amplitude, fold_radius), power_joint, fold_joint,
                                          replace_radius(phase, amplitude, equal_power),
                                          replace_radius(phase, amplitude, equal_fold)))
        matching[active] = budget
        denom = amplitude.square().sum((-2, -1)).clamp_min(1e-10)
        edits = [((z - sample).abs().square().sum((-2, -1)) / denom).sqrt() for z in (power_joint, fold_joint)]
        evidence[active] = torch.stack((variance.mean((1, 2)), coherence.mean((1, 2)), *ratios, *edits), -1)
    return raw, phased, outputs, active, matching, evidence, rotation * rms


def isolated(raw, phased, epsilon, frame, common):
    wave = complex_iq(epsilon)
    noise = wave / frame
    present = wave.abs().amax((-2, -1), keepdim=True) > 0.
    reference = raw - (noise - noise.mean(-1, keepdim=True)) * frame
    magnitude = raw.abs()
    matched = radius_path(magnitude, reference.abs(), common)
    full = radius_path(magnitude, reference.abs(), torch.where(present, 2., 0.))
    outputs = (replace_radius(raw, magnitude, matched), replace_radius(phased, magnitude, matched),
               replace_radius(raw, magnitude, full), replace_radius(phased, magnitude, full))
    return torch.stack(outputs), (radial_ratio(magnitude, matched), radial_ratio(magnitude, full))


def render(iq, epsilon, baseline, tracker, models, *, threshold, measurements=None):
    ref, _ = baseline(iq)
    raw, phase, actual, active, budget, values, frame = received(iq, tracker, models[0], threshold)
    exact, (matched, full) = isolated(raw, phase, epsilon, frame, budget)
    if measurements is not None:
        table = torch.cat((active[:, None], values[:, :4], budget[:, 0], matched[:, None],
                           full[:, None], values[:, -2:]), dim=1)
        measurements.append(np.asarray(table.tolist(), dtype=np.float64))
    options = torch.cat((phase.unsqueeze(0), actual, exact))
    return torch.cat((torch.stack((iq, ref)), torch.stack((options.real, options.imag), dim=-2))), int(active.sum())


def original(rt, contract):
    root = Path(rt['run_root']).parent / parent.RUN
    if read(root / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Require the frozen model with the exact two-route source contract')
    models, threshold = load_models(root, contract, DEVICE, parent.RECIPES)
    return root, models[-1], threshold


@torch.no_grad()
def smoke(rt):
    (_, source, _), contract = setup(rt, DEVICE, smoke=True)
    _, model, threshold = original(rt, contract)
    _, _, bound, baseline, tracker = source_control(rt, 'full', DEVICE)
    if bound != contract:
        raise ValueError('Unexpected frozen classifier contract')
    rng = torch.Generator(device=DEVICE).manual_seed(2026112496)
    distorted = real_iq(generic_observation(complex_iq(source), rng))
    keep = corruption_score(coordinates(tracker, source)[0]).argmin().view(1)
    choose = corruption_score(coordinates(tracker, distorted)[0]).argmax().view(1)
    clean, damaged = source[keep], distorted[choose]
    noise = torch.cat((torch.zeros_like(clean), damaged - source[choose]))
    measurements = []
    iq, active = render(torch.cat((clean, damaged)), noise, baseline, tracker, (model,),
                        threshold=threshold, measurements=measurements)
    reader = load_reader(Path(rt['run_root']), contract, DEVICE)
    logits = reader(iq.flatten(0, 1))[0]
    if (active != 1 or logits.shape != (2 * len(MODES), 10) or not torch.isfinite(logits).all()
            or not torch.equal(iq[2:, 0], clean.expand_as(iq[2:, 0]))
            or measurements[0].shape != (2, len(FIELDS))):
        raise ValueError('Expected finite real checkpoint outputs and exact clean bypass')
    timing = {}
    for tag, wave in (('clean', clean), ('generic_disturbed', damaged)):
        samples = []
        for _ in range(22):
            torch.cuda.synchronize()
            t0 = perf_counter()
            received(wave, tracker, model, threshold)
            torch.cuda.synchronize()
            samples.append(1000 * (perf_counter() - t0))
        timing[tag] = float(np.median(samples[6:]))
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', diagnostic_only=True, V_reads=0,
          target_reads=0, timing_b1_received_matrix_ms=timing))


def evaluate(rt, route):
    binding = source_control(rt, route, DEVICE)
    prior, model, threshold = original(rt, binding[2])
    previous_ids, previous, marks = read_frozen(prior, route, parent.MODES)
    inherited = [parent.MODES.index(mode) for mode in CONTEXT]
    records = []

    def finalize(root, route, source, chosen, ids, decisions, metrics, modes):
        if ids[chosen].tolist() != previous_ids or metrics != marks:
            raise ValueError('Inherited physical source panel or guard changed')
        if any(not np.array_equal(decisions[view][:len(CONTEXT)], previous[view][inherited]) for view in VIEWS):
            raise ValueError('Inherited decisions differ from the frozen phase study')
        measure = np.concatenate(records).reshape(len(VIEWS), len(chosen), len(FIELDS))
        rows = {}
        for name, scene in zip(VIEWS, measure, strict=True):
            result = {}
            for rx, group in zip((*SOURCE, 'all'), (*np.split(scene, len(SOURCE)), scene), strict=True):
                active = group[group[:, 0] > 0.]
                matched = active[active[:, 5] > 1e-7]
                values = dict(zip(FIELDS[1:], np.median(active[:, 1:], axis=0).tolist(), strict=True)) if len(active) else {}
                result[rx] = dict(active=len(active), equal_nonzero=len(matched),
                                  exact_equal_reached=int((np.abs(matched[:, 6] - matched[:, 5]) < 5e-6).sum()),
                                  **values)
            rows[name] = result
        write(root / f'{route}-radius-evidence.json', dict(by_view=rows, diagnostic_only=True))
        freeze(root, route, source, chosen, ids, decisions, metrics, modes)

    predict(rt, route, panel=panel, modes=MODES, binding=binding, frozen_models=(model,),
            replay_fn=replay_trace, render=partial(render, threshold=threshold, measurements=records),
            require_clean_bypass=True, finalize=finalize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + key, type=Path, required=True)
    for key in ('dispatch', 'smoke', 'evaluate', 'score'):
        worker = actions.add_parser(key)
        worker.add_argument('runtime', type=Path)
        if key != 'dispatch':
            worker.add_argument('name', choices=ROUTES if key == 'evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args, run=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action == 'dispatch':
        dispatch(rt)
    elif action == 'score':
        score_frozen(rt, MODES, comparisons=COMPARISONS)
    else:
        activate(rt)
        if action == 'smoke':
            smoke(rt)
        else:
            evaluate(rt, args['name'])


if __name__ == '__main__':
    main()
