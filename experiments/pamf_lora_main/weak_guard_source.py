"""Score a receiver-only guard-band extension of frozen rank-four restoration."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze, score_frozen
from .complex_vector_source import load_reader, publish, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .mirror_refine_source import load_model
from .packet_runner import schedule
from .refine_runner import start
from .runner import activate
from .spectral_covariance_source import RECIPES, RUN as RANK_RUN
from .spectral_unroll import ROUTES
from .weak_guard import (FIT_SEED, GUARD_FEATURES, HELD_DOSES, HELD_SEED, GuardFloor,
                         WeakGuard, apply_scores, generic_noise, monotone_scores, proposed_doses)


RUN = '20261124-phase1-lora-rank-guard-extension-source-m2-r01'
SCHEMA = 'lora_weak_guard_v1'
ISOLATION = dict(F_reads=0, V_reads=0, TX_reads=0, target_reads=0, practical_reads=0, trace_reads=0)
MODES = ('direct', 'reference', 'rank_old', 'guard_quarter', 'guard_half', 'guard_variance')
PAIRS = (('guard_quarter', 'rank_old'), ('guard_half', 'rank_old'), ('guard_variance', 'rank_old'),
         ('guard_half', 'guard_quarter'), ('guard_variance', 'guard_quarter'))


def parent_model(root, contract, device):
    parent = root.parent / RANK_RUN
    if read(parent / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the validated source-only rank-four checkpoint')
    return load_model(parent, RECIPES['rank'], contract, device)


def collect(extractor, iq):
    return torch.cat([extractor(part) for part in iq.split(80)])


@torch.no_grad()
def source_fit(rt, *, smoke=False):
    root, device = Path(rt['run_root']), 'cuda:0'
    (training, calibration, held), contract = setup(rt, device, smoke=smoke)
    feature = GuardFloor().to(device)
    rng = torch.Generator(device=device).manual_seed(FIT_SEED)
    start_time = perf_counter()
    clean = collect(feature, training).cpu()
    generic = torch.cat([collect(feature, real_iq(generic_noise(complex_iq(training), rng))).cpu()
                         for _ in range(2)])
    midpoint, dispersion, weight, offset = monotone_scores(generic, clean)
    threshold = float(apply_scores(collect(feature, calibration), midpoint.to(device),
                                   dispersion.to(device), weight.to(device), offset).max()) + .01
    model, old_threshold = parent_model(root, contract, device)
    record = dict(schema=SCHEMA, status='SOURCE_FROZEN', contract=contract, parent=RANK_RUN,
                  n_fit=len(training), n_cal=len(calibration), n_held=len(held), fit_seed=FIT_SEED,
                  held_seed=HELD_SEED, midpoint=midpoint.tolist(), dispersion=dispersion.tolist(),
                  weight=weight.tolist(), offset=offset, threshold=threshold, old_threshold=old_threshold,
                  worker_seconds=perf_counter()-start_time, **ISOLATION)
    gate = WeakGuard(record).to(device).eval()
    tracker = PacketRestorer('consensus').to(device).eval()
    held_rng = torch.Generator(device=device).manual_seed(HELD_SEED)
    if smoke:
        _, _, verified, reference, _ = source_control(rt, 'full', device)
        if contract != verified:
            raise ValueError('Expected matching rank and frozen reader IQ contract')
        old_clean = corruption_score(coordinates(tracker, calibration)[0]) > old_threshold
        clean = calibration[~old_clean][:2]
        perturbation = real_iq(generic_noise(complex_iq(calibration), held_rng, amplitude=.025))
        old_perturbed = corruption_score(coordinates(tracker, perturbation)[0]) > old_threshold
        extra = perturbation[gate(perturbation) & ~old_perturbed][:2]
        if len(clean) != 2 or not len(extra):
            raise ValueError('Weak guard smoke needs clean bypass and extension examples')
        batch = torch.cat((clean, extra))
        outputs, active = render(batch, None, reference, tracker, (model, gate), old_threshold=old_threshold)
        logits = load_reader(root, contract, device)(outputs.flatten(0, 1))[0]
        if active != len(extra) or logits.shape != (len(batch) * len(MODES), 10) or not torch.isfinite(logits).all():
            raise ValueError('Weak guard real-checkpoint no-query smoke failed')
        if gate(calibration).any() or not torch.equal(outputs[2:, :2], clean.expand_as(outputs[2:, :2])):
            raise ValueError('Clean bypass failed on calibration smoke')
        if not torch.equal(outputs[2, 2:], extra) or torch.equal(outputs[3, 2:], extra):
            raise ValueError('Guard extension changed the wrong smoke readout')
        write(root / 'checkpoint_smoke.json', dict(status='PASS', prechecked=active, source_U=len(batch),
                                                    variants=len(MODES), target_reads=0))
        print('LORA_WEAK_GUARD_CHECKPOINT_PASS', active, flush=True)
        return

    diagnostics = {}
    for dose in (0., *HELD_DOSES):
        evaluated = held if not dose else real_iq(generic_noise(complex_iq(held), held_rng, amplitude=dose))
        old, new = [], []
        for part in evaluated.split(48):
            y = coordinates(tracker, part)[0]
            old.append(corruption_score(y) > old_threshold)
            new.append(gate(part))
        before, after = torch.cat(old), torch.cat(new)
        diagnostics[str(dose)] = dict(old=int(before.sum()), guard=int(after.sum()), extra=int((after & ~before).sum()))
    record['held_counts'] = diagnostics
    write(root / 'weak-guard.json', record)
    print('LORA_WEAK_GUARD_SOURCE_FROZEN', len(training), len(calibration), diagnostics, flush=True)


def frozen_guard(root, contract, device):
    record = read(root / 'weak-guard.json')
    if (record['contract'] != contract or record['schema'] != SCHEMA or record['status'] != 'SOURCE_FROZEN'
            or record['parent'] != RANK_RUN or any(record[k] != value for k, value in ISOLATION.items())):
        raise ValueError('Unexpected source-only weak guard provenance')
    for key in ('midpoint', 'dispersion', 'weight'):
        if len(record[key]) != GUARD_FEATURES or not np.isfinite(record[key]).all():
            raise ValueError('Malformed source weak guard')
    if min(record['dispersion']) <= 0 or not np.isfinite([record['offset'], record['threshold']]).all():
        raise ValueError('Invalid source weak-guard calibration')
    return WeakGuard(record).to(device).eval(), record['old_threshold']


@torch.no_grad()
def recover(iq, tracker, model, gate, old_threshold):
    y, phase, q, rms = coordinates(tracker, iq)
    measure = corruption_score(y)
    old = measure > old_threshold
    seen = gate(iq)
    extension = seen & ~old
    masks = old | extension
    if not bool(masks.any()):
        return [iq] * 4, 0
    old_output = iq.clone()
    if bool(old.any()):
        refined = model(y[old], q[old], phase[old])[-1]
        delta = torch.lerp(y[old], refined, .5) - y[old]
        old_output[old] += real_iq(delta * phase[old] * rms[old])
    if not bool(extension.any()):
        return [old_output] * 4, int(old.sum())
    estimate = model(y[extension], q[extension], phase[extension])[-1]
    doses = proposed_doses(measure, old_threshold, old, seen)
    arms = [old_output]
    for dose in doses[1:]:
        result = old_output.clone()
        weight = dose[extension, None, None]
        delta = (estimate - y[extension]) * weight
        result[extension] += real_iq(delta * phase[extension] * rms[extension])
        arms.append(result)
    return arms, int(masks.sum())


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, old_threshold):
    model, gate = models
    reference, _ = baseline(iq)
    arms, active = recover(iq, tracker, model, gate, old_threshold)
    return torch.stack((iq, reference, *arms)), active


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    contract = binding[2]
    model, threshold = parent_model(root, contract, device)
    gate, expected = frozen_guard(root, contract, device)
    if threshold != expected:
        raise ValueError('Parent and frozen weak guard disagree on the old threshold')
    score(rt, route, panel=panel, modes=MODES, render=partial(render, old_threshold=threshold),
          binding=binding, frozen_models=(model, gate), require_clean_bypass=False,
          replay_fn=receive_only, finalize=freeze, reader_chunk=144)


@torch.no_grad()
def timing(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    model, threshold = parent_model(root, contract, device)
    gate, expected = frozen_guard(root, contract, device)
    if threshold != expected:
        raise ValueError('Unexpected old threshold in timing')
    tracker = PacketRestorer('consensus').to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026113157)
    generic = real_iq(generic_noise(complex_iq(cal), rng, amplitude=.02))
    old = corruption_score(coordinates(tracker, generic)[0]) > threshold
    available = torch.where(gate(generic) & ~old)[0]
    times = {}
    cases = [('clean', cal[:1])]
    if len(available):
        cases.append(('generic_extra_trigger', generic[available[:1]]))
    for name, value in cases:
        samples = []
        for _ in range(32):
            torch.cuda.synchronize()
            start_time = perf_counter()
            _, active = recover(value, tracker, model, gate, threshold)
            torch.cuda.synchronize()
            samples.append((perf_counter()-start_time)*1000)
        times[name] = dict(median_ms=float(np.median(samples[8:])), active=active)
    write(root / 'b1.json', dict(n_measurements=24, timing=times, gpu=torch.cuda.get_device_name(),
                               from_rank_parameters=sum(p.numel() for p in model.parameters()), gate_coefficients=9))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    launch = partial(start, module=rt['runner'])
    stages = [('smoke', ('source',)), ('fit', ('source',)), ('timing', ('source',)), ('evaluate', ROUTES)]
    try:
        for action, rows in stages:
            schedule(rt, [(action, row) for row in rows], launch)
        if launch(rt, 'score', 'source').wait():
            raise RuntimeError('Independent weak guard source scoring failed')
        write(root / 'state.json', dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root / 'state.json', dict(status='FAILED', error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for flag in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + flag, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        stage = commands.add_parser(action)
        stage.add_argument('runtime', type=Path)
        if action != 'dispatch':
            stage.add_argument('row', choices=ROUTES if action=='evaluate' else ('source',))
    opts = vars(parser.parse_args())
    action = opts.pop('action')
    if action == 'publish':
        publish(**opts, run=RUN, runner=__spec__.name)
        return
    rt = read(opts['runtime'])
    if action == 'dispatch':
        dispatch(rt)
    elif action == 'score':
        score_frozen(rt, modes=MODES, comparisons=PAIRS)
    else:
        activate(rt)
        if action == 'evaluate':
            evaluate(rt, opts['row'])
        elif action == 'timing':
            timing(rt)
        else:
            source_fit(rt, smoke=action=='smoke')


if __name__ == '__main__':
    main()
