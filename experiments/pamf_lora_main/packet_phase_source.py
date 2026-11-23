"""Compare packet phase tracking and matched joint waveform recovery on source LoRa."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, panel, receive_only, setup
from .chirp_spectrum_source import dispatch, freeze, read_frozen, score_frozen
from .complex_vector_source import load_reader, publish, score
from .data import VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import source_control
from .mirror_refine_source import load_model
from .natural_detail import natural_objective, paired_echo_inputs
from .packet_phase import PacketPhaseTracker, frame_inputs
from .pilot_joint import PilotJoint
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import MIRROR_MODES, MIRROR_RECIPES, MIRROR_RUN, STEPS, TAIL_WEIGHT, WIDTHS


RUN = '20261124-phase1-lora-packet-phase-tracker-source-m3-r01'
RECIPE = FitRecipe('packet_phase', STEPS, partial(PilotJoint, equalize=False, widths=WIDTHS, companion='mirror'),
                   colored_observation, PilotJoint.forward)
MODES = ('direct', 'reference', 'mirror_coarse', 'mirror', 'mirror_half', 'curvature_only',
         'local_frozen_half', 'joint_frozen', 'joint_frozen_half', 'trained_coarse',
         'trained_joint', 'trained_joint_half')
PAIRS = (('curvature_only', 'direct'), ('local_frozen_half', 'mirror_half'),
         ('joint_frozen_half', 'local_frozen_half'), ('joint_frozen', 'mirror'),
         ('joint_frozen_half', 'mirror_half'), ('trained_coarse', 'mirror_coarse'),
         ('trained_joint', 'mirror'), ('trained_joint_half', 'mirror_half'))


def parent_model(root, contract, device):
    parent = root.parent / MIRROR_RUN
    if read(parent / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the frozen source-only mirror model')
    return load_model(parent, MIRROR_RECIPES['mirror'], contract, device)


def writeback(iq, values, frame, active):
    wave, rot, _, rms = frame
    output = iq.clone()
    output[active] += real_iq((values - wave) * rot * rms)
    return output


@torch.no_grad()
def render(iq, _unused, baseline, _tracker, models, *, phase_tracker, threshold, weights=None):
    reference, _ = baseline(iq)
    old = phase_tracker.base.track(iq)
    active = corruption_score(frame_inputs(old)[0]) > threshold
    if not bool(active.any()):
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    old = tuple(value[active] for value in old)
    gamma = phase_tracker.curvature(old[0], old[2])
    local = frame_inputs(phase_tracker.refine(old))
    joint = frame_inputs(phase_tracker.refine(old, gamma))
    plain = frame_inputs(old)
    a, b = models

    def solve(model, frame):
        y, rot, q, _ = frame
        coarse, fine = model(y, q, rot)
        return coarse, fine, torch.lerp(y, fine, .5)

    coarse, base, base_half = [writeback(iq, w, plain, active) for w in solve(a, plain)]
    phase_only = iq.clone()
    angle = -gamma[:, None] * phase_tracker.quadratic
    phase_only[active] = real_iq(complex_iq(iq[active]) * torch.polar(torch.ones_like(angle), angle))
    local_half = writeback(iq, solve(a, local)[-1], local, active)
    frozen, frozen_half = [writeback(iq, w, joint, active) for w in solve(a, joint)[1:]]
    learned = [writeback(iq, w, joint, active) for w in solve(b, joint)]
    if weights is not None:
        weights[0] += gamma.abs().sum()
        weights[1] += (gamma != 0).sum()
        weights[2] += gamma.numel()
    return torch.stack((iq, reference, coarse, base, base_half, phase_only,
                        local_half, frozen, frozen_half, *learned)), int(active.sum())


@torch.no_grad()
def checkpoint_smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, calibration, _), contract = setup(rt, device, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if expected != contract:
        raise ValueError('Mismatched real LoRa source-only contract')
    parent, threshold = parent_model(root, contract, device)
    candidate = RECIPE.model().to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026112403)
    batch = real_iq(colored_observation(complex_iq(calibration[:2]), rng))
    x, active = render(batch, None, baseline, tracker, (parent, candidate),
                       phase_tracker=PacketPhaseTracker().to(device), threshold=-1.)
    logits = load_reader(root, contract, device)(x.flatten(0, 1))[0]
    if active != len(batch) or logits.shape != (len(MODES) * len(batch), 10) or not torch.isfinite(logits).all():
        raise ValueError('Real no-query packet phase checkpoint smoke failed')
    write(root / 'checkpoint_smoke.json', dict(status='PASS', source_U=len(batch), variants=len(MODES),
                                              inherited_threshold=threshold, target_reads=0))
    print('LORA_PACKET_PHASE_REAL_CHECKPOINT_PASS', flush=True)


@torch.no_grad()
def timing(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, calibration, held), contract = setup(rt, device)
    candidate, threshold = load_model(root, RECIPE, contract, device)
    tracker = PacketPhaseTracker().to(device)
    rng = torch.Generator(device=device).manual_seed(2026112403)
    clean = calibration[:1]
    disturbed = real_iq(colored_observation(complex_iq(calibration[:12]), rng))
    observed = corruption_score(frame_inputs(tracker.base.track(disturbed))[0])
    corrupt = disturbed[observed.argmax().view(1)]
    samples = {}
    for name, iq in (('source_clean', clean), ('generic_disturbed', corrupt)):
        timings = []
        for index in range(32):
            torch.cuda.synchronize()
            start = perf_counter()
            initial = tracker.base.track(iq)
            active = corruption_score(frame_inputs(initial)[0]) > threshold
            result = iq
            if bool(active.any()):
                frame = tuple(v[active] for v in initial)
                joint = frame_inputs(tracker.refine(frame, tracker.curvature(frame[0], frame[2])))
                y, orientation, q, _ = joint
                fine = candidate(y, q, orientation)[-1]
                result = writeback(iq, torch.lerp(y, fine, .5), joint, active)
            torch.cuda.synchronize()
            elapsed = (perf_counter() - start) * 1000
            if not torch.isfinite(result).all():
                raise ValueError('Nonfinite timed output')
            if index >= 8:
                timings.append(elapsed)
        samples[name] = dict(active=int(active.sum()), median_ms=float(np.median(timings)))
    write(root / 'b1.json', dict(parameters=sum(p.numel() for p in candidate.parameters()),
                                timing=samples, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                                n_measurements=24))
    phase_holdout(root, held, tracker)


def phase_holdout(root, held, tracker):
    device = held.device
    rng = torch.Generator(device=device).manual_seed(2026112422)
    results = {key: [] for key in ('base', 'local', 'joint', 'quadratic')}
    for iq in held.split(20):
        oracle = tracker.base.track(iq)
        delta = .9 * (2 * torch.rand((len(iq), 6, 1), device=device, generator=rng) - 1)
        gamma = .48 * (2 * torch.rand((len(iq), 1), device=device, generator=rng) - 1)
        angle = delta * tracker.time + gamma[:, None] * tracker.quadratic
        source = complex_iq(iq)
        displaced = source * torch.polar(torch.ones_like(angle), angle)
        iq_in = real_iq(colored_observation(displaced, rng))
        initial = tracker.base.track(iq_in)
        fitted = tracker.curvature(initial[0], initial[2])
        zero = tracker.curvature(oracle[0], oracle[2])
        results['quadratic'].append((fitted - zero - gamma).abs().flatten().cpu())
        for name, frame in (('base', initial), ('local', tracker.refine(initial)),
                            ('joint', tracker.refine(initial, fitted))):
            relative = frame[1] * oracle[1].conj()
            # The midpoint difference cancels the even quadratic term.
            expected = torch.polar(torch.ones_like(delta.squeeze(-1)), -delta.squeeze(-1) * (2 / 255))
            frequency_error = (relative[..., 128] * relative[..., 127].conj() * expected).angle()
            frequency_error *= 256000 / (2 * torch.pi)
            results[name].append(frequency_error.abs().flatten().cpu())
    write(root / 'source-u-phase.json', dict(held_packets=len(held), readings={
        name: torch.cat(data).quantile(torch.tensor([.5, .9, .99])).tolist() for name, data in results.items()},
          quantiles=[.5, .9, .99], frequency_unit='Hz', quadratic_unit='radians',
          reference='original clean local slope plus generic applied phase',
          held_seed=2026112422, received_only=True, target_reads=0,
          injected_linear_radian_bound=.9, injected_quadratic_radian_bound=.48))


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    baseline, threshold = parent_model(root, binding[2], device)
    candidate, new_threshold = load_model(root, RECIPE, binding[2], device)
    if threshold != new_threshold:
        raise ValueError('Expected identical original source U gate')
    stats = torch.zeros(3, device=device)
    render_fn = partial(render, phase_tracker=PacketPhaseTracker().to(device), threshold=threshold, weights=stats)
    score(rt, route, panel=panel, modes=MODES, render=render_fn, binding=binding,
          frozen_models=(baseline, candidate), replay_fn=receive_only, finalize=freeze, reader_chunk=144)
    total_abs, nonzero, n = stats.tolist()
    write(root / f'{route}-phase.json', dict(active_packets=int(n), nonzero=int(nonzero),
                                             mean_abs_curvature=total_abs / max(n, 1)))


def independent_score(rt):
    root = Path(rt['run_root'])
    for route in ROUTES:
        ids, current, _ = read_frozen(root, route, MODES)
        old_ids, past, _ = read_frozen(root.parent / MIRROR_RUN, route, MIRROR_MODES)
        if ids != old_ids:
            raise ValueError('Source packet panel differs from the matched old mirror')
        for view in VIEWS:
            for arm in MODES[:5]:
                if not np.array_equal(current[view][MODES.index(arm)], past[view][MIRROR_MODES.index(arm)]):
                    raise ValueError(f'Frozen prior endpoint disagreement on {route}/{view}/{arm}')
    score_frozen(rt, modes=MODES, comparisons=PAIRS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for action in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        stage = commands.add_parser(action)
        stage.add_argument('runtime', type=Path)
        if action != 'dispatch':
            names = ROUTES if action == 'evaluate' else (RECIPE.name,) if action in ('smoke', 'fit') else ('source',)
            stage.add_argument('name', choices=names)
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        publish(**options, run=RUN, runner=__spec__.name)
        return
    rt = read(options['runtime'])
    if action == 'dispatch':
        dispatch(rt, (RECIPE.name,), initial_stages=(('checkpoint_smoke', ('source',)),))
    elif action == 'score':
        independent_score(rt)
    else:
        activate(rt)
        if action == 'checkpoint_smoke':
            checkpoint_smoke(rt)
        elif action == 'evaluate':
            evaluate(rt, options['name'])
        elif action == 'timing':
            timing(rt)
        else:
            tracker = PacketPhaseTracker().to('cuda:0')
            def calculate(model, _original, iq, rng, **kwargs):
                return natural_objective(model, tracker, iq, rng, preserve=.35, make_pair=paired_echo_inputs,
                                         tail_weight=TAIL_WEIGHT, **kwargs)
            # The original tracker in fit also supplies the unchanged source-only gate.
            fit(rt, smoke=action == 'smoke', recipe=RECIPE, receiver_batches=True, calculate=calculate)


if __name__ == '__main__':
    main()
