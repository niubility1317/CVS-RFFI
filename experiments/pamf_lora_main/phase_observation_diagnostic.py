"""Compare received and isolated LoRa phase directions at matched physical edits."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from . import received_update_source as frozen
from .channel_blind import corruption_score
from .channel_blind_source import panel, setup
from .complex_vector_source import load_reader, score as predict
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import source_control
from .local_phase import project
from .local_wiener_source import freeze, publish, read_frozen, score_frozen
from .noise_floor_diagnostic import ROUTES, dispatch
from .noise_split_diagnostic import replay_trace
from .received_update import update_coordinates
from .reference_cancel import generic_observation
from .runner import activate
from .spectral_block_source import load_models


RUN = '20261124-diagnostic-lora-phase-observation-source-m2-r01'
CONTEXT = ('direct', 'reference', 'directional_coarse', 'directional_final')
RECEIVED = ('phase_margin_off', 'learned_time', 'repeat_time', 'learned_equal', 'repeat_equal')
ISOLATED = ('exact_phase_equal', 'exact_phase_10', 'exact_phase_20', 'exact_phase_all')
MODES = (*CONTEXT, *RECEIVED, *ISOLATED)
COMPARISONS = (('learned_equal', 'repeat_equal'), ('exact_phase_equal', 'learned_equal'),
               ('exact_phase_equal', 'repeat_equal'), ('exact_phase_20', 'exact_phase_10'),
               ('exact_phase_all', 'exact_phase_20'),
               *((mode, base) for mode in (*RECEIVED, *ISOLATED) for base in ('direct', 'reference', 'directional_final')))
FIELDS = ('eligible_received', 'noise_present', 'learned_time_rms', 'repeat_time_rms', 'equal_rms',
          'exact_equal_rms', 'exact_10_rms', 'exact_20_rms', 'exact_all_rms',
          'calibrated_phase_gains', 'relaxed_phase_gains', 'learned_cosine', 'repeat_cosine')
DEVICE = 'cuda:0'


def phase_ratio(raw, phased):
    error = (phased - raw).abs().square().mean((-2, -1))
    return (error / raw.abs().square().mean((-2, -1)).clamp_min(1e-10)).sqrt()


def phase_budget(raw, angle, ratio):
    """Rotate along the short arc, limiting physical waveform RMS."""
    power = raw.abs().square()
    bound = torch.as_tensor(ratio, device=raw.device, dtype=raw.real.dtype).square()
    target = bound * power.sum((-2, -1), keepdim=True)

    def moved(scale):
        return (4 * power * (angle * scale * .5).sin().square()).sum((-2, -1), keepdim=True)

    hi = torch.ones_like(target)
    needs_limit = moved(hi) > target
    lo = torch.zeros_like(hi)
    for _ in range(18):
        mid = (lo + hi) * .5
        allowed = moved(mid) <= target
        lo, hi = torch.where(allowed, mid, lo), torch.where(allowed, hi, mid)
    scale = torch.where(needs_limit, lo, 1.)
    scale = torch.where(target > 0., scale, 0.)
    actual = angle * scale
    return torch.where(actual != 0., raw * torch.polar(torch.ones_like(actual), actual), raw)


def repeat_angle(observed, quality):
    """Compare two disjoint sets of the other five actually received chirps."""
    weights = quality.clamp_min(.01).square()
    weighted = observed * weights
    first = sum(weighted.roll(i, 1) for i in (1, 3, 5))
    second = sum(weighted.roll(i, 1) for i in (2, 4))
    agreement = (first * second.conj()).real / (first.abs() * second.abs()).clamp_min(1e-8)
    direction = first / first.abs().clamp_min(1e-8) + second / second.abs().clamp_min(1e-8)
    delta = (direction * observed.conj()).angle()
    return delta * agreement.clamp(0., 1.)


def protected_angle(delta, raw):
    time = torch.linspace(-1., 1., raw.shape[-1], device=raw.device)
    return project(delta, raw.abs().square(), torch.ones_like(time), time)


def received(iq, tracker, model, threshold):
    """All outputs and the matching budget depend only on the current received IQ."""
    y, rotation, quality, rms = update_coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    original = complex_iq(iq)
    outputs = original.expand(len(CONTEXT) - 2 + len(RECEIVED), *original.shape).clone()
    equal_budget = y.real.new_zeros((len(iq), 1, 1))
    calibration = y.real.new_zeros((len(iq), 2))
    if active.any():
        observed, q, level = y[active], quality[active], rms[active]
        raw = original[active]
        coarse, final, details, _ = model.states(observed, q, level)
        _, relaxed, relaxed_details, _ = model.states(observed, q, level, phase_margin=False)
        received_waves = [raw + (wave - observed) * rotation[active] * level
                          for wave in (coarse, final, relaxed)]
        learnt_delta = protected_angle((received_waves[-1] * raw.conj()).angle(), raw)
        repeat_delta = protected_angle(repeat_angle(observed, q), raw)
        learned = phase_budget(raw, learnt_delta, .10)
        repeat = phase_budget(raw, repeat_delta, .10)
        budget = torch.minimum(phase_ratio(raw, learned), phase_ratio(raw, repeat)).view(-1, 1, 1)
        outputs[:, active] = torch.stack((*received_waves, learned, repeat, phase_budget(raw, learnt_delta, budget),
                                          phase_budget(raw, repeat_delta, budget)))
        equal_budget[active] = budget
        phase_gains = [(result[-1][-1][..., 1] > 0).float().mean((1, 2))
                       for result in (details, relaxed_details)]
        calibration[active] = torch.stack(phase_gains, -1)
    return original, outputs, active, equal_budget, calibration, rotation * rms


def isolated(raw, epsilon, frame, budget):
    """Diagnostic-only exact noncarrier noise removal, followed by phase replacement."""
    noise = complex_iq(epsilon)
    in_frame = noise / frame
    reference = raw - (in_frame - in_frame.mean(-1, keepdim=True)) * frame
    present = noise.abs().amax((-2, -1), keepdim=True) > 0.
    angle = torch.where(present, (reference * raw.conj()).angle(), 0.)
    # The known noise is never used to derive the receiver matching budget.
    return torch.stack([phase_budget(raw, angle, ratio) for ratio in (budget, .10, .20, 2.)])


def alignment(raw, candidate, reference):
    left, right = candidate - raw, reference - raw
    inner = (left.conj() * right).real.sum((-2, -1))
    denominator = (left.abs().square().sum((-2, -1)) * right.abs().square().sum((-2, -1))).sqrt()
    return inner / denominator.clamp_min(1e-8)


def render(iq, epsilon, baseline, tracker, models, *, threshold, measurements=None):
    base, _ = baseline(iq)
    raw, public, active, budget, calibration, frame = received(iq, tracker, models[0], threshold)
    diagnostics = isolated(raw, epsilon, frame, budget)
    if measurements is not None:
        dose = phase_ratio(raw, torch.cat((public[3:5], diagnostics)))
        ratio = [dose[i] for i in range(len(dose))]
        values = torch.stack((active, epsilon.abs().amax((1, 2, 3)) > 0., *ratio[:2],
                              budget[:, 0, 0], *ratio[2:], *calibration.unbind(-1),
                              alignment(raw, public[5], diagnostics[0]),
                              alignment(raw, public[6], diagnostics[0])), -1)
        measurements.append(np.asarray(values.tolist(), dtype=np.float64))
    waves = torch.cat((public, diagnostics))
    return torch.cat((torch.stack((iq, base)), torch.stack((waves.real, waves.imag), dim=-2))), int(active.sum())


def original(rt, contract):
    parent = Path(rt['run_root']).parent / frozen.RUN
    if read(parent / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected both previously frozen source routes')
    models, threshold = load_models(parent, contract, DEVICE, frozen.RECIPES)
    return parent, models[-1], threshold


@torch.no_grad()
def smoke(rt):
    (_, source, _), contract = setup(rt, DEVICE, smoke=True)
    _, model, threshold = original(rt, contract)
    _, _, bound, baseline, tracker = source_control(rt, 'full', DEVICE)
    if bound != contract:
        raise ValueError('Source data and reader contracts differ')
    rng = torch.Generator(device=DEVICE).manual_seed(2026112459)
    distorted = real_iq(generic_observation(complex_iq(source), rng))
    keep = corruption_score(update_coordinates(tracker, source)[0]).argmin().view(1)
    choose = corruption_score(update_coordinates(tracker, distorted)[0]).argmax().view(1)
    clean, damaged = source[keep], distorted[choose]
    iq = torch.cat((clean, damaged))
    noise = torch.cat((torch.zeros_like(clean), damaged - source[choose]))
    measurements = []
    options, active = render(iq, noise, baseline, tracker, (model,), threshold=threshold, measurements=measurements)
    logits = load_reader(Path(rt['run_root']), contract, DEVICE)(options.flatten(0, 1))[0]
    if (active != 1 or logits.shape != (2 * len(MODES), 10) or not torch.isfinite(logits).all()
            or not torch.equal(options[2:, 0], clean[0].expand_as(options[2:, 0]))
            or measurements[0].shape != (2, len(FIELDS))):
        raise ValueError('Expected a finite active real checkpoint smoke and identity clean')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', diagnostic_only=True, V_reads=0, target_reads=0))


def evaluate(rt, route):
    binding = source_control(rt, route, DEVICE)
    parent, model, threshold = original(rt, binding[2])
    previous_ids, previous, marks = read_frozen(parent, route, frozen.MODES)
    indices = [frozen.MODES.index(mode) for mode in CONTEXT]
    evidence = []

    def finalize(root, route, source, chosen, ids, decisions, metrics, modes):
        if ids[chosen].tolist() != previous_ids or metrics != marks:
            raise ValueError('The source panel or frozen clean guard changed')
        if any(not np.array_equal(decisions[view][:len(CONTEXT)], previous[view][indices]) for view in VIEWS):
            raise ValueError('Expected bitwise parity of all inherited source decisions')
        values = np.concatenate(evidence).reshape(len(VIEWS), len(chosen), len(FIELDS))
        rows = {}
        for view, scene in zip(VIEWS, values, strict=True):
            rows[view] = {}
            for key, cells in zip((*SOURCE, 'all'), (*np.split(scene, len(SOURCE)), scene), strict=True):
                active = cells[cells[:, 0] > 0]
                matched = active[active[:, 4] > 1e-6]
                medians = dict(zip(FIELDS[2:], np.median(active[:, 2:], axis=0).tolist(), strict=True)) if len(active) else {}
                rows[view][key] = dict(active=len(active), equal_nonzero=len(matched),
                                       exact_equal_reached=int((np.abs(matched[:, 5] - matched[:, 4]) < 5e-6).sum()),
                                       **medians)
        write(root / f'{route}-phase-evidence.json', dict(by_view=rows, diagnostic_only=True))
        freeze(root, route, source, chosen, ids, decisions, metrics, modes)

    predict(rt, route, panel=panel, modes=MODES, binding=binding, frozen_models=(model,), replay_fn=replay_trace,
            render=partial(render, threshold=threshold, measurements=evidence), require_clean_bypass=False,
            finalize=finalize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    entry = actions.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        entry.add_argument('--' + key, type=Path, required=True)
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
