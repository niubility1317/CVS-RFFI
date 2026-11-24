"""Isolate routing quality from the available LoRa mixture edit directions."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from . import spectral_mixture_source as parent
from .channel_blind import corruption_score
from .channel_blind_source import panel, setup
from .complex_vector_source import load_reader, score
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze, publish, read_frozen, score_frozen
from .noise_floor_diagnostic import ROUTES, dispatch
from .noise_split_diagnostic import replay_trace
from .packet_physics import chirp_windows
from .phase_dose_bursts_source import burst_observation
from .runner import activate
from .spectral_mixture import COMPONENTS
from .variance_transport import physical_norm


RUN = '20261124-diagnostic-lora-mixture-directions-source-m2-r01'
DEVICE = 'cuda:0'
CONTEXT = ('direct', 'reference', 'pooled_half', 'split_half', 'guarded_half')
MODE_SUFFIX = ('received_equal', 'noise_free_equal', 'clean_equal', 'expert_equal',
               'exact_equal', 'exact_noncarrier')
MODES = (*CONTEXT, *MODE_SUFFIX)
COMPARISONS = (('noise_free_equal', 'received_equal'), ('clean_equal', 'received_equal'),
               ('expert_equal', 'received_equal'), ('expert_equal', 'pooled_half'),
               ('exact_equal', 'expert_equal'), ('exact_noncarrier', 'exact_equal'),
               *((mode, control) for mode in MODES[2:] for control in ('direct', 'reference')))
FIELDS = ('active', 'received_noise_free_tv', 'received_clean_tv', 'noise_free_clean_tv',
          'received_clean_same', 'noise_free_clean_same', 'noise_free_expert', 'rms_lock_error',
          'received_projection', 'noise_free_projection', 'clean_projection', 'expert_projection', 'budget')


def replay_isolated(route, view, row):
    received, noise = replay_trace(route, view, row)
    return received, np.stack((noise, chirp_windows(row[0])))


def lock(edit, budget):
    measured = edit.abs().square().mean((-2, -1), keepdim=True).sqrt()
    return edit * (budget / measured.clamp_min(1e-8))


def paired_fold_weights(model, experts, weights):
    even, odd = model.synthesize(torch.einsum('gbk,bkcd->gbcd', weights[:2], experts))
    return torch.where(model.route_even, odd, even)


def oracle_readouts(iq, attached, tracker, model, threshold):
    noise, clean = attached.unbind(1)
    tracked, phase, _, rms = coordinates(tracker, iq)
    active = corruption_score(tracked) > threshold
    raw = complex_iq(iq)
    frame = phase * rms
    exact = -complex_iq(noise) / frame
    noncarrier = (exact - exact.mean(-1, keepdim=True)) * frame
    out = iq.expand(len(MODES) - 2, *iq.shape).clone()
    out[-1] = real_iq(raw + noncarrier)
    telemetry = iq.new_zeros((len(iq), len(FIELDS)))
    if not active.any():
        return out, int(active.sum()), telemetry

    y, scale, basis = (value[active] for value in (tracked, rms, frame))
    original, desired = raw[active], noncarrier[active]
    parent_edits, _, _ = model.changes(y, scale)
    fixed = parent_edits[[1, 4, 5]] * basis
    pooled, observed, _ = fixed.unbind(0)
    budget = pooled.abs().square().mean((1, 2), keepdim=True).sqrt()
    spectrum = model.spectrum(y)
    _, experts = model.prior_directions(spectrum, model.noise(spectrum))

    clear = coordinates(tracker, real_iq(original - complex_iq(noise[active])))[0]
    reference = coordinates(tracker, clean[active])[0]
    seen, noise_free, pristine = (model.routing(value)[0] for value in (y, clear, reference))
    oracle_edits = [paired_fold_weights(model, experts, value) * basis for value in (noise_free, pristine)]
    equal = lock(torch.stack((observed, *oracle_edits)), budget)
    # Select an expert by waveform overlap alone at the same received-IQ RMS.
    basis_edits = model.synthesize(experts) * basis[:, None]
    modes = lock(basis_edits, budget[:, None])
    overlaps = (modes * desired[:, None].conj()).real.sum((-2, -1))
    expert = overlaps.argmax(-1)
    best = modes[torch.arange(len(modes), device=modes.device), expert]
    ideal = lock(desired, budget)
    comparisons = torch.cat((equal, best.unsqueeze(0), ideal.unsqueeze(0)))

    def tv(first, second):
        return .5 * (first[:2] - second[:2]).abs().sum(-1).mean(0)

    lock_error = (physical_norm(comparisons, 1.) - budget).abs().amax(0).flatten()
    target_energy = desired.abs().square().sum((1, 2)).clamp_min(1e-8)
    projections = ((comparisons[:4] * desired.conj()).real.sum((-2, -1)) / target_energy).mT
    telemetry[active] = torch.cat((torch.stack((torch.ones_like(lock_error),
        tv(seen, noise_free), tv(seen, pristine), tv(noise_free, pristine),
        (seen[-1].argmax(-1) == pristine[-1].argmax(-1)),
        (noise_free[-1].argmax(-1) == pristine[-1].argmax(-1)), expert, lock_error), 1),
        projections, (budget / physical_norm(original, 1.).clamp_min(1e-8))[:, 0]), dim=1)
    physical = original + torch.cat((fixed, comparisons))
    out[:-1, active] = torch.stack((physical.real, physical.imag), dim=-2)
    return out, int(active.sum()), telemetry


def render(iq, attached, baseline, tracker, models, *, threshold, measurement=None):
    reference, _ = baseline(iq)
    out, active, evidence = oracle_readouts(iq, attached, tracker, models[0], threshold)
    if measurement is not None:
        measurement.append(np.asarray(evidence.tolist()))
    return torch.cat((torch.stack((iq, reference)), out)), active


def source_model(rt, contract):
    original = Path(rt['run_root']).parent / parent.RUN
    if read(original/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Require the frozen scored mixture source study')
    threshold = parent.load_gate(original)
    return original, parent.load(original, contract, threshold), threshold


@torch.no_grad()
def smoke(rt):
    (_, source, _), contract = setup(rt, DEVICE, smoke=True)
    _, model, threshold = source_model(rt, contract)
    _, _, expected, baseline, tracker = source_control(rt, 'full', DEVICE)
    if expected != contract:
        raise ValueError('Expected a common source contract for all controls')
    rng = torch.Generator(device=DEVICE).manual_seed(2026112481)
    damaged = real_iq(burst_observation(complex_iq(source), rng))
    quiet = corruption_score(coordinates(tracker, source)[0]).argmin().view(1)
    active = corruption_score(coordinates(tracker, damaged)[0]).argmax().view(1)
    raw = torch.cat((source[quiet], damaged[active]))
    clean = torch.cat((source[quiet], source[active]))
    attachment = torch.stack((raw - clean, clean), dim=1)
    measurements = []
    waves, count = render(raw, attachment, baseline, tracker, (model,), threshold=threshold, measurement=measurements)
    logits = load_reader(Path(rt['run_root']), contract, DEVICE)(waves.flatten(0, 1))[0]
    if (count != 1 or logits.shape != (2 * len(MODES), 10) or not torch.isfinite(logits).all()
            or not torch.equal(waves[2:, 0], raw[0].expand_as(waves[2:, 0]))
            or measurements[0][:, FIELDS.index('rms_lock_error')].max() > 5e-6):
        raise ValueError('Expected a clean bypass and same-RMS isolated reference directions')
    write(Path(rt['run_root'])/'smoke.json', dict(status='PASS', diagnostic_only=True, V_reads=0, target_reads=0))


def evaluate(rt, route):
    binding = source_control(rt, route, DEVICE)
    original, model, threshold = source_model(rt, binding[2])
    previous_ids, previous, states = read_frozen(original, route, parent.MODES)
    ordered = [parent.MODES.index(mode) for mode in CONTEXT]
    records = []

    def finish(root, route, source, chosen, ids, predicted, metrics, modes):
        if ids[chosen].tolist() != previous_ids or metrics != states:
            raise ValueError('Comparisons require the original physical IDs and clean guard')
        if any(not np.array_equal(predicted[view][:len(CONTEXT)], previous[view][ordered]) for view in VIEWS):
            raise ValueError('The prior received-only controls must replay exactly')
        readings = np.concatenate(records).reshape(len(VIEWS), len(SOURCE), -1, len(FIELDS))
        result = {}
        for name, rows in zip(VIEWS, readings, strict=True):
            summary = {}
            for rx, data in zip(SOURCE, rows, strict=True):
                active = data[data[:, 0] > 0.]
                if len(active) and active[:, FIELDS.index('rms_lock_error')].max() > 5e-6:
                    raise ValueError('The paired diagnostics must have matched IQ RMS')
                index = FIELDS.index('noise_free_expert')
                values = dict(zip(FIELDS[1:], np.median(active[:, 1:], axis=0).tolist(), strict=True)) if len(active) else {}
                selector = np.bincount(active[:, index].astype(int), minlength=COMPONENTS).tolist()
                matches = [int(active[:, FIELDS.index(key)].sum())
                           for key in ('received_clean_same', 'noise_free_clean_same')]
                summary[rx] = dict(active=len(active), medians=values, selected_experts=selector, matched_counts=matches)
            result[name] = summary
        write(root/f'{route}-isolated-estimates.json', dict(by_view=result, diagnostic_only=True))
        freeze(root, route, source, chosen, ids, predicted, metrics, modes)

    score(rt, route, panel=panel, modes=MODES, binding=binding, frozen_models=(model,),
          replay_fn=replay_isolated, render=partial(render, threshold=threshold, measurement=records),
          require_clean_bypass=True, finalize=finish)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + key, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        if action != 'dispatch':
            worker.add_argument('name', choices=ROUTES if action == 'evaluate' else ('source',))
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
