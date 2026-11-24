"""Isolate exact LoRa band energy and band direction at matched physical RMS."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .band_allocation import apply_gain, band_energy, band_masks, fft, ifft
from .channel_blind import corruption_score
from .channel_blind_source import panel, setup
from .complex_vector_source import load_reader, score
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze, publish, read_frozen, score_frozen
from .noise_floor_diagnostic import ROUTES, dispatch
from .noise_split_diagnostic import replay_trace
from .phase_dose_bursts_source import burst_observation
from .runner import activate
from .spectral_mixture_diagnostic import RUN as PREDECESSOR, MODES as PREVIOUS_MODES, source_model


RUN = '20261124-diagnostic-lora-band-allocation-source-m2-r01'
DEVICE = 'cuda:0'
MODES = ('direct', 'reference', 'anchor', 'exact_band_energy', 'exact_band_direction', 'exact_equal',
         'exact_near', 'exact_middle', 'exact_far', 'exact_weak', 'exact_noncarrier')
COMPARISONS = (('exact_band_energy', 'anchor'), ('exact_band_direction', 'anchor'),
               ('exact_equal', 'exact_band_direction'), ('exact_equal', 'exact_band_energy'),
               ('exact_noncarrier', 'exact_equal'),
               *((mode, 'anchor') for mode in MODES[6:10]),
               *((mode, control) for mode in MODES[2:] for control in ('direct', 'reference')))
FIELDS = ('active', 'matched', 'anchor_near_share', 'anchor_mid_share', 'anchor_far_share',
          'exact_near_share', 'exact_mid_share', 'exact_far_share', 'near_cosine', 'mid_cosine',
          'far_cosine', 'overall_cosine', 'max_rms_error')


def substitutions(anchor, exact, rms, bands):
    original = band_energy(anchor, rms, bands)
    known = band_energy(exact, rms, bands)
    valid = ((original > 1e-16) & (known > 1e-16)).all(-1, keepdim=True)
    anchor_sum = original.sum(-1, keepdim=True)
    known_sum = known.sum(-1, keepdim=True)
    scaled_exact = known * (anchor_sum / known_sum.clamp_min(1e-16))

    def with_energies(proposal, base, target):
        adjusted = apply_gain(proposal, (target / base.clamp_min(1e-16)).sqrt(), bands)
        return torch.where(valid.unsqueeze(-1), adjusted, anchor)

    band_energy_only = with_energies(anchor, original, scaled_exact)
    band_direction_only = with_energies(exact, known, original)
    equal_exact = with_energies(exact, known, scaled_exact)
    selectors = (bands[0].bool(), bands[1].bool(), bands[2].bool(), bands[:2].any(0))
    replacements = tuple(torch.where(mask, band_direction_only, anchor) for mask in selectors)
    edits = torch.stack((anchor, band_energy_only, band_direction_only, equal_exact, *replacements))
    dot_product = ((anchor * exact.conj()).real * rms.square()).sum(-2)
    inner = dot_product @ bands.T
    block_agreement = inner / (original * known).sqrt().clamp_min(1e-16)
    total_agreement = inner.sum(-1, keepdim=True) / (anchor_sum * known_sum).sqrt().clamp_min(1e-16)
    telemetry = torch.cat((valid, original / anchor_sum.clamp_min(1e-16),
                           known / known_sum.clamp_min(1e-16), block_agreement, total_agreement), -1)
    return edits, telemetry


def oracle_readouts(iq, noise, tracker, model, threshold):
    y, phase, _, rms = coordinates(tracker, iq)
    frame = phase * rms
    exact = -complex_iq(noise) / frame
    exact = exact - exact.mean(-1, keepdim=True)
    full_exact = real_iq(exact * frame)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    out = iq.expand(len(MODES) - 2, *iq.shape).clone()
    out[-1] = iq + full_exact
    records = iq.new_zeros((len(iq), len(FIELDS)))
    if not count:
        return out, count, records
    shape = model.changes(y[active], rms[active])[0][1]
    bands = band_masks(iq.device).to(iq)
    replacements, evidence = substitutions(fft(shape), fft(exact[active]), rms[active], bands)
    # Keep the existing exact whole-packet normalization for the readout that
    # was already evaluated on the frozen source panel.
    physical = ifft(replacements) * frame[active]
    reused_budget = (shape * frame[active]).abs().square().mean((1, 2), keepdim=True).sqrt()
    desired = exact[active] * frame[active]
    normalized = desired * (reused_budget / desired.abs().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8))
    physical[0] = shape * frame[active]
    physical[3] = torch.where(evidence[:, :1, None].bool(), normalized, physical[0])
    error = (physical.abs().square().mean((2, 3)).sqrt()
             - reused_budget.flatten()[None]).abs().amax(0)
    if (error > 5e-6).any() or not torch.isfinite(physical).all():
        raise ValueError('Expected matched physical IQ RMS for all isolated directions')
    out[:-1, active] = iq[active] + torch.stack((physical.real, physical.imag), dim=-2)
    records[active] = torch.cat((torch.ones_like(error[:, None]), evidence, error[:, None]), -1)
    return out, count, records


def render(iq, attached, baseline, tracker, models, *, threshold, measurement=None):
    reference, _ = baseline(iq)
    waves, count, telemetry = oracle_readouts(iq, attached, tracker, models[0], threshold)
    if measurement is not None:
        measurement.append(np.asarray(telemetry.tolist()))
    return torch.cat((torch.stack((iq, reference)), waves)), count


@torch.no_grad()
def smoke(rt):
    (_, source, _), contract = setup(rt, DEVICE, smoke=True)
    _, model, threshold = source_model(rt, contract)
    _, _, expected, baseline, tracker = source_control(rt, 'full', DEVICE)
    if expected != contract:
        raise ValueError('Expected the same source contracts in the isolated controls')
    rng = torch.Generator(device=DEVICE).manual_seed(2026112481)
    disturbed = real_iq(burst_observation(complex_iq(source), rng))
    quiet = corruption_score(coordinates(tracker, source)[0]).argmin().view(1)
    active = corruption_score(coordinates(tracker, disturbed)[0]).argmax().view(1)
    iq = torch.cat((source[quiet], disturbed[active]))
    attached = torch.cat((torch.zeros_like(source[quiet]), disturbed[active] - source[active]))
    waves, count = render(iq, attached, baseline, tracker, (model,), threshold=threshold)
    logits = load_reader(Path(rt['run_root']), contract, DEVICE)(waves.flatten(0, 1))[0]
    if count != 1 or logits.shape != (2 * len(MODES), 10) or not torch.isfinite(logits).all():
        raise ValueError('Expected a clean and active real-checkpoint diagnostic')
    if not torch.equal(waves[2:, 0], iq[0].expand_as(waves[2:, 0])):
        raise ValueError('Expected exact clean passthrough in the isolated readouts')
    write(Path(rt['run_root'])/'smoke.json', dict(status='PASS', diagnostic_only=True, V_reads=0, target_reads=0))


def evaluate(rt, route):
    binding = source_control(rt, route, DEVICE)
    _, model, threshold = source_model(rt, binding[2])
    predecessor = Path(rt['run_root']).parent / PREDECESSOR
    old_ids, old_predictions, old_metrics = read_frozen(predecessor, route, PREVIOUS_MODES)
    previous = dict(direct='direct', reference='reference', anchor='pooled_half',
                    exact_equal='exact_equal', exact_noncarrier='exact_noncarrier')
    rows = []

    def finalize(root, route, source, chosen, ids, predictions, metrics, modes):
        if ids[chosen].tolist() != old_ids or metrics != old_metrics:
            raise ValueError('The source panel and guard must match the prior isolated study')
        for view in VIEWS:
            for mode, original in previous.items():
                first = predictions[view][modes.index(mode)]
                second = old_predictions[view][PREVIOUS_MODES.index(original)]
                if not np.array_equal(first, second):
                    raise ValueError(f'Isolated readout replay changed for {view}/{mode}')
        readings = np.concatenate(rows).reshape(len(VIEWS), len(SOURCE), -1, len(FIELDS))
        by_view = {}
        for view, cells in zip(VIEWS, readings, strict=True):
            by_view[view] = {}
            for receiver, panel_data in zip(SOURCE, cells, strict=True):
                active_rows = panel_data[panel_data[:, 0] > 0.]
                medians = dict(zip(FIELDS[2:], np.median(active_rows[:, 2:], axis=0).tolist(), strict=True)) if len(active_rows) else {}
                by_view[view][receiver] = dict(active=len(active_rows),
                                               matched=int(active_rows[:, 1].sum()), **medians)
        write(root/f'{route}-isolated-estimates.json', dict(diagnostic_only=True, by_view=by_view))
        freeze(root, route, source, chosen, ids, predictions, metrics, modes)

    score(rt, route, panel=panel, modes=MODES, binding=binding, frozen_models=(model,),
          replay_fn=replay_trace, render=partial(render, threshold=threshold, measurement=rows),
          require_clean_bypass=True, finalize=finalize)


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
