"""Validate first/last projection direction blending on source LoRa IQ only."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.g_lineage import STAGES, load as load_g
from experiments.cric_dura_comparison.io import read, tensor, write
from .data import SOURCE, VIEWS
from .dura_repair import no_tone
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_dura_observed_shape import ObservedShape
from .full_packet_views import source_contract, source_role
from .full_supervised import construct as recognizer
from .packet_physics import chirp_windows, residual_view
from .residual_dura import ROUTE, SEED
from .residual_dura_gate import load as load_gate
from .runner import activate


MODES = ('direct', 'gated_last', 'gated_first_last')
PER_RX = 400


def rms(value):
    return value.abs().square().mean(-1, keepdim=True).sqrt()


def blend_direction(first, last):
    """Average unit directions with the last step's original RMS."""
    first_rms, last_rms = rms(first), rms(last)
    middle = first / first_rms.clamp_min(1e-7) + last / last_rms.clamp_min(1e-7)
    middle_rms = rms(middle)
    eligible = (first_rms > 1e-7) & (last_rms > 1e-7) & (middle_rms > 1e-7)
    return torch.where(eligible, middle * (last_rms / middle_rms.clamp_min(1e-7)), last)


@torch.no_grad()
def source_readouts(iq, model, gate, projector):
    active = gate(iq)
    last, blended = iq.clone(), iq.clone()
    dose_error = 0.
    if active.any():
        selected = iq[active]
        received = selected.flatten(0, 1)
        outputs = model.stage_outputs(received)[0]
        if len(outputs) < 2:
            raise ValueError('At least two frozen DURA stages required')
        coordinates = projector.coordinates(received)
        observed, rotation, scale = coordinates
        shape = no_tone(observed)
        first_delta = .5 * (projector.parts(outputs[0], coordinates)[0] - shape)
        last_delta = .5 * (projector.parts(outputs[-1], coordinates)[0] - shape)
        combined = blend_direction(first_delta, last_delta)
        dose_error = float((rms(combined) - rms(last_delta)).abs().max())
        for output, delta in ((last, last_delta), (blended, combined)):
            wave = delta * rotation * scale
            output[active] = (received + torch.stack((wave.real, wave.imag), dim=1)).reshape_as(selected)
    return (iq, last, blended), int(active.sum()), dose_error


def confirmation_indices(receivers):
    earlier = [(np.random.default_rng(s), n) for s, n in ((2026112341, 120), (2026112344, 120),
               (2026112345, 400), (2026112347, 240), (2026112351, 250), (2026112359, 300))]
    first, second, confirmation = (np.random.default_rng(s) for s in (2026112363, 2026112431, 2026112435))
    groups = []
    for receiver in SOURCE:
        complete = np.flatnonzero(receivers == receiver)
        remaining = complete
        for rng, n in earlier:
            remaining = np.setdiff1d(remaining, rng.choice(remaining, n, replace=False))
        available = np.setdiff1d(complete, first.choice(remaining, 160, replace=False))
        held = second.choice(available, 320, replace=False)
        groups.append(np.sort(confirmation.choice(np.setdiff1d(available, held), PER_RX, replace=False)))
    return np.concatenate(groups)


def practical(scene, packet):
    iq, receiver, identity = packet
    return residual_view(iq, scene, receiver, identity, 'source-V')


@torch.no_grad()
def validate(runtime, stage, gate_path, output):
    activate(runtime)
    root = Path(runtime['run_root'])
    source = Path(runtime['source_root'])
    contract = source_contract(source)
    residual = read(root / 'source_pairs' / 'contract.json')
    if residual != dict(contract, practical=ROUTE):
        raise ValueError('Matched residual source bank required')
    device = 'cuda:0'
    stage_root = root / 'G' / 'lineage' / stage
    state = read(stage_root / 'complete.json')
    if (state['status'] != 'SOURCE_FROZEN' or state['stage'] != stage or state['updates'] != 4940
            or any(state[key] for key in ('F_reads', 'TX_labels_read', 'target_reads'))):
        raise ValueError('Source stage must finish and freeze before the blend audit')
    model, proof = load_g(stage_root / 'final.pt', residual, 'LU', device)
    if proof['stage'] != stage or proof['model_seed'] != SEED:
        raise ValueError('Wrong registered frozen DURA stage')
    model.requires_grad_(False)
    gate = load_gate(gate_path, contract, device)
    projector = ObservedShape().to(device).eval()
    iq, receivers, identities, _ = source_role(source, contract, 'V')
    chosen = confirmation_indices(receivers)
    rows = list(zip(iq[chosen], receivers[chosen], identities[chosen], strict=True))
    predictions, waveforms, metrics = {}, {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene in VIEWS:
            observed = (np.stack([chirp_windows(row[0]) for row in rows]) if scene == 'clean' else
                        np.stack(list(pool.map(partial(practical, scene), rows, chunksize=12))))
            parts, active, error = [], 0, 0.
            for batch in tensor(observed, device).split(32):
                values, count, drift = source_readouts(batch, model, gate, projector)
                if not all(torch.isfinite(value).all() for value in values):
                    raise ValueError('Nonfinite source stage projection')
                parts.append(torch.stack(values).cpu())
                active += count
                error = max(error, drift)
            waveforms[scene] = torch.cat(parts, dim=1)
            metrics[scene] = dict(active=active, maximum_relative_rms_error=error)
    f = recognizer(device).eval().requires_grad_(False)
    weight = complete_f(root.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(weight, map_location=device, weights_only=True), strict=True)
    for scene, waveform in waveforms.items():
        parts = []
        for batch in waveform.split(24, dim=1):
            logits = f(batch.flatten(0, 1).to(device))[0]
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite frozen source Tri-set logits')
            parts.append(logits.argmax(1).reshape(len(MODES), batch.shape[1]).cpu().tolist())
        predictions[scene] = np.concatenate(parts, axis=1)
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = metadata['y'][chosen]
    for scene, labels in predictions.items():
        valid = labels == truth
        metrics[scene].update(correct=valid.sum(1).tolist(),
                              by_rx=valid.reshape(len(MODES), len(SOURCE), PER_RX).sum(-1).tolist())
    write(output, dict(status='SOURCE_STAGE_BLEND_VALIDATED', stage=stage, seed=SEED,
                       modes=list(MODES), source_n=len(chosen), target_reads=0,
                       truth_after_source_predictions=True, views=metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES[2:], required=True)
    parser.add_argument('--gate', dest='gate_path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    args['runtime'] = read(args['runtime'])
    validate(**args)
