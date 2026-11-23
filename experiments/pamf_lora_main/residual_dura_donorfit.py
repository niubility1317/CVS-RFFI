"""Fit a packet-local complex spectral estimator from paired LoRa source IQ."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .data import SOURCE, VIEWS
from .dura_repair import CLEAN_PASS_Q, no_tone
from .full_dura_observed_shape import ObservedShape
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .packet_physics import chirp_windows
from .packet_runner import schedule
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import RUN as PARENT, SEED, ResidualView
from .residual_dura_bandfit import band_masks, correction, fft, holdout_indices, matched_bands
from .residual_dura_calibration import source_model, source_state
from .residual_dura_gate import load as load_gate
from .residual_dura_gate_run import RUN as GATE_RUN
from .residual_dura_stage_blend import PER_RX, confirmation_indices, practical
from .runner import activate


RUN = '20261123-phase1-lora-dura-r69-donor-source-m1-r01'
MODES = ('direct', 'r69', 'quarter_unfitted', 'donor', 'quarter_donor', 'joint', 'quarter_joint', 'oracle')
TRAIN_SEED = 2026112481
VALIDATION_SEED = 2026112482
RIDGE = .01
FIT_PER_RX = (200, 800)


def paired_indices(root, contract):
    rng = np.random.default_rng(TRAIN_SEED)
    groups, offset = [], 0
    for role, count in zip(('L_s', 'U_s'), FIT_PER_RX, strict=True):
        _, rx, _, _ = source_role(root, contract, role)
        groups.extend(offset + rng.choice(np.flatnonzero(rx == name), count, replace=False) for name in SOURCE)
        offset += len(rx)
    return np.sort(np.concatenate(groups))


def validation_indices(rx):
    first, third = confirmation_indices(rx), holdout_indices(rx)
    second = np.random.default_rng(2026112476)
    fourth = np.random.default_rng(2026112479)
    fifth = np.random.default_rng(VALIDATION_SEED)
    chosen = []
    for receiver in SOURCE:
        eligible = np.setdiff1d(np.flatnonzero(rx == receiver), first)
        preserved = second.choice(eligible, PER_RX, replace=False)
        eligible = np.setdiff1d(eligible, np.concatenate((preserved, third)))
        previous = fourth.choice(eligible, PER_RX, replace=False)
        chosen.append(np.sort(fifth.choice(np.setdiff1d(eligible, previous), PER_RX, replace=False)))
    return np.concatenate(chosen)


def packet_features(received, projector, model, band):
    y = received.flatten(0, 1)
    coordinates = projector.coordinates(y)
    observed = no_tone(coordinates[0])
    original = correction(projector, model(y), coordinates, observed)
    observed_fft = fft(observed)
    restored_fft = fft(observed + original)
    _, _, quality, *_ = projector.tracker.track(received)
    mask = torch.eye(6, dtype=torch.bool, device=y.device)
    values, indices = quality[:, None, :].expand(-1, 6, -1).masked_fill(mask, -torch.inf).topk(2, dim=-1)
    packet = torch.arange(len(received), device=y.device)[:, None, None]
    borrowed = restored_fft[:, band].reshape(len(received), 6, -1)[packet, indices].flatten(0, 1)
    features = torch.stack((observed_fft[:, band], restored_fft[:, band], borrowed[:, 0], borrowed[:, 1]), dim=-1)
    strata = 2 * (quality >= CLEAN_PASS_Q).long() + (values[..., 0] >= CLEAN_PASS_Q).long()
    weights = values.clamp_min(1e-6).square()
    weights = (weights / weights.sum(-1, keepdim=True)).flatten(0, 1)
    naive_donor = (borrowed * weights[..., None]).sum(1)
    return features, strata.flatten(), observed_fft, original, coordinates, naive_donor


def gram_update(gram, cross, counts, features, truth, stratum):
    features, truth = features.to(gram.dtype), truth.to(cross.dtype)
    for group in range(len(counts)):
        local = stratum == group
        x, y = features[local], truth[local]
        gram[group] += torch.einsum('nbj,nbk->bjk', x.conj(), x)
        cross[group] += torch.einsum('nbj,nb->bj', x.conj(), y)
        counts[group] += local.sum()


def ridge_solution(gram, cross):
    normalizer = gram.diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(1e-12)
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + RIDGE * normalizer[..., None, None] * identity, cross[..., None])[..., 0]


@torch.no_grad()
def fit(rt, smoke=False):
    parent, paths, contract, paired = source_state(rt)
    device = 'cuda:0'
    model = source_model(parent, 'r69', paired, device).requires_grad_(False)
    view = ResidualView(paths)
    if not view.ready():
        raise ValueError('Expected complete paired source arrays')
    indices = paired_indices(Path(rt['source_root']), contract)
    if smoke:
        indices = indices[:24]
    projector = ObservedShape().to(device).eval()
    band = torch.stack(band_masks(device)).any(0)
    gram = torch.zeros(4, int(band.sum()), 4, 4, dtype=torch.complex128, device=device)
    cross = torch.zeros_like(gram[..., 0])
    counts = torch.zeros(4, dtype=torch.long, device=device)
    for draw in range(2):
        for offset in range(0, len(indices), 24):
            batch = indices[offset:offset + 24]
            received = tensor(view.received[draw, batch], device)
            clean = tensor(view.clean[batch], device).flatten(0, 1)
            features, group, *_ = packet_features(received, projector, model, band)
            target = fft(no_tone(projector.coordinates(clean)[0]))[:, band]
            gram_update(gram, cross, counts, features, target, group)
        print('DONOR_FIT_DRAW', draw, counts.tolist(), flush=True)
    if not (torch.isfinite(gram).all() and torch.isfinite(cross).all()) or gram.abs().sum() == 0:
        raise ValueError('Missing finite source covariance')
    root = Path(rt['run_root'])
    proof = dict(status='SOURCE_SMOKE_PASSED' if smoke else 'SOURCE_FROZEN', seed=SEED,
                 packets=len(indices), draws=2, stratum_counts=counts.tolist(), G_updates=0,
                 F_reads=0, TX_labels_read=0, target_reads=0)
    if smoke:
        write(root / 'smoke.json', proof)
    else:
        if (counts == 0).any():
            raise ValueError('Receiver quality stratum absent from the paired fit')
        weights = {'joint': ridge_solution(gram, cross),
                   'donor': ridge_solution(gram[..., 2:, 2:], cross[..., 2:])}
        if not all(torch.isfinite(value).all() for value in weights.values()):
            raise ValueError('Nonfinite calibrated source weights')
        torch.save(dict(proof=proof, paired=paired, weights={k: v.cfloat().cpu() for k, v in weights.items()}),
                   root / 'calibrator.pt')
        write(root / 'complete.json', proof)


def readouts(received, clean, model, projector, gate, masks, weights):
    enabled = gate(received)
    outputs = [received, *(received.clone() for _ in MODES[1:])]
    if not enabled.any():
        return torch.stack(outputs), 0, 0.
    selected = received[enabled]
    band = masks[0] | masks[1]
    x, group, received_fft, original, coords, naive = packet_features(selected, projector, model, band)
    target = fft(no_tone(projector.coordinates(clean[enabled].flatten(0, 1))[0]))
    base = fft(original)

    def fitted(method, features):
        return (features * weights[method][group]).sum(dim=-1)

    def direction(spectrum):
        target_delta = torch.zeros_like(base)
        target_delta[:, band] = spectrum - received_fft[:, band]
        return matched_bands(base, target_delta, masks, (0, 1))

    def quarter(coefficients):
        return matched_bands(base, .75 * base + .25 * coefficients, masks, (0, 1))

    donor = direction(fitted('donor', x[..., 2:]))
    joint = direction(fitted('joint', x))
    coefficients = (quarter(direction(naive)), donor, quarter(donor), joint, quarter(joint), direction(target[:, band]))
    base_rms = base.abs().square().mean(-1).sqrt()
    drift = max(float((value.abs().square().mean(-1).sqrt() - base_rms).abs().max()) for value in coefficients)
    deltas = [original, *(torch.fft.ifft(value, dim=-1, norm='ortho') for value in coefficients)]
    _, rotation, scale = coords
    y = selected.flatten(0, 1)
    for output, delta in zip(outputs[1:], deltas, strict=True):
        wave = delta * rotation * scale
        output[enabled] = (y + torch.stack((wave.real, wave.imag), dim=1)).reshape_as(selected)
    return torch.stack(outputs), int(enabled.sum()), drift


@torch.no_grad()
def score(rt):
    from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
    from .full_supervised import construct as recognizer

    parent, _, contract, paired = source_state(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    device = 'cuda:0'
    saved = torch.load(root / 'calibrator.pt', map_location='cpu', weights_only=True)
    proof = read(root / 'complete.json')
    if (proof != saved['proof'] or paired != saved['paired'] or proof['status'] != 'SOURCE_FROZEN'
            or proof['packets'] != len(SOURCE) * sum(FIT_PER_RX) or proof['seed'] != SEED
            or any(proof[key] for key in ('G_updates', 'F_reads', 'TX_labels_read', 'target_reads'))):
        raise ValueError('Expected independently frozen source-only calibration')
    weights = {key: value.to(device) for key, value in saved['weights'].items()}
    model = source_model(parent, 'r69', paired, device).requires_grad_(False)
    projector = ObservedShape().to(device).eval()
    gate = load_gate(parent.parent / GATE_RUN / 'gate.json', contract, device)
    predictor = recognizer(device).eval().requires_grad_(False)
    ckpt = complete_f(parent.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    predictor.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    iq, rx, ids, _ = source_role(source, contract, 'V')
    selected = validation_indices(rx)
    packets = list(zip(iq[selected], rx[selected], ids[selected], strict=True))
    clean = tensor(np.stack([chirp_windows(packet[0]) for packet in packets]), device)
    masks = band_masks(device)
    predictions, metrics = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene in VIEWS:
            received = (clean if scene == 'clean' else
                        tensor(np.stack(list(pool.map(partial(practical, scene), packets, chunksize=18))), device))
            parts, active, drift = [], 0, 0.
            for y, x in zip(received.split(20), clean.split(20), strict=True):
                views, count, mismatch = readouts(y, x, model, projector, gate, masks, weights)
                logits = predictor(views.flatten(0, 1))[0]
                if mismatch > 1e-6 or not torch.isfinite(logits).all():
                    raise ValueError('Invalid frozen source donor prediction or dose')
                parts.append(logits.argmax(1).reshape(len(MODES), len(y)).cpu().tolist())
                active += count
                drift = max(drift, mismatch)
            predictions[scene] = np.concatenate(parts, axis=1)
            metrics[scene] = dict(active=active, maximum_rms_error=drift)
            print('DONOR_SOURCE_PREDICTED', scene, active, flush=True)
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = metadata['y'][selected]
    correct = {scene: value == truth for scene, value in predictions.items()}
    for scene, correct_rows in correct.items():
        metrics[scene].update(correct=correct_rows.sum(1).tolist(),
                              by_rx=correct_rows.reshape(len(MODES), len(SOURCE), PER_RX).sum(-1).tolist())
    six = np.stack([correct[scene] for scene in VIEWS[1:]]).sum(0)
    low = correct['practical_low_urban'].astype(np.int16)
    rng = np.random.default_rng(VALIDATION_SEED)
    boot = np.concatenate([rng.integers(i * PER_RX, (i + 1) * PER_RX, (1800, PER_RX))
                           for i in range(len(SOURCE))], axis=1)

    def interval(table):
        return [np.quantile(row[boot].sum(1), (.025, .975)).tolist() for row in table - table[1]]

    write(root / 'source_scores.json', dict(status='SOURCE_SCORED', seed=SEED, modes=list(MODES),
          n=len(selected), target_reads=0, truth_after_predictions=True, views=metrics,
          six=six.sum(1).tolist(), six_ci=interval(six), low_ci=interval(low)))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    launch = partial(start, module=rt['runner'])
    try:
        for phase, action in (('SOURCE_SMOKE', 'smoke'), ('SOURCE_FITTING', 'fit'), ('FROZEN_SOURCE_SCORING', 'score')):
            write(state, dict(status=phase), exclusive=False)
            schedule(rt, [(action, 'all')], launch)
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    ctx = read(context)
    rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / PARENT
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Residual R69 release receipt differs')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[6, 3, 2, 0, 4, 1, 5], max_workers=1, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for kind in ('dispatch', 'smoke', 'fit', 'score'):
        command = commands.add_parser(kind)
        command.add_argument('runtime', type=Path)
        command.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.kind == 'publish':
        publish(**{key: value for key, value in vars(args).items() if key != 'kind'})
    else:
        runtime = read(args.runtime)
        if args.kind == 'dispatch':
            dispatch(runtime)
        else:
            activate(runtime)
            if args.kind == 'score':
                score(runtime)
            else:
                fit(runtime, smoke=args.kind == 'smoke')
