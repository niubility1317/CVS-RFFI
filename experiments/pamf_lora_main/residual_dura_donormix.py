"""Calibrate limited R69 donor mixing against paired dechirped source directions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .data import SOURCE, VIEWS
from .dura_repair import no_tone
from .full_dura_observed_shape import ObservedShape
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .packet_physics import chirp_windows
from .packet_runner import schedule
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import RUN as PARENT, SEED, ResidualView
from .residual_dura_bandfit import band_masks, fft, holdout_indices, power
from .residual_dura_calibration import source_model, source_state
from .residual_dura_donorfit import FIT_PER_RX, paired_indices, packet_features, validation_indices
from .residual_dura_gate import load as load_gate
from .residual_dura_gate_run import RUN as GATE_RUN
from .residual_dura_stage_blend import PER_RX, confirmation_indices, practical
from .runner import activate


RUN = '20261123-phase1-lora-dura-r69-donormix-source-m1-r01'
MODES = ('direct', 'r69', 'quarter', 'global', 'conditioned', 'oracle_half_range', 'oracle_full_range', 'oracle_clean')
GRID = (0., .125, .25, .375, .5)
GROUPS = 8
ANGLE_AGREEMENT = 3**.5 / 2
PSEUDO_CHIRPS = 128


def sixth_panel(rx):
    first = confirmation_indices(rx)
    third = holdout_indices(rx)
    shared = np.concatenate((first, third, validation_indices(rx)))
    second, fourth = (np.random.default_rng(seed) for seed in (2026112476, 2026112479))
    selected = []
    for receiver in SOURCE:
        eligible = np.setdiff1d(np.flatnonzero(rx == receiver), first)
        prior_second = second.choice(eligible, PER_RX, replace=False)
        prior_fourth = fourth.choice(np.setdiff1d(eligible, np.concatenate((prior_second, third))),
                                     PER_RX, replace=False)
        remaining = np.setdiff1d(eligible, np.concatenate((shared, prior_second, prior_fourth)))
        if len(remaining) != PER_RX:
            raise ValueError('Expected 400 source V packets per receiver outside the five registered panels')
        selected.append(remaining)
    return np.concatenate(selected)


def unit(value):
    length = power(value).sqrt()
    return value / length.clamp_min(1e-7)[:, None], length


def real_dot(left, right):
    return (left.conj() * right).real.sum(-1)


def direction_angles(base, donor, target):
    u, base_norm = unit(base)
    v, donor_norm = unit(donor)
    t, target_norm = unit(target)
    # With no donor direction, all blend weights reduce to the base direction.
    v = torch.where((donor_norm > 1e-7)[:, None], v, u)
    valid = (base_norm > 1e-7) & (target_norm > 1e-7)
    return u, v, base_norm, real_dot(u, t), real_dot(v, t), real_dot(u, v).clamp(-1., 1.), valid


def angular_value(a, b, agreement, coefficient):
    norm = (1 - 2 * (1 - agreement) * coefficient * (1 - coefficient)).clamp_min(0).sqrt()
    angle = (a + coefficient * (b - a)) / norm.clamp_min(1e-7)
    return torch.where(norm > 1e-7, angle, a)


def oracle_weight(a, b, agreement, valid, cap):
    denominator = (1 - agreement) * (a + b)
    stable = denominator.abs() > 1e-7
    stationary = ((b - agreement * a) / torch.where(stable, denominator, torch.ones_like(denominator))).clamp(0., cap)
    candidates = torch.stack((torch.zeros_like(a), torch.full_like(a, cap), torch.where(stable, stationary, 0.)),
                             dim=1)
    scores = angular_value(a[:, None], b[:, None], agreement[:, None], candidates)
    best = candidates.gather(1, scores.argmax(1, keepdim=True)).squeeze(1)
    return torch.where(valid, best, 0.)


def source_bands(received, clean, model, projector, masks):
    provision = masks[0] | masks[1]
    _, group, received_fft, original, coords, donor = packet_features(received, projector, model, provision)
    base = fft(original)
    target = fft(no_tone(projector.coordinates(clean.flatten(0, 1))[0])) - received_fft
    estimate = torch.zeros_like(base)
    estimate[:, provision] = donor - received_fft[:, provision]
    bands = [direction_angles(base[:, m], estimate[:, m], target[:, m]) for m in masks]
    return bands, group, base, target, original, coords


def calibration(counts, angular_sums):
    pooled = angular_sums.sum(1) / counts.sum(1).clamp_min(1)[:, None]
    conditioned = (angular_sums + PSEUDO_CHIRPS * pooled[:, None]) / (counts[..., None] + PSEUDO_CHIRPS)
    grid = torch.tensor(GRID, device=counts.device)
    return grid[pooled.argmax(-1)], grid[conditioned.argmax(-1)]


@torch.no_grad()
def fit(rt, smoke=False):
    parent, paths, contract, paired = source_state(rt)
    device = 'cuda:0'
    model = source_model(parent, 'r69', paired, device).requires_grad_(False)
    projector = ObservedShape().to(device).eval()
    masks = band_masks(device)
    view = ResidualView(paths)
    if not view.ready():
        raise ValueError('Incomplete paired source residual data')
    indices = paired_indices(Path(rt['source_root']), contract)
    if smoke:
        indices = indices[:24]
    grid = torch.tensor(GRID, device=device)
    sums = torch.zeros(2, GROUPS, len(GRID), device=device, dtype=torch.float64)
    counts = torch.zeros(2, GROUPS, device=device, dtype=torch.long)
    for draw in range(2):
        for offset in range(0, len(indices), 24):
            batch = indices[offset:offset + 24]
            received, clean = tensor(view.received[draw, batch], device), tensor(view.clean[batch], device)
            bands, group, *_ = source_bands(received, clean, model, projector, masks)
            for i, (_, _, _, a, b, agreement, valid) in enumerate(bands):
                cell = 2 * group + (agreement >= ANGLE_AGREEMENT).long()
                delta = angular_value(a[:, None], b[:, None], agreement[:, None], grid[None]) - a[:, None]
                if not torch.isfinite(delta).all():
                    raise ValueError('Nonfinite source angular increment')
                sums[i].index_add_(0, cell[valid], delta[valid].double())
                counts[i] += torch.bincount(cell[valid], minlength=GROUPS)
        print('DONOR_MIX_FIT_DRAW', draw, counts.tolist(), flush=True)
    if (counts.sum(1) == 0).any():
        raise ValueError('No valid paired source directions for a frequency band')
    root = Path(rt['run_root'])
    global_weight, conditional = calibration(counts, sums)
    proof = dict(status='SOURCE_SMOKE_PASSED' if smoke else 'SOURCE_FROZEN', seed=SEED,
                 packets=len(indices), draws=2, counts=counts.tolist(), global_weight=global_weight.tolist(),
                 conditional=conditional.tolist(), G_updates=0, F_reads=0, TX_labels_read=0, target_reads=0)
    write(root / ('smoke.json' if smoke else 'calibration.json'), dict(source_contract=paired, **proof))


def mix_vector(original, u, v, norm, alpha):
    direction, length = unit((1 - alpha[:, None]) * u + alpha[:, None] * v)
    use = (length > 1e-7) & (norm > 1e-7)
    return torch.where(use[:, None], direction * norm[:, None], original)


def readouts(received, clean, model, projector, gate, masks, global_weight, conditional):
    active = gate(received)
    outputs = [received, *(received.clone() for _ in MODES[1:])]
    if not active.any():
        return torch.stack(outputs), 0, 0.
    observed = received[active]
    bands, group, base, target, original, coords = source_bands(observed, clean[active], model, projector, masks)
    alternatives = [base.clone() for _ in MODES[2:]]
    for i, (u, v, norm, a, b, agreement, valid) in enumerate(bands):
        cell = 2 * group + (agreement >= ANGLE_AGREEMENT).long()
        choices = (torch.full_like(a, .25), global_weight[i].expand_as(a), conditional[i, cell],
                   oracle_weight(a, b, agreement, valid, .5), oracle_weight(a, b, agreement, valid, 1.))
        mask = masks[i]
        for coeff, alpha in zip(alternatives[:-1], choices, strict=True):
            coeff[:, mask] = mix_vector(base[:, mask], u, v, norm, alpha)
        truth, truth_norm = unit(target[:, mask])
        alternatives[-1][:, mask] = torch.where((truth_norm > 1e-7)[:, None], truth * norm[:, None], base[:, mask])
    base_rms = power(base).sqrt()
    error = max(float((power(value).sqrt() - base_rms).abs().max() / 16) for value in alternatives)
    deltas = [original] + [torch.fft.ifft(value, dim=-1, norm='ortho') for value in alternatives]
    y = observed.flatten(0, 1)
    _, rotation, scale = coords
    for output, delta in zip(outputs[1:], deltas, strict=True):
        wave = delta * rotation * scale
        output[active] = (y + torch.stack((wave.real, wave.imag), dim=1)).reshape_as(observed)
    return torch.stack(outputs), len(observed), error


@torch.no_grad()
def score(rt):
    from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
    from .full_supervised import construct as recognizer

    parent, _, contract, paired = source_state(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    device = 'cuda:0'
    state = read(root / 'calibration.json')
    if (state['source_contract'] != paired or state['status'] != 'SOURCE_FROZEN' or state['seed'] != SEED
            or state['packets'] != sum(FIT_PER_RX) * len(SOURCE) or state['draws'] != 2
            or any(state[k] for k in ('G_updates', 'F_reads', 'TX_labels_read', 'target_reads'))):
        raise ValueError('Expected frozen source-only mixture')
    global_weight, conditional = (torch.tensor(state[key], device=device) for key in ('global_weight', 'conditional'))
    model = source_model(parent, 'r69', paired, device).requires_grad_(False)
    projector = ObservedShape().to(device).eval()
    masks = band_masks(device)
    gate = load_gate(parent.parent / GATE_RUN / 'gate.json', contract, device)
    f = recognizer(device).eval().requires_grad_(False)
    ckpt = complete_f(parent.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    iq, rx, ids, _ = source_role(source, contract, 'V')
    selected = sixth_panel(rx)
    packets = list(zip(iq[selected], rx[selected], ids[selected], strict=True))
    clean = tensor(np.stack([chirp_windows(row[0]) for row in packets]), device)
    predictions, metrics = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene in VIEWS:
            observed = (clean if scene == 'clean' else
                        tensor(np.stack(list(pool.map(partial(practical, scene), packets, chunksize=18))), device))
            parts, count, maximum = [], 0, 0.
            for y, x in zip(observed.split(20), clean.split(20), strict=True):
                waveforms, changed, error = readouts(y, x, model, projector, gate, masks, global_weight, conditional)
                logits = f(waveforms.flatten(0, 1))[0]
                if error > 1e-6 or not torch.isfinite(logits).all():
                    raise ValueError('Invalid mixture dose or frozen source prediction')
                parts.append(logits.argmax(1).reshape(len(MODES), len(y)).cpu().tolist())
                count += changed
                maximum = max(maximum, error)
            predictions[scene] = np.concatenate(parts, axis=1)
            metrics[scene] = dict(active=count, maximum_rms_error=maximum)
            print('DONOR_MIX_SOURCE_PREDICTED', scene, count, flush=True)
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = metadata['y'][selected]
    hits = {scene: labels == truth for scene, labels in predictions.items()}
    for scene, mask in hits.items():
        metrics[scene].update(correct=mask.sum(1).tolist(),
                              by_rx=mask.reshape(len(MODES), len(SOURCE), PER_RX).sum(-1).tolist())
    bootstrap = np.random.default_rng(2026112483)
    draws = np.concatenate([bootstrap.integers(j * PER_RX, (j + 1) * PER_RX, (1800, PER_RX))
                            for j in range(len(SOURCE))], axis=1)

    def ci(table):
        return [np.quantile(row[draws].sum(1), (.025, .975)).tolist() for row in table - table[1]]

    six = np.stack([hits[v] for v in VIEWS[1:]]).sum(0)
    low = hits['practical_low_urban'].astype(np.int16)
    write(root / 'source_scores.json', dict(status='SOURCE_SCORED', seed=SEED, modes=list(MODES),
          n=len(selected), target_reads=0, truth_after_predictions=True, views=metrics,
          six=six.sum(1).tolist(), six_ci=ci(six), low_ci=ci(low)))


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
    config = read(context)
    rt = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / PARENT
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Matching residual R69 source receipt required')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=1, max_compute_processes=4)
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
