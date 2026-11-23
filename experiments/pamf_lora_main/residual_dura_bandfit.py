"""Matched R69 continuation with paired clean supervision in dechirped bands."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.g_lineage import skeleton
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
from .residual_dura_calibration import BATCH, LR, MICRO, STEPS, source_model, source_state
from .residual_dura_gate import load as load_gate
from .residual_dura_gate_run import RUN as GATE_RUN
from .residual_dura_stage_blend import PER_RX, confirmation_indices, practical
from .runner import activate


RUN = '20261123-phase1-lora-dura-r69-bands-source-m3-r01'
ARMS = {'shape_anchor': (), 'band_9_32': (1,), 'bands_1_32': (0, 1)}
MODES = ('direct', 'r69', *ARMS, 'band_9_32_matched', 'bands_1_32_matched', 'clean_oracle_matched')
TRAIN_ORDER = 2026112478
HOLDOUT_ORDER = 2026112477
ANCHOR_WEIGHT = .25
DIRECTION_WEIGHT = .8
ENERGY_FLOOR = 1e-4


def band_masks(device):
    absolute_bin = torch.fft.fftfreq(256, device=device).abs() * 256
    return (absolute_bin >= 1) & (absolute_bin <= 8), (absolute_bin >= 9) & (absolute_bin <= 32)


def power(vector):
    return vector.abs().square().sum(dim=-1)


def fft(vector):
    return torch.fft.fft(vector, dim=-1, norm='ortho')


def correction(projector, output, coordinates, direct_shape):
    projected = projector.parts(output, coordinates)[0]
    return .5 * (projected - direct_shape)


def clean_direction(projector, clean, direct_shape):
    return fft(no_tone(projector.coordinates(clean)[0]) - direct_shape)


def band_losses(prediction, parent, reference, masks, chosen):
    far = ~(masks[0] | masks[1])
    fidelity = power(prediction[:, far] - parent[:, far]) / (power(parent[:, far]) + ENERGY_FLOOR)
    energy, direction = [], []
    for index, mask in enumerate(masks):
        student, base, truth = prediction[:, mask], parent[:, mask], reference[:, mask]
        original = power(base)
        rate = (power(student) + ENERGY_FLOOR).sqrt() - (original + ENERGY_FLOOR).sqrt()
        energy.append(rate.square() / (original + ENERGY_FLOOR))
        if index in chosen:
            truth_norm = power(truth).sqrt()
            expected = truth * (original.sqrt() / truth_norm.clamp_min(1e-7))[:, None]
            valid = truth_norm > 1e-7
            direction.append(valid * power(student - expected) / (original + ENERGY_FLOOR))
    anchor = fidelity + torch.stack(energy).mean(0)
    band = torch.stack(direction).mean() if direction else prediction.real.new_zeros(())
    return anchor.mean(), band


def losses(student, parent, teacher, projector, received, clean, masks, arm):
    estimates = student.stage_outputs(received)[0]
    coordinates = projector.coordinates(received)
    direct = no_tone(coordinates[0])
    with torch.no_grad():
        desired = teacher.stage_outputs(received)[0][1:]
        base = fft(correction(projector, parent(received), coordinates, direct))
        clean_target = clean_direction(projector, clean, direct)
    candidate = fft(correction(projector, estimates[-1], coordinates, direct))
    reconstruction = projector(received, clean, estimates, desired).mean()
    anchor, direction = band_losses(candidate, base, clean_target, masks, ARMS[arm])
    return reconstruction, anchor, direction


def fit(rt, arm, smoke=False):
    if arm not in ARMS:
        raise ValueError('Registered frequency-band continuation required')
    parent_root, paths, _, paired = source_state(rt)
    device = 'cuda:0'
    student = source_model(parent_root, 'r69', paired, device).train().requires_grad_(True)
    parent = source_model(parent_root, 'r69', paired, device).requires_grad_(False)
    teacher = source_model(parent_root, 'r65', paired, device).requires_grad_(False)
    masks = band_masks(device)
    projector = ObservedShape().to(device).eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
    view = ResidualView(paths)
    if not view.ready():
        raise ValueError('Incomplete paired source arrays')
    order = np.random.default_rng(TRAIN_ORDER).permutation(len(view))[:STEPS * BATCH].reshape(STEPS, BATCH)
    steps = 1 if smoke else STEPS
    for step, ids in enumerate(order[:steps], start=1):
        arrays = view.batch(ids, 1 + (step - 1) // (STEPS // 2))
        y = tensor(arrays['received'], device)
        clean = tensor(arrays['clean'], device)
        direction_weight = DIRECTION_WEIGHT * min(step / 150, 1.)
        optimizer.zero_grad(set_to_none=True)
        measurements = torch.zeros(3, device=device)
        for i, (received, truth) in enumerate(zip(y.split(MICRO), clean.split(MICRO), strict=True)):
            values = losses(student, parent, teacher, projector, received, truth, masks, arm)
            reconstruction, anchor, direction = values
            total = reconstruction + ANCHOR_WEIGHT * anchor + direction_weight * direction
            if not torch.isfinite(total):
                raise ValueError(f'Nonfinite band loss at step {step}, microbatch {i}')
            (total * (MICRO / BATCH)).backward()
            measurements += torch.stack([value.detach() for value in values]) * (MICRO / BATCH)
        if step == 1:
            for branch in (student.frozen.coarse, student.frozen.estimator, student.prior):
                norms = torch.stack([p.grad.norm() for p in branch.parameters() if p.grad is not None])
                if not (torch.isfinite(norms).all() and (norms > 0).any()):
                    raise ValueError('Expected finite nonzero gradients in every R69 branch')
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print('BAND_FIT_STEP', arm, step, direction_weight, measurements.tolist(), flush=True)
    root = Path(rt['run_root'])
    proof = dict(status='SOURCE_SMOKE_PASSED' if smoke else 'SOURCE_FROZEN', arm=arm, stage='r69',
                 updates=steps, seed=SEED, F_reads=0, TX_labels_read=0, target_reads=0,
                 source_exposures=steps * BATCH, labelled_exposures=int(view.is_L[order[:steps]].sum()))
    if smoke:
        write(root / 'smoke.json', proof)
    else:
        folder = root / arm
        folder.mkdir(exist_ok=False)
        torch.save(dict(proof=proof, source_contract=paired, model=student.state_dict()), folder / 'final.pt')
        write(folder / 'complete.json', proof)


def load_arm(root, arm, paired, device):
    folder = root / arm
    proof = read(folder / 'complete.json')
    frozen = torch.load(folder / 'final.pt', map_location='cpu', weights_only=True)
    if (proof != frozen['proof'] or paired != frozen['source_contract'] or proof['status'] != 'SOURCE_FROZEN'
            or proof['stage'] != 'r69' or proof['seed'] != SEED or proof['arm'] != arm or proof['updates'] != STEPS
            or any(proof[k] for k in ('F_reads', 'TX_labels_read', 'target_reads'))):
        raise ValueError('Expected completed same-contract R69 source continuation')
    model = skeleton('r69')
    model.load_state_dict(frozen['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def holdout_indices(rx):
    original = confirmation_indices(rx)
    second = np.random.default_rng(2026112476)
    fresh = np.random.default_rng(HOLDOUT_ORDER)
    chosen = []
    for name in SOURCE:
        eligible = np.setdiff1d(np.flatnonzero(rx == name), original)
        earlier = second.choice(eligible, PER_RX, replace=False)
        chosen.append(np.sort(fresh.choice(np.setdiff1d(eligible, earlier), PER_RX, replace=False)))
    return np.concatenate(chosen)


def matched_bands(base, direction, masks, chosen):
    adjusted = base.clone()
    for index in chosen:
        mask = masks[index]
        new, old = direction[:, mask], base[:, mask]
        new_length = power(new).sqrt()
        scale = power(old).sqrt() / new_length.clamp_min(1e-7)
        adjusted[:, mask] = torch.where((new_length > 1e-7)[:, None], new * scale[:, None], old)
    return adjusted


def readouts(received, clean, models, projector, gate, masks):
    enabled = gate(received)
    outputs = [received, *(received.clone() for _ in MODES[1:])]
    if not enabled.any():
        return torch.stack(outputs), 0, 0.
    selected = received[enabled]
    y = selected.flatten(0, 1)
    coordinates = projector.coordinates(y)
    direct = no_tone(coordinates[0])
    native = [correction(projector, model(y), coordinates, direct) for model in models]
    base, _, mid, both = (fft(delta) for delta in native)
    all_bands = (0, 1)
    paired = clean_direction(projector, clean[enabled].flatten(0, 1), direct)
    matched = (matched_bands(base, mid, masks, (1,)), matched_bands(base, both, masks, all_bands),
               matched_bands(base, paired, masks, all_bands))
    base_norm = power(base).sqrt()
    mismatch = max(float((power(v).sqrt() - base_norm).abs().max() / 16) for v in matched)
    _, rotation, scale = coordinates
    deltas = native + [torch.fft.ifft(value, norm='ortho') for value in matched]
    for output, delta in zip(outputs[1:], deltas, strict=True):
        waveform = delta * rotation * scale
        output[enabled] = (y + torch.stack((waveform.real, waveform.imag), dim=1)).reshape_as(selected)
    return torch.stack(outputs), int(enabled.sum()), mismatch


@torch.no_grad()
def score(rt):
    from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
    from .full_supervised import construct as recognizer

    parent, _, contract, paired = source_state(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    device = 'cuda:0'
    models = [source_model(parent, 'r69', paired, device).requires_grad_(False),
              *(load_arm(root, arm, paired, device) for arm in ARMS)]
    gate = load_gate(parent.parent / GATE_RUN / 'gate.json', contract, device)
    projector = ObservedShape().to(device).eval()
    predictor = recognizer(device).eval().requires_grad_(False)
    weight = complete_f(parent.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    predictor.load_state_dict(torch.load(weight, map_location=device, weights_only=True), strict=True)
    iq, rx, ids, _ = source_role(source, contract, 'V')
    chosen = holdout_indices(rx)
    physical = list(zip(iq[chosen], rx[chosen], ids[chosen], strict=True))
    clean = tensor(np.stack([chirp_windows(row[0]) for row in physical]), device)
    masks = band_masks(device)
    predictions, metrics = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene in VIEWS:
            received = (clean if scene == 'clean' else
                        tensor(np.stack(list(pool.map(partial(practical, scene), physical, chunksize=18))), device))
            counts, drift, parts = 0, 0., []
            for y, x in zip(received.split(20), clean.split(20), strict=True):
                views, active, mismatch = readouts(y, x, models, projector, gate, masks)
                logits = predictor(views.flatten(0, 1))[0]
                if mismatch > 1e-6 or not torch.isfinite(logits).all():
                    raise ValueError('Invalid frozen source band output or matched dose')
                parts.append(logits.argmax(1).reshape(len(MODES), len(y)).cpu().tolist())
                counts += active
                drift = max(drift, mismatch)
            predictions[scene] = np.concatenate(parts, axis=1)
            metrics[scene] = dict(active=counts, maximum_rms_error=drift)
            print('BAND_SOURCE_PREDICTED', scene, counts, flush=True)
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = metadata['y'][chosen]
    hits = {scene: values == truth for scene, values in predictions.items()}
    for scene, values in hits.items():
        metrics[scene].update(correct=values.sum(1).tolist(),
                              by_rx=values.reshape(len(MODES), len(SOURCE), PER_RX).sum(-1).tolist(),
                              gained=(values & ~values[1]).sum(1).tolist(),
                              lost=(~values & values[1]).sum(1).tolist())
    six = np.stack([hits[scene] for scene in VIEWS[1:]]).sum(0)
    low = hits['practical_low_urban'].astype(np.int8)
    rng = np.random.default_rng(HOLDOUT_ORDER)
    bootstrap = np.concatenate([rng.integers(i * PER_RX, (i + 1) * PER_RX, (1800, PER_RX))
                                for i in range(len(SOURCE))], axis=1)

    def intervals(table, reference):
        return [np.quantile(row[bootstrap].sum(1), (.025, .975)).tolist() for row in table - table[reference]]

    write(root / 'source_scores.json', dict(status='SOURCE_SCORED', modes=list(MODES), seed=SEED,
          n=len(chosen), target_reads=0, truth_after_predictions=True, views=metrics,
          practical_correct=six.sum(1).tolist(), six_ci_vs_r69=intervals(six, 1), low_ci_vs_r69=intervals(low, 1),
          six_ci_vs_shape=intervals(six, 2), low_ci_vs_shape=intervals(low, 2)))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    launch = partial(start, module=rt['runner'])
    try:
        write(state, dict(status='SOURCE_SMOKE'))
        schedule(rt, [('smoke', 'bands_1_32')], launch)
        write(state, dict(status='SOURCE_FITTING'), exclusive=False)
        schedule(rt, [('fit', arm) for arm in ARMS], launch)
        write(state, dict(status='FROZEN_SOURCE_SCORING'), exclusive=False)
        schedule(rt, [('score', 'all')], launch)
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    config = read(context)
    rt = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / PARENT
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Matching residual source parent receipt required')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 0, 2, 4, 5, 1], max_workers=3, max_compute_processes=4)
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
                fit(runtime, args.name, smoke=args.kind == 'smoke')
