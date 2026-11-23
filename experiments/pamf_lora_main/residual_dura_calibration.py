"""Source-only matched continuation of compact LoRa DURA with action calibration."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.g_lineage import load, skeleton
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_action_restore.evaluation import probe_error, teacher_action
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_r17_compact.models import stages
from experiments.pamf_restoration.models import complex_iq

from .data import SOURCE, VIEWS
from .dura_repair import no_tone
from .full_dura_lineage_data import in_observed_phase
from .full_dura_observed_shape import ObservedShape
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_contract, source_role
from .packet_physics import chirp_windows, residual_view
from .packet_runner import schedule
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import ROUTE, RUN as PARENT, SEED, ResidualView
from .residual_dura_gate import load as load_gate
from .residual_dura_gate_run import RUN as GATE_RUN
from .residual_dura_stage_blend import PER_RX, confirmation_indices, practical
from .runner import activate


RUN = '20261123-phase1-lora-dura-calibration-source-m2-r01'
ARMS = ('shape_only', 'calibrated')
MODES = ('direct', 'r44', *ARMS, 'calibrated_equal_r44_rms')
STEPS = 600
BATCH = 128
MICRO = 64
MAX_ANCHOR = .20
SAMPLING_SEED = 2026112473
LR = 6e-5


def source_state(rt):
    parent = Path(rt['parent_root'])
    contract = source_contract(Path(rt['source_root']))
    pairs = parent / 'source_pairs'
    paired = read(pairs / 'contract.json')
    if paired != dict(contract, practical=ROUTE):
        raise ValueError('Matching source residual pairs required')
    return parent, pairs, contract, paired


def source_model(parent, stage, paired, device):
    folder = parent / 'G' / 'lineage' / stage
    state = read(folder / 'complete.json')
    if state['status'] != 'SOURCE_FROZEN' or state['stage'] != stage or state['updates'] != 4940:
        raise ValueError('Matching complete source parent required')
    model, proof = load(folder / 'final.pt', paired, 'LU', device)
    if proof['stage'] != stage or proof['model_seed'] != SEED:
        raise ValueError('Wrong source lineage or seed')
    return model


def losses(student, teacher, objective, arrays, selected, probes, device):
    received, clean = (tensor(arrays[k][selected], device) for k in ('received', 'clean'))
    context = student.estimate_context(received)
    values = stages(student.prior, received, context)[0]
    with torch.no_grad():
        desired = teacher.stage_outputs(received)[0]
    waveform = objective(received, clean, values, desired).mean()
    _, action, variance = context
    target = teacher_action(arrays, selected, device)
    noise = tensor(arrays['variance'][selected], device)
    forward = probe_error(action, target, probes).mean()
    uncertainty = (variance.log() - noise.log()).square().mean()
    return waveform, forward, uncertainty


def fit(rt, arm, *, smoke=False):
    parent, pairs, _, paired = source_state(rt)
    if arm not in ARMS:
        raise ValueError('Registered calibration arm required')
    device = 'cuda:0'
    student = source_model(parent, 'r44', paired, device).train().requires_grad_(True)
    teacher = source_model(parent, 'r38', paired, device).requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
    waveform = ObservedShape().to(device).eval()
    view = ResidualView(pairs)
    if not view.ready():
        raise ValueError('Incomplete source training pair arrays')
    order = np.random.default_rng(SAMPLING_SEED).permutation(len(view))[:STEPS * BATCH]
    generator = torch.Generator(device=device).manual_seed(SAMPLING_SEED)
    updates = 1 if smoke else STEPS
    for step, indices in enumerate(order.reshape(STEPS, BATCH)[:updates], start=1):
        arrays = view.batch(indices, 1 + (step - 1) // (STEPS // 2), action=True)
        multiplier = MAX_ANCHOR * min(step / 200, 1.) if arm == 'calibrated' else 0.
        optimizer.zero_grad(set_to_none=True)
        metrics = torch.zeros(3, device=device)
        for offset in range(0, BATCH, MICRO):
            sl = slice(offset, offset + MICRO)
            probes = [torch.complex(*torch.randn((2, MICRO, 256), device=device, generator=generator)) / 2**.5
                      for _ in range(2)]
            current = losses(student, teacher, waveform, arrays, sl, probes, device)
            shape, forward, variance = current
            total = shape + multiplier * (forward + .1 * variance)
            if not torch.isfinite(total):
                raise ValueError('Nonfinite calibration loss')
            (total * (MICRO / BATCH)).backward()
            metrics += torch.stack([part.detach() for part in current]) * (MICRO / BATCH)
        if step == 1:
            for module in (student.frozen.estimator, student.frozen.coarse, student.prior):
                norms = torch.stack([p.grad.norm() for p in module.parameters() if p.grad is not None])
                if not (torch.isfinite(norms).all() and (norms > 0).any()):
                    raise ValueError('Calibration must reach each active branch')
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print('CALIBRATION_STEP', arm, step, multiplier, metrics.tolist(), flush=True)
    proof = dict(status='SOURCE_SMOKE_PASSED' if smoke else 'SOURCE_FROZEN', arm=arm, stage='r44',
                 updates=updates, seed=SEED, F_reads=0, TX_labels_read=0, target_reads=0,
                 labelled_exposures=int(view.is_L[order[:updates * BATCH]].sum()),
                 source_exposures=updates * BATCH)
    root = Path(rt['run_root'])
    if smoke:
        write(root / 'smoke.json', proof)
    else:
        output = root / arm
        output.mkdir(exist_ok=False)
        torch.save(dict(proof=proof, source_contract=paired, model=student.state_dict()), output / 'final.pt')
        write(output / 'complete.json', proof)


def fitted_model(root, arm, paired, device):
    state = read(root / arm / 'complete.json')
    saved = torch.load(root / arm / 'final.pt', map_location='cpu', weights_only=True)
    if (state != saved['proof'] or saved['source_contract'] != paired or state['status'] != 'SOURCE_FROZEN'
            or state['arm'] != arm or state['updates'] != STEPS or state['seed'] != SEED
            or any(state[k] for k in ('F_reads', 'TX_labels_read', 'target_reads'))):
        raise ValueError('Incomplete or noncomparable calibration weights')
    model = skeleton('r44')
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def readouts(received, models, projector, gate):
    mask = gate(received)
    outputs = [received, *(received.clone() for _ in MODES[1:])]
    if not mask.any():
        return torch.stack(outputs), 0, 0.
    selected = received[mask]
    flat = selected.flatten(0, 1)
    observed, rotation, scale = coordinates = projector.coordinates(flat)
    direct = no_tone(observed)
    deltas = [.5 * (projector.parts(model(flat), coordinates)[0] - direct) for model in models]
    reference_rms = deltas[0].abs().square().mean(-1, keepdim=True).sqrt()
    calibrated_rms = deltas[-1].abs().square().mean(-1, keepdim=True).sqrt()
    matched = torch.where(calibrated_rms > 1e-7,
                          deltas[-1] * (reference_rms / calibrated_rms.clamp_min(1e-7)), deltas[0])
    drift = float((matched.abs().square().mean(-1, keepdim=True).sqrt() - reference_rms).abs().max())
    for output, delta in zip(outputs[1:], (*deltas, matched), strict=True):
        correction = delta * rotation * scale
        output[mask] = (flat + torch.stack((correction.real, correction.imag), dim=1)).reshape_as(selected)
    return torch.stack(outputs), int(mask.sum()), drift


def low_pair(row):
    iq, receiver, pid = row
    received, signal, noise = residual_view(iq, 'practical_low_urban', receiver, pid, 'source-V',
                                            components=True)
    clean = in_observed_phase(chirp_windows(iq), received)
    noise_variance = .5 * np.mean(np.abs(noise)**2, axis=-1)
    signal_iq = np.stack((signal.real, signal.imag), axis=1).astype(np.float32)
    return received, clean, signal_iq, noise_variance.astype(np.float32)


def validate_action(models, pairs, device):
    by_model = []
    for model in models:
        columns = []
        for start in range(0, len(pairs), 16):
            y, clean, signal, noise = (np.stack(value) for value in zip(*pairs[start:start + 16], strict=True))
            y = tensor(y.reshape(-1, 2, 256), device)
            reference = complex_iq(tensor(clean.reshape(-1, 2, 256), device))
            signal = complex_iq(tensor(signal.reshape(-1, 2, 256), device))
            noise = tensor(noise.reshape(-1), device).clamp_min(1e-8)
            _, action, variance = model.estimate_context(y)
            fitted = action(reference)
            signal_power = signal.abs().square().mean(-1).clamp_min(1e-12)
            forward_nmse = (signal - fitted).abs().square().mean(-1) / signal_power
            fitted_power = fitted.abs().square().mean(-1)
            variance_bias = 10 * torch.log10(variance / noise)
            ratio = SCALE**2 * fitted_power / (reference.abs().square().mean(-1).clamp_min(1e-12) * variance)
            columns.append(torch.stack((forward_nmse, fitted_power / signal_power, variance_bias, ratio)).cpu())
        medians = torch.cat(columns, dim=1).quantile(.5, dim=1)
        by_model.append(dict(zip(('forward_nmse', 'power_ratio', 'variance_bias_db', 'data_to_prior'),
                                 medians.tolist(), strict=True)))
    return dict(zip(('r44', *ARMS), by_model, strict=True))


@torch.no_grad()
def score(rt):
    from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
    from .full_supervised import construct as recognizer

    parent, _, contract, paired = source_state(rt)
    root = Path(rt['run_root'])
    source = Path(rt['source_root'])
    device = 'cuda:0'
    models = [source_model(parent, 'r44', paired, device).requires_grad_(False),
              *(fitted_model(root, arm, paired, device) for arm in ARMS)]
    gate = load_gate(parent.parent / GATE_RUN / 'gate.json', contract, device)
    projector = ObservedShape().to(device).eval()
    recognizer_model = recognizer(device).eval().requires_grad_(False)
    checkpoint = complete_f(parent.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    recognizer_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    iq, rx, ids, _ = source_role(source, contract, 'V')
    chosen = confirmation_indices(rx)
    rows = list(zip(iq[chosen], rx[chosen], ids[chosen], strict=True))
    predictions, metrics = {}, {}
    action_calibration = None
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene in VIEWS:
            if scene == 'practical_low_urban':
                paired_low = list(pool.map(low_pair, rows, chunksize=18))
                observed = np.stack([row[0] for row in paired_low])
                action_calibration = validate_action(models, paired_low, device)
            elif scene == 'clean':
                observed = np.stack([chirp_windows(row[0]) for row in rows])
            else:
                observed = np.stack(list(pool.map(partial(practical, scene), rows, chunksize=18)))
            parts, active, maximum = [], 0, 0.
            for chunk in tensor(observed, device).split(24):
                views, count, drift = readouts(chunk, models, projector, gate)
                logits = recognizer_model(views.flatten(0, 1))[0]
                if not torch.isfinite(logits).all() or drift > 1e-6:
                    raise ValueError('Nonfinite recognizer or mismatched recovery dose')
                parts.append(logits.argmax(1).reshape(len(MODES), len(chunk)).cpu().tolist())
                active += count
                maximum = max(maximum, drift)
            predictions[scene] = np.concatenate(parts, axis=1)
            metrics[scene] = dict(active=active, maximum_relative_rms_error=maximum)
            print('CALIBRATION_SOURCE_PREDICTED', scene, active, flush=True)
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = metadata['y'][chosen]
    for scene, decisions in predictions.items():
        valid = decisions == truth
        reference = valid[MODES.index('shape_only')]
        metrics[scene].update(correct=valid.sum(1).tolist(),
                              gained=(valid & ~reference).sum(1).tolist(),
                              lost=(~valid & reference).sum(1).tolist(),
                              by_rx=valid.reshape(len(MODES), len(SOURCE), PER_RX).sum(-1).tolist())
    write(root / 'source_scores.json', dict(status='SOURCE_SCORED', seed=SEED, stages=list(MODES),
                                           n=len(chosen), truth_after_predictions=True, target_reads=0,
                                           views=metrics, low_urban_action_calibration=action_calibration))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    launch = partial(start, module=rt['runner'])
    try:
        write(state, dict(status='SOURCE_SMOKE'))
        schedule(rt, [('smoke', 'calibrated')], launch)
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
              parent_root=str(parent), allowed_gpus=[6, 0, 2, 4, 5, 1], max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for kind in ('dispatch', 'smoke', 'fit', 'score'):
        work = commands.add_parser(kind)
        work.add_argument('runtime', type=Path)
        work.add_argument('name', nargs='?')
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
