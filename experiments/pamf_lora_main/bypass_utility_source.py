"""Test a full practical extension of the frozen complex-response half-dose pipeline."""

import argparse
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write

from .bypass_utility import Proposals, Utility, actions
from .data import SOURCE, VIEWS
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_dura_detail import COMPLEX_RESPONSE_RUN, load_detail, phase_baseline_models
from .full_dura_observed_shape import ObservedShape
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_restore_draw import RUN as DRAW_RUN, load_draw
from .full_supervised import construct as recognizer, handles
from .packet_runner import schedule
from .publish import release, release_commit
from .r17_anchored_source import validation_indices as anchor_panel
from .refine_runner import start
from .residual_dura import RUN as RESIDUAL_RUN, SEED
from .runner import activate
from .spectral_unroll import bind


RUN = '20261123-phase1-lora-dura-signed-utility-source-m1-r01'
SEED_CONTROL = SEED + 745
TRAIN = 3200
EPOCHS = 36
PER_RX = 300
MODES = ('direct', 'complex_response_half', 'detector_response', 'detector_r17',
         'utility_response', 'utility_r17', 'joint', 'paired_waveform_oracle')
ISOLATION = dict(F_reads=0, TX_reads=0, V_reads=0, target_reads=0)


def contract_and_models(rt, device):
    parent = Path(rt['parent_root']).parent
    source, cache, contract = handles(dict(rt, parent_root=str(parent / DATA_RUN)))
    response_root = parent / COMPLEX_RESPONSE_RUN
    base, prior = phase_baseline_models(response_root, contract, SEED, device)
    response = load_detail(response_root, contract, SEED, device)
    r17, ancestry, _ = bind(rt, 'full', device)
    if ancestry != contract:
        raise ValueError('Full practical recovery ancestors must use the same source contract')
    models = Proposals(base.eval(), prior.eval(), response, r17.eval().requires_grad_(False),
                       ObservedShape().to(device).eval())
    return source, cache, contract, models


@torch.no_grad()
def collect(clean, arrays, models, indices, device):
    features, eligibility, scores, labels = [], [], [], []
    for slot, samples in enumerate(arrays):
        for group in np.array_split(indices, 1 + len(indices) // 48):
            observed = tensor(samples[group], device)
            ground = observed if slot == 0 else tensor(clean[group], device)
            waves, valid, view, _ = models.candidates(observed)
            gain = models.gains(waves, ground)
            if not (torch.isfinite(view).all() and torch.isfinite(gain).all()):
                raise FloatingPointError('Invalid signed source gain or feature')
            labels.append(torch.full((len(group),), float(slot != 0), device=device))
            features.append(view)
            eligibility.append(valid)
            scores.append(gain)
    return tuple(torch.cat(part) for part in (features, eligibility, scores, labels))


@torch.no_grad()
def clean_detection(model, models, clean, indices, device):
    detection = []
    for group in np.array_split(indices, 1 + len(indices) // 256):
        features, _ = models.features(tensor(clean[group], device))
        detection.append(model.detection(features))
    return torch.cat(detection)


def fit(rt, *, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(SEED_CONTROL)
    _, cache, contract, models = contract_and_models(rt, device)
    clean = np.load(cache / 'U_s.npy', mmap_mode='r', allow_pickle=False)[0]
    draws = (np.load(cache / 'U_s_practical.npy', mmap_mode='r', allow_pickle=False),
             load_draw(Path(rt['parent_root']).parent / DRAW_RUN, 'U_s'))
    n = contract['counts']['U_s']
    if any(a.shape != (n, 6, 2, 256) or a.dtype != np.float32 for a in (clean, *draws)):
        raise ValueError('Incomplete registered U source views')
    order = np.random.default_rng(SEED_CONTROL).permutation(n)
    train, calibration = (order[:48], order[48:96]) if smoke else (order[:TRAIN], order[TRAIN:])
    inputs, eligible, gain, labels = collect(clean, (clean, *draws), models, train, device)
    model = Utility().to(device)
    model.center.copy_(inputs.mean(0))
    model.scale.copy_(inputs.std(0).clamp_min(.01))
    optim = torch.optim.AdamW(model.parameters(), lr=.004, weight_decay=.01)
    generator = torch.Generator(device=device).manual_seed(SEED_CONTROL)
    epochs = 1 if smoke else EPOCHS
    for epoch in range(1, epochs + 1):
        total = torch.zeros(2, device=device)
        for rows in torch.randperm(len(labels), device=device, generator=generator).split(256):
            score = model.detection(inputs[rows])
            y = labels[rows]
            detect_loss = F.binary_cross_entropy_with_logits(score, y, weight=torch.where(y == 0, 2., 1.))
            predicted = model.utility(inputs[rows])
            fit_loss = F.smooth_l1_loss(predicted, gain[rows].clamp(-1., 1.), beta=.1, reduction='none')
            valid = eligible[rows]
            risk_loss = (valid * fit_loss).sum() / valid.sum().clamp_min(1)
            value = detect_loss + risk_loss
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite detector or signed expected gain objective')
            optim.zero_grad(set_to_none=True)
            value.backward()
            optim.step()
            total += len(rows) * torch.stack((detect_loss.detach(), risk_loss.detach()))
        print('UTILITY_FIT', epoch, *(total / len(labels)).tolist(), flush=True)
    fits = clean_detection(model, models, clean, train, device)
    held = clean_detection(model, models, clean, calibration, device)
    model.boundary.copy_(torch.maximum(fits.max(), held.max()) + 1e-4)
    if not torch.isfinite(model.boundary) or (held > model.boundary).any():
        raise ValueError('Independent source clean calibration failed')
    root = Path(rt['run_root'])
    receipt = dict(status='PASS' if smoke else 'SOURCE_FROZEN', seed=SEED, epochs=epochs,
                   train_packets=len(train), calibration_packets=len(calibration), **ISOLATION)
    if smoke:
        write(root / 'smoke.json', receipt)
    else:
        torch.save(dict(receipt=receipt, contract=contract, utility=model.state_dict()), root / 'model.pt')
        write(root / 'complete.json', receipt)


def validation_indices(rx):
    excluded = anchor_panel(rx)
    rng = np.random.default_rng(SEED_CONTROL + 1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == name), excluded),
                                             PER_RX, replace=False)) for name in SOURCE])


def readouts(received, waves, eligible, detection, predicted, oracle):
    rows = torch.arange(len(received), device=received.device)
    first_only, second_only = eligible.clone(), eligible.clone()
    first_only[:, 1] = False
    second_only[:, 0] = False
    cap = detection.to(torch.int64)
    chosen = torch.stack((cap * eligible[:, 0], 2 * cap * eligible[:, 1],
                          actions(first_only, detection, predicted), actions(second_only, detection, predicted),
                          actions(eligible, detection, predicted), actions(eligible, detection, oracle)))
    return torch.cat((received[None], waves[0:1], waves[chosen, rows]), dim=0), chosen


@torch.no_grad()
def score(rt):
    device = 'cuda:0'
    source, cache, contract, models = contract_and_models(rt, device)
    root = Path(rt['run_root'])
    state = torch.load(root / 'model.pt', map_location=device, weights_only=True)
    expected = dict(status='SOURCE_FROZEN', seed=SEED, epochs=EPOCHS, train_packets=TRAIN,
                    calibration_packets=contract['counts']['U_s'] - TRAIN, **ISOLATION)
    if state['contract'] != contract or state['receipt'] != expected or read(root / 'complete.json') != expected:
        raise ValueError('Fixed and source-only full practical gate required')
    selector = Utility().to(device).eval().requires_grad_(False)
    selector.load_state_dict(state['utility'], strict=True)
    f = recognizer(device).eval().requires_grad_(False)
    f_path = complete_f(root.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(f_path, map_location=device, weights_only=True), strict=True)
    _, rx, ids, _ = source_role(source, contract, 'V')
    chosen = validation_indices(rx)
    archive = np.load(cache / 'V.npy', mmap_mode='r', allow_pickle=False)
    if archive.shape != (len(VIEWS), len(ids), 6, 2, 256) or archive.dtype != np.float32:
        raise ValueError('Fixed full practical V arrays differ from the physical order')
    predictions, metrics = {}, {}
    for i, view in enumerate(VIEWS):
        parts, hits, old = [], np.zeros((len(MODES) - 2, 2), dtype=np.int64), 0
        forecast, measured, detector_active = [], [], 0
        for group in np.array_split(chosen, 1 + len(chosen) // 32):
            y, clean = (tensor(archive[index, group], device) for index in (i, 0))
            candidates, eligible, features, accepted = models.candidates(y)
            disturbed = selector.detection(features) > selector.boundary
            estimate, truth = selector.utility(features), models.gains(candidates, clean)
            waves, chosen_actions = readouts(y, candidates, eligible, disturbed, estimate, truth)
            logits = f(waves.flatten(0, 1))[0]
            if not (torch.isfinite(logits).all() and torch.isfinite(estimate).all()):
                raise ValueError('Nonfinite frozen utility readout')
            parts.append(logits.argmax(1).reshape(len(MODES), len(group)).cpu().tolist())
            hits += np.asarray(torch.stack([(chosen_actions == val).sum(1) for val in (1, 2)],
                                           dim=1).cpu().tolist())
            old += int(accepted.sum())
            detector_active += int(disturbed.sum())
            forecast.extend(estimate[eligible].tolist())
            measured.extend(truth[eligible].tolist())
        predictions[view] = np.concatenate(parts, axis=1)
        sign_match = (np.asarray(forecast) > 0) == (np.asarray(measured) > 0)
        metrics[view] = dict(old_active=old, detector_active=detector_active, actions=hits.tolist(),
                             eligible_gain_sign_match=float(sign_match.mean()) if len(sign_match) else None)
        print('UTILITY_PREDICTED', view, old, detector_active, flush=True)
    write(root / 'predictions.json', dict(ids=ids[chosen].tolist(), rows={v: p.tolist() for v, p in predictions.items()}))
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as m:
        truth = m['y'][chosen]
    results = {view: item == truth for view, item in predictions.items()}
    for view, hit in results.items():
        metrics[view].update(correct=hit.sum(1).tolist(), by_rx=hit.reshape(len(MODES), len(SOURCE), PER_RX).sum(-1).tolist())
    six = np.stack([results[v] for v in VIEWS[1:]]).sum(0)
    low = results['practical_low_urban'].astype(int)
    rng = np.random.default_rng(SEED_CONTROL + 2)
    sample = np.concatenate([rng.integers(i * PER_RX, (i + 1) * PER_RX, (1600, PER_RX))
                             for i in range(len(SOURCE))], axis=1)
    ci = {k: [np.quantile(row[sample].sum(1), (.025, .975)).tolist() for row in v - v[1]]
          for k, v in (('six', six), ('low', low))}
    write(root / 'source-scores.json', dict(status='SOURCE_SCORED', n=len(chosen), modes=list(MODES),
          metrics=metrics, six=six.sum(1).tolist(), paired_ci_vs_baseline=ci, target_reads=0,
          truth_after_predictions=True))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch = partial(start, module=rt['runner'])
    try:
        for action in ('smoke', 'fit', 'score'):
            write(root / 'state.json', dict(status=action.upper()), exclusive=False)
            schedule(rt, [(action, 'full')], launch)
        write(root / 'state.json', dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    cfg = read(context)
    rt = {k: cfg[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / RESIDUAL_RUN
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Registered LoRa source ancestry required')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=1,
              max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for kind in ('dispatch', 'smoke', 'fit', 'score'):
        worker = commands.add_parser(kind)
        worker.add_argument('runtime', type=Path)
        worker.add_argument('name', nargs='?')
    opts = vars(parser.parse_args())
    kind = opts.pop('kind')
    if kind == 'publish':
        publish(**opts)
    else:
        rt = read(opts['runtime'])
        if kind == 'dispatch':
            dispatch(rt)
        else:
            activate(rt)
            if kind == 'score':
                score(rt)
            else:
                fit(rt, smoke=kind == 'smoke')
