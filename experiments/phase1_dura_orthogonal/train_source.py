"""Train a DURA correction using only permitted paired Phase1 source IQ."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.model import OrthogonalRecovery, DualDomainRecovery

FIT_SIZE = 8192
TRAIN_EPOCHS = 5
DUAL_DOMAIN_EPOCHS = 20
SOURCE_EPOCHS = (21, 22, 23)
SCHEMA = 'phase1_dura_orthogonal_source_v1'


@torch.no_grad()
def extract_context(dura, x):
    context = dura.estimate_context(x)
    stages, _, action, _ = dura.stage_outputs(x, context)
    anchor = stages[-1]
    y = torch.complex(x[:, 0], x[:, 1])
    z = torch.complex(anchor[:, 0], anchor[:, 1])
    residual = action.adjoint(y - action(z))
    return anchor, torch.stack((residual.real, residual.imag), dim=1), context[2]


def source_loss(output, anchor, clean, *, objective='iq'):
    if output.shape != clean.shape or anchor.shape != clean.shape or output.shape[1:] != (2, 256):
        raise ValueError('Matching source IQ windows required')
    if objective not in ('iq', 'dura_guard', 'band_guard', 'nonlinear_guard'):
        raise ValueError('A registered source-IQ objective is required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    error = (output-clean).square().mean((1, 2)) / energy
    change = (output-anchor).square().mean((1, 2)) / energy
    per = error + .02*change
    if objective in ('dura_guard', 'band_guard', 'nonlinear_guard'):
        baseline_error = (anchor.detach()-clean).square().mean((1, 2)) / energy
        excess = (error-baseline_error).clamp_min(0)
        per = per + .002*excess/(baseline_error+1e-4)
    if objective == 'band_guard':
        delta = output-clean
        complex_delta = torch.complex(delta[:, 0], delta[:, 1])
        spectrum = torch.fft.fft(complex_delta, dim=-1, norm='ortho')
        freq = torch.fft.fftfreq(256, device=output.device).abs()
        band = (freq >= .125) & (freq < .375)
        per = per + .5*spectrum[:, band].abs().square().sum(-1)/(512*energy)
    if objective == 'nonlinear_guard':
        scale = energy.sqrt()[:, None, None]
        restored_normal = output/scale
        clean_normal = clean/scale
        def response(z):
            power = z.square().sum(dim=1, keepdim=True)
            return z * (power/(1+power))
        per = per + .1*(response(restored_normal)-response(clean_normal)).square().mean((1, 2))
    return per.mean()


def load_source_training(root, eval_ids):
    complete = core.read(root/'runs'/core.F_RUN/'F'/core.F_ROW/'source/complete.json')
    contract = core.read(complete['config']['source_contract'])
    cache, clean, roles = core.data(root, contract)
    rx = np.load(cache/'train/rx.npy', allow_pickle=False)
    day = np.load(cache/'train/day.npy', allow_pickle=False)
    if rx.shape != roles.shape or day.shape != roles.shape:
        raise ValueError('Matching public source receiver and day IDs required')
    eligible = np.flatnonzero(~(np.isin(rx, (4, 6, 8)) & (day == 3)))
    eligible = np.setdiff1d(eligible, eval_ids, assume_unique=False)
    if len(eligible) < FIT_SIZE or not np.isin(roles[eligible], ('L_s', 'U_s')).all():
        raise ValueError('Sufficient source training IQ is required')
    fit = np.sort(np.random.default_rng(2026110611).choice(eligible, size=FIT_SIZE, replace=False))
    fresh = root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
    if (fresh/'train/physical_ids.npy').resolve() != (cache/'train/physical_ids.npy').resolve():
        raise ValueError('Source augmentations must share original physical IDs')
    return cache, fresh, clean, fit


@torch.no_grad()
def prepare(root, dura, clean, fresh, fit, device, batch_size, architecture='orthogonal'):
    chunks = []
    scene_counts = {}
    for epoch in SOURCE_EPOCHS:
        folder = fresh/'epochs'/f'{epoch:02d}'
        metadata = core.read(folder/'complete.json')
        if (metadata.get('count') != core.COUNT or
                metadata.get('namespace') != f'learned-action/train/epoch{epoch:02d}' or
                any(metadata.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
            raise ValueError('Approved F-free source augmentation required')
        received = np.load(folder/'received.npy', mmap_mode='r', allow_pickle=False)
        with np.load(folder/'groups.npz', allow_pickle=False) as group:
            scene = group['scene'].copy()
        if received.shape != clean.shape or scene.shape != (core.COUNT,):
            raise ValueError('Complete source IQ cache required')
        for start in range(0, len(fit), batch_size):
            ids = fit[start:start+batch_size]
            x, y = core.tensor(received[ids], device), core.tensor(clean[ids], device)
            if architecture == 'dual_domain':
                anchor, direction, variance = x, torch.zeros_like(x), torch.ones(len(x), device=x.device)
            else:
                anchor, direction, variance = extract_context(dura, x)
            values = tuple(t.detach().cpu() for t in (x, anchor, direction, variance, y))
            if not all(torch.isfinite(v).all() for v in values) or (values[3] <= 0).any():
                raise FloatingPointError('Finite source IQ and context required')
            chunks.append(values)
        for view in core.VIEWS:
            scene_counts[view] = scene_counts.get(view, 0) + int(np.count_nonzero(scene[fit] == view))
        print('ORTHOGONAL_SOURCE_PREP', epoch, len(fit), flush=True)
    return tuple(torch.cat(v, 0) for v in zip(*chunks)), scene_counts


@torch.no_grad()
def evaluate_source(model, dura, ids, rx, scene, received, clean, device, batch_size):
    model.eval()
    chunks = []
    for start in range(0, len(ids), batch_size):
        index = ids[start:start+batch_size]
        x, y = core.tensor(received[index], device), core.tensor(clean[index], device)
        anchor, direction, variance = extract_context(dura, x)
        output = model(x, anchor, direction, variance)
        energy = y.square().mean((1, 2)).clamp_min(1e-8)
        chunks.append(torch.stack(((anchor-y).square().mean((1, 2))/energy,
                                   (output-y).square().mean((1, 2))/energy), dim=1).cpu())
    values = np.from_dlpack(torch.cat(chunks).contiguous()).copy()
    results = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source receiver and scene cells required')
            results[name+'/'+view] = dict(count=int(mask.sum()), anchor_nmse=float(values[mask, 0].mean()),
                                         candidate_nmse=float(values[mask, 1].mean()))
    return values, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-batch', type=int, default=32)
    parser.add_argument('--smoke-fit', type=int, default=64)
    parser.add_argument('--objective', choices=('iq', 'dura_guard', 'band_guard', 'nonlinear_guard'), default='iq')
    parser.add_argument('--architecture', choices=('orthogonal', 'dual_domain'), default='orthogonal')
    args = parser.parse_args()
    if args.architecture == 'dual_domain' and args.objective != 'iq':
        raise ValueError('Direct dual-domain model uses received-IQ regularization only')
    process_started = time.monotonic()
    training_environments(args.device, os.environ)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('An available CPU or explicitly selected CUDA device is required')
    if not 1 <= args.prepare_batch <= 128 or not 1 <= args.smoke_fit <= FIT_SIZE:
        raise ValueError('Bounded source preparation and smoke sizes required')
    if args.out.exists():
        raise FileExistsError('Unique source output required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026110612)
    ids, rx, scene, _, source_received, source_clean = source_inputs(args.root)
    _, fresh, clean, fit = load_source_training(args.root, ids)
    if args.smoke:
        fit = fit[:args.smoke_fit]
    dura = frozen_dura(args.root, str(device)) if args.architecture == 'orthogonal' else None
    preparation_started = time.monotonic()
    data, counts = prepare(args.root, dura, clean, fresh, fit, str(device), args.prepare_batch, args.architecture)
    preparation_seconds = time.monotonic() - preparation_started
    transfer_started = time.monotonic()
    data = tuple(value.to(device) for value in data)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    transfer_seconds = time.monotonic() - transfer_started
    model = (OrthogonalRecovery(width=32) if args.architecture == 'orthogonal'
             else DualDomainRecovery(width=32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    generator = torch.Generator().manual_seed(2026110613)
    losses = []
    with torch.no_grad():
        initial = model(*(value[:8] for value in data[:4]))
    if not torch.equal(initial, data[1][:8]):
        raise ValueError('New correction must initially reproduce frozen DURA')
    started = time.monotonic()
    epochs = TRAIN_EPOCHS if args.architecture == 'orthogonal' else DUAL_DOMAIN_EPOCHS
    for epoch in range(1, (1 if args.smoke else epochs)+1):
        model.train()
        order = torch.randperm(len(data[0]), generator=generator).to(device)
        total = torch.zeros((), device=device)
        for start in range(0, len(order), 32):
            indices = order[start:start+32]
            x, anchor, direction, variance, y = (v[indices] for v in data)
            output = model(x, anchor, direction, variance, validate=False)
            loss = source_loss(output, anchor, y, objective=args.objective)
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite IQ-only source loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            second_head = model.step if args.architecture == 'orthogonal' else model.spectral_head
            if (model.wave.weight.grad is None or second_head.weight.grad is None or
                    not torch.isfinite(model.wave.weight.grad).all() or not torch.isfinite(second_head.weight.grad).all()):
                raise ValueError('Both learned correction paths require finite gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach() * len(indices)
        losses.append(float(total.cpu())/len(order))
        print('ORTHOGONAL_SOURCE_EPOCH', epoch, 'loss', losses[-1], 'elapsed', round(time.monotonic()-started, 1), flush=True)
    if dura is not None and any(p.requires_grad or p.grad is not None for p in dura.parameters()):
        raise ValueError('DURA must stay frozen')
    if args.smoke:
        core.write(args.out/'smoke.json', dict(status='SOURCE_TRAINING_SMOKE_PASS', commit=args.commit,
            fit_ids=len(fit), exposure=len(data[0]), losses=losses, device=str(device),
            objective=args.objective, architecture=args.architecture, prepare_batch=args.prepare_batch,
            DURA_training_reads=int(dura is not None), preparation_seconds=preparation_seconds,
            transfer_seconds=transfer_seconds, training_seconds=time.monotonic()-started,
            end_to_end_seconds=time.monotonic()-process_started, TX_labels_read=0, F_weights_read=0,
            F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True))
        return
    checkpoint = dict(schema=SCHEMA, commit=args.commit, model=model.state_dict(), width=32,
                      epochs=epochs, fit_ids=fit, counts=counts, losses=losses, architecture=args.architecture,
                      device=str(device), objective=args.objective, prepare_batch=args.prepare_batch,
                      DURA_training_reads=int(dura is not None),
                      TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                      target_reads=0, DURA_frozen=True)
    with (args.out/'final.pt').open('xb') as f:
        torch.save(checkpoint, f)
    if dura is None:
        dura = frozen_dura(args.root, str(device))
    _, groups = evaluate_source(model, dura, ids, rx, scene, source_received, source_clean,
                                str(device), args.prepare_batch)
    core.write(args.out/'source_summary.json', dict(status='SOURCE_TRAINED', commit=args.commit,
        fit_ids=len(fit), epochs=epochs, architecture=args.architecture,
        DURA_training_reads=int(args.architecture == 'orthogonal'),
        exposure=len(data[0]), counts=counts, losses=losses,
        device=str(device), objective=args.objective, prepare_batch=args.prepare_batch,
        preparation_seconds=preparation_seconds,
        transfer_seconds=transfer_seconds,
        source_groups=groups, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, DURA_frozen=True, source_IQ_metrics_not_identity_accuracy=True))
    print('ORTHOGONAL_SOURCE_TRAINED', args.commit, flush=True)


if __name__ == '__main__':
    main()
