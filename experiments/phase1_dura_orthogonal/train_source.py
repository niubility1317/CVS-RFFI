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
from experiments.phase1_dura_orthogonal.model import OrthogonalRecovery, DualDomainRecovery, InternalDuraRecovery
from experiments.phase1_dura_orthogonal.adaptive_inverse import AdaptiveInverseRecovery
from experiments.phase1_dura_orthogonal.timevarying_inverse import TimeVaryingInverseRecovery
from experiments.phase1_dura_orthogonal.dura_timevarying import DuraTimeVaryingRecovery
from experiments.phase1_dura_orthogonal.noise_gated_dura import NoiseGatedDuraTimeVarying, GATE_LOG10_LOW, GATE_LOG10_HIGH
from experiments.phase1_dura_orthogonal.moment_relational import moment_relational_loss, MOMENT_WEIGHT, LAGS

FIT_SIZE = 8192
TRAIN_EPOCHS = 5
DUAL_DOMAIN_EPOCHS = 20
SOURCE_EPOCHS = (21, 22, 23)
SCHEMA = 'phase1_dura_orthogonal_source_v1'
DIRECT_ARCHITECTURES = ('dual_domain', 'adaptive_inverse', 'timevarying_inverse')
ACTION_OBJECTIVES = ('sim_action_guard', 'sim_action_fixed_prior')


def configure_internal_trainability(dura, objective):
    dura.requires_grad_(True)
    if objective == 'sim_action_fixed_prior':
        dura.prior.requires_grad_(False)


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


def grouped_view_indices(base_indices, fit_count):
    """Index matching physical records in three view-major source arrays."""
    return torch.cat(tuple(base_indices + offset*fit_count for offset in range(3)))


def crossview_dispersion(output, clean):
    """Mean within-record three-view variance in units of clean IQ energy."""
    if (output.ndim != 3 or output.shape[1:] != (2, 256) or clean.shape != output.shape
            or len(output) == 0 or len(output) % 3):
        raise ValueError('Three matched view-major IQ batches required')
    count = len(output)//3
    views = output.reshape(3, count, 2, 256)
    center = views.mean(0, keepdim=True)
    energy = clean[:count].square().mean((1, 2)).clamp_min(1e-8)
    return ((views-center).square().mean((0, 2, 3))/energy).mean()


def action_response_loss(estimated, reference, probe):
    """Compare real-linear actions on two fixed unlabeled complex probes."""
    losses = []
    for value in (probe, 1j*probe):
        with torch.no_grad():
            target = reference(value)
        prediction = estimated(value)
        scale = target.abs().square().mean(-1).clamp_min(1e-6)
        relative = (prediction-target).abs().square().mean(-1)/scale
        losses.append(torch.log1p(relative))
    return torch.stack(losses, dim=1).mean()


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
def prepare(root, dura, clean, fresh, fit, device, batch_size, architecture='orthogonal', include_action=False):
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
        if include_action:
            kernel = np.load(folder/'kernel.npy', mmap_mode='r', allow_pickle=False)
            coefficients = np.load(folder/'coefficients.npy', mmap_mode='r', allow_pickle=False)
            if (kernel.shape != (core.COUNT, 2, 64) or coefficients.shape != (core.COUNT, 4, 256)
                    or kernel.dtype != np.float32 or coefficients.dtype != np.float32):
                raise ValueError('Complete source simulator action arrays required')
        with np.load(folder/'groups.npz', allow_pickle=False) as group:
            scene = group['scene'].copy()
        if received.shape != clean.shape or scene.shape != (core.COUNT,):
            raise ValueError('Complete source IQ cache required')
        for start in range(0, len(fit), batch_size):
            ids = fit[start:start+batch_size]
            x, y = core.tensor(received[ids], device), core.tensor(clean[ids], device)
            if architecture in DIRECT_ARCHITECTURES:
                anchor, direction, variance = x, torch.zeros_like(x), torch.ones(len(x), device=x.device)
            else:
                anchor, direction, variance = extract_context(dura, x)
            fields = (x, anchor, direction, variance, y)
            if include_action:
                fields += (core.tensor(kernel[ids], device), core.tensor(coefficients[ids], device))
            values = tuple(t.detach().cpu() for t in fields)
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
    parser.add_argument('--objective', choices=('iq', 'dura_guard', 'band_guard', 'nonlinear_guard', 'crossview_guard', 'sim_action_guard', 'sim_action_fixed_prior', 'moment_guard'), default='iq')
    parser.add_argument('--architecture', choices=('orthogonal', 'dual_domain', 'dura_internal', 'adaptive_inverse', 'timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate'), default='orthogonal')
    args = parser.parse_args()
    if args.architecture in DIRECT_ARCHITECTURES and args.objective != 'iq':
        raise ValueError('Direct dual-domain model uses received-IQ regularization only')
    if args.architecture == 'dura_timevarying' and args.objective != 'dura_guard':
        raise ValueError('DURA time-varying correction requires DURA guard')
    if args.architecture == 'dura_timevarying_gate' and args.objective not in ('dura_guard', 'moment_guard'):
        raise ValueError('Gated correction requires a registered DURA guard')
    if args.objective == 'moment_guard' and args.architecture != 'dura_timevarying_gate':
        raise ValueError('Source moment relation applies only to gated DURA correction')
    if args.architecture == 'dura_internal' and args.objective not in ('dura_guard', *ACTION_OBJECTIVES):
        raise ValueError('Internal DURA refinement requires fixed DURA guard')
    if args.objective == 'crossview_guard' and args.architecture != 'orthogonal':
        raise ValueError('Cross-view training is registered only for orthogonal correction')
    if args.objective in ACTION_OBJECTIVES and args.architecture != 'dura_internal':
        raise ValueError('Simulator action supervision requires internal DURA refinement')
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
    dura = frozen_dura(args.root, str(device)) if args.architecture not in DIRECT_ARCHITECTURES else None
    preparation_started = time.monotonic()
    data, counts = prepare(args.root, dura, clean, fresh, fit, str(device), args.prepare_batch,
                           args.architecture, include_action=args.objective in ACTION_OBJECTIVES)
    preparation_seconds = time.monotonic() - preparation_started
    transfer_started = time.monotonic()
    data = tuple(value.to(device) for value in data)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    transfer_seconds = time.monotonic() - transfer_started
    if args.objective == 'crossview_guard':
        count = len(fit)
        if (len(data[4]) != 3*count or not torch.equal(data[4][:count], data[4][count:2*count])
                or not torch.equal(data[4][:count], data[4][2*count:])):
            raise ValueError('Three source views must share physical clean IQ in matching order')
    if args.architecture == 'orthogonal':
        model = OrthogonalRecovery(width=32).to(device)
    elif args.architecture == 'dual_domain':
        model = DualDomainRecovery(width=32).to(device)
    elif args.architecture == 'adaptive_inverse':
        model = AdaptiveInverseRecovery(width=32, taps=33).to(device)
    elif args.architecture == 'timevarying_inverse':
        model = TimeVaryingInverseRecovery(width=32, taps=9).to(device)
    elif args.architecture == 'dura_timevarying':
        model = DuraTimeVaryingRecovery(width=32, taps=9).to(device)
    elif args.architecture == 'dura_timevarying_gate':
        model = NoiseGatedDuraTimeVarying(width=32, taps=9).to(device)
    else:
        model = InternalDuraRecovery(frozen_dura(args.root, str(device))).to(device)
        configure_internal_trainability(model.dura, args.objective)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=1e-5 if args.architecture == 'dura_internal' else .0003,
        weight_decay=0. if args.architecture == 'dura_internal' else .001)
    generator = torch.Generator().manual_seed(2026110613)
    if args.objective in ACTION_OBJECTIVES:
        probe_generator = torch.Generator().manual_seed(2026110614)
        probe_parts = torch.randn((1, 2, 256), generator=probe_generator).to(device)
        probe = torch.complex(probe_parts[:, 0], probe_parts[:, 1])
    losses = []
    action_losses = []
    moment_losses = []
    initial_batch = min(args.prepare_batch, len(fit)) if args.architecture == 'dura_internal' else 8
    with torch.no_grad():
        initial = model(*(value[:initial_batch] for value in data[:4]))
    if args.architecture == 'dura_internal':
        torch.testing.assert_close(initial, data[1][:initial_batch], rtol=1e-5, atol=1e-6)
    elif not torch.equal(initial, data[1][:initial_batch]):
        raise ValueError('New correction must initially reproduce frozen DURA')
    started = time.monotonic()
    epochs = DUAL_DOMAIN_EPOCHS if args.architecture in DIRECT_ARCHITECTURES else TRAIN_EPOCHS
    for epoch in range(1, (1 if args.smoke else epochs)+1):
        model.train()
        if args.architecture == 'dura_internal':
            model.dura.eval()
        crossview = args.objective == 'crossview_guard'
        order = torch.randperm(len(fit) if crossview else len(data[0]), generator=generator).to(device)
        total = torch.zeros((), device=device)
        action_total = torch.zeros((), device=device)
        moment_total = torch.zeros((), device=device)
        for start in range(0, len(order), 10 if crossview else 32):
            base = order[start:start+(10 if crossview else 32)]
            indices = grouped_view_indices(base, len(fit)) if crossview else base
            fields = tuple(v[indices] for v in data)
            x, anchor, direction, variance, y = fields[:5]
            if args.objective in ACTION_OBJECTIVES:
                from experiments.phase1_dura_orthogonal.oracle_action_source_probe import true_action
                context = model.dura.estimate_context(x)
                output = model.dura.stage_outputs(x, context)[0][-1]
                reference = true_action(type(context[1]), fields[5], fields[6])
                action_loss = action_response_loss(context[1], reference, probe.expand(len(x), -1))
            else:
                output = model(x, anchor, direction, variance, validate=False)
            base_objective = 'dura_guard' if crossview or args.objective in ACTION_OBJECTIVES or args.objective == 'moment_guard' else args.objective
            loss = source_loss(output, anchor, y, objective=base_objective)
            if crossview:
                loss = loss + .1*crossview_dispersion(output, y)
            if args.objective == 'moment_guard':
                moment_loss = moment_relational_loss(output, y)
                loss = loss + MOMENT_WEIGHT*moment_loss
                moment_total += moment_loss.detach()*len(indices)
            if args.objective in ACTION_OBJECTIVES:
                loss = loss + .01*action_loss
                action_total += action_loss.detach()*len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite IQ-only source loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.architecture == 'dura_internal':
                parts = (model.dura.frozen,) if args.objective == 'sim_action_fixed_prior' else (model.dura.frozen, model.dura.prior)
                for part in parts:
                    if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in part.parameters()):
                        raise ValueError('Trainable DURA parts require nonzero gradients')
                if args.objective == 'sim_action_fixed_prior' and any(
                        p.requires_grad or p.grad is not None for p in model.dura.prior.parameters()):
                    raise ValueError('Fixed original priors must remain frozen')
            else:
                second_head = (model.step if args.architecture == 'orthogonal' else
                               model.filter_head if args.architecture in ('adaptive_inverse', 'timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate') else model.spectral_head)
                if (model.wave.weight.grad is None or second_head.weight.grad is None or
                        not torch.isfinite(model.wave.weight.grad).all() or not torch.isfinite(second_head.weight.grad).all()):
                    raise ValueError('Both learned correction paths require finite gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach() * len(indices)
        losses.append(float(total.cpu())/(3*len(order) if crossview else len(order)))
        if args.objective in ACTION_OBJECTIVES:
            action_losses.append(float(action_total.cpu())/len(order))
        if args.objective == 'moment_guard':
            moment_losses.append(float(moment_total.cpu())/len(order))
        print('ORTHOGONAL_SOURCE_EPOCH', epoch, 'loss', losses[-1], 'elapsed', round(time.monotonic()-started, 1), flush=True)
    if dura is not None and any(p.requires_grad or p.grad is not None for p in dura.parameters()):
        raise ValueError('DURA must stay frozen')
    if args.smoke:
        core.write(args.out/'smoke.json', dict(status='SOURCE_TRAINING_SMOKE_PASS', commit=args.commit,
            fit_ids=len(fit), exposure=len(data[0]), losses=losses, device=str(device),
            objective=args.objective, architecture=args.architecture, prepare_batch=args.prepare_batch,
            candidate_dura_trainable=args.architecture == 'dura_internal',
            source_moment_weight=MOMENT_WEIGHT if args.objective == 'moment_guard' else None,
            source_moment_lags=list(LAGS) if args.objective == 'moment_guard' else None,
            source_moment_losses=moment_losses if args.objective == 'moment_guard' else None,
            noise_gate_low=GATE_LOG10_LOW if args.architecture == 'dura_timevarying_gate' else None,
            noise_gate_high=GATE_LOG10_HIGH if args.architecture == 'dura_timevarying_gate' else None,
        adaptive_filter_taps=(33 if args.architecture == 'adaptive_inverse' else
                                  9 if args.architecture in ('timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate') else None),
        candidate_prior_frozen=args.objective == 'sim_action_fixed_prior',
            source_views_per_record=3 if args.objective == 'crossview_guard' else None,
            crossview_weight=.1 if args.objective == 'crossview_guard' else None,
            source_simulator_action_used=args.objective in ACTION_OBJECTIVES,
            source_simulator_action_weight=.01 if args.objective in ACTION_OBJECTIVES else None,
            source_probe_seed=2026110614 if args.objective in ACTION_OBJECTIVES else None,
            source_action_response_losses=action_losses if args.objective in ACTION_OBJECTIVES else None,
            DURA_training_reads=int(dura is not None), preparation_seconds=preparation_seconds,
            transfer_seconds=transfer_seconds, training_seconds=time.monotonic()-started,
            end_to_end_seconds=time.monotonic()-process_started, TX_labels_read=0, F_weights_read=0,
            F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True))
        return
    checkpoint = dict(schema=SCHEMA, commit=args.commit, model=model.state_dict(), width=32,
                      epochs=epochs, fit_ids=fit, counts=counts, losses=losses, architecture=args.architecture,
                      candidate_dura_trainable=args.architecture == 'dura_internal',
            source_moment_weight=MOMENT_WEIGHT if args.objective == 'moment_guard' else None,
            source_moment_lags=list(LAGS) if args.objective == 'moment_guard' else None,
            source_moment_losses=moment_losses if args.objective == 'moment_guard' else None,
            noise_gate_low=GATE_LOG10_LOW if args.architecture == 'dura_timevarying_gate' else None,
            noise_gate_high=GATE_LOG10_HIGH if args.architecture == 'dura_timevarying_gate' else None,
                      adaptive_filter_taps=(33 if args.architecture == 'adaptive_inverse' else
                                  9 if args.architecture in ('timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate') else None),
                      candidate_prior_frozen=args.objective == 'sim_action_fixed_prior',
                      source_views_per_record=3 if args.objective == 'crossview_guard' else None,
                      crossview_weight=.1 if args.objective == 'crossview_guard' else None,
                      source_simulator_action_used=args.objective in ACTION_OBJECTIVES,
                      source_simulator_action_weight=.01 if args.objective in ACTION_OBJECTIVES else None,
                      source_probe_seed=2026110614 if args.objective in ACTION_OBJECTIVES else None,
                      source_action_response_losses=action_losses if args.objective in ACTION_OBJECTIVES else None,
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
        candidate_dura_trainable=args.architecture == 'dura_internal',
            source_moment_weight=MOMENT_WEIGHT if args.objective == 'moment_guard' else None,
            source_moment_lags=list(LAGS) if args.objective == 'moment_guard' else None,
            source_moment_losses=moment_losses if args.objective == 'moment_guard' else None,
            noise_gate_low=GATE_LOG10_LOW if args.architecture == 'dura_timevarying_gate' else None,
            noise_gate_high=GATE_LOG10_HIGH if args.architecture == 'dura_timevarying_gate' else None,
            adaptive_filter_taps=(33 if args.architecture == 'adaptive_inverse' else
                                  9 if args.architecture in ('timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate') else None),
            candidate_prior_frozen=args.objective == 'sim_action_fixed_prior',
        source_views_per_record=3 if args.objective == 'crossview_guard' else None,
        crossview_weight=.1 if args.objective == 'crossview_guard' else None,
        source_simulator_action_used=args.objective in ACTION_OBJECTIVES,
        source_simulator_action_weight=.01 if args.objective in ACTION_OBJECTIVES else None,
        source_probe_seed=2026110614 if args.objective in ACTION_OBJECTIVES else None,
        source_action_response_losses=action_losses if args.objective in ACTION_OBJECTIVES else None,
        DURA_training_reads=int(args.architecture not in DIRECT_ARCHITECTURES),
        exposure=len(data[0]), counts=counts, losses=losses,
        device=str(device), objective=args.objective, prepare_batch=args.prepare_batch,
        preparation_seconds=preparation_seconds,
        transfer_seconds=transfer_seconds,
        source_groups=groups, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, DURA_frozen=True, source_IQ_metrics_not_identity_accuracy=True))
    print('ORTHOGONAL_SOURCE_TRAINED', args.commit, flush=True)


if __name__ == '__main__':
    main()
