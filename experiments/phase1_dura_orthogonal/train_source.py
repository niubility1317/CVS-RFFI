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
from experiments.phase1_dura_orthogonal.iq_geometry import iq_geometry_loss, GEOMETRY_WEIGHT, GEOMETRY_TEMPERATURE
from experiments.phase1_dura_orthogonal.joint_ssl_iq import SourceIQEncoder, cross_view_ssl_loss, frozen_feature_loss, SSL_FEATURE_WEIGHT, SSL_OFFDIAGONAL_WEIGHT, SSL_FEATURE_DIM
from experiments.phase1_dura_orthogonal.local_ssl_iq import PatchIQEncoder, patch_cross_view_loss, frozen_patch_feature_loss
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
from experiments.phase1_dura_orthogonal.direct_spectral_student import DirectSpectralStudent
from experiments.phase1_dura_orthogonal.dura_experts import DuraExperts, EXPERT_COUNT
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache, save_cache
from experiments.phase1_dura_orthogonal.posterior_teacher_contract import load_posterior_teacher, TEACHER_SOURCE, TEACHER_COMMIT

FIT_SIZE = 8192
WIDE_FIT_SIZE = 32768
TRAIN_EPOCHS = 5
DUAL_DOMAIN_EPOCHS = 20
SOURCE_EPOCHS = (21, 22, 23)
SCHEMA = 'phase1_dura_orthogonal_source_v1'
DIRECT_ARCHITECTURES = ('dual_domain', 'adaptive_inverse', 'timevarying_inverse', 'direct_spectral_student')
ACTION_OBJECTIVES = ('sim_action_guard', 'sim_action_fixed_prior')
SSL_TEACHER_SOURCE = '20261106-phase1-dura-joint-iq-ssl-source-s2026092701-r01'
SSL_TEACHER_COMMIT = '63ab9a08fe9f1fb076168467360e85cbc70ccafe'


def validate_ssl_teacher(summary, state, fit, *, smoke=False):
    expected = dict(commit=SSL_TEACHER_COMMIT, objective='ssl_joint_guard', architecture='dura_multiscale',
                    TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                    target_reads=0, DURA_frozen=True, source_ssl_seed=2026110615)
    if summary.get('status') != 'SOURCE_TRAINED' or any(
            obj.get(k) != v for obj in (summary, state) for k, v in expected.items()):
        raise ValueError('Fixed source-only SSL teacher provenance required')
    teacher_fit = np.asarray(state.get('fit_ids', []))
    if len(teacher_fit) != FIT_SIZE or not np.array_equal(teacher_fit[:len(fit)], fit) or (
            not smoke and len(fit) != FIT_SIZE):
        raise ValueError('SSL teacher must use exactly the matching source fit IDs')
    if not isinstance(state.get('source_ssl_aux_state'), dict):
        raise ValueError('Frozen source SSL auxiliary encoder state required')


def load_ssl_teacher(root, fit, device, *, smoke=False):
    folder = root/'runs'/SSL_TEACHER_SOURCE
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    validate_ssl_teacher(summary, state, fit, smoke=smoke)
    aux = SourceIQEncoder()
    aux.load_state_dict(state['source_ssl_aux_state'], strict=True)
    return aux.to(device).eval().requires_grad_(False)


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
    if objective not in ('iq', 'dura_guard', 'band_guard', 'nonlinear_guard', 'distill_iq', 'tail_excess_guard'):
        raise ValueError('A registered source-IQ objective is required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    error = (output-clean).square().mean((1, 2)) / energy
    change = (output-anchor).square().mean((1, 2)) / energy
    per = error + (.5 if objective == 'distill_iq' else .02)*change
    if objective in ('dura_guard', 'band_guard', 'nonlinear_guard', 'tail_excess_guard'):
        baseline_error = (anchor.detach()-clean).square().mean((1, 2)) / energy
        excess = (error-baseline_error).clamp_min(0)
        per = per + .002*excess/(baseline_error+1e-4)
    if objective == 'tail_excess_guard':
        # Emphasize the worst quarter of source-relative excess without TX labels.
        # The anchor is detached above; all coefficients are fixed before testing.
        positive_excess = torch.where(error > baseline_error, error-baseline_error, torch.zeros_like(error))
        tail_count = max(1, (len(positive_excess) + 3)//4)
        return per.mean() + .5*torch.topk(positive_excess, tail_count).values.mean()
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


def posterior_teacher_loss(output, teacher, clean):
    if output.shape != teacher.shape or clean.shape != output.shape or output.shape[1:] != (2, 256):
        raise ValueError('Matching source posterior teacher IQ required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    return ((output-teacher).square().mean((1, 2))/energy).mean()


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


def select_source_fit(rx, day, roles, eval_ids, fit_size):
    if rx.shape != roles.shape or day.shape != roles.shape:
        raise ValueError('Matching public source receiver and day IDs required')
    eligible = np.flatnonzero(~(np.isin(rx, (4, 6, 8)) & (day == 3)))
    eligible = np.setdiff1d(eligible, eval_ids, assume_unique=False)
    if fit_size not in (FIT_SIZE, WIDE_FIT_SIZE) or len(eligible) < fit_size or not np.isin(roles[eligible], ('L_s', 'U_s')).all():
        raise ValueError('Sufficient source training IQ is required')
    return np.sort(np.random.default_rng(2026110611).choice(eligible, size=fit_size, replace=False))


def load_source_training(root, eval_ids, fit_size=FIT_SIZE):
    complete = core.read(root/'runs'/core.F_RUN/'F'/core.F_ROW/'source/complete.json')
    contract = core.read(complete['config']['source_contract'])
    cache, clean, roles = core.data(root, contract)
    rx = np.load(cache/'train/rx.npy', allow_pickle=False)
    day = np.load(cache/'train/day.npy', allow_pickle=False)
    fit = select_source_fit(rx, day, roles, eval_ids, fit_size)
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
    parser.add_argument('--source-context-cache', type=Path)
    parser.add_argument('--save-source-context-cache', type=Path)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--objective', choices=('iq', 'dura_guard', 'band_guard', 'nonlinear_guard', 'crossview_guard', 'sim_action_guard', 'sim_action_fixed_prior', 'moment_guard', 'geometry_guard', 'ssl_joint_guard', 'ssl_fixed_guard', 'distill_iq', 'posterior_teacher_guard', 'ssl_crosshalf_guard', 'tail_excess_guard', 'wide_source_guard', 'wide_source_distill', 'wide_mixture_guard'), default='iq')
    parser.add_argument('--architecture', choices=('orthogonal', 'dual_domain', 'dura_internal', 'adaptive_inverse', 'timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate', 'dura_multiscale', 'direct_spectral_student', 'dura_experts'), default='orthogonal')
    args = parser.parse_args()
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    if args.source_context_cache and (args.smoke or (args.architecture in DIRECT_ARCHITECTURES and args.architecture != 'direct_spectral_student') or args.objective in ACTION_OBJECTIVES):
        raise ValueError('Source context cache applies only to full DURA-context training without action supervision')
    if args.save_source_context_cache and (args.smoke or args.source_context_cache or args.objective != 'wide_source_guard' or not args.deterministic):
        raise ValueError('Cache writing requires a fresh deterministic wide-source run')
    if args.architecture in DIRECT_ARCHITECTURES and args.architecture != 'direct_spectral_student' and args.objective != 'iq':
        raise ValueError('Direct dual-domain model uses received-IQ regularization only')
    if (args.architecture == 'direct_spectral_student') != (args.objective in ('distill_iq', 'wide_source_distill')):
        raise ValueError('Direct spectral student requires fixed source-IQ distillation')
    if args.architecture == 'dura_timevarying' and args.objective != 'dura_guard':
        raise ValueError('DURA time-varying correction requires DURA guard')
    if args.architecture == 'dura_multiscale' and args.objective not in ('dura_guard', 'ssl_joint_guard', 'ssl_fixed_guard', 'posterior_teacher_guard', 'ssl_crosshalf_guard', 'tail_excess_guard', 'wide_source_guard'):
        raise ValueError('Multiscale correction requires a registered DURA guard')
    if args.objective == 'wide_source_guard' and args.architecture != 'dura_multiscale':
        raise ValueError('Wide source guard requires multiscale DURA correction')
    if (args.architecture == 'dura_experts') != (args.objective == 'wide_mixture_guard'):
        raise ValueError('DURA experts require the registered wide mixture guard')
    if args.objective == 'tail_excess_guard' and args.architecture != 'dura_multiscale':
        raise ValueError('Tail excess guard requires multiscale DURA correction')
    if args.objective in ('ssl_joint_guard', 'ssl_fixed_guard', 'ssl_crosshalf_guard') and args.architecture != 'dura_multiscale':
        raise ValueError('Source IQ SSL is registered only for multiscale correction')
    if args.objective == 'posterior_teacher_guard' and args.architecture != 'dura_multiscale':
        raise ValueError('Source posterior teacher applies only to multiscale DURA correction')
    if args.architecture == 'dura_timevarying_gate' and args.objective not in ('dura_guard', 'moment_guard', 'geometry_guard'):
        raise ValueError('Gated correction requires a registered DURA guard')
    if args.objective in ('moment_guard', 'geometry_guard') and args.architecture != 'dura_timevarying_gate':
        raise ValueError('Source relation objectives apply only to gated DURA correction')
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
    fit_size = WIDE_FIT_SIZE if args.objective in ('wide_source_guard', 'wide_source_distill', 'wide_mixture_guard') else FIT_SIZE
    _, fresh, clean, fit = load_source_training(args.root, ids, fit_size=fit_size)
    if args.smoke:
        fit = fit[:args.smoke_fit]
    dura = frozen_dura(args.root, str(device)) if args.architecture not in DIRECT_ARCHITECTURES or args.architecture == 'direct_spectral_student' else None
    preparation_started = time.monotonic()
    if args.source_context_cache:
        data, counts = load_cache(args.source_context_cache, fit, args.prepare_batch, str(device),
                                  deterministic=args.deterministic)
    else:
        prepare_arch = 'orthogonal' if args.architecture == 'direct_spectral_student' else args.architecture
        data, counts = prepare(args.root, dura, clean, fresh, fit, str(device), args.prepare_batch,
                               prepare_arch, include_action=args.objective in ACTION_OBJECTIVES)
        if args.save_source_context_cache:
            save_cache(args.save_source_context_cache, data, counts, fit, args.prepare_batch,
                       str(device), deterministic=True)
    if args.objective == 'posterior_teacher_guard':
        data = (*data, load_posterior_teacher(args.root, fit, smoke=args.smoke))
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
    elif args.architecture == 'dura_multiscale':
        model = DuraMultiscaleSpectral(width=32, taps=9).to(device)
    elif args.architecture == 'dura_experts':
        model = DuraExperts(width=32, taps=9).to(device)
    elif args.architecture == 'direct_spectral_student':
        model = DirectSpectralStudent(width=32, taps=9).to(device)
    else:
        model = InternalDuraRecovery(frozen_dura(args.root, str(device))).to(device)
        configure_internal_trainability(model.dura, args.objective)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=1e-5 if args.architecture == 'dura_internal' else .0003,
        weight_decay=0. if args.architecture == 'dura_internal' else .001)
    aux = None
    aux_optimizer = None
    if args.objective in ('ssl_joint_guard', 'ssl_crosshalf_guard'):
        torch.manual_seed(2026110615)
        aux = (PatchIQEncoder() if args.objective == 'ssl_crosshalf_guard' else SourceIQEncoder()).to(device)
        aux_optimizer = torch.optim.AdamW(aux.parameters(), lr=.0003, weight_decay=.001)
    elif args.objective == 'ssl_fixed_guard':
        aux = load_ssl_teacher(args.root, fit, device, smoke=args.smoke)
    generator = torch.Generator().manual_seed(2026110613)
    if args.objective in ACTION_OBJECTIVES:
        probe_generator = torch.Generator().manual_seed(2026110614)
        probe_parts = torch.randn((1, 2, 256), generator=probe_generator).to(device)
        probe = torch.complex(probe_parts[:, 0], probe_parts[:, 1])
    losses = []
    action_losses = []
    moment_losses = []
    geometry_losses = []
    ssl_crossview_losses, ssl_feature_losses = [], []
    initial_batch = min(args.prepare_batch, len(fit)) if args.architecture == 'dura_internal' else 8
    with torch.no_grad():
        initial = model(*(value[:initial_batch] for value in data[:4]))
    if args.architecture == 'dura_internal':
        torch.testing.assert_close(initial, data[1][:initial_batch], rtol=1e-5, atol=1e-6)
    elif not torch.equal(initial, data[0 if args.architecture == 'direct_spectral_student' else 1][:initial_batch]):
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
        geometry_total = torch.zeros((), device=device)
        ssl_crossview_total = torch.zeros((), device=device)
        ssl_feature_total = torch.zeros((), device=device)
        for start in range(0, len(order), 10 if crossview else 32):
            base = order[start:start+(10 if crossview else 32)]
            indices = grouped_view_indices(base, len(fit)) if crossview else base
            fields = tuple(v[indices] for v in data)
            x, anchor, direction, variance, y = fields[:5]
            if aux_optimizer is not None:
                aux.train().requires_grad_(True)
                aux_optimizer.zero_grad(set_to_none=True)
                ssl_crossview = (patch_cross_view_loss(aux, x, y) if args.objective == 'ssl_crosshalf_guard'
                                 else cross_view_ssl_loss(aux(x), aux(y)))
                if not torch.isfinite(ssl_crossview):
                    raise FloatingPointError('Finite source SSL loss required')
                ssl_crossview.backward()
                torch.nn.utils.clip_grad_norm_(aux.parameters(), 1., error_if_nonfinite=True)
                aux_optimizer.step()
                aux_optimizer.zero_grad(set_to_none=True)
                aux.eval().requires_grad_(False)
                ssl_crossview_total += ssl_crossview.detach()*len(indices)
            if args.objective in ACTION_OBJECTIVES:
                from experiments.phase1_dura_orthogonal.oracle_action_source_probe import true_action
                context = model.dura.estimate_context(x)
                output = model.dura.stage_outputs(x, context)[0][-1]
                reference = true_action(type(context[1]), fields[5], fields[6])
                action_loss = action_response_loss(context[1], reference, probe.expand(len(x), -1))
            else:
                output = model(x, anchor, direction, variance, validate=False)
            base_objective = ('distill_iq' if args.objective == 'wide_source_distill' else
                              'dura_guard' if crossview or args.objective in ACTION_OBJECTIVES or args.objective in ('moment_guard', 'geometry_guard', 'ssl_joint_guard', 'ssl_fixed_guard', 'posterior_teacher_guard', 'ssl_crosshalf_guard', 'wide_source_guard', 'wide_mixture_guard') else args.objective)
            loss = source_loss(output, anchor, y, objective=base_objective)
            if args.objective == 'posterior_teacher_guard':
                loss = loss + .5*posterior_teacher_loss(output, fields[5], y)
            if aux is not None:
                ssl_feature = (frozen_patch_feature_loss(aux, output, y) if args.objective == 'ssl_crosshalf_guard'
                               else frozen_feature_loss(aux, output, y))
                loss = loss + SSL_FEATURE_WEIGHT*ssl_feature
                ssl_feature_total += ssl_feature.detach()*len(indices)
            if crossview:
                loss = loss + .1*crossview_dispersion(output, y)
            if args.objective == 'geometry_guard':
                geometry_loss = iq_geometry_loss(output, y)
                loss = loss + GEOMETRY_WEIGHT*geometry_loss
                geometry_total += geometry_loss.detach()*len(indices)
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
            elif args.architecture == 'dura_experts':
                if any(head.weight.grad is None or not torch.isfinite(head.weight.grad).all()
                       for expert in model.experts for head in (expert.wave, expert.filter_head)):
                    raise ValueError('All expert correction paths require finite gradients')
            else:
                second_head = (model.step if args.architecture == 'orthogonal' else
                               model.filter_head if args.architecture in ('adaptive_inverse', 'timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate', 'dura_multiscale', 'direct_spectral_student') else model.spectral_head)
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
        if args.objective == 'geometry_guard':
            geometry_losses.append(float(geometry_total.cpu())/len(order))
        if aux is not None:
            if aux_optimizer is not None:
                ssl_crossview_losses.append(float(ssl_crossview_total.cpu())/len(order))
            ssl_feature_losses.append(float(ssl_feature_total.cpu())/len(order))
        print('ORTHOGONAL_SOURCE_EPOCH', epoch, 'loss', losses[-1], 'elapsed', round(time.monotonic()-started, 1), flush=True)
    if dura is not None and any(p.requires_grad or p.grad is not None for p in dura.parameters()):
        raise ValueError('DURA must stay frozen')
    ssl_metadata = dict(source_ssl_feature_weight=SSL_FEATURE_WEIGHT,
        source_ssl_offdiagonal_weight=SSL_OFFDIAGONAL_WEIGHT, source_ssl_feature_dim=SSL_FEATURE_DIM,
        source_ssl_seed=2026110615, source_ssl_parameters=sum(p.numel() for p in aux.parameters()),
        source_ssl_crossview_losses=ssl_crossview_losses, source_ssl_feature_losses=ssl_feature_losses) if aux is not None else {}
    if args.objective == 'ssl_crosshalf_guard':
        ssl_metadata.update(source_ssl_patch_length=128, source_ssl_cross_half=True)
    if args.objective == 'ssl_fixed_guard':
        if any(p.requires_grad or p.grad is not None for p in aux.parameters()):
            raise ValueError('Source SSL teacher must remain frozen')
        ssl_metadata.update(source_ssl_teacher=SSL_TEACHER_SOURCE,
                            source_ssl_teacher_commit=SSL_TEACHER_COMMIT, source_ssl_aux_trainable=False)
    cache_metadata = dict(source_context_cache=str(args.source_context_cache) if args.source_context_cache else None,
                          deterministic_algorithms=args.deterministic)
    if args.objective in ('wide_source_guard', 'wide_source_distill', 'wide_mixture_guard'):
        cache_metadata.update(source_fit_size=WIDE_FIT_SIZE,
                              source_context_cache_written=str(args.save_source_context_cache) if args.save_source_context_cache else None)
    if args.architecture == 'dura_experts':
        cache_metadata.update(source_expert_count=EXPERT_COUNT)
    if args.objective == 'tail_excess_guard':
        cache_metadata.update(source_tail_fraction=.25, source_tail_weight=.5)
    if args.architecture == 'direct_spectral_student':
        cache_metadata.update(source_iq_teacher='dura_v1_R65_E20_4940', source_iq_teacher_weight=.5,
                              inference_received_iq_only=True)
    if args.objective == 'posterior_teacher_guard':
        cache_metadata.update(source_posterior_teacher=TEACHER_SOURCE,
                              source_posterior_teacher_commit=TEACHER_COMMIT,
                              source_posterior_teacher_weight=.5, posterior_teacher_inference_reads=0)
    if args.smoke:
        core.write(args.out/'smoke.json', dict(status='SOURCE_TRAINING_SMOKE_PASS', commit=args.commit,
            fit_ids=len(fit), exposure=len(data[0]), losses=losses, device=str(device),
            objective=args.objective, architecture=args.architecture, prepare_batch=args.prepare_batch,
            candidate_dura_trainable=args.architecture == 'dura_internal',
            spectral_freq_bins=17 if args.architecture in ('dura_multiscale', 'direct_spectral_student') else None,
            source_geometry_weight=GEOMETRY_WEIGHT if args.objective == 'geometry_guard' else None,
            source_geometry_temperature=GEOMETRY_TEMPERATURE if args.objective == 'geometry_guard' else None,
            source_geometry_losses=geometry_losses if args.objective == 'geometry_guard' else None,
            source_moment_weight=MOMENT_WEIGHT if args.objective == 'moment_guard' else None,
            source_moment_lags=list(LAGS) if args.objective == 'moment_guard' else None,
            source_moment_losses=moment_losses if args.objective == 'moment_guard' else None,
            noise_gate_low=GATE_LOG10_LOW if args.architecture in ('dura_timevarying_gate', 'dura_multiscale') else None,
            noise_gate_high=GATE_LOG10_HIGH if args.architecture in ('dura_timevarying_gate', 'dura_multiscale') else None,
        adaptive_filter_taps=(33 if args.architecture == 'adaptive_inverse' else
                                  9 if args.architecture in ('timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate', 'dura_multiscale', 'direct_spectral_student') else None),
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
            F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True, **ssl_metadata, **cache_metadata))
        return
    checkpoint = dict(schema=SCHEMA, commit=args.commit, model=model.state_dict(), width=32,
                      epochs=epochs, fit_ids=fit, counts=counts, losses=losses, architecture=args.architecture,
                      candidate_dura_trainable=args.architecture == 'dura_internal',
            spectral_freq_bins=17 if args.architecture in ('dura_multiscale', 'direct_spectral_student') else None,
            source_geometry_weight=GEOMETRY_WEIGHT if args.objective == 'geometry_guard' else None,
            source_geometry_temperature=GEOMETRY_TEMPERATURE if args.objective == 'geometry_guard' else None,
            source_geometry_losses=geometry_losses if args.objective == 'geometry_guard' else None,
            source_moment_weight=MOMENT_WEIGHT if args.objective == 'moment_guard' else None,
            source_moment_lags=list(LAGS) if args.objective == 'moment_guard' else None,
            source_moment_losses=moment_losses if args.objective == 'moment_guard' else None,
            noise_gate_low=GATE_LOG10_LOW if args.architecture in ('dura_timevarying_gate', 'dura_multiscale') else None,
            noise_gate_high=GATE_LOG10_HIGH if args.architecture in ('dura_timevarying_gate', 'dura_multiscale') else None,
                      adaptive_filter_taps=(33 if args.architecture == 'adaptive_inverse' else
                                  9 if args.architecture in ('timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate', 'dura_multiscale', 'direct_spectral_student') else None),
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
                      target_reads=0, DURA_frozen=True, **ssl_metadata, **cache_metadata)
    if aux is not None:
        checkpoint['source_ssl_aux_state'] = aux.state_dict()
    with (args.out/'final.pt').open('xb') as f:
        torch.save(checkpoint, f)
    if dura is None:
        dura = frozen_dura(args.root, str(device))
    _, groups = evaluate_source(model, dura, ids, rx, scene, source_received, source_clean,
                                str(device), args.prepare_batch)
    core.write(args.out/'source_summary.json', dict(status='SOURCE_TRAINED', commit=args.commit,
        fit_ids=len(fit), epochs=epochs, architecture=args.architecture,
        candidate_dura_trainable=args.architecture == 'dura_internal',
            spectral_freq_bins=17 if args.architecture in ('dura_multiscale', 'direct_spectral_student') else None,
            source_geometry_weight=GEOMETRY_WEIGHT if args.objective == 'geometry_guard' else None,
            source_geometry_temperature=GEOMETRY_TEMPERATURE if args.objective == 'geometry_guard' else None,
            source_geometry_losses=geometry_losses if args.objective == 'geometry_guard' else None,
            source_moment_weight=MOMENT_WEIGHT if args.objective == 'moment_guard' else None,
            source_moment_lags=list(LAGS) if args.objective == 'moment_guard' else None,
            source_moment_losses=moment_losses if args.objective == 'moment_guard' else None,
            noise_gate_low=GATE_LOG10_LOW if args.architecture in ('dura_timevarying_gate', 'dura_multiscale') else None,
            noise_gate_high=GATE_LOG10_HIGH if args.architecture in ('dura_timevarying_gate', 'dura_multiscale') else None,
            adaptive_filter_taps=(33 if args.architecture == 'adaptive_inverse' else
                                  9 if args.architecture in ('timevarying_inverse', 'dura_timevarying', 'dura_timevarying_gate', 'dura_multiscale', 'direct_spectral_student') else None),
            candidate_prior_frozen=args.objective == 'sim_action_fixed_prior',
        source_views_per_record=3 if args.objective == 'crossview_guard' else None,
        crossview_weight=.1 if args.objective == 'crossview_guard' else None,
        source_simulator_action_used=args.objective in ACTION_OBJECTIVES,
        source_simulator_action_weight=.01 if args.objective in ACTION_OBJECTIVES else None,
        source_probe_seed=2026110614 if args.objective in ACTION_OBJECTIVES else None,
        source_action_response_losses=action_losses if args.objective in ACTION_OBJECTIVES else None,
        DURA_training_reads=int(args.architecture not in DIRECT_ARCHITECTURES or args.architecture == 'direct_spectral_student'),
        exposure=len(data[0]), counts=counts, losses=losses,
        device=str(device), objective=args.objective, prepare_batch=args.prepare_batch,
        preparation_seconds=preparation_seconds,
        transfer_seconds=transfer_seconds,
        source_groups=groups, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, DURA_frozen=True, source_IQ_metrics_not_identity_accuracy=True, **ssl_metadata, **cache_metadata))
    print('ORTHOGONAL_SOURCE_TRAINED', args.commit, flush=True)


if __name__ == '__main__':
    main()
