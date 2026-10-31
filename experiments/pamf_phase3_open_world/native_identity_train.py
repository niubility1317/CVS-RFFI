"""Source-first CE/KD and fixed masked-PL training of the complete native F0.

Only its original 232993-parameter reader is deployed.  No r03 body or new
identity head is inserted.  The original F0, r03, final PCD and DURA remain
independent frozen references; target pseudo-labels are fixed before updates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch.nn import functional as F

from . import source_open_embedding_train as native
from . import star_signal_open_train as signal_base
from . import star_representation_cache as cache
from . import receiver_input_train as inputs
from . import pcd_cluster_registry_run as reg
from . import fresh_pcd_train as fresh

RUN_ID = '20261031-phase3-native-identity-s2026092701-r01'
SCHEMA = 'pamf_phase3_native_identity_train_v1'
STATUS = 'NATIVE_IDENTITY_TRAIN_COMPLETE'
ROW_STATUS = 'NATIVE_IDENTITY_STUDENT_FROZEN'
ARMS = ('SOURCE_ID', 'TARGET_ID')
STEPS = 444
L_COUNT, L_BATCH, U_COUNT, U_BATCH = 6300, 128, 576, 64
L_SEED, U_SEED = 2026103160, 2026103112
REFERENCE_CODE = reg.REFERENCE_CODE
FIELDS = TRAIN_FIELDS = {'run_id', 'code_commit', 'reference_config', 'output_root', 'steps', 'lr'}
READER_SINC = 'core.id_backbone.sinc.'
READER_HEAD = 'core.id_backbone.cls_head.weight'
READER_PARAMETERS, TRAINABLE_PARAMETERS, OUTSIDE_PARAMETERS = 232993, 232945, 667729
LOGIT_RTOL, LOGIT_ATOL = 1e-5, 2e-5
ZEROS = ('target_Q_reads', 'validation_reads', 'novel_validation_reads', 'private_truth_reads',
         'U_truth_reads', 'original_U_s_reads', 'source_Us_reads', 'OE_reads',
         'Sinc_optimizer_steps', 'physical_clip_calls', 'source_density_fit_updates', 'threshold_fit_updates')
_read, _write = reg._read, reg._write
fixed_mask = inputs.fixed_mask


def _source_config(c):
    return c['reference_config']['source_control_config']['reference_config']['train_config']


def _native_config(c):
    # Existing strict constructor loads original E50; the 2220 mapping only
    # satisfies its historical construction schema, not this run's 444 budget.
    return signal_base._native_config(_source_config(c)['native_config'])


def validate_config(c):
    if (not isinstance(c, dict) or set(c) != FIELDS or c.get('run_id') != RUN_ID
            or re.fullmatch('[0-9a-f]{40}', str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps'] != STEPS
            or type(c.get('lr')) is not float or c['lr'] != .0001
            or not isinstance(c.get('output_root'), str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field native identity config required; no truth/Q/K inputs')
    r = reg.fresh.validate_config(c['reference_config'])
    if r['code_commit'] != REFERENCE_CODE:
        raise ValueError('Original completed fresh-world f4b5 configuration required')
    source = _source_config(c)
    roots = [r['output_root'], r['train_config']['output_root'], r['train_config']['U_public_root'],
             source['IQ_root'], source['output_root'], source['native_config']['source_output']]
    roots.extend(v['public_root'] for v in r['Q_conditions'].values())
    out = Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):
        raise ValueError('Immutable identity output overlaps reference/model/public data')
    return c


def _reference(c):
    return reg._reference(dict(reference_config=c['reference_config']))


def _backend(c, device):
    return reg._backend(dict(reference_config=c['reference_config']), device)


def _load_PCD(c, device):
    return reg._load_model(dict(reference_config=c['reference_config']), device)


def _snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _unchanged(model, before, label):
    now = model.state_dict()
    if set(now) != set(before) or any(not torch.equal(v, now[k].detach().cpu()) for k, v in before.items()):
        raise ValueError(label + ' parameters or persistent buffers changed')


def _initial_model(c, pm, device):
    model = native.make_trainable_model(_native_config(c), device).eval().requires_grad_(False)
    reader = model.id_backbone.encoder
    reader.requires_grad_(True)
    for name, p in reader.named_parameters():
        if name.startswith(READER_SINC):
            p.requires_grad_(False)
    parameters = dict(reader.named_parameters())
    sinc = {READER_SINC + leaf for leaf in ('low_coordinate', 'band_coordinate')}
    if (sum(p.numel() for p in reader.parameters()) != READER_PARAMETERS
            or sum(p.numel() for p in model.parameters()) - READER_PARAMETERS != OUTSIDE_PARAMETERS
            or sum(p.numel() for p in model.parameters() if p.requires_grad) != TRAINABLE_PARAMETERS
            or set(n for n, p in parameters.items() if not p.requires_grad) != sinc
            or any(parameters[n].shape != (24, 1) for n in sinc)
            or parameters[READER_HEAD].shape != (6, 160)
            or any(p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError('Complete original native reader/default head with only Sinc48 frozen required')
    n = _source_config(c)['native_config']
    for key in ('source_output', 'source_contract', 'base_release', 'model_seed'):
        if model._phase3_parent_identity.get(key) != n[key] or pm['parent_F0_identity'].get(key) != n[key]:
            raise ValueError('Native constructor and frozen original F0 ancestry differ: ' + key)
    expected_handle = dict(classes=native.CLASSES, source_rxs=native.RX, source_days=[1, 2, 3],
                           split_seed=392005, counts=dict(L_s=6300, U_s=56700, V=27000), source_contract=n['source_contract'])
    if model._phase3_source_handle != expected_handle:
        raise ValueError('Original native source data roles differ')
    return model


def native_logits(model, x):
    """Unmodified native features -> original jointly trained default head."""
    reader = model.id_backbone.encoder
    if model.training or any(v.training for v in reader.modules()):
        raise ValueError('Native student must stay eval; autograd remains enabled by its caller')
    with torch.autocast(device_type=x.device.type, enabled=False):
        logits = reader.classify_features(reader.features(x))
    if logits.shape != (len(x), 6) or logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite full-six-class native FP32 default logits required')
    return logits


def _tensor(a, device, dtype=torch.float32):
    return torch.tensor(np.asarray(a).tolist(), device=device, dtype=dtype)


def source_loss(logits, y, teacher_logits):
    ce = F.cross_entropy(logits, y, label_smoothing=0.)
    kd = 4 * F.kl_div(F.log_softmax(logits / 2, dim=1), F.softmax(teacher_logits.detach() / 2, dim=1), reduction='batchmean')
    return dict(loss=ce + kd, CE=ce, KD=kd)


def pseudo_loss(first, second, pseudo, mask):
    # Divide by the full sampled batch, including rejected rows.
    rows = .5 * (F.cross_entropy(first, pseudo.detach(), reduction='none') +
                 F.cross_entropy(second, pseudo.detach(), reduction='none'))
    return (rows * mask.detach().to(rows.dtype)).sum() / len(rows)


def _gradients(model):
    reader = model.id_backbone.encoder; b = reader.core.id_backbone
    norm = lambda params: sum(float(p.grad.detach().double().square().sum().cpu()) for p in params if p.grad is not None) ** .5
    result = dict(head=norm([b.cls_head.weight]), shallow=norm((b.t1 if hasattr(b, 't1') else b.t2).parameters()),
                  middle=norm(b.t2.parameters()), Sinc=norm(b.sinc.parameters()),
                  body=norm(p for n, p in reader.named_parameters() if n != READER_HEAD and not n.startswith(READER_SINC)))
    reader_ids = {id(p) for p in reader.parameters()}
    if (any(p.grad is not None for p in b.sinc.parameters())
            or any(p.grad is not None for p in model.parameters() if id(p) not in reader_ids)
            or any(not np.isfinite(v) for v in result.values())
            or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in reader.parameters())):
        raise FloatingPointError('Nonfinite native gradient or a frozen parameter received a gradient')
    return result


def _student_smoke(model, b, device):
    from .star_open_feature_export import _synthetic
    before = _snapshot(model)
    try:
        with torch.random.fork_rng(devices=native._cuda_devices(model)):
            raw = _synthetic(device)
            with torch.no_grad():
                x = b['dura'][1](raw)[0].detach().clone()
                original = b['reader'].classify_features(b['reader'].features(x))
            logits = native_logits(model, x)
            if not torch.allclose(logits.detach(), original, rtol=LOGIT_RTOL, atol=LOGIT_ATOL):
                raise ValueError('Initial complete native student logits differ from actual original F0')
            F.cross_entropy(logits, torch.tensor([0, 1], device=device)).backward()
            gradients = _gradients(model)
            if any(gradients[k] <= 0 for k in ('head', 'body', 'shallow', 'middle')):
                raise ValueError('Native default head and shallow/middle body gradients must be live')
            _unchanged(model, before, 'No-step native smoke')
            return dict(status='PASS', actual_original_F0_equal=True,
                initial_logit_max_abs_difference=float((logits.detach() - original).abs().max().cpu()),
                logit_rtol=LOGIT_RTOL, logit_atol=LOGIT_ATOL, gradient_norms=gradients,
                Sinc_gradients_none=True, outside_reader_gradients_none=True, parameters_buffers_unchanged=True,
                optimizer_updates=0, real_data_reads=0, synthetic_student_forward_rows=2,
                synthetic_F_forward_rows=2, G_synthetic_forward_rows=2)
    finally:
        model.zero_grad(set_to_none=True)


def _L(c, pm):
    role = cache.load_role(Path(_source_config(c)['IQ_root']) / 'L', 'L')
    m = role['manifest']
    if (m['count'] != L_COUNT or m['parent_F0_identity'] != pm['parent_F0_identity']
            or m['dura_identity'] != pm['dura_identity'] or role['teacher_logits'].shape != (L_COUNT, 6)):
        cache.close_role(role)
        raise ValueError('Original source L P1/labels/cached original-F0 teacher required')
    return role


def _public_U(c):
    return fresh._public(c['reference_config']['train_config'])


def _U(c, Lids):
    t = c['reference_config']['train_config']; root = Path(t['output_root'])
    done = _read(root / 'complete.json'); manifest = done.get('U_manifest', {})
    public = None
    try:
        declared, ids, public = _public_U(c)
        if (done.get('status') != fresh.STATUS or done.get('config') != t or done.get('code_commit') != REFERENCE_CODE
                or done.get('source_only') is not False or done.get('target_U_reads') != U_COUNT
                or done.get('original_models_unchanged') is not True
                or any(manifest.get(k) != v or declared.get(k) != v for k, v in
                       dict(count=U_COUNT, capsule_id=t['U_capsule_id'], split_id=t['U_split_id']).items())):
            raise ValueError('Original completed fixed U capsule/split/count ancestry differs')
        with np.load(root / 'U_features.npz', allow_pickle=False) as z:
            if set(z.files) != {'ids', 'views', 'parent_z', 'neighbors'} or not np.array_equal(z['ids'], ids):
                raise ValueError('Original public/cache opaque U IDs/order must match')
            views = z['views'].copy()  # Parent z and neighbor values are not read.
        if (ids.shape != (U_COUNT,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != U_COUNT
                or np.any(np.char.strip(ids) == '') or set(ids.tolist()) & set(Lids.tolist())
                or views.shape != (3, U_COUNT, 2, 256) or views.dtype != np.float32 or not np.isfinite(views).all()):
            raise ValueError('Finite disjoint original U P1 weak and two phase/gain views required')
        return dict(ids=ids.copy(), views=views, manifest=manifest)
    finally:
        if hasattr(public, '_mmap'):
            public._mmap.close()


def _teacher(b, pcd, U, state, tau, device):
    # Existing frozen original F0 + final PCD predicted-ratio operating point.
    return inputs._teacher(b, pcd, dict(ids=U['ids'], restored=U['views'][0]), state, tau, device)


def _identity(c, arm, pm, source_manifest, U_manifest):
    target = arm == 'TARGET_ID'
    return dict(run_id=RUN_ID, code_commit=c['code_commit'], arm=arm,
        checkpoint=str(Path(c['output_root']) / arm / 'final.pt'), updates=STEPS,
        reader_parameters=READER_PARAMETERS, identity_parameters=READER_PARAMETERS, signal_parameters=READER_PARAMETERS,
        trainable_parameters=TRAINABLE_PARAMETERS, Sinc_frozen=True, frozen_Sinc_parameters=48,
        unused_wrapper_parameters=OUTSIDE_PARAMETERS, identity_head_parameters=960,
        identity_head_updates=STEPS, identity_head_jointly_trained=True, identity_head_replaced=False,
        original_F0_frozen=True, initialization='independent_complete_original_F0_E50',
        identity_weight_source_only=not target, source_only=not target, target_U_used=target,
        target_U_reads=U_COUNT if target else 0, U_manifest=U_manifest if target else None,
        parent_F0_identity=pm['parent_F0_identity'], parent_PCD_identity=pm['model_identities']['TARGET_PCD'],
        dura_identity=pm['dura_identity'], source_L_manifest=source_manifest,
        system_PCD_has_target_U_ancestor=True, inherited_PCD_target_U_count=U_COUNT,
        system_source_only=False, identity_input_path='cached_DURA_P1',
        deployment_state='reader_only', target_REG=False)


def _fixed_student(model, initial):
    current = model.state_dict()
    fixed = (set(dict(model.named_buffers())) & set(current)) | {
        n for n, p in model.named_parameters() if not p.requires_grad}
    if any(not torch.equal(initial[k], current[k].detach().cpu()) for k in fixed):
        raise ValueError('Native student Sinc/outside parameters/persistent buffers changed')


def _reader_changed(reader, initial):
    parameters = dict(reader.named_parameters())
    head = not torch.equal(initial[READER_HEAD], parameters[READER_HEAD].detach().cpu())
    body = any(not torch.equal(initial[n], p.detach().cpu()) for n, p in parameters.items()
               if n != READER_HEAD and not n.startswith(READER_SINC))
    if not head or not body:
        raise ValueError('Original default head and native body must actually update together')


def _loop(c, arm, model, L, U, teacher, device, folder, check):
    target = arm == 'TARGET_ID'; rng = np.random.default_rng(L_SEED)
    ur = np.random.default_rng(U_SEED) if target else None
    source = (Path(c['output_root']) / 'SOURCE_ID/steps.jsonl').open(encoding='utf-8') if target else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=c['lr'], weight_decay=.0001, eps=1e-8)
    trace = []; usage = {k: set() for k in ('L', 'U')}
    try:
        with (folder / 'steps.jsonl').open('x', encoding='utf-8') as log:
            for step in range(1, STEPS + 1):
                ix = rng.choice(L_COUNT, L_BATCH, replace=False)
                if source is not None:
                    original = json.loads(source.readline())
                    if original['step'] != step or original['L_indices'] != ix.tolist():
                        raise ValueError('Matched SOURCE_ID source L trajectory differs')
                opt.zero_grad(set_to_none=True)
                logits = native_logits(model, _tensor(L['restored'][ix], device))
                terms = source_loss(logits, _tensor(L['metadata']['y'][ix], device, torch.long),
                                    _tensor(L['teacher_logits'][ix], device))
                pl = logits.sum() * 0.; ui = np.empty(0, np.int64); accepted = 0; u_rows = 0; hist = [0] * 6
                if target:
                    ui = ur.choice(U_COUNT, U_BATCH, replace=False); accepted = int(teacher['mask'][ui].sum())
                    hist = np.bincount(teacher['pseudo'][ui][teacher['mask'][ui]], minlength=6).tolist()
                    if accepted:
                        first, second = (native_logits(model, _tensor(U['views'][v, ui], device)) for v in (1, 2))
                        pl = pseudo_loss(first, second, _tensor(teacher['pseudo'][ui], device, torch.long),
                                         _tensor(teacher['mask'][ui], device, torch.bool))
                        u_rows = 2 * U_BATCH
                total = terms['loss'] + pl
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError('Nonfinite native CE/KD/fixed-PL objective')
                total.backward(); gradients = _gradients(model)
                norm = float(torch.nn.utils.clip_grad_norm_(params, 5.))
                if not np.isfinite(norm):
                    raise FloatingPointError('Nonfinite native ordinary gradient norm')
                opt.step(); check()
                usage['L'].update(map(int, ix)); usage['U'].update(map(int, ui))
                row = dict(step=step, arm=arm, L_indices=ix.tolist(), U_indices=ui.tolist(), source_indices_equal=True,
                    loss_total=float(total.detach().cpu()), CE=float(terms['CE'].detach().cpu()), KD=float(terms['KD'].detach().cpu()),
                    PL=float(pl.detach().cpu()), CE_weight=1., KD_weight=1., KD_temperature=2., PL_weight=1.,
                    U_accepted=accepted, U_count=len(ui), U_accepted_prediction_histogram=hist,
                    fixed_teacher=True, gradient_norms=gradients, gradient_norm_before_clip=norm,
                    ordinary_clip_calls=step, student_optimizer_updates=step, identity_head_updates=step,
                    Sinc_optimizer_steps=0, physical_clip_calls=0, source_L_exposures=L_BATCH * step,
                    target_U_sampled_exposures=U_BATCH * step if target else 0,
                    student_L_forward_rows=L_BATCH, student_U_forward_rows=u_rows,
                    student_forward_rows=L_BATCH + u_rows, strong_forward_skipped=bool(target and not accepted))
                trace.append(row); log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                if step == 1 or step % 100 == 0 or step == STEPS:
                    print('NATIVE_IDENTITY_PROGRESS', arm, step, STEPS, row['loss_total'], accepted, flush=True)
        if source is not None and source.readline():
            raise ValueError('SOURCE_ID trajectory contains extra steps')
        return trace, {k: len(v) for k, v in usage.items()}
    finally:
        if source is not None:
            source.close()


def train(c, device='cuda:0'):
    validate_config(c); out = Path(c['output_root'])
    if out.exists():
        raise FileExistsError('New immutable native identity training root required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    pm, state, tau, _ = _reference(c); b = _backend(c, device); pcd, pcd_identity = _load_PCD(c, device)
    if (pcd_identity != pm['model_identities']['TARGET_PCD'] or b['reader_identity'] != pm['parent_F0_identity']
            or b['dura_identity'] != pm['loaded_dura_identity']):
        raise ValueError('Actual original F0/r03/G/final-PCD identities differ')
    originals = (b['reader'], b['model'], b['dura'][0], pcd)
    if any(p.requires_grad or p.grad is not None or p.dtype != torch.float32 for m in originals for p in m.parameters()) or any(v.training for m in originals for v in m.modules()):
        raise ValueError('Independent original references must be frozen FP32 eval without gradients')
    snapshots = [(m, _snapshot(m)) for m in originals]; fixed_state = deepcopy(state); fixed_tau = tau.copy()
    def check():
        for model, before in snapshots:
            _unchanged(model, before, 'Original F0/r03/G/PCD')
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError('Frozen original model received a gradient')
        if not reg._same(state, fixed_state) or not np.array_equal(tau, fixed_tau):
            raise ValueError('Original PCD source density/thresholds changed')
    checkpoint = reg._smoke(b, pcd, device); check()
    if checkpoint.get('status') != 'PASS' or checkpoint.get('actual_F0_r03_DURA_PCD_checkpoints') is not True:
        raise ValueError('Actual frozen checkpoints synthetic no-query smoke must pass before L/U')
    student = _initial_model(c, pm, device)
    initial = _snapshot(student); initial_reader = _snapshot(student.id_backbone.encoder)
    original_reader = _snapshot(b['reader'])
    if set(initial_reader) != set(original_reader) or any(not torch.equal(v, original_reader[k]) for k, v in initial_reader.items()):
        raise ValueError('Complete student reader must exactly copy original F0, including default head and buffers')
    gradient = _student_smoke(student, b, device); check()
    out.mkdir(parents=True); _write(out / 'resolved_config.json', c)
    _write(out / 'smoke.json', dict(checkpoint=checkpoint, native=gradient))
    L = None; rows = {}; U = teacher = None
    try:
        L = _L(c, pm)
        for arm in ARMS:
            target = arm == 'TARGET_ID'
            if target:
                if not (out / 'SOURCE_ID/complete.json').is_file():
                    raise ValueError('SOURCE_ID must finish and freeze before any U values open')
                U = _U(c, L['ids']); teacher = _teacher(b, pcd, U, state, tau, device); check()
                fresh.body._npz(out / 'U_teacher.npz', {k: v for k, v in teacher.items() if k != 'features'})
                _write(out / 'U_manifest.json', U['manifest'])
                student = _initial_model(c, pm, device)
            _unchanged(student, initial, 'Independent original-F0 student initialization')
            folder = out / arm; folder.mkdir()
            trace, unique = _loop(c, arm, student, L, U, teacher, device, folder, check)
            _fixed_student(student, initial); _reader_changed(student.id_backbone.encoder, initial_reader)
            network = _snapshot(student.id_backbone.encoder)
            if any(not bool(torch.isfinite(v).all()) for v in network.values()):
                raise ValueError('Finite full native reader state required')
            check(); identity = _identity(c, arm, pm, L['manifest'], U['manifest'] if target else None)
            student.eval().requires_grad_(False)
            torch.save(dict(schema=SCHEMA, config=c, identity=identity, reader_state=network,
                            parent_F0_identity=pm['parent_F0_identity'], source_L_manifest=L['manifest']), folder / 'final.pt')
            row = dict(schema=SCHEMA, status=ROW_STATUS, run_id=RUN_ID, code_commit=c['code_commit'], config=c,
                arm=arm, identity=identity, updates=STEPS, identity_head_updates=STEPS,
                identity_weight_source_only=not target, source_only=not target, target_U_used=target,
                target_U_reads=U_COUNT if target else 0, U_manifest=U['manifest'] if target else None,
                source_L_manifest=L['manifest'], initial_state_equal_to_original_F0=True,
                student_Sinc_unchanged=True, student_body_changed=True, student_head_changed=True,
                model_buffers_unchanged=True, outside_reader_unchanged=True, original_models_unchanged=True,
                source_L_exposures=L_BATCH * STEPS, target_U_sampled_exposures=U_BATCH * STEPS if target else 0,
                U_accepted_exposures=sum(r['U_accepted'] for r in trace),
                student_forward_rows=sum(r['student_forward_rows'] for r in trace),
                student_L_forward_rows=L_BATCH * STEPS, student_U_forward_rows=sum(r['student_U_forward_rows'] for r in trace),
                strong_forward_skipped_steps=sum(r['strong_forward_skipped'] for r in trace),
                unique_training_rows=unique, first_step=trace[0], last_step=trace[-1],
                optimizer=dict(name='AdamW', lr=.0001, weight_decay=.0001, eps=1e-8, ordinary_clip=5., ordinary_clip_calls=STEPS),
                identity_head_replaced=False, final_phase3_complete=False, **{k: 0 for k in ZEROS})
            _write(folder / 'complete.json', row); rows[arm] = row
            print('NATIVE_IDENTITY_ARM_COMPLETE', arm, flush=True)
            del student
        done = dict(schema=SCHEMA, status=STATUS, run_id=RUN_ID, code_commit=c['code_commit'], config=c,
            arms=list(ARMS), rows=rows, identities={a: rows[a]['identity'] for a in ARMS}, updates_per_arm=STEPS,
            total_optimizer_updates=2 * STEPS, new_target_UDA_updates=STEPS,
            source_sampling_matched=True, SOURCE_completed_before_U_open=True, source_completed_before_U_open=True,
            source_L_reads=L_COUNT, cached_L_teacher_logits_reads=L_COUNT, target_U_reads=U_COUNT,
            source_L_manifest=L['manifest'], U_manifest=U['manifest'], parent_F0_identity=pm['parent_F0_identity'],
            parent_PCD_identity=pcd_identity, dura_identity=pm['dura_identity'], reference_prediction_identity=pm,
            source_only=False, source_identity_weights_source_only=True, system_PCD_has_target_U_ancestor=True,
            inherited_PCD_target_U_count=U_COUNT, all_students_initialized_from_original_F0=True,
            all_student_Sinc_unchanged=True, all_student_buffers_unchanged=True, all_student_outside_reader_unchanged=True,
            all_student_bodies_changed=True, all_student_heads_changed=True, original_models_unchanged=True,
            fixed_teacher_mask=True, U_teacher_accepted=int(teacher['mask'].sum()),
            U_teacher_acceptance=float(teacher['mask'].mean()),
            U_teacher_prediction_histogram=np.bincount(teacher['pseudo'], minlength=6).tolist(),
            U_teacher_accepted_prediction_histogram=np.bincount(teacher['pseudo'][teacher['mask']], minlength=6).tolist(),
            pseudo_label_histograms_are_accuracy=False, target_P1_cache_reused=True, new_physical_observations=0,
            real_F_forward_rows=U_COUNT, real_PCD_forward_rows=U_COUNT, real_r03_forward_rows=0, G_forward_rows=0,
            real_student_forward_rows=sum(row['student_forward_rows'] for row in rows.values()),
            synthetic_F_forward_rows=checkpoint['synthetic_F_forward_rows'] + gradient['synthetic_F_forward_rows'],
            synthetic_r03_forward_rows=2, synthetic_PCD_forward_rows=2,
            G_synthetic_forward_rows=checkpoint['G_synthetic_forward_rows'] + gradient['G_synthetic_forward_rows'],
            synthetic_student_forward_rows=gradient['synthetic_student_forward_rows'],
            reader_parameters=READER_PARAMETERS, trainable_parameters_per_student=TRAINABLE_PARAMETERS,
            unused_wrapper_parameters=OUTSIDE_PARAMETERS, deployment_reader_only=True, identity_head_jointly_trained=True,
            identity_head_replaced=False, original_F0_frozen=True, L_sampling_seed=L_SEED, U_sampling_seed=U_SEED,
            formal_target_accessed=False, target_REG=False, final_phase3_complete=False, **{k: 0 for k in ZEROS})
        check(); _write(out / 'complete.json', done); print(STATUS, flush=True)
        return done
    finally:
        if L is not None:
            cache.close_role(L)


def load_trained_model(c, arm, device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:
        raise ValueError('Only SOURCE_ID or TARGET_ID native students may load')
    root = Path(c['output_root'])
    try:
        done = _read(root / 'complete.json')
        rows = {a: _read(root / a / 'complete.json') for a in ARMS}
    except (OSError, ValueError) as exc:
        raise ValueError('Both completed native identity students required') from exc
    expected = dict(schema=SCHEMA, status=STATUS, run_id=RUN_ID, code_commit=c['code_commit'], config=c,
        arms=list(ARMS), updates_per_arm=STEPS, total_optimizer_updates=2 * STEPS,
        source_sampling_matched=True, SOURCE_completed_before_U_open=True, original_models_unchanged=True,
        all_student_Sinc_unchanged=True, all_student_buffers_unchanged=True, all_student_outside_reader_unchanged=True,
        all_student_bodies_changed=True, all_student_heads_changed=True, target_U_reads=U_COUNT,
        all_students_initialized_from_original_F0=True, final_phase3_complete=False, **{k: 0 for k in ZEROS})
    if any(done.get(k) != v for k, v in expected.items()) or done.get('rows') != rows:
        raise ValueError('Complete matched source-first native identity training required')
    pm, _, _, _ = _reference(c)
    if any(done.get(k) != v for k, v in dict(parent_F0_identity=pm['parent_F0_identity'],
            parent_PCD_identity=pm['model_identities']['TARGET_PCD'], dura_identity=pm['dura_identity']).items()):
        raise ValueError('Actual original F0/PCD/G ancestry differs')
    for a, row in rows.items():
        target = a == 'TARGET_ID'
        identity = _identity(c, a, pm, done['source_L_manifest'], done['U_manifest'] if target else None)
        required = dict(schema=SCHEMA, status=ROW_STATUS, run_id=RUN_ID, code_commit=c['code_commit'], config=c,
            arm=a, identity=identity, updates=STEPS, identity_head_updates=STEPS, source_only=not target,
            target_U_used=target, target_U_reads=U_COUNT if target else 0, initial_state_equal_to_original_F0=True,
            student_Sinc_unchanged=True, student_body_changed=True, student_head_changed=True,
            model_buffers_unchanged=True, outside_reader_unchanged=True, original_models_unchanged=True,
            final_phase3_complete=False, **{k: 0 for k in ZEROS})
        if any(row.get(k) != v for k, v in required.items()) or done['identities'].get(a) != identity:
            raise ValueError('Native row identity/default-head/freeze state differs')
    model = _initial_model(c, pm, device); reader = model.id_backbone.encoder
    value = torch.load(root / arm / 'final.pt', map_location='cpu', weights_only=False)
    identity = rows[arm]['identity']
    if (value.get('schema') != SCHEMA or value.get('config') != c or value.get('identity') != identity
            or value.get('parent_F0_identity') != pm['parent_F0_identity']
            or value.get('source_L_manifest') != done['source_L_manifest']):
        raise ValueError('Strict original F0 reader checkpoint lineage required')
    saved = value.get('reader_state'); initial = _snapshot(reader)
    if (not isinstance(saved, dict) or set(saved) != set(initial)
            or any(not torch.is_tensor(v) or v.shape != initial[k].shape or v.dtype != initial[k].dtype
                   or not bool(torch.isfinite(v).all()) for k, v in saved.items())):
        raise ValueError('Exact complete finite native reader state with original shapes/dtypes required')
    fixed = (set(dict(reader.named_buffers())) & set(initial)) | {
        k for k, p in reader.named_parameters() if k.startswith(READER_SINC)}
    if any(not torch.equal(initial[k], saved[k]) for k in fixed):
        raise ValueError('Native Sinc and persistent buffers must remain identical to original F0')
    try:
        reader.load_state_dict(saved, strict=True)
    except RuntimeError as exc:
        raise ValueError('Strict full native reader and original default-head state required') from exc
    _reader_changed(reader, initial)
    model.eval().requires_grad_(False); model._native_identity = identity
    return model, identity


def main():
    print(json.dumps(dict(event='NATIVE_IDENTITY_TRAIN_PROCESS', pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv)), flush=True)
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); train(_read(args.config), args.device)


if __name__ == '__main__':
    main()
