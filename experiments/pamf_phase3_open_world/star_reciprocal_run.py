"""U-only reciprocal adapters, frozen registries, then independent future Q."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

from . import star_neighborhood_run as previous
from . import star_reciprocal_core as core
from . import star_reciprocal_adapter as adapter
from . import star_relative_target as public
from .development_transfer import _snapshot_model, _check_model
from .source_open_export import _tensor
from .star_consensus_score import _ids

RUN_ID = '20261030-phase3-star-reciprocal-adapter-s2026092701-r01'
SCHEMA = 'pamf_phase3_star_reciprocal_prediction_v1'
STATUS = 'STAR_RECIPROCAL_PREDICTIONS_COMPLETE'
U_STATUS = 'STAR_RECIPROCAL_U_STATE_FROZEN'
ARMS = ('self_pl', 'reciprocal')
STATES = ('DA0_REG0', 'DA1_REG0', 'DA0_REG1', 'DA1_REG1')
DAS = ('DA0', 'DA1')
FIELDS = previous.TARGET_FIELDS
PARENT = '20261029-phase3-star-neighborhood-s2026092701-r01'
PARENT_COMMIT = 'cb58217de28eb2cb5536d2392aef268b79006877'
PRED_FIELDS = {'ids', 'U_ids', 'arms', 'state_names', 'prediction', 'closed_prediction', 'baseline_closed_prediction', 'knownness'} | {
    'joint_'+arm+'_'+name for arm in ARMS for name in STATES}
DIAG_FIELDS = {'ids', 'assignments', 'candidate_mask', 'component_labels', 'component_sizes', 'registered_component_indices', 'neighbors', 'reciprocal'}
PSEUDO_FIELDS = {'ids', 'labels', 'selected', 'trusted', 'expanded'}
_read, _write, _npz = previous._read, previous._write, previous._npz


def validate_config(c):
    if (not isinstance(c, dict) or set(c) != FIELDS or c.get('run_id') != RUN_ID
            or not isinstance(c.get('code_commit'), str) or re.fullmatch('[0-9a-f]{40}', c['code_commit']) is None
            or type(c.get('batch_size')) is not int or c['batch_size'] != 128
            or any(not isinstance(c[k], str) or not c[k].strip() for k in FIELDS-{'batch_size'})):
        raise ValueError('Only fixed public U/Q handles and frozen source state allowed; no truth/K/quotas')
    fixed = dict(U_capsule_id='p3dev-star-g10-6e1292b62ab09286ebf1a0df', U_split_id='83c5cc1fe2f2a1512dbc1287',
        Q_capsule_id='p3dev-star-g10-71a5a0dec23ac6ccc3b479c2', Q_split_id='05de65cd4818f3e5e33ced5c')
    if any(c[k] != v for k, v in fixed.items()):
        raise ValueError('Only the existing RX24-1 G10 U384/Q864 roles allowed')
    for role, folder in (('U', 'discovery'), ('Q', 'evaluation')):
        p = Path(c[role+'_public_root'])
        if p.name != 'public' or p.parent.name != folder:
            raise ValueError('Independent discovery and future-query public roles required')
    if Path(c['source_root']).name != 'source' or Path(c['source_root']).parent.name != PARENT:
        raise ValueError('Only the fixed original-F0 aggregate source bundle allowed')
    if any(Path(c['output_root']).resolve().is_relative_to(Path(c[k]).resolve()) for k in ('source_root', 'U_public_root', 'Q_public_root')):
        raise ValueError('Output cannot replace source or public data')
    return c


def _unit(raw):
    x = np.asarray(raw, np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if x.ndim != 2 or x.shape[1] != 160 or not np.isfinite(x).all() or np.any(norm < 1e-12):
        raise ValueError('Nonzero finite raw fused160 required')
    return (x/norm).astype(np.float32)


def _margin(b1, l1, b0, l0, prior):
    d1 = previous.base.core.descriptor(b1, l1, prior['geometry']['P1'])
    d0 = previous.base.core.descriptor(b0, l0, prior['geometry']['P0'])
    pair = previous.paired.core.pair_descriptors(d1, d0)
    raw = previous.paired.core.score_readouts(pair, prior['readouts'])[:, 3]
    return previous.class_core.apply(raw, l1.argmax(1), prior['thresholds'])[:, 1]


def _encode(backend, values, device):
    with torch.no_grad():
        b0, l0, _ = public._encode(backend['reader'], _tensor(values, device))
        b1, l1, _ = public._encode(backend['reader'], public._current(values, 0, len(values), device, backend['dura']))
    return b0, l0, b1, l1


def _adapted(model, b1, classify, device):
    raw, logits = adapter.predict(model, b1['fused160'], classify, device)
    result = dict(b1); result['fused160'] = raw
    return result, logits


def _smoke(backend, prior, device):
    from .star_channel import observe
    mixture = _read(Path(__file__).resolve().parents[2]/'configs/pamf_phase2_practical_mixture_20261028.json')
    t = np.arange(256)
    x = np.asarray([observe(np.stack((np.cos(w*t), np.sin(w*t))), physical_id='reciprocal-synthetic-'+str(i),
        receiver='1-19', day='2021_03_08', episode=dict(id='MIX01', scene_seed=2026103001, channel_seed=2026104001), mixture=mixture)[0]
        for i, w in enumerate((.07, .11))], np.float32)
    b0, l0, b1, l1 = _encode(backend, x, device)
    with torch.enable_grad():
        raw = _tensor(b1['fused160'], device).requires_grad_(True)
        logits = backend['reader'].classify_features(raw)
        # A logit contrast avoids the zero derivative of squared cosine logits
        # at a perfectly aligned synthetic feature (a valid frozen head).
        grad = torch.autograd.grad((logits[:,0]+.37*logits[:,1]).sum(), raw)[0]
    head_logits = np.asarray(logits.detach().cpu().tolist(), np.float32)
    if (not bool(torch.isfinite(grad).all()) or float(grad.norm()) <= 0
            or not np.allclose(head_logits, l1, rtol=1e-5, atol=5e-5)):
        raise ValueError('Actual frozen original head must retain finite feature gradient and original logits')
    _margin(b1, l1, b0, l0, prior)
    return dict(status='PASS', actual_F0_G_checkpoint=True, original_default_head_input_gradient_norm=float(grad.norm()),
        default_head_roundtrip_max_abs=float(np.abs(head_logits-l1).max()), source_or_target_real_rows_read=0,
        F0_forward_rows=4, G_forward_rows=2, default_head_feature_rows=2, neural_updates=0)


def _save_registry(folder, arm, da, state, diag, ids):
    prefix = arm+'_'+da
    _npz(folder/(prefix+'.npz'), {k: state[k] for k in ('centers', 'radii')})
    _npz(folder/(prefix+'_diagnostics.npz'), dict(ids=ids, **{k: diag[k] for k in DIAG_FIELDS-{'ids'}}))
    _write(folder/(prefix+'_diagnostics.json'), dict(components=diag['components'], registered_count=diag['registered_count']))


def run(c, device='cuda:0'):
    validate_config(c); out = Path(c['output_root'])
    if out.exists():
        raise FileExistsError('New immutable reciprocal output root required')
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    prior = previous.load_source(c['source_root'])
    if prior['marker']['code_commit'] != PARENT_COMMIT:
        raise ValueError('Fixed original aggregate source code identity required')
    backend = previous._target_backend(prior, c, device)
    before = copy.deepcopy(prior); snapshots = previous._snapshot(backend)
    def check():
        previous._check_models(snapshots)
        if not previous.relative_source._same(prior, before):
            raise ValueError('Frozen source geometry/readout/threshold changed')
    smoke = _smoke(backend, prior, device); check()
    out.mkdir(parents=True); (out/'adaptation').mkdir(); _write(out/'smoke.json', smoke)
    cfg = dict(c, U_expected_count=384, Q_expected_count=864)
    _, idsU, iq = public._open(cfg, 'U')
    b0_rows, b1_rows, l0_rows, l1_rows = [], [], [], []
    try:
        for start in range(0, len(idsU), 128):
            b0, l0, b1, l1 = _encode(backend, iq[start:start+128], device)
            b0_rows.append(b0); b1_rows.append(b1); l0_rows.append(l0); l1_rows.append(l1)
            print('STAR_RECIPROCAL_U_PROGRESS', min(start+128, len(idsU)), len(idsU), flush=True)
    finally:
        if hasattr(iq, '_mmap'): iq._mmap.close()
    b0 = {k: np.concatenate([v[k] for v in b0_rows]) for k in b0_rows[0]}
    b1 = {k: np.concatenate([v[k] for v in b1_rows]) for k in b1_rows[0]}
    l0, l1 = np.concatenate(l0_rows), np.concatenate(l1_rows)
    unit0 = _unit(b1['fused160']); margin0 = _margin(b1, l1, b0, l0, prior)
    _npz(out/'adaptation/U_evidence.npz', dict(ids=idsU, unit=unit0, logits=l1, margins=margin0))
    pseudo = {arm: core.pseudolabels(unit0, l1, margin0, idsU, arm) for arm in ARMS}
    models, states, summaries, adapter_snaps = {}, {}, {}, {}
    for arm in ARMS:
        _npz(out/'adaptation'/('pseudo_'+arm+'.npz'), dict(ids=idsU, **{k: pseudo[arm][k] for k in PSEUDO_FIELDS-{'ids'}}))
        folder = out/('adapter_'+arm); folder.mkdir()
        model, trace, summary = adapter.fit(b1['fused160'], pseudo[arm]['labels'], backend['reader'].classify_features, device)
        models[arm] = model; summaries[arm] = summary
        with (folder/'model.pt').open('xb') as f: torch.save(model.state_dict(), f)
        with (folder/'steps.jsonl').open('x', encoding='utf-8') as f:
            for row in trace: f.write(json.dumps(row, allow_nan=False)+'\n')
        _write(folder/'summary.json', summary); adapter_snaps[arm] = _snapshot_model(model)
        if summary['steps']:
            b_adapt, l_adapt = _adapted(model, b1, backend['reader'].classify_features, device)
            margin1 = _margin(b_adapt, l_adapt, b0, l0, prior); unit1 = _unit(b_adapt['fused160'])
        else:
            b_adapt, l_adapt, margin1, unit1 = b1, l1, margin0, unit0
        states[arm] = {}
        for da, unit, margin in (('DA0', unit0, margin0), ('DA1', unit1, margin1)):
            state, diag = core.fit_registry(unit, margin, margin0, pseudo[arm]['trusted'], idsU)
            states[arm][da] = state
            _save_registry(out/'adaptation', arm, da, state, diag, idsU)
        check()
        print('STAR_RECIPROCAL_ADAPTER_FROZEN', arm, summary['status'], summary['steps'], 'pseudo', summary['selected_count'], flush=True)
    counts = {a: {da: len(states[a][da]['centers']) for da in DAS} for a in ARMS}
    frozen_states = copy.deepcopy(states)
    _write(out/'adaptation/complete.json', dict(status=U_STATUS, schema=SCHEMA, run_id=RUN_ID, code_commit=c['code_commit'],
        arms=list(ARMS), U_count=384, Q_rows_read=0, truth_read=False, target_label_inputs=0, true_K_inputs=0,
        source_updates=0, threshold_updates=0, prototype_counts=counts, adapter_summaries=summaries,
        pseudo_statistics={a: pseudo[a]['statistics'] for a in ARMS}, registries_frozen_before_Q=True))
    # No U features or graph is consulted by the subsequent row-wise Q path.
    del b0_rows, b1_rows, l0_rows, l1_rows, b0, b1, l0, l1, unit0, unit1, b_adapt, l_adapt, margin0, margin1, pseudo
    _, idsQ, iq = public._open(cfg, 'Q')
    if set(idsU.tolist()) & set(idsQ.tolist()):
        if hasattr(iq, '_mmap'): iq._mmap.close()
        raise ValueError('U and future Q IDs overlap')
    n = len(idsQ)
    p = dict(ids=idsQ, U_ids=idsU, arms=np.asarray(ARMS), state_names=np.asarray(STATES),
        prediction=np.empty((2, 4, n), np.int64), closed_prediction=np.empty((2, 2, n), np.int64),
        baseline_closed_prediction=np.empty(n, np.int64), knownness=np.empty((2, 4, n), np.float64))
    for arm in ARMS:
        for name in STATES:
            da = name[:3]; k = counts[arm][da] if name.endswith('REG1') else 0
            p['joint_'+arm+'_'+name] = np.empty((n, 6+k), np.float64)
    try:
        for start in range(0, n, 128):
            stop = min(start+128, n)
            b0, l0, b1, l1 = _encode(backend, iq[start:stop], device)
            margin0 = _margin(b1, l1, b0, l0, prior); unit0 = _unit(b1['fused160'])
            p['baseline_closed_prediction'][start:stop] = l1.argmax(1)
            for ai, arm in enumerate(ARMS):
                if summaries[arm]['steps']:
                    b_adapt, l_adapt = _adapted(models[arm], b1, backend['reader'].classify_features, device)
                    margin1 = _margin(b_adapt, l_adapt, b0, l0, prior); unit1 = _unit(b_adapt['fused160'])
                else:
                    b_adapt, l_adapt, margin1, unit1 = b1, l1, margin0, unit0
                for si, name in enumerate(STATES):
                    da = int(name.startswith('DA1')); unit, logits, margin = (unit1, l_adapt, margin1) if da else (unit0, l1, margin0)
                    pred, closed, knownness, joint = core.infer(unit, logits, margin, states[arm]['DA'+str(da)], name.endswith('REG1'))
                    p['prediction'][ai, si, start:stop] = pred; p['closed_prediction'][ai, da, start:stop] = closed
                    p['knownness'][ai, si, start:stop] = knownness; p['joint_'+arm+'_'+name][start:stop] = joint
            check()
            for arm in ARMS: _check_model(models[arm], adapter_snaps[arm])
            if not previous.relative_source._same(states, frozen_states): raise ValueError('Frozen U registry changed during Q')
            print('STAR_RECIPROCAL_Q_PROGRESS', stop, n, flush=True)
    finally:
        if hasattr(iq, '_mmap'): iq._mmap.close()
    _npz(out/'predictions.npz', p)
    costs = dict(F0_real_full_forward_rows=2*(384+864), F0_synthetic_full_forward_rows=4,
        G_real_forward_rows=384+864, G_synthetic_forward_rows=2, default_head_smoke_feature_rows=2,
        adapter_training={a: summaries[a] for a in ARMS},
        additional_adapter_inference_rows_per_arm={a: 384+864 if summaries[a]['steps'] else 0 for a in ARMS},
        additional_default_head_inference_rows_per_arm={a: 384+864 if summaries[a]['steps'] else 0 for a in ARMS}, source_IQ_or_feature_row_reads=0,
        source_F0_head_G_updates=0, prior_models_stored=4, prior_models_scored_per_evidence=4,
        prior_evidence_real_row_calls=(1+sum(summaries[a]['steps']>0 for a in ARMS))*(384+864), prior_evidence_synthetic_row_calls=2,
        F0_parameters=232993, G_parameters=266951, stored_readout_parameters=4*3233,
        adapter_parameters_per_arm=10432, simultaneous_two_adapter_total_neural_parameters=533740,
        single_adapter_resident_neural_parameters=523308, B1_latency_measured=False, ancestor_costs_preserved=True)
    done = dict(schema=SCHEMA, status=STATUS, run_id=RUN_ID, code_commit=c['code_commit'], config=c,
        arms=list(ARMS), states=list(STATES), primary_arm='reciprocal', primary_state='DA1_REG1',
        condition='P1', view_family='practical_cross_receiver', classification_path='P1', rejection_input_paths=['P1','P0'],
        count=864, U_count=384, U_day='2021-03-08', Q_day='2021-03-15', prototype_counts=counts,
        source_frozen_before_U=True, registry_frozen_before_Q=True, truth_read=False, query_fit=False,
        algorithm_target_label_inputs=0, true_K_inputs=0, new_threshold_calibrations=0, formal_target_reads=0,
        source_parameters_updated=0, identity_head_updates=0, source_bank_deployed=False,
        all_new_IQ_satellite=True, original_F0_identity=backend['reader_identity'], dura_identity=backend['dura_identity'],
        source_prior_run=PARENT, source_prior_code=PARENT_COMMIT, costs=costs,
        final_phase3_complete=False, full_goal_status='ACTIVE', seconds=time.perf_counter()-started)
    _write(out/'complete.json', done); print(STATUS, flush=True)
    return done


def _predictions(root):
    root = Path(root); done = _read(root/'complete.json'); c = done.get('config'); validate_config(c)
    if (done.get('schema') != SCHEMA or done.get('status') != STATUS or done.get('run_id') != RUN_ID
            or done.get('code_commit') != c['code_commit'] or done.get('arms') != list(ARMS) or done.get('states') != list(STATES)
            or done.get('count') != 864 or done.get('U_count') != 384 or done.get('condition') != 'P1'
            or any(done.get(k) is not True for k in ('source_frozen_before_U','registry_frozen_before_Q','all_new_IQ_satellite'))
            or done.get('truth_read') is not False or done.get('query_fit') is not False
            or any(done.get(k) != 0 for k in ('algorithm_target_label_inputs','true_K_inputs','new_threshold_calibrations','formal_target_reads','source_parameters_updated','identity_head_updates'))):
        raise ValueError('All eight frozen source-free predictions required before any private truth')
    frozen = _read(root/'adaptation/complete.json')
    if (frozen.get('status') != U_STATUS or frozen.get('Q_rows_read') != 0 or frozen.get('truth_read') is not False
            or frozen.get('code_commit') != done['code_commit'] or frozen.get('prototype_counts') != done['prototype_counts']):
        raise ValueError('All U adapters and arbitrary-K registries must freeze before Q')
    with np.load(root/'predictions.npz', allow_pickle=False) as z:
        if set(z.files) != PRED_FIELDS: raise ValueError('Exact public eight-state prediction fields required')
        p = {k: z[k].copy() for k in z.files}
    for name, n in (('ids',864),('U_ids',384)):
        if not _ids(p[name], n): raise ValueError('Unique aligned opaque IDs required')
    if (set(p['ids'].tolist()) & set(p['U_ids'].tolist()) or p['arms'].tolist() != list(ARMS) or p['state_names'].tolist() != list(STATES)
            or p['prediction'].shape != (2,4,864) or p['prediction'].dtype != np.int64
            or p['knownness'].shape != (2,4,864) or p['knownness'].dtype != np.float64 or not np.isfinite(p['knownness']).all()
            or p['closed_prediction'].shape != (2,2,864) or p['closed_prediction'].dtype != np.int64
            or p['baseline_closed_prediction'].shape != (864,) or p['baseline_closed_prediction'].dtype != np.int64
            or np.any((p['closed_prediction']<0)|(p['closed_prediction']>=6))):
        raise ValueError('Aligned independent U/Q and complete current default-head predictions required')
    diag, states, pseudo = {}, {}, {}
    for ai, arm in enumerate(ARMS):
        if not np.array_equal(p['closed_prediction'][ai,0],p['baseline_closed_prediction']):
            raise ValueError('DA0 must retain the actual original F0 baseline')
        with np.load(root/'adaptation'/('pseudo_'+arm+'.npz'), allow_pickle=False) as z:
            if set(z.files) != PSEUDO_FIELDS: raise ValueError('Exact public pseudolabel masks required')
            pseudo[arm] = {k:z[k].copy() for k in z.files}
        ps = pseudo[arm]
        if (not np.array_equal(ps['ids'], p['U_ids']) or ps['labels'].shape != (384,) or ps['labels'].dtype != np.int64
                or np.any((ps['labels'] < -1)|(ps['labels'] > 5)) or any(ps[k].shape != (384,) or ps[k].dtype != bool for k in ('selected','trusted','expanded'))
                or not np.array_equal(ps['selected'],ps['labels']>=0) or not np.array_equal(ps['expanded'],ps['selected']&~ps['trusted'])
                or np.any(ps['trusted']&~ps['selected'])):
            raise ValueError('Fixed valid U-only pseudo masks required')
        summary = _read(root/('adapter_'+arm)/'summary.json')
        expected_steps = 200 if ps['selected'].any() else 0
        if (summary != frozen['adapter_summaries'][arm] or summary.get('steps') != expected_steps
                or summary.get('status') != ('COMPLETE' if expected_steps else 'SKIPPED_NO_PSEUDOLABELS')
                or summary.get('U_count') != 384 or summary.get('selected_count') != int(ps['selected'].sum())
                or summary.get('parameters') != 10432 or summary.get('model_seed') != 2026092701
                or summary.get('F0_updates') != 0 or summary.get('G_updates') != 0 or summary.get('finite') is not True):
            raise ValueError('Actual fixed U-only adapter budget and frozen F0/G required')
        steps = [json.loads(x) for x in (root/('adapter_'+arm)/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(steps) != expected_steps or any(x.get('step') != i+1 or not all(np.isfinite(x[k]) for k in ('loss_ce','loss_gram','loss_total','grad_norm','max_pre_normalization_residual')) for i,x in enumerate(steps)):
            raise ValueError('Complete finite adapter step trace required')
        model = adapter.Adapter()
        weights = torch.load(root/('adapter_'+arm)/'model.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in weights.values()):
            raise ValueError('Finite FP32 residual-adapter deployment weights required')
        states[arm], diag[arm] = {}, {}
        for da in DAS:
            prefix = arm+'_'+da
            with np.load(root/'adaptation'/(prefix+'.npz'), allow_pickle=False) as z:
                if set(z.files) != {'centers','radii'}: raise ValueError('Only frozen anonymous aggregates allowed')
                state = dict(schema=core.REGISTRY_SCHEMA, **{k:z[k].copy() for k in z.files})
            with np.load(root/'adaptation'/(prefix+'_diagnostics.npz'), allow_pickle=False) as z:
                if set(z.files) != DIAG_FIELDS: raise ValueError('Exact U registry diagnostics required')
                d = {k:z[k].copy() for k in z.files}
            d.update(_read(root/'adaptation'/(prefix+'_diagnostics.json')))
            k = len(state['centers'])
            if (state['centers'].shape != (k,160) or state['centers'].dtype != np.float64 or state['radii'].shape != (k,)
                    or state['radii'].dtype != np.float64 or not np.isfinite(state['centers']).all() or not np.isfinite(state['radii']).all()
                    or np.any(state['radii']<1e-6) or done['prototype_counts'][arm][da] != k or d['registered_count'] != k
                    or not np.array_equal(d['ids'],p['U_ids']) or d['assignments'].shape != (384,) or d['assignments'].dtype != np.int64
                    or np.any((d['assignments']<-1)|(d['assignments']>=k)) or d['candidate_mask'].shape != (384,)
                    or d['candidate_mask'].dtype != bool or np.any((d['assignments']>=0)&~d['candidate_mask'])):
                raise ValueError('Aligned arbitrary-K U-only anonymous state required')
            states[arm][da], diag[arm][da] = state, d
        for si, name in enumerate(STATES):
            da = int(name.startswith('DA1')); k = len(states[arm]['DA'+str(da)]['centers']) if name.endswith('REG1') else 0
            joint = p['joint_'+arm+'_'+name]
            if (joint.shape != (864,6+k) or joint.dtype != np.float64 or not np.isfinite(joint).all()
                    or not np.array_equal(joint.max(1),p['knownness'][ai,si])
                    or not np.array_equal(np.where(joint.max(1)>=0,joint.argmax(1),-1),p['prediction'][ai,si])
                    or not np.array_equal(joint[:,:6].argmax(1),p['closed_prediction'][ai,da])):
                raise ValueError('Actual default-head joint decisions and evidence must reconstruct')
        for da in range(2):
            j0=p['joint_'+arm+'_DA'+str(da)+'_REG0'];j1=p['joint_'+arm+'_DA'+str(da)+'_REG1'];accepted=j0.max(1)>=0
            if (not np.array_equal(j0,j1[:,:6]) or not np.array_equal(p['prediction'][ai,da+2,accepted],p['prediction'][ai,da,accepted])
                    or np.any((p['prediction'][ai,da+2,~accepted]>=0)&(p['prediction'][ai,da+2,~accepted]<6))
                    or (j1.shape[1]>6 and np.any(j1[accepted,6:]>=0))):
                raise ValueError('Registration must preserve accepted old labels and gate anonymous decisions')
    if not np.array_equal(p['prediction'][0,0],p['prediction'][1,0]) or not np.array_equal(p['knownness'][0,0],p['knownness'][1,0]):
        raise ValueError('Both arms require the same untouched DA0_REG0 comparison')
    return done,p,diag,states,pseudo


def main():
    print(json.dumps(dict(event='STAR_RECIPROCAL_RUN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run']);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    run(_read(a.config),a.device)


if __name__=='__main__':main()
