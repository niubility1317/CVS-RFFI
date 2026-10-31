"""Frozen fresh-PCD U-only SHIFT/CORAL moments and cached three-path scores."""
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
from . import pcd_moment_core as core
from . import fresh_dura_run as base
from . import pcd_cluster_registry_run as registry

fresh=base.fresh
RUN_ID='20261031-phase3-pcd-moment-s2026092701-r01'
REFERENCE_CODE=base.REFERENCE_CODE
SCHEMA='pamf_phase3_pcd_moment_prediction_v1'
STATUS='PCD_MOMENT_PREDICTIONS_COMPLETE'
MODELS_STATUS='PCD_MOMENT_MODELS_FROZEN'
ADAPTERS_STATUS='PCD_MOMENT_ADAPTERS_FROZEN'
CALIBRATION_STATUS='PCD_MOMENT_THRESHOLDS_INHERITED'
U_REFERENCE_RUN='20261031-phase3-pcd-cluster-registry-s2026092701-r01'
U_REFERENCE_CODE='16e6f04e7475827e7c8e24b398fb029a39a8138c'
U_REFERENCE_SCHEMA='pamf_phase3_pcd_cluster_registry_prediction_v1'
U_REFERENCE_STATUS='PCD_CLUSTER_REGISTRY_PREDICTIONS_COMPLETE'
METHODS=core.METHODS
PRIMARY_METHOD=core.PRIMARY_METHOD
PRIMARY_INDEX=core.PRIMARY_INDEX
MODEL_NAMES=('E0','PCD')
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(fresh.COUNTS)
Q_COUNT=10238
U_COUNT=576
FIELDS={'run_id','code_commit','reference_config','U_reference_root','Q_conditions','output_root','batch_size'}
U_FIELDS={'ids','features','default_logits','closed','base_scores'}
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=fresh.KNOWN_FIELDS,fresh.NOVEL_FIELDS,fresh.TARGET_FIELDS
_read,_write,_same,_save=base._read,base._write,base._same,base._save
_backend,_load_model,_smoke=base._backend,base._load_model,registry._smoke


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('output_root','U_reference_root'))):
        raise ValueError('Exact seven-field moment config required; no truth, IQ, K or registry overrides')
    ref=fresh.validate_config(c['reference_config'])
    if ref['code_commit']!=REFERENCE_CODE or c['Q_conditions']!=ref['Q_conditions']:raise ValueError('Original fresh f4b5 and identical clean/star public handles required')
    source=ref['source_control_config']['reference_config']['train_config'];out=Path(c['output_root']).resolve();u=Path(c['U_reference_root']).resolve()
    if u==Path(ref['train_config']['output_root']).resolve():raise ValueError('Final-PCD discovery export required; fresh parent_z is not final PCD')
    roots=[ref['output_root'],ref['train_config']['output_root'],ref['train_config']['U_public_root'],source['IQ_root'],source['output_root'],c['U_reference_root']]
    roots.extend(v['public_root'] for v in c['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable moment output overlaps an input')
    return c


def _reference(c):
    # Reuse only this helper's source-state validation; discard its own-run
    # calibration wrapper and construct an explicitly inherited current wrapper.
    pm,states,old_tau,parent_cal=base._reference(c)
    tau=np.concatenate((old_tau,old_tau[2:4],old_tau[2:4]),axis=0);reports={}
    for index,name in enumerate(METHODS):
        origin=base.METHODS[index if index<4 else 2+(index%2)]
        reports[name]=deepcopy(parent_cal['methods'][origin])
    cal=dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        methods=reports,conditional_thresholds=tau.tolist(),known_only=True,unknown_inputs_used=False,thresholds_refitted=False,
        unique_source_threshold_values=24,threshold_values=48,threshold_bytes=384,parent_calibration_identity=deepcopy(parent_cal['parent_calibration_identity']))
    return pm,states,tau,cal


def _U_header(c,pm):
    root=Path(c['U_reference_root']);m=_read(root/'complete.json')
    required=dict(schema=U_REFERENCE_SCHEMA,status=U_REFERENCE_STATUS,run_id=U_REFERENCE_RUN,code_commit=U_REFERENCE_CODE,
        reference_prediction_identity=pm,model_identity=pm['model_identities']['TARGET_PCD'],parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'])
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Completed final-PCD discovery feature ancestry differs; no old registry state is accepted')
    manifest=m['U_manifest'];train=c['reference_config']['train_config']
    expected=dict(data_status='VALIDATED_ONCE',count=U_COUNT,capsule_id=train['U_capsule_id'],split_id=train['U_split_id'])
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Same validated fresh star-U capsule/split/count required')
    handle={k:manifest[k] for k in expected}
    # Deliberately retain no cluster count, labels, assignments or registry ID.
    ancestry=dict(run_id=m['run_id'],code_commit=m['code_commit'],status=m['status'],root=str(root),model_identity=m['model_identity'],
        parent_F0_identity=m['parent_F0_identity'],dura_identity=m['dura_identity'],U_manifest=handle)
    return handle,ancestry


def _U(c,states,tau):
    with np.load(Path(c['U_reference_root'])/'discovery/U.npz',allow_pickle=False) as z:
        if set(z.files)!=U_FIELDS|{'assignment'}:raise ValueError('Final-PCD U export must expose the declared five public members')
        a={key:z[key].copy() for key in U_FIELDS}  # Never deserialize assignment.
    ids=a['ids']
    if ids.shape!=(U_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=U_COUNT or np.any(np.char.strip(ids)==''):raise ValueError('Complete unique discovery opaque IDs required')
    core._features(a['features']);closed=core._closed(a['closed'],U_COUNT)
    if a['features'].shape!=(U_COUNT,160) or a['default_logits'].shape!=(U_COUNT,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all() or not np.array_equal(closed,a['default_logits'].argmax(1)):raise ValueError('Final PCD units and same-U original-F0 predictions required')
    if a['base_scores'].shape!=(U_COUNT,2) or a['base_scores'].dtype!=np.float64 or not np.isfinite(a['base_scores']).all():raise ValueError('Two original PCD margins required')
    expected=fresh._raw(states['PCD'],a['features'],closed)-tau[2:4,closed].T
    if not np.allclose(a['base_scores'],expected,rtol=1e-12,atol=1e-9):raise ValueError('Final PCD source density/threshold margins differ')
    return a


def _select(a):return base._select(a)


def _raw(states,z,closed,adapters=None):
    original=base._raw(states,z,closed)
    if adapters is None:return np.column_stack((original,original[:,2:4],original[:,2:4]))
    return np.column_stack((original,*(core.raw_scores(states['PCD'],z[:,1],closed,adapters[name]) for name in core.ARMS)))


def _combine(a,states,tau,adapters=None):
    if adapters is None:
        return dict(a,raw_scores=np.column_stack((a['raw_scores'],a['raw_scores'][:,2:4],a['raw_scores'][:,2:4])),
            scores=np.column_stack((a['scores'],a['scores'][:,2:4],a['scores'][:,2:4])))
    raw=np.column_stack([core.raw_scores(states['PCD'],a['signal_features'][:,1],a['closed'],adapters[name]) for name in core.ARMS])
    return dict(a,raw_scores=np.column_stack((a['raw_scores'],raw)),scores=np.column_stack((a['scores'],raw-tau[4:,a['closed']].T)))


def _validate(a,role):
    n=COUNTS['V' if role=='known' else 'novel'] if role in ('known','novel') else Q_COUNT
    fields=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS if role=='novel' else TARGET_FIELDS
    if (set(a)!=fields or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n or np.any(np.char.strip(a['ids'])=='')
            or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))):raise ValueError('Complete original F0 identity/opaque IDs and exact public fields required')
    if 'baseline_closed' in fields and (a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):raise ValueError('Unchanged original F0 same-DURA baseline required')
    z=a['signal_features']
    if z.shape!=(n,2,160) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1.,rtol=0,atol=2e-6):raise ValueError('Two original frozen E0/final-PCD unit features required')
    if any(a[k].shape!=(n,8) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores')):raise ValueError('Eight finite frozen density readouts required')
    if role in ('known','novel'):
        if a['raw_signal_features'].shape!=(n,160) or a['raw_signal_features'].dtype!=np.float32 or not np.isfinite(a['raw_signal_features']).all():raise ValueError('Original cached F0 raw features required')
        if any(a[k].shape!=(n,) or a[k].dtype!=np.int64 or not np.isin(a[k],v).all() for k,v in (('rx_i',(1,3,4,6,8)),('day_i',(1,2,3)))):raise ValueError('Original source RX/day namespace required')
        if role=='known' and (a['y'].shape!=(n,) or a['y'].dtype!=np.int64 or not np.isin(a['y'],range(6)).all() or a['scenario'].shape!=(n,) or a['scenario'].dtype.kind!='U' or not np.isin(a['scenario'],fresh.risk.SCENES).all()):raise ValueError('Known V labels/scenarios only required')
    return a


def _source_rows(root,role):
    if role not in ('known','novel'):raise ValueError('Fixed source known/novel role required')
    with np.load(Path(root)/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,role)


def _target_rows(root,condition):
    if condition not in CONDITIONS[1:]:raise ValueError('Fixed future clean/star DURA condition required')
    with np.load(Path(root)/'targets'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,'Q')


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable PCD moment prediction root required')
    pm,states,tau,cal=_reference(c);Umanifest,Uidentity=_U_header(c,pm)
    b=_backend(c,device);pcd,identity=_load_model(c,device)
    identities=dict(E0=pm['model_identities']['E0'],PCD=pm['model_identities']['TARGET_PCD'])
    if b['model_identity']!=identities['E0'] or identity!=identities['PCD'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Actual frozen F0/source-r03/final-PCD/G ancestry differs')
    models=(b['reader'],b['model'],pcd,b['dura'][0])
    if any(p.requires_grad or p.dtype!=torch.float32 for m in models for p in m.parameters()) or any(v.training for m in models for v in m.modules()):raise ValueError('Original four checkpoints must stay frozen FP32 eval')
    snapshots=[(m,fresh.best._snapshot_model(m)) for m in models];frozen=deepcopy(states);fixed_tau=tau.copy();tau.setflags(write=False)
    def check():
        for m,before in snapshots:fresh.best._check_model(m,before)
        if not _same(states,frozen) or not np.array_equal(tau,fixed_tau):raise ValueError('Original frozen source density/threshold state changed')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    smoke=_smoke(b,pcd,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_r03_DURA_PCD_checkpoints') is not True:raise ValueError('Actual four-checkpoint no-query smoke required before U values')
    out.mkdir();_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    U=_U(c,states,tau);adapters,diag=core.fit_adapters(states['PCD'],U['features'],U['default_logits'],U['closed'],U['base_scores']);check()
    Uids=U['ids'].copy();del U
    for name,state in states.items():_save(out/'models'/(name+'.npz'),state)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],model_identities=identities,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],models_frozen=True,models_refitted=False,thresholds_refitted=False,source_V_fit=False,novel_fit=False,target_fit=False))
    _save(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir();_write(out/'calibration/complete.json',cal)
    for name,adapter in adapters.items():_save(out/'adapters'/(name+'.npz'),adapter)
    with (out/'adapters/U_ids.npy').open('xb') as stream:np.save(stream,Uids,allow_pickle=False)
    metadata=dict(schema=SCHEMA,status=ADAPTERS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],model_identity=identity,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],U_reference_identity=Uidentity,U_manifest=Umanifest,
        diagnostics=diag,fit_status=diag['status'],adaptation_status=diag['status'],fit_succeeded=diag['fit_succeeded'],adapters_frozen_before_Q=True,
        query_fit=False,U_only=True,U_count=U_COUNT,U_IDs_ref='adapters/U_ids.npy',source_state_refitted=False,thresholds_refitted=False,optimizer_updates=0)
    _write(out/'adapters/complete.json',metadata);fixed_adapters=deepcopy(adapters)
    print('PCD_MOMENT_ADAPTERS_FROZEN',diag['status'],diag['selected_count'],diag['effective_sample_size'],flush=True)
    used=set(Uids.tolist());source_ids=set();paired=None;source_count=0;query_count=0
    for role in ('known','novel'):
        a=_select(fresh._source_rows(c['reference_config']['output_root'],'source_DURA',role));ids=set(a['ids'].tolist())
        if ids&(used|source_ids):raise ValueError('Source V/novel and fitted U must be physically disjoint')
        source_ids.update(ids);a=_validate(_combine(a,states,tau),'known' if role=='known' else 'novel')
        _save(out/(role+'_predictions.npz'),a);source_count+=len(a['ids']);check();print('PCD_MOMENT_SOURCE_SAVED',role,len(a['ids']),flush=True)
    for condition in CONDITIONS[1:]:
        a=_select(fresh._target_rows(c['reference_config']['output_root'],condition));ids=set(a['ids'].tolist())
        if ids&(used|source_ids):raise ValueError('Future Q must be physically disjoint from U and source evaluation')
        if paired is None:paired=a['ids'].copy()
        elif not np.array_equal(paired,a['ids']):raise ValueError('Paired clean/star future physical IDs/order differ')
        a=_validate(_combine(a,states,tau,adapters),'Q');_save(out/'targets'/condition/'predictions.npz',a);query_count+=len(a['ids'])
        check()
        if not _same(adapters,fixed_adapters):raise ValueError('Frozen U moment adapter changed during query inference')
        print('PCD_MOMENT_TARGET_SAVED',condition,len(a['ids']),flush=True)
    check()
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        model_names=list(MODEL_NAMES),conditions=list(CONDITIONS),reference_prediction_identity=pm,model_identities=identities,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],U_reference_identity=Uidentity,U_manifest=Umanifest,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS[1:]},Q_manifests=deepcopy(pm['Q_manifests']),
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,adapters_frozen_before_Q=True,all_thresholds_frozen_before_U_and_Q=True,
        source_control_readouts_reused_exactly=True,classification_unchanged=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,
        source_V_fit=False,source_NV_fit=False,Q_fit=False,new_neural_updates=0,optimizer_updates=0,source_density_models_fitted=0,new_threshold_sets_fitted=0,
        target_U_reads=U_COUNT,U_feature_rows_read=U_COUNT,U_ID_metadata_reads=U_COUNT,U_statistic_fit_calls=1,adaptation_arms=list(core.ARMS),
        adaptation_status=diag['status'],fit_succeeded=diag['fit_succeeded'],source_cached_prediction_rows=source_count,target_Q_cached_prediction_rows=query_count,
        source_L_reads=0,source_OE_reads=0,source_Us_reads=0,original_U_s_reads=0,source_IQ_reads=0,target_Q_IQ_reads=0,
        real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,
        synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],synthetic_signal_forward_rows=smoke['synthetic_signal_forward_rows'],synthetic_r03_forward_rows=2,synthetic_PCD_forward_rows=2,G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],
        threshold_values=48,threshold_bytes=384,unique_source_threshold_values=24,unique_source_threshold_bytes=192,new_threshold_values=0,
        primary_adapter_values=26560,primary_adapter_bytes=212480,primary_deployed_neural_parameters=834665,
        inherited_PCD_target_U_count=576,PCD_has_target_U_ancestor=True,source_bank_deployed=False,U_bank_deployed=False,source_only=False,target_DA=True,target_REG=False,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,formal_target_accessed=False,full_goal_status='ACTIVE',final_phase3_complete=False)
    _write(out/'complete.json',done);print(STATUS,diag['status'],flush=True);return done


evaluate=run


def main():
    print(json.dumps(dict(event='PCD_MOMENT_PREDICT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
