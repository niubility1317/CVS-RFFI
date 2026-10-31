"""Frozen source-Us student with exact SOURCE/PCD controls and truth-blind queries."""
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
from . import fresh_world_run as fresh

RUN_ID='20261031-phase3-source-us-replay-s2026092701-r01'
REFERENCE_CODE='f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7'
SCHEMA='pamf_phase3_source_us_replay_prediction_v1'
STATUS='SOURCE_US_REPLAY_PREDICTIONS_COMPLETE'
MODELS_STATUS='SOURCE_US_REPLAY_DENSITIES_FROZEN'
CALIBRATION_STATUS='SOURCE_US_REPLAY_THRESHOLDS_FROZEN'
MODEL_NAMES=('SOURCE','OLD_PCD','SOURCE_US')
METHODS=tuple(a+'_'+b for a in MODEL_NAMES for b in ('pooled_ratio','predicted_ratio'))
PRIMARY_METHOD=METHODS[-1]
CONDITIONS=fresh.CONDITIONS
COUNTS=dict(fresh.COUNTS)
Q_COUNT=10238
US_COUNT=56700
FIELDS={'run_id','code_commit','train_config','source_control_config','Q_conditions','output_root','batch_size'}
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=fresh.KNOWN_FIELDS,fresh.NOVEL_FIELDS,fresh.TARGET_FIELDS
best,density,risk,source_eval=fresh.best,fresh.density,fresh.risk,fresh.source_eval
_read,_write,_same=fresh._read,fresh._write,fresh._same


def _train():
    from . import source_us_train
    return source_us_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact seven-field source-Us evaluation required; no truth or role overrides')
    t=_train().validate_config(c['train_config']);p=fresh.validate_config(c['source_control_config'])
    if (t['run_id']!=RUN_ID or t['code_commit']!=c['code_commit'] or p['code_commit']!=REFERENCE_CODE
            or t['reference_config']!=p['source_control_config'] or c['Q_conditions']!=p['Q_conditions']):
        raise ValueError('Source-only training, exact fresh controls and the same future public handles required')
    out=Path(c['output_root']).resolve()
    roots=[t['output_root'],t['Us_public_root'],p['output_root'],t['reference_config']['output_root'],
        t['reference_config']['reference_config']['train_config']['IQ_root']]+[v['public_root'] for v in c['Q_conditions'].values()]
    if any(out.is_relative_to(Path(x).resolve()) or Path(x).resolve().is_relative_to(out) for x in roots):
        raise ValueError('Immutable evaluation output cannot overlap model, source or public inputs')
    return c


def _npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _reference(c):
    p=c['source_control_config'];root=Path(p['output_root']);m=_read(root/'complete.json')
    expected=dict(schema=fresh.SCHEMA,status=fresh.STATUS,run_id=fresh.RUN_ID,code_commit=REFERENCE_CODE,config=p,
        all_three_DURA_conditions_complete=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,
        classification_unchanged=True,all_thresholds_frozen_before_novel_and_Q=True,private_truth_reads=0)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Complete exact fresh SOURCE/PCD reference required')
    parent,_,_,_=fresh._reference(p)
    if m['source_control_prediction_identity']!=parent:raise ValueError('Fresh source-control ancestry differs')
    fresh._identity(m['model_identities']['TARGET_PCD'],p,parent)
    fit=_read(root/'models/complete.json')
    if fit.get('status')!=fresh.MODELS_STATUS or fit.get('source_V_fit') is not False or fit.get('identities')!=m['model_identities']:
        raise ValueError('Frozen source-fitted SOURCE/PCD models required')
    states={n:_npz(root/'models'/file) for n,file in [('SOURCE','SOURCE.npz'),('OLD_PCD','TARGET_PCD.npz')]}
    for v in states.values():density.validate_state(v,'density_ratio')
    tau=_npz(root/'thresholds.npz')['conditional'][2:6].copy();cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or cal.get('status')!=fresh.CALIBRATION_STATUS
            or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False):raise ValueError('Four exact frozen control thresholds required')
    reports={}
    for j,name in enumerate(METHODS[:4]):
        report=cal['methods'][fresh.METHODS[j+2]]
        if report['conditional_thresholds']!=tau[j].tolist():raise ValueError('Control calibration and thresholds differ')
        reports[name]=deepcopy(report)
    return m,states,tau,dict(methods=reports)


def _identity(i,c,pm):
    steps=_train().STEPS
    expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm='SOURCE_US',updates=steps,projection_updates=steps,
        signal_parameters=334721,identity_head_updates=0,Sinc_frozen=True,source_only=True,
        target_U_used=False,target_U_reads=0,input_path='cached_P1_IQ_only',source_Us_used=True,source_Us_count=US_COUNT,
        source_Us_sampled_exposures=128*steps,new_body_updates=steps,new_projection_updates=steps)
    if (any(i.get(k)!=v for k,v in expected.items()) or i.get('parent_signal_identity')!=pm['model_identities']['E0']
            or i.get('parent_F0_identity')!=pm['parent_F0_identity'] or i.get('dura_identity')!=pm['dura_identity']):
        raise ValueError('Frozen source-only r03 plus source-Us student identity required')
    return i


def _training_metadata(c,pm):
    m=_read(Path(c['train_config']['output_root'])/'complete.json');i=m.get('identity',{})
    expected=dict(status=_train().STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c['train_config'],updates=_train().STEPS,
        total_optimizer_updates=_train().STEPS,native_updates=0,identity_head_updates=0,source_only=True,target_U_used=False,
        target_U_reads=0,source_sampling_matched=True,Us_ground_truth_reads=0,private_truth_reads=0)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Complete source-Us training required')
    _identity(i,c,pm)
    return m,i


def _backend(c,device):return fresh._backend(c['source_control_config'],device)
def _load_model(c,device):return _train().load_trained_model(c['train_config'],device)
def _iq_role(c,role,pm):return fresh._iq_role(c['source_control_config'],role,pm['source_control_prediction_identity'])
def _public(c,name):return fresh._public(c['source_control_config'],name)
def _raw(state,z,closed):return fresh._raw(state,z,closed)


def _controls(a):
    return {k:(v[:,2:6].copy() if k in ('raw_scores','scores') else v[:,1:3].copy() if k=='signal_features' else v.copy()) for k,v in a.items()}


def _exact_controls(a,parent):
    old=_controls(parent)
    if set(a)!=set(old):raise ValueError('Old control public fields changed')
    for k,v in old.items():
        actual=a[k][:,:4] if k in ('scores','raw_scores') else a[k][:,:2] if k=='signal_features' else a[k]
        if not np.array_equal(actual,v):raise ValueError('SOURCE/OLD_PCD identity, features or decisions changed')


def _validate(a,role):
    n=COUNTS['V' if role=='known' else 'novel'] if role in ('known','novel') else Q_COUNT
    if len(a['ids'])!=n:raise ValueError('Complete fixed physical role required')
    return fresh._validate(a,role)


def _source_rows(root,condition,role):
    if condition!='source_DURA' or role not in ('known','novel'):raise ValueError('Fixed source role required')
    return _validate(_npz(Path(root)/'source'/condition/(role+'_predictions.npz')),role)


def _target_rows(root,condition):
    if condition not in CONDITIONS[1:]:raise ValueError('Fixed future-DURA condition required')
    return _validate(_npz(Path(root)/'targets'/condition/'predictions.npz'),'Q')


def _forward(b,model,values,device):
    x=np.asarray(values)
    if x.dtype!=np.float32 or x.shape!=(len(x),2,256) or not np.isfinite(x).all():raise ValueError('Finite public full256 IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        iq=torch.tensor(x.tolist(),dtype=torch.float32,device=device);restored=b['dura'][1](iq)[0].detach().clone()
        logits=b['reader'].classify_features(b['reader'].features(restored));z=model(restored)
    logits=np.asarray(logits.detach().cpu().tolist(),np.float32);z=np.asarray(z.detach().cpu().tolist(),np.float32)
    if (logits.shape!=(len(x),6) or z.shape!=(len(x),160) or not np.isfinite(logits).all() or not np.isfinite(z).all()
            or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1,atol=2e-6,rtol=0)):raise ValueError('One same-DURA F0/new-student pass required')
    closed=logits.argmax(1).astype(np.int64)
    return dict(default_logits=logits,closed=closed,baseline_closed=closed.copy(),signal_features=z)


def _smoke(b,model,device):
    from .star_open_feature_export import _synthetic
    values=np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32);_forward(b,model,values,device)
    return dict(status='PASS',actual_F0_SOURCE_US_DURA_checkpoints=True,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,real_data_reads=0,optimizer_updates=0)


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable source-Us predictions required')
    pm,states,old_tau,old_cal=_reference(c);training,declared=_training_metadata(c,pm)
    b=_backend(c,device);model,identity=_load_model(c,device);_identity(identity,c,pm)
    if (identity!=declared or b['model_identity']!=pm['model_identities']['E0'] or b['reader_identity']!=pm['parent_F0_identity']
            or b['dura_identity']!=pm['loaded_dura_identity']):raise ValueError('Actual checkpoint/F0/DURA ancestry differs')
    if sum(v.numel() for v in model.parameters())!=334721 or any(v.requires_grad or v.dtype!=torch.float32 for v in model.parameters()) or any(v.training for v in model.modules()):raise ValueError('Frozen SOURCE_US checkpoint required before source reads')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,best._snapshot_model(m)) for m in (b['reader'],b['dura'][0],b['model'],model)]
    def check():
        for m,s in snapshots:best._check_model(m,s)
    smoke=_smoke(b,model,device);check()
    if smoke.get('status')!='PASS':raise ValueError('Actual no-query checkpoint smoke must precede source inputs')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    used=set();source_rows=0;diagnostics={}
    for role in ('L','OE'):
        iq=_iq_role(c,role,pm)
        try:
            ids=set(iq['ids'].tolist())
            if ids&used:raise ValueError('Source L/OE roles overlap')
            used.update(ids);z=source_eval._project(model,iq['restored'],device);source_rows+=len(z);m=iq['metadata']
            if role=='L':known,diagnostics['known']=density.fit_known(z,m['y'],m['rx_i'],m['day_i'])
            else:states['SOURCE_US'],diagnostics['background']=density.fit_background(known,z,m['rx_i'],m['day_i'])
            check()
        finally:source_eval._close_role(iq)
    frozen=deepcopy(states)
    for name,state in states.items():best._save(out/'models'/(name+'.npz'),state)
    identities=dict(SOURCE=pm['model_identities']['SOURCE'],OLD_PCD=pm['model_identities']['TARGET_PCD'],SOURCE_US=identity)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],identities=identities,
        diagnostics=diagnostics,models_frozen=True,source_density_models_fitted=1,source_V_fit=False,novel_fit=False,target_Q_fit=False,controls_reused_exactly=True))
    def frozen_check():
        check()
        if not _same(states,frozen):raise ValueError('Frozen density statistics changed')
    def source(role):
        nonlocal source_rows
        parent=fresh._source_rows(c['source_control_config']['output_root'],'source_DURA',role);a=_controls(parent)
        iq=_iq_role(c,'V' if role=='known' else 'novel',pm)
        try:
            source_eval._align(iq,a,'V' if role=='known' else 'novel');ids=set(a['ids'].tolist())
            if ids&used:raise ValueError('Source fitting, V and NV must be disjoint')
            used.update(ids);z=source_eval._project(model,iq['restored'],device);source_rows+=len(z)
            return dict(a,signal_features=np.concatenate((a['signal_features'],z[:,None]),axis=1),raw_scores=np.column_stack((a['raw_scores'],_raw(states['SOURCE_US'],z,a['closed']))))
        finally:source_eval._close_role(iq)
    V=source('known');tau=np.empty((6,6),np.float64);tau[:4]=old_tau;reports=deepcopy(old_cal['methods'])
    for j in (4,5):
        fitted,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],V['raw_scores'][:,j],V['rx_i'],V['scenario']);tau[j]=fitted['conditional_thresholds'];reports[METHODS[j]]=report
    best._save(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),methods=reports,
        conditional_thresholds=tau.tolist(),primary_method=PRIMARY_METHOD,known_only=True,unknown_inputs_used=False,target_reads=0,source_V_threshold_fit=True,source_V_neural_fit=False,source_V_density_fit=False,group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES)))
    fixed_tau=tau.copy()
    def finish(a,role):
        a['scores']=a['raw_scores']-tau[:,a['closed']].T
        parent=fresh._source_rows(c['source_control_config']['output_root'],'source_DURA',role)
        a['scores'][:,:4]=parent['scores'][:,2:6];_validate(a,role);_exact_controls(a,parent)
        best._save(out/'source/source_DURA'/(role+'_predictions.npz'),a);frozen_check()
    finish(V,'known');N=source('novel');finish(N,'novel')
    manifests={};paired=None;query_rows=0
    for condition in CONDITIONS[1:]:
        iq=None
        try:
            manifest,ids,iq=_public(c,condition)
            if set(ids.tolist())&used:raise ValueError('Future Q overlaps source fitting or calibration')
            if paired is not None and not np.array_equal(ids,paired):raise ValueError('Clean/star canonical query order differs')
            paired=ids.copy();blocks=[]
            for start in range(0,len(ids),128):blocks.append(_forward(b,model,iq[start:start+128],device))
            current={k:np.concatenate([a[k] for a in blocks]) for k in blocks[0]};parent=fresh._target_rows(c['source_control_config']['output_root'],condition)
            if (not np.array_equal(ids,parent['ids']) or not np.array_equal(current['closed'],parent['closed'])
                    or not np.allclose(current['default_logits'],parent['default_logits'],rtol=2e-5,atol=1e-5)):raise ValueError('Same-input original F0 predictions differ from cached controls')
            a=_controls(parent);z=current['signal_features'];a['signal_features']=np.concatenate((a['signal_features'],z[:,None]),axis=1)
            a['raw_scores']=np.column_stack((a['raw_scores'],_raw(states['SOURCE_US'],z,a['closed'])));old_scores=a['scores'];a['scores']=a['raw_scores']-tau[:,a['closed']].T;a['scores'][:,:4]=old_scores
            _validate(a,'Q');_exact_controls(a,parent);best._save(out/'targets'/condition/'predictions.npz',a);manifests[condition]=manifest;query_rows+=len(ids);frozen_check()
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    if not np.array_equal(tau,fixed_tau):raise ValueError('Thresholds changed after NV/query access')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        reference_prediction_identity=pm,model_identities=identities,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],Q_manifests=manifests,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={n:Q_COUNT for n in CONDITIONS[1:]},source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_density_models_fitted=1,new_threshold_sets_fitted=2,threshold_values=36,threshold_bytes=tau.nbytes,models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,original_models_unchanged=True,frozen_parameters_buffers_unchanged=True,source_states_unchanged=True,source_control_readouts_reused_exactly=True,
        source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,prediction_optimizer_updates=0,
        classification_unchanged=True,identity_baseline_path='same_DURA_output_original_F0',private_truth_reads=0,U_truth_reads=0,target_U_reads=0,target_truth_reads=0,source_novel_truth_reads=0,
        source_IQ_reads=source_rows,target_Q_IQ_reads=query_rows,real_signal_forward_rows=source_rows+query_rows,real_F_forward_rows=query_rows,G_forward_rows=query_rows,
        synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],synthetic_signal_forward_rows=smoke['synthetic_signal_forward_rows'],G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],
        SOURCE_US_weights_source_only=True,SOURCE_US_original_Us_used=True,OLD_PCD_target_U_ancestor=576,new_target_UDA_updates=0,target_REG=False,optimizer_updates=0,
        source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',training_identity=identity,
        training_costs={k:v for k,v in training.items() if k.endswith('_rows') or k.endswith('_updates') or k.endswith('_exposures')})
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


def _prediction_gate(root):
    root=Path(root);done=_read(root/'complete.json');c=validate_config(done['config']);pm,old_states,old_tau,old_cal=_reference(c);_,identity=_training_metadata(c,pm)
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],reference_prediction_identity=pm,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,original_models_unchanged=True,
        source_states_unchanged=True,source_control_readouts_reused_exactly=True,classification_unchanged=True,private_truth_reads=0,target_U_reads=0,U_truth_reads=0,target_REG=False,new_target_UDA_updates=0)
    if Path(c['output_root']).resolve()!=root.resolve() or any(done.get(k)!=v for k,v in required.items()):raise ValueError('All six readouts and three frozen DURA paths required before truth')
    states={n:_npz(root/'models'/(n+'.npz')) for n in MODEL_NAMES}
    for name,state in states.items():
        density.validate_state(state,'density_ratio')
        if name in old_states and not _same(state,old_states[name]):raise ValueError('Control density changed')
    fit=_read(root/'models/complete.json');identities=dict(SOURCE=pm['model_identities']['SOURCE'],OLD_PCD=pm['model_identities']['TARGET_PCD'],SOURCE_US=identity)
    if (fit.get('status')!=MODELS_STATUS or fit.get('identities')!=identities or done.get('model_identities')!=identities or fit.get('models_frozen') is not True
            or fit.get('source_V_fit') is not False or fit.get('novel_fit') is not False or fit.get('target_Q_fit') is not False):raise ValueError('Frozen source-only fit identities required')
    threshold=_npz(root/'thresholds.npz');tau=threshold.get('conditional');cal=_read(root/'calibration/complete.json')
    if (set(threshold)!={'conditional'} or tau.shape!=(6,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or not np.array_equal(tau[:4],old_tau)
            or cal.get('status')!=CALIBRATION_STATUS or cal.get('conditional_thresholds')!=tau.tolist() or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False):raise ValueError('Six legal frozen known-only thresholds required')
    for j,name in enumerate(METHODS):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist() or (j<4 and cal['methods'][name]!=old_cal['methods'][name]):raise ValueError('Calibration routing/control changed')
    rows={'source':{role:_source_rows(root,'source_DURA',role) for role in ('known','novel')}}
    rows.update({n:_target_rows(root,n) for n in CONDITIONS[1:]})
    used=set()
    for role,a in [('known',rows['source']['known']),('novel',rows['source']['novel']),*[(n,rows[n]) for n in CONDITIONS[1:]]]:
        parent=fresh._source_rows(c['source_control_config']['output_root'],'source_DURA',role) if role in ('known','novel') else fresh._target_rows(c['source_control_config']['output_root'],role)
        _exact_controls(a,parent)
        raw=np.column_stack([_raw(states[n],a['signal_features'][:,k],a['closed']) for k,n in enumerate(MODEL_NAMES)])
        if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-8) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('Density raw scores or class-threshold routing changed')
        if role in ('known','novel'):
            ids=set(a['ids'].tolist())
            if used&ids:raise ValueError('Known/novel physical IDs overlap')
            used.update(ids)
        elif set(a['ids'].tolist())&used:raise ValueError('Source/query physical IDs overlap')
    if not np.array_equal(rows['clean_DURA']['ids'],rows['star_DURA']['ids']):raise ValueError('Paired future IDs differ')
    return done,cal,rows


run=evaluate


def main():
    print(json.dumps(dict(event='SOURCE_US_REPLAY_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
