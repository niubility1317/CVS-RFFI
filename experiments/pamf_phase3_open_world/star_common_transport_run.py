"""One frozen CCD student extends the six completed body-prototype readouts."""
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
from . import star_body_prototype_run as previous
from . import star_body_prototype_score as previous_score
from . import star_density_core as density

RUN_ID='20261031-phase3-common-transport-s2026092701-r01'
SCHEMA='pamf_phase3_common_transport_prediction_v1'
STATUS='COMMON_TRANSPORT_PREDICTIONS_COMPLETE'
MODELS_STATUS='COMMON_TRANSPORT_DENSITIES_FROZEN'
CALIBRATION_STATUS='COMMON_TRANSPORT_THRESHOLDS_FROZEN'
PARENT_CODE='4273e88392abe0717c1bd94fca1370d19c67cbb3'
METHODS=previous.METHODS+('CCD_pooled_ratio','CCD_predicted_ratio')
PRIMARY_INDEX=7
PRIMARY_METHOD=METHODS[PRIMARY_INDEX]
MODEL_NAMES=previous.MODEL_NAMES+('TARGET_CCD',)
CONDITIONS,COUNTS,Q_COUNT=previous.CONDITIONS,previous.COUNTS,previous.Q_COUNT
U_COUNT,U_CAPSULE_ID,U_SPLIT_ID=previous.U_COUNT,previous.U_CAPSULE_ID,previous.U_SPLIT_ID
FIELDS=previous.FIELDS
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=previous.KNOWN_FIELDS,previous.NOVEL_FIELDS,previous.TARGET_FIELDS
_read,_write,_same,_raw=previous._read,previous._write,previous._same,previous._raw
risk,source_eval=previous.risk,previous.source_eval
_save=previous.best._save


def _train():
    from . import star_common_transport_train
    return star_common_transport_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field common-transport evaluation required; no truth or role override')
    t=_train().validate_config(c['train_config']);p=previous.validate_config(c['reference_config'])
    if t['run_id']!=RUN_ID or t['code_commit']!=c['code_commit'] or t['reference_config']!=p or p['code_commit']!=PARENT_CODE:
        raise ValueError('Only the fixed completed body-r02 controls and actual CCD training may be used')
    out=Path(c['output_root']).resolve();base=p['reference_config']
    roots=[t['output_root'],t['U_public_root'],p['output_root'],p['train_config']['output_root'],base['output_root'],base['train_config']['IQ_root']]+[v['public_root'] for v in base['Q_conditions'].values()]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('New immutable output cannot overlap source, model or public inputs')
    return c


def _reference(c):
    """Validate frozen parent aggregates without opening its novel or Q rows."""
    p=c['reference_config'];root=Path(p['output_root']);m=_read(root/'complete.json')
    required=dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=previous.RUN_ID,code_commit=PARENT_CODE,config=p,
        method_names=list(previous.METHODS),primary_method=previous.PRIMARY_METHOD,known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,
        all_conditions_through_DURA=True,classification_unchanged=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,
        private_truth_reads=0,target_truth_reads=0,source_novel_truth_reads=0,U_truth_reads=0,target_U_reads=0)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Exact completed six-column body-r02 controls required')
    original,base_state,base_tau,base_cal=previous._reference(p)
    if (m['reference_prediction_identity']!=original or m['parent_F0_identity']!=original['parent_F0_identity']
            or m['dura_identity']!=original['dura_identity'] or m['loaded_dura_identity']!=original['loaded_dura_identity']):
        raise ValueError('Original source r03/F0/DURA ancestry must remain unchanged')
    fit=_read(root/'models/complete.json');cal=_read(root/'calibration/complete.json')
    if (fit.get('status')!=previous.MODELS_STATUS or fit.get('run_id')!=previous.RUN_ID or fit.get('code_commit')!=PARENT_CODE
            or fit.get('neural_states_frozen') is not True or fit.get('source_V_fit') is not False or fit.get('target_Q_fit') is not False
            or cal.get('status')!=previous.CALIBRATION_STATUS or cal.get('run_id')!=previous.RUN_ID or cal.get('code_commit')!=PARENT_CODE
            or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False or cal.get('target_reads')!=0):
        raise ValueError('Frozen parent densities and known-only calibration required')
    identities=previous_score._identities(p,m,fit,original);declared,Uids=previous._training_metadata(p,original)
    if declared!=identities:raise ValueError('Actual inherited training identities differ from parent feature axes')
    states={name:previous_score._arrays(root/'models'/(name+'.npz')) for name in previous.MODEL_NAMES}
    for state in states.values():density.validate_state(state,'density_ratio')
    if not _same(states['E0'],base_state):raise ValueError('Original source E0 density changed')
    saved=previous_score._arrays(root/'thresholds.npz');tau=saved.get('conditional')
    if (set(saved)!={'conditional'} or tau.dtype!=np.float64 or tau.shape!=(6,6) or not np.isfinite(tau).all()
            or not np.array_equal(tau[:2],base_tau) or cal.get('conditional_thresholds')!=tau.tolist() or set(cal.get('methods',{}))!=set(previous.METHODS)):
        raise ValueError('All thirty-six inherited source thresholds must remain exact')
    for j,name in enumerate(previous.METHODS):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist():raise ValueError('Parent threshold report differs')
        if j<2 and cal['methods'][name]!=base_cal['methods'][previous.parent.METHODS[(1,3)[j]]]:raise ValueError('Original E0 source risk reports differ')
    return m,states,tau,cal,Uids


def _identity(i,c,pm):
    base=pm['reference_prediction_identity']
    expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm='TARGET_CCD',updates=444,projection_updates=444,new_projection_updates=444,new_body_updates=444,
        projection_cumulative_updates=2664,signal_feature_cumulative_updates=13764,open_signal_updates=444,signal_encoder_updates=444,native_updates=0,Sinc_optimizer_steps=0,
        encoder_frozen=False,Sinc_frozen=True,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,trainable_parameters=334673,
        body_trainable_parameters=231985,trainable_projection_parameters=102688,input_path='cached_P1_IQ_only',source_only=False,target_U_used=True,target_U_reads=U_COUNT,
        CCD_weight=.1,CCD_start_step=101)
    if (any(i.get(k)!=v for k,v in expected.items()) or i.get('parent_signal_identity')!=base['model_identity']
            or i.get('parent_F0_identity')!=pm['parent_F0_identity'] or i.get('dura_identity')!=pm['dura_identity']):raise ValueError('Strict independent source-r03 CCD student identity required')
    return i


def _training_metadata(c,pm):
    root=Path(c['train_config']['output_root']);m=_read(root/'complete.json');i=m.get('identity',{})
    required=dict(schema='pamf_phase3_common_transport_train_v1',status='COMMON_TRANSPORT_TRAINING_COMPLETE',run_id=RUN_ID,code_commit=c['code_commit'],
        config=c['train_config'],arm='TARGET_CCD',updates=444,total_optimizer_updates=444,source_only=False,target_U_used=True,target_U_reads=U_COUNT,
        control_training_run_id=previous.TRAIN_RUN_ID,control_training_code_commit=previous.TRAIN_CODE,
        source_sampling_matched=True,U_sampling_matched=True,original_models_unchanged=True,
        all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,all_student_buffers_unchanged=True)
    if any(m.get(k)!=v for k,v in required.items()) or m.get('parent_old_control_prediction')!=pm:raise ValueError('Complete single CCD training, matched sampling and exact frozen-control ancestry required')
    _identity(i,c,pm);manifest=m.get('U_manifest',{})
    if any(manifest.get(k)!=v for k,v in dict(count=U_COUNT,capsule_id=c['train_config']['U_capsule_id'],split_id=c['train_config']['U_split_id']).items()):raise ValueError('Fixed star-U training identity required')
    with np.load(root/'U_features.npz',allow_pickle=False) as a:ids=a['ids'].copy()
    if ids.shape!=(U_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=U_COUNT or np.any(np.char.strip(ids)==''):raise ValueError('Complete opaque discovery IDs required')
    keys=('real_parent_signal_forward_rows','real_student_signal_forward_rows','G_forward_rows','real_F_forward_rows')
    costs={k:m[k] for k in keys}|{k:v for k,v in m.items() if k.startswith('synthetic_') or k=='G_synthetic_forward_rows'}
    return i,ids,costs


def _backend(c,device):return previous._backend(c['reference_config'],device)
def _load_model(c,device):return _train().load_trained_model(c['train_config'],device)
def _iq_role(c,role,header):return previous._iq_role(c['reference_config'],role,header['model_identities']['E0'])
def _public(c,name):return previous._public(c['reference_config'],name)


def _exact_controls(a,old):
    if set(a)!=set(old):raise ValueError('All inherited public prediction fields must be preserved')
    for key,v in old.items():
        actual=a[key][:,:6] if key in ('scores','raw_scores') else a[key][:,:3] if key=='signal_features' else a[key]
        if not np.array_equal(actual,v):raise ValueError('Six inherited readouts, three feature axes and original identity must remain exact')


def _validate(a,role):
    old=dict(a,signal_features=a['signal_features'][:,:3],raw_scores=a['raw_scores'][:,:6],scores=a['scores'][:,:6]);previous._validate(old,role)
    n=len(a['ids']);z=a['signal_features']
    if (z.shape!=(n,4,160) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1.,atol=2e-6,rtol=0)
            or any(a[k].shape!=(n,8) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):raise ValueError('Four unit160 feature axes and eight complete readouts required')
    return a


def _source_rows(root,condition,role):
    if condition!='source_DURA' or role not in ('known','novel'):raise ValueError('Fixed source condition and role required')
    return _validate(previous_score._arrays(Path(root)/'source'/condition/(role+'_predictions.npz')),role)


def _target_rows(root,condition):
    if condition not in CONDITIONS[1:]:raise ValueError('Fixed future condition required')
    return _validate(previous_score._arrays(Path(root)/'targets'/condition/'predictions.npz'),'Q')


def _forward(b,model,values,device,reference=False):
    values=np.asarray(values)
    if values.dtype!=np.float32 or values.shape!=(len(values),2,256) or not np.isfinite(values).all():raise ValueError('Public full256 FP32 IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        x=torch.tensor(values.tolist(),dtype=torch.float32,device=device);restored=b['dura'][1](x)[0].detach().clone()
        if restored.dtype!=torch.float32 or restored.shape!=x.shape or not bool(torch.isfinite(restored).all()):raise ValueError('One shared finite DURA output required')
        logits=b['reader'].classify_features(b['reader'].features(restored));z=model(restored)
        ref=b['model'](restored) if reference else None
    logits=np.asarray(logits.cpu().tolist(),np.float32);z=np.asarray(z.cpu().tolist(),np.float32)
    if logits.shape!=(len(values),6) or z.shape!=(len(values),160) or not np.isfinite(logits).all() or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6):raise ValueError('Actual original F0 and CCD unit160 outputs required')
    if ref is not None and (ref.shape!=(len(values),160) or not bool(torch.isfinite(ref).all())):raise ValueError('Actual original r03 smoke failed')
    return dict(default_logits=logits,closed=logits.argmax(1).astype(np.int64),signal_features=z)


def _smoke(b,model,device):
    from .star_open_feature_export import _synthetic
    x=np.asarray(_synthetic(device).cpu().tolist(),np.float32);_forward(b,model,x,device,reference=True)
    return dict(status='PASS',actual_F0_signal_DURA_checkpoints=True,actual_CCD_checkpoint=True,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,
        synthetic_parent_signal_forward_rows=2,synthetic_CCD_signal_forward_rows=2,real_data_reads=0,optimizer_updates=0)


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable common-transport output required')
    pm,states,old_tau,old_cal,old_Uids=_reference(c);b=_backend(c,device);model,identity=_load_model(c,device);_identity(identity,c,pm)
    if b['model_identity']!=pm['model_identities']['E0'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Actual original F0/r03/G must match inherited controls')
    if sum(p.numel() for p in model.parameters())!=334721 or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Actual full frozen CCD SignalOpen required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,previous.best._snapshot_model(m)) for m in (b['reader'],b['model'],b['dura'][0],model)]
    def check():
        for m,before in snapshots:previous.best._check_model(m,before)
    smoke=_smoke(b,model,device);check()
    declared,Uids,training_costs=_training_metadata(c,pm)
    if declared!=identity or not np.array_equal(Uids,old_Uids):raise ValueError('Actual CCD checkpoint and the inherited fixed U role must match')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    used=set(Uids.tolist());known=None;diag={}
    for role in ('L','OE'):
        iq=_iq_role(c,role,pm)
        try:
            ids=set(iq['ids'].tolist())
            if ids&used:raise ValueError('Source fit and target-U physical roles overlap')
            used.update(ids);z=source_eval._project(model,iq['restored'],device);m=iq['metadata']
            if role=='L':known,diag['known']=density.fit_known(z,m['y'],m['rx_i'],m['day_i'])
            else:states['TARGET_CCD'],diag['background']=density.fit_background(known,z,m['rx_i'],m['day_i'])
            check();print('COMMON_TRANSPORT_SOURCE_PROGRESS',role,len(ids),flush=True)
        finally:source_eval._close_role(iq)
    frozen=deepcopy(states);identities=dict(pm['model_identities'],TARGET_CCD=identity)
    for name,state in states.items():_save(out/'models'/(name+'.npz'),state)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],identities=identities,
        neural_states_frozen=True,source_fit_inputs_only=True,source_V_fit=False,novel_fit=False,target_Q_fit=False,source_density_models_fitted=1,
        normalizer_fit_role='L',background_fit_role='OE',control_readouts_reused_exactly=True,TARGET_CCD_has_target_U_ancestor=True,diagnostics=diag))
    def frozen_check():
        check()
        if not _same(states,frozen):raise ValueError('Frozen new or inherited density changed')
    def source(role):
        a=previous._source_rows(c['reference_config']['output_root'],'source_DURA',role)
        previous_score._validate_rows(a,len(a['ids']),KNOWN_FIELDS if role=='known' else NOVEL_FIELDS,{k:states[k] for k in previous.MODEL_NAMES},old_tau)
        iq=_iq_role(c,'V' if role=='known' else 'novel',pm)
        try:
            source_eval._align(iq,a,'V' if role=='known' else 'novel');ids=set(a['ids'].tolist())
            if ids&used:raise ValueError('Source validation, novel and fitting roles overlap')
            used.update(ids);z=source_eval._project(model,iq['restored'],device);raw=np.column_stack((a['raw_scores'],_raw(states['TARGET_CCD'],z,a['closed'])))
            print('COMMON_TRANSPORT_SOURCE_PROGRESS',role,len(ids),flush=True)
            return dict(a,signal_features=np.concatenate((a['signal_features'],z[:,None]),axis=1),raw_scores=raw)
        finally:source_eval._close_role(iq)
    V=source('known');tau=np.empty((8,6),np.float64);tau[:6]=old_tau;methods=deepcopy(old_cal['methods'])
    for j in (6,7):
        state,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],V['raw_scores'][:,j],V['rx_i'],V['scenario']);tau[j]=state['conditional_thresholds'];methods[METHODS[j]]=report
    _save(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        conditional_thresholds=tau.tolist(),methods=methods,group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),known_only=True,unknown_inputs_used=False,
        source_V_neural_fit=False,source_V_density_fit=False,target_reads=0,new_threshold_sets_fitted=2,control_readouts_reused_exactly=True))
    V['scores']=V['raw_scores']-tau[:,V['closed']].T;_validate(V,'known');_save(out/'source/source_DURA/known_predictions.npz',V);frozen_tau=tau.copy();frozen_check()
    print(CALIBRATION_STATUS,flush=True)
    N=source('novel');N['scores']=N['raw_scores']-tau[:,N['closed']].T;_validate(N,'novel');_save(out/'source/source_DURA/novel_predictions.npz',N);frozen_check()
    manifests={}
    for condition in CONDITIONS[1:]:
        old=previous._target_rows(c['reference_config']['output_root'],condition);iq=None
        try:
            manifest,ids,iq=_public(c,condition)
            if set(ids.tolist())&used or not np.array_equal(ids,old['ids']):raise ValueError('Future Q IDs must match controls and be disjoint from all fitting roles')
            blocks=[]
            for start in range(0,len(ids),128):blocks.append(_forward(b,model,iq[start:start+128],device));print('COMMON_TRANSPORT_Q_PROGRESS',condition,min(start+128,len(ids)),len(ids),flush=True)
            a={k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]}
            if not np.array_equal(a['closed'],old['closed']) or not np.allclose(a['default_logits'],old['default_logits'],rtol=2e-5,atol=1e-5):raise ValueError('Actual same-G F0 baseline differs from completed r02 controls')
            raw=np.column_stack((old['raw_scores'],_raw(states['TARGET_CCD'],a['signal_features'],old['closed'])))
            result=dict(old,signal_features=np.concatenate((old['signal_features'],a['signal_features'][:,None]),axis=1),raw_scores=raw,scores=raw-tau[:,old['closed']].T)
            _exact_controls(result,old);_validate(result,'Q');_save(out/'targets'/condition/'predictions.npz',result);manifests[condition]=manifest;frozen_check()
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Thresholds changed after novel/query access')
    real_signal=sum(COUNTS.values())+2*Q_COUNT
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        model_identities=identities,reference_prediction_identity=pm,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=b['dura_identity'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS[1:]},Q_manifests=manifests,
        source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,classification_unchanged=True,
        control_readouts_reused_exactly=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,source_density_models_fitted=1,new_threshold_sets_fitted=2,
        source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,private_truth_reads=0,target_U_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,original_U_s_reads=0,
        new_neural_updates=0,new_candidate_optimizer_updates=444,new_body_updates=444,new_projection_updates=444,Sinc_optimizer_steps=0,inherited_control_training_updates=888,
        source_IQ_reads=sum(COUNTS.values()),target_Q_IQ_reads=2*Q_COUNT,U_ID_metadata_reads=U_COUNT,training_target_U_count=U_COUNT,
        real_signal_forward_rows=real_signal,real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,old_control_real_signal_forward_rows=0,
        synthetic_parent_signal_forward_rows=2,synthetic_CCD_signal_forward_rows=2,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,
        training_costs=training_costs,combined_new_real_signal_forward_rows=training_costs['real_parent_signal_forward_rows']+training_costs['real_student_signal_forward_rows']+real_signal,
        combined_new_G_forward_rows=training_costs['G_forward_rows']+2*Q_COUNT,combined_new_real_F_forward_rows=training_costs['real_F_forward_rows']+2*Q_COUNT,
        threshold_values=48,threshold_bytes=384,new_threshold_values=12,reused_threshold_values=36,training_only_states_deployed=False,source_only=False,target_DA=True,target_REG=False,
        formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    frozen_check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


run=evaluate


def main():
    print(json.dumps(dict(event='COMMON_TRANSPORT_EVALUATION_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
