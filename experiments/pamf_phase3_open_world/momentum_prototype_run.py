"""Frozen student evaluation after label-free fixed/EMA teacher adaptation."""
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
from . import source_us_run as control

RUN_ID='20261031-phase3-momentum-prototype-s2026092701-r01'
SCHEMA='pamf_phase3_momentum_prototype_prediction_v1'
STATUS='MOMENTUM_PROTOTYPE_PREDICTIONS_COMPLETE'
MODELS_STATUS='MOMENTUM_PROTOTYPE_DENSITIES_FROZEN'
CALIBRATION_STATUS='MOMENTUM_PROTOTYPE_THRESHOLDS_FROZEN'
MODEL_NAMES=('SOURCE','OLD_PCD','FIXED_TEACHER','EMA_TEACHER')
ARMS=MODEL_NAMES[2:]
METHODS=tuple(a+'_'+b for a in MODEL_NAMES for b in ('pooled_ratio','predicted_ratio'))
PRIMARY_METHOD=METHODS[-1]
CONDITIONS=control.CONDITIONS
COUNTS=dict(control.COUNTS)
Q_COUNT=control.Q_COUNT
FIELDS=set(control.FIELDS)
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=control.KNOWN_FIELDS,control.NOVEL_FIELDS,control.TARGET_FIELDS
fresh,best,density,risk,source_eval=control.fresh,control.best,control.density,control.risk,control.source_eval
_read,_write,_same=control._read,control._write,control._same
_npz,_reference,_controls,_exact_controls=control._npz,control._reference,control._controls,control._exact_controls
_backend,_iq_role,_public,_raw=control._backend,control._iq_role,control._public,control._raw


def _train():
    from . import momentum_prototype_train
    return momentum_prototype_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact public-only momentum evaluation configuration required')
    t=_train().validate_config(c['train_config']);p=fresh.validate_config(c['source_control_config'])
    if (t['run_id']!=RUN_ID or t['code_commit']!=c['code_commit'] or t['reference_config']!=p
            or p['code_commit']!=control.REFERENCE_CODE or c['Q_conditions']!=p['Q_conditions']):
        raise ValueError('Actual two-arm training, original controls and paired public future handles required')
    out=Path(c['output_root']).resolve()
    roots=[t['output_root'],p['output_root'],p['train_config']['output_root'],p['train_config']['U_public_root'],
           p['source_control_config']['reference_config']['train_config']['IQ_root']]+[v['public_root']for v in c['Q_conditions'].values()]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out)for v in roots):
        raise ValueError('Immutable prediction output overlaps existing model/data inputs')
    return c


def _identity(i,arm,c,pm):
    expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,updates=444,signal_parameters=334721,
                  Sinc_frozen=True,source_only=False,target_U_used=True,target_U_reads=576)
    if (any(i.get(k)!=v for k,v in expected.items()) or i.get('parent_signal_identity')!=pm['model_identities']['E0']
            or i.get('parent_F0_identity')!=pm['parent_F0_identity'] or i.get('dura_identity')!=pm['dura_identity']):
        raise ValueError('Independent original-r03/target-U576 student ancestry required')
    return i


def _training_metadata(c,pm):
    t=c['train_config'];m=_read(Path(t['output_root'])/'complete.json')
    expected=dict(status=_train().STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=t,arms=list(ARMS),
        updates_per_arm=444,total_optimizer_updates=888,target_U_reads=576,source_control_source_only=True,
        source_control_frozen_before_training=True,source_sampling_matched=True,original_models_unchanged=True,
        all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,
        target_Q_reads=0,private_truth_reads=0,U_truth_reads=0,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()) or m.get('source_control_identity')!=pm['model_identities']['SOURCE']:
        raise ValueError('Both complete source-matched training arms must precede inference')
    identities={arm:_identity(m['identities'][arm],arm,c,pm) for arm in ARMS}
    return m,identities


def _validate(a,role):
    n=COUNTS['V'if role=='known'else'novel']if role in('known','novel')else Q_COUNT
    if role in('known','novel'):
        source_eval._validate_rows({k:(v[:,0]if k=='signal_features'else v)for k,v in a.items()if k!='raw_signal_features'},role,len(METHODS),True)
        if a['raw_signal_features'].shape!=(n,160) or a['raw_signal_features'].dtype!=np.float32 or not np.isfinite(a['raw_signal_features']).all():
            raise ValueError('Original source raw features must remain unchanged')
    elif (set(a)!=TARGET_FIELDS or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n or np.any(np.char.strip(a['ids'])=='')
            or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
            or a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):
        raise ValueError('Complete same-DURA original F0 query predictions required')
    z=a['signal_features']
    if (z.shape!=(n,4,160) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1,rtol=0,atol=2e-6)
            or any(a[k].shape!=(n,8) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all()for k in('raw_scores','scores'))):
        raise ValueError('Four frozen student coordinates and eight finite readouts required')
    return a


def _source_rows(root,condition,role):
    if condition!='source_DURA'or role not in('known','novel'):raise ValueError('Fixed source role required')
    return _validate(_npz(Path(root)/'source'/condition/(role+'_predictions.npz')),role)


def _target_rows(root,condition):
    if condition not in CONDITIONS[1:]:raise ValueError('Fixed paired future condition required')
    return _validate(_npz(Path(root)/'targets'/condition/'predictions.npz'),'Q')


def _forward(b,models,values,device):
    x=np.asarray(values)
    if x.dtype!=np.float32 or x.shape!=(len(x),2,256) or not np.isfinite(x).all():raise ValueError('Finite public full256 IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        iq=torch.tensor(x.tolist(),dtype=torch.float32,device=device);p=b['dura'][1](iq)[0].detach().clone()
        logits=b['reader'].classify_features(b['reader'].features(p));z=[models[arm](p)for arm in ARMS]
    logits=np.asarray(logits.cpu().tolist(),np.float32);z=np.stack([np.asarray(v.cpu().tolist(),np.float32)for v in z],axis=1)
    if logits.shape!=(len(x),6)or z.shape!=(len(x),2,160)or not np.isfinite(logits).all()or not np.isfinite(z).all():raise ValueError('Same restored input must reach original F0 and both new students')
    closed=logits.argmax(1).astype(np.int64)
    return dict(default_logits=logits,closed=closed,baseline_closed=closed.copy(),signal_features=z)


def _smoke(b,models,device):
    from .star_open_feature_export import _synthetic
    x=np.asarray(_synthetic(device).cpu().tolist(),np.float32);_forward(b,models,x,device)
    return dict(status='PASS',actual_F0_two_students_DURA_checkpoints=True,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=4,G_synthetic_forward_rows=2,real_data_reads=0)


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable prediction root required')
    pm,states,old_tau,old_cal=_reference(c);training,declared=_training_metadata(c,pm);b=_backend(c,device);models={};new_ids={}
    for arm in ARMS:
        model,identity=_train().load_trained_model(c['train_config'],arm,device);_identity(identity,arm,c,pm)
        if identity!=declared[arm]or sum(p.numel()for p in model.parameters())!=334721 or any(p.requires_grad for p in model.parameters())or any(m.training for m in model.modules()):raise ValueError('Frozen complete deployable student required')
        models[arm]=model;new_ids[arm]=identity
    if b['model_identity']!=pm['model_identities']['E0']or b['reader_identity']!=pm['parent_F0_identity']or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Actual original r03/F0/DURA ancestors differ')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,best._snapshot_model(m))for m in(b['reader'],b['model'],b['dura'][0],*models.values())]
    def check():
        for m,s in snapshots:best._check_model(m,s)
    smoke=_smoke(b,models,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    used=set();source_rows=0;diagnostics={}
    for role in('L','OE'):
        a=_iq_role(c,role,pm)
        try:
            ids=set(a['ids'].tolist())
            if ids&used:raise ValueError('Source fit roles overlap')
            used.update(ids);meta=a['metadata']
            for arm,model in models.items():
                z=source_eval._project(model,a['restored'],device);source_rows+=len(z)
                if role=='L':states[arm],diagnostics[arm]=density.fit_known(z,meta['y'],meta['rx_i'],meta['day_i'])
                else:states[arm],diagnostics[arm]['background']=density.fit_background(states[arm],z,meta['rx_i'],meta['day_i'])
                check()
        finally:source_eval._close_role(a)
    frozen=deepcopy(states)
    identities=dict(SOURCE=pm['model_identities']['SOURCE'],OLD_PCD=pm['model_identities']['TARGET_PCD'],**new_ids)
    for arm,state in states.items():best._save(out/'models'/(arm+'.npz'),state)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],identities=identities,diagnostics=diagnostics,
        models_frozen=True,source_density_models_fitted=2,source_V_fit=False,novel_fit=False,target_Q_fit=False,controls_reused_exactly=True))
    def frozen_check():
        check()
        if not _same(states,frozen):raise ValueError('Frozen density changed')
    def source(role):
        nonlocal source_rows
        old=fresh._source_rows(c['source_control_config']['output_root'],'source_DURA',role);a=_controls(old);iq=_iq_role(c,'V'if role=='known'else'novel',pm)
        try:
            source_eval._align(iq,a,'V'if role=='known'else'novel');ids=set(a['ids'].tolist())
            if ids&used:raise ValueError('Source fit/calibration/novel roles overlap')
            used.update(ids);values=[];raw=[]
            for arm,model in models.items():
                z=source_eval._project(model,iq['restored'],device);source_rows+=len(z);values.append(z);raw.append(_raw(states[arm],z,a['closed']))
            return dict(a,signal_features=np.concatenate((a['signal_features'],np.stack(values,axis=1)),axis=1),raw_scores=np.column_stack((a['raw_scores'],*raw)))
        finally:source_eval._close_role(iq)
    V=source('known');tau=np.empty((8,6),np.float64);tau[:4]=old_tau;reports=deepcopy(old_cal['methods'])
    for j in range(4,8):
        fitted,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],V['raw_scores'][:,j],V['rx_i'],V['scenario']);tau[j]=fitted['conditional_thresholds'];reports[METHODS[j]]=report
    best._save(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),methods=reports,
        conditional_thresholds=tau.tolist(),primary_method=PRIMARY_METHOD,known_only=True,unknown_inputs_used=False,target_reads=0,source_V_threshold_fit=True,source_V_neural_fit=False,
        source_V_density_fit=False,group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES)))
    fixed_tau=tau.copy()
    def finish(a,role):
        old=fresh._source_rows(c['source_control_config']['output_root'],'source_DURA',role);a['scores']=a['raw_scores']-tau[:,a['closed']].T;a['scores'][:,:4]=old['scores'][:,2:6]
        _validate(a,role);_exact_controls(a,old);best._save(out/'source/source_DURA'/(role+'_predictions.npz'),a);frozen_check()
    finish(V,'known');finish(source('novel'),'novel')
    with np.load(Path(c['source_control_config']['train_config']['output_root'])/'U_features.npz',allow_pickle=False)as f:u_ids=set(f['ids'].tolist())
    manifests={};paired=None;query_rows=0
    for condition in CONDITIONS[1:]:
        iq=None
        try:
            manifest,ids,iq=_public(c,condition)
            if set(ids.tolist())&(used|u_ids):raise ValueError('Future Q overlaps source or discovery U')
            if paired is not None and not np.array_equal(ids,paired):raise ValueError('Paired clean/star physical order differs')
            paired=ids.copy();blocks=[_forward(b,models,iq[start:start+128],device)for start in range(0,len(ids),128)]
            current={k:np.concatenate([v[k]for v in blocks])for k in blocks[0]};old=fresh._target_rows(c['source_control_config']['output_root'],condition)
            if not np.array_equal(ids,old['ids'])or not np.array_equal(current['closed'],old['closed'])or not np.allclose(current['default_logits'],old['default_logits'],rtol=2e-5,atol=1e-5):raise ValueError('Same-input original F0 predictions differ from frozen controls')
            a=_controls(old);z=current['signal_features'];a['signal_features']=np.concatenate((a['signal_features'],z),axis=1)
            a['raw_scores']=np.column_stack((a['raw_scores'],*[_raw(states[arm],z[:,j],a['closed'])for j,arm in enumerate(ARMS)]));prior=a['scores'];a['scores']=a['raw_scores']-tau[:,a['closed']].T;a['scores'][:,:4]=prior
            _validate(a,'Q');_exact_controls(a,old);best._save(out/'targets'/condition/'predictions.npz',a);manifests[condition]=manifest;query_rows+=len(ids);frozen_check()
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    if not np.array_equal(tau,fixed_tau):raise ValueError('Thresholds changed after novel/query access')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        reference_prediction_identity=pm,model_identities=identities,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],Q_manifests=manifests,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={n:Q_COUNT for n in CONDITIONS[1:]},source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_density_models_fitted=2,new_threshold_sets_fitted=4,threshold_values=48,threshold_bytes=tau.nbytes,models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,original_models_unchanged=True,frozen_parameters_buffers_unchanged=True,source_states_unchanged=True,
        source_control_readouts_reused_exactly=True,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,prediction_optimizer_updates=0,classification_unchanged=True,
        identity_baseline_path='same_DURA_output_original_F0',private_truth_reads=0,U_truth_reads=0,target_U_reads=0,target_truth_reads=0,source_novel_truth_reads=0,
        real_signal_forward_rows=source_rows+2*query_rows,real_F_forward_rows=query_rows,G_forward_rows=query_rows,source_IQ_forward_rows=source_rows,target_Q_IQ_reads=query_rows,
        synthetic_F_forward_rows=2,synthetic_signal_forward_rows=4,G_synthetic_forward_rows=2,new_target_UDA_updates=888,target_REG=False,optimizer_updates=0,
        auxiliary_teacher_deployed=False,prototype_deployed=False,source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',
        training_identities=new_ids,training_costs={k:v for k,v in training.items()if k.endswith('_rows')or k.endswith('_updates')or k.endswith('_exposures')})
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


def _prediction_gate(root):
    root=Path(root);done=_read(root/'complete.json');c=validate_config(done['config']);pm,old_states,old_tau,old_cal=_reference(c);_,new_ids=_training_metadata(c,pm)
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],reference_prediction_identity=pm,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,original_models_unchanged=True,
        source_states_unchanged=True,source_control_readouts_reused_exactly=True,classification_unchanged=True,private_truth_reads=0,target_U_reads=0,U_truth_reads=0,target_REG=False,new_target_UDA_updates=888)
    if Path(c['output_root']).resolve()!=root.resolve()or any(done.get(k)!=v for k,v in required.items()):raise ValueError('All eight readouts and three DURA conditions required before truth')
    states={arm:_npz(root/'models'/(arm+'.npz'))for arm in MODEL_NAMES}
    for arm,state in states.items():
        density.validate_state(state,'density_ratio')
        if arm in old_states and not _same(state,old_states[arm]):raise ValueError('Control density changed')
    fit=_read(root/'models/complete.json');identities=dict(SOURCE=pm['model_identities']['SOURCE'],OLD_PCD=pm['model_identities']['TARGET_PCD'],**new_ids)
    if fit.get('status')!=MODELS_STATUS or fit.get('identities')!=identities or done.get('model_identities')!=identities or fit.get('models_frozen')is not True or fit.get('source_V_fit')is not False or fit.get('novel_fit')is not False or fit.get('target_Q_fit')is not False:raise ValueError('Frozen source-only density fit identities differ')
    threshold=_npz(root/'thresholds.npz');tau=threshold.get('conditional');cal=_read(root/'calibration/complete.json')
    if set(threshold)!={'conditional'}or tau.shape!=(8,6)or tau.dtype!=np.float64 or not np.isfinite(tau).all()or not np.array_equal(tau[:4],old_tau)or cal.get('status')!=CALIBRATION_STATUS or cal.get('conditional_thresholds')!=tau.tolist()or cal.get('known_only')is not True or cal.get('unknown_inputs_used')is not False:raise ValueError('Eight legal known-only threshold vectors required')
    for j,name in enumerate(METHODS):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist()or(j<4 and cal['methods'][name]!=old_cal['methods'][name]):raise ValueError('Calibration routing/control differs')
    rows={'source':{role:_source_rows(root,'source_DURA',role)for role in('known','novel')}};rows.update({n:_target_rows(root,n)for n in CONDITIONS[1:]});used=set()
    for role,a in[('known',rows['source']['known']),('novel',rows['source']['novel']),*[(n,rows[n])for n in CONDITIONS[1:]]]:
        old=fresh._source_rows(c['source_control_config']['output_root'],'source_DURA',role)if role in('known','novel')else fresh._target_rows(c['source_control_config']['output_root'],role);_exact_controls(a,old)
        raw=np.column_stack([_raw(states[arm],a['signal_features'][:,j],a['closed'])for j,arm in enumerate(MODEL_NAMES)])
        if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-8)or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('Reconstructed density/threshold readout differs')
        if role in('known','novel'):
            ids=set(a['ids'].tolist())
            if used&ids:raise ValueError('Source known/novel overlap')
            used.update(ids)
        elif set(a['ids'].tolist())&used:raise ValueError('Source/future overlap')
    if not np.array_equal(rows['clean_DURA']['ids'],rows['star_DURA']['ids']):raise ValueError('Clean/star future IDs differ')
    return done,cal,rows


run=evaluate
def main():
    print(json.dumps(dict(event='MOMENTUM_PROTOTYPE_PREDICT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(_read(a.config),a.device)
if __name__=='__main__':main()
