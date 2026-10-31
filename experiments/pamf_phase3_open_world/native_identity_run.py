"""Frozen native F0 identity students with the unchanged final PCD open branch."""
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
from . import receiver_input_run as base
from . import star_macro12_risk as risk

RUN_ID='20261031-phase3-native-identity-s2026092701-r01'
SCHEMA='pamf_phase3_native_identity_prediction_v1'
STATUS='NATIVE_IDENTITY_PREDICTIONS_COMPLETE'
MODELS_STATUS='NATIVE_IDENTITY_MODELS_FROZEN'
CALIBRATION_STATUS='NATIVE_IDENTITY_THRESHOLDS_FROZEN'
ARMS=('OLD_PCD','SOURCE_ID','TARGET_ID')
NEW_ARMS=ARMS[1:]
METHODS=tuple(a+'_'+m for a in ARMS for m in ('pooled_ratio','predicted_ratio'))
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=5
CONDITIONS=base.CONDITIONS
ROW_KEYS=base.parent.ROW_KEYS
COUNTS=dict(V=27000,novel=3150)
Q_COUNT=10238
U_COUNT=576
FIELDS={'run_id','code_commit','train_config','reference_config','output_root','batch_size'}
_read,_write,_same,_npz=base._read,base._write,base._same,base._npz
_save,_validate,_old=base._save,base._validate,base._old
_reference,_parent_rows,_public=base._reference,base._parent_rows,base._public


def _train():
    from . import native_identity_train
    return native_identity_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or c.get('batch_size')!=128 or type(c.get('batch_size')) is not int or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact native-identity evaluation fields required')
    r=base.parent.fresh.validate_config(c['reference_config']);t=_train().validate_config(c['train_config'])
    if t['reference_config']!=r or t['code_commit']!=c['code_commit']:raise ValueError('Same fresh PCD and current native training required')
    out=Path(c['output_root']).resolve()
    roots=[t['output_root'],r['output_root'],r['train_config']['output_root'],r['train_config']['U_public_root'],r['source_control_config']['reference_config']['train_config']['IQ_root']]
    roots.extend(v['public_root'] for v in r['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('New immutable prediction root overlaps inputs')
    return c


def _training(c,pm):
    t=_train();root=Path(c['train_config']['output_root']);done=_read(root/'complete.json')
    required=dict(schema=t.SCHEMA,status=t.STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c['train_config'],total_optimizer_updates=888,
        source_sampling_matched=True,SOURCE_completed_before_U_open=True,target_U_reads=576,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0,
        original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,final_phase3_complete=False)
    if any(done.get(k)!=v for k,v in required.items()) or set(done.get('identities',{}))!=set(NEW_ARMS):raise ValueError('Both independent native identity students must finish before prediction')
    for arm in NEW_ARMS:
        row=_read(root/arm/'complete.json');i=done['identities'][arm]
        if (row!=done.get('rows',{}).get(arm) or row.get('status')!=t.ROW_STATUS or row.get('identity')!=i or row.get('updates')!=444
                or i.get('parent_F0_identity')!=pm['parent_F0_identity'] or i.get('arm')!=arm):raise ValueError('Native identity ancestry or completed row differs')
    return done


def _rows(root,role,arm):
    if arm not in ARMS or role not in ('known','novel'):raise ValueError('Fixed native identity source arm/role required')
    return _validate(_npz(Path(root)/'source'/arm/(role+'_predictions.npz')),COUNTS['V' if role=='known' else 'novel'],role)


def _target_rows(root,condition,arm):
    if arm not in ARMS or condition not in CONDITIONS[1:]:raise ValueError('Fixed native identity future condition required')
    return _validate(_npz(Path(root)/'targets'/condition/arm/'predictions.npz'),Q_COUNT,'Q')


def _logits(model,values,device):
    blocks=[]
    for start in range(0,len(values),128):
        with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
            x=torch.tensor(np.asarray(values[start:start+128]).tolist(),device=device,dtype=torch.float32)
            v=_train().native_logits(model,x)
        blocks.append(np.asarray(v.detach().cpu().tolist(),np.float32))
    result=np.concatenate(blocks)
    if result.shape!=(len(values),6) or not np.isfinite(result).all():raise ValueError('Finite native default six-class logits required')
    return result


def _forward(b,pcd,models,values,device):
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        x=torch.tensor(np.asarray(values).tolist(),device=device,dtype=torch.float32);p1=b['dura'][1](x)[0].detach().clone()
        original=b['reader'].classify_features(b['reader'].features(p1));z=pcd(p1)
        own={a:_train().native_logits(m,p1) for a,m in models.items()}
    return np.asarray(original.detach().cpu().tolist(),np.float32),np.asarray(z.detach().cpu().tolist(),np.float32),{a:np.asarray(v.detach().cpu().tolist(),np.float32) for a,v in own.items()}


def _smoke(b,pcd,models,device):
    from .star_open_feature_export import _synthetic
    original=base.parent._smoke(b,pcd,device)
    logits,z,own=_forward(b,pcd,models,np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32),device)
    if original['status']!='PASS' or logits.shape!=(2,6) or z.shape!=(2,160) or any(v.shape!=(2,6) or not np.isfinite(v).all() for v in own.values()):raise ValueError('Actual trained native identities/F0/PCD/G no-query smoke failed')
    return dict(status='PASS',real_data_reads=0,optimizer_updates=0,synthetic_F_forward_rows=4,synthetic_identity_student_forward_rows=4,
        synthetic_PCD_forward_rows=4,synthetic_r03_forward_rows=2,G_synthetic_forward_rows=4)


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable native identity predictions required')
    pm,state,old_tau,old_cal=_reference(c);training=_training(c,pm);b=base._backend(c,device);pcd,pcd_identity=base._load_model(c,device)
    models={}
    if (pcd_identity!=pm['model_identities']['TARGET_PCD'] or b['reader_identity']!=pm['parent_F0_identity']
            or b['dura_identity']!=pm['loaded_dura_identity']):raise ValueError('Original F0/final PCD/DURA identities differ')
    for arm in NEW_ARMS:
        models[arm],i=_train().load_trained_model(c['train_config'],arm,device)
        if i!=training['identities'][arm] or any(p.requires_grad for p in models[arm].parameters()):raise ValueError('Frozen trained native identity student required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,base.parent.fresh.best._snapshot_model(m)) for m in (b['reader'],b['model'],b['dura'][0],pcd,*models.values())];fixed=deepcopy(state)
    def check():
        for m,before in snapshots:base.parent.fresh.best._check_model(m,before)
        if not _same(state,fixed):raise ValueError('Frozen PCD source density changed')
    smoke=_smoke(b,pcd,models,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    _save(out/'models/density.npz',state)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],identities=training['identities'],
        parent_PCD_identity=pcd_identity,neural_states_frozen=True,new_density_fits=0,source_V_fit=False,novel_fit=False,target_Q_fit=False))
    source_ids=set()
    def source(role):
        old=_old(_parent_rows(c,'source_V' if role=='known' else 'source_NV'));rows={'OLD_PCD':old}
        iq=base.parent.fresh._iq_role(c['reference_config'],'V' if role=='known' else 'novel',pm)
        try:
            base.parent.fresh.source_eval._align(iq,old,'V' if role=='known' else 'novel')
            ids=set(old['ids'].tolist())
            if ids&source_ids:raise ValueError('Known calibration and novel roles overlap')
            source_ids.update(ids)
            for arm in NEW_ARMS:
                a=deepcopy(old);a['default_logits']=_logits(models[arm],iq['restored'],device);a['closed']=a['default_logits'].argmax(1).astype(np.int64)
                a['raw_scores']=base.parent.fresh._raw(state,a['signal_features'],a['closed']);rows[arm]=a
            check();return rows
        finally:base.parent.fresh.source_eval._close_role(iq)
    known=source('known');tau=np.empty((6,6),np.float64);tau[:2]=old_tau
    reports={METHODS[j]:deepcopy(old_cal['methods'][base.parent.fresh.METHODS[4+j]]) for j in range(2)}
    for index,arm in enumerate(NEW_ARMS,1):
        a=known[arm]
        for j in range(2):
            fit,report=risk.calibrate_known(a['ids'],a['y'],a['closed'],a['baseline_closed'],a['raw_scores'][:,j],a['rx_i'],a['scenario'])
            tau[2*index+j]=fit['conditional_thresholds'];reports[METHODS[2*index+j]]=report
    fixed_tau=tau.copy();_save(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),
        methods=reports,conditional_thresholds=tau.tolist(),known_only=True,unknown_inputs_used=False,new_threshold_sets_fitted=4,source_V_threshold_fit=True))
    def save_source(rows,role):
        for j,arm in enumerate(ARMS):
            a=rows[arm];a['scores']=a['raw_scores']-tau[2*j:2*j+2,a['closed']].T;_validate(a,COUNTS['V' if role=='known' else 'novel'],role)
            _save(out/'source'/arm/(role+'_predictions.npz'),a)
    save_source(known,'known');save_source(source('novel'),'novel')
    manifests={};paired=None
    for condition in CONDITIONS[1:]:
        old=_old(_parent_rows(c,'Q_'+condition));manifest,ids,iq=_public(c,condition)
        try:
            if not np.array_equal(ids,old['ids']) or set(ids.tolist())&source_ids:raise ValueError('Exact future IDs and disjoint source roles required')
            if paired is not None and not np.array_equal(ids,paired):raise ValueError('Paired clean/star future order differs')
            paired=ids.copy();blocks={arm:[] for arm in NEW_ARMS}
            for start in range(0,len(ids),128):
                original,z,own=_forward(b,pcd,models,iq[start:start+128],device);stop=start+len(original)
                if (not np.allclose(original,old['baseline_logits'][start:stop],rtol=1e-5,atol=2e-5) or not np.array_equal(original.argmax(1),old['baseline_closed'][start:stop])
                        or not np.allclose(z,old['signal_features'][start:stop],rtol=1e-5,atol=2e-6)):raise ValueError('Same-DURA original identity or frozen PCD control features differ')
                for arm,v in own.items():blocks[arm].append(v)
            rows={'OLD_PCD':old}
            for arm in NEW_ARMS:
                a=deepcopy(old);a['default_logits']=np.concatenate(blocks[arm]);a['closed']=a['default_logits'].argmax(1).astype(np.int64)
                a['raw_scores']=base.parent.fresh._raw(state,a['signal_features'],a['closed']);rows[arm]=a
            for j,arm in enumerate(ARMS):
                a=rows[arm];a['scores']=a['raw_scores']-tau[2*j:2*j+2,a['closed']].T;_validate(a,Q_COUNT,'Q');_save(out/'targets'/condition/arm/'predictions.npz',a)
            manifests[condition]=manifest;check();print('NATIVE_IDENTITY_Q_COMPLETE',condition,len(ids),flush=True)
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    if not np.array_equal(tau,fixed_tau):raise ValueError('Source thresholds changed after unknown/query access')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,arm_order=list(ARMS),conditions=list(CONDITIONS),
        reference_prediction_identity=pm,training_identity=training,student_identities=training['identities'],parent_F0_identity=pm['parent_F0_identity'],parent_PCD_identity=pcd_identity,dura_identity=pm['dura_identity'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={n:Q_COUNT for n in CONDITIONS[1:]},Q_manifests=manifests,models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,controls_reused_exactly=True,classification_unchanged=False,identity_baseline_path='same_DURA_original_F0',
        new_density_fits=0,new_threshold_sets_fitted=4,new_threshold_values=24,reused_threshold_values=12,threshold_values=36,threshold_bytes=288,
        private_truth_reads=0,U_truth_reads=0,target_U_reads=0,target_truth_reads=0,source_novel_truth_reads=0,source_IQ_reads=sum(COUNTS.values()),target_Q_IQ_reads=2*Q_COUNT,
        real_identity_student_forward_rows=2*sum(COUNTS.values())+4*Q_COUNT,real_F_forward_rows=2*Q_COUNT,real_PCD_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,
        prediction_optimizer_updates=0,new_student_optimizer_updates=888,new_target_UDA_updates=444,original_F0_unchanged=True,frozen_PCD_unchanged=True,
        student_identity_parameters=232993,trainable_identity_parameters=232945,primary_deployed_neural_parameters=834665,PCD_U_ancestor=576,SOURCE_ID_system_source_only=False,
        source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,target_REG=False,full_goal_status='ACTIVE',final_phase3_complete=False,**{k:v for k,v in smoke.items() if k.startswith('synthetic_') or k=='G_synthetic_forward_rows'})
    check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


def _prediction_gate(root):
    root=Path(root);done=_read(root/'complete.json');c=validate_config(done['config']);pm,state,old_tau,old_cal=_reference(c);training=_training(c,pm)
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,arm_order=list(ARMS),conditions=list(CONDITIONS),
        reference_prediction_identity=pm,training_identity=training,student_identities=training['identities'],parent_PCD_identity=pm['model_identities']['TARGET_PCD'],models_frozen_before_V=True,
        all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,controls_reused_exactly=True,
        classification_unchanged=False,new_density_fits=0,new_threshold_sets_fitted=4,private_truth_reads=0,U_truth_reads=0,target_U_reads=0,prediction_optimizer_updates=0,final_phase3_complete=False)
    if Path(c['output_root']).resolve()!=root.resolve() or any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Complete six-readout/three-condition native identity predictions required')
    fit=_read(root/'models/complete.json');cal=_read(root/'calibration/complete.json');saved=_npz(root/'thresholds.npz');tau=saved.get('conditional')
    headers=dict(schema=SCHEMA,run_id=RUN_ID,code_commit=c['code_commit'])
    if (any(fit.get(k)!=v or cal.get(k)!=v for k,v in headers.items()) or not _same(_npz(root/'models/density.npz'),state)
            or fit.get('status')!=MODELS_STATUS or fit.get('identities')!=training['identities'] or fit.get('parent_PCD_identity')!=done['parent_PCD_identity']
            or fit.get('neural_states_frozen') is not True or fit.get('new_density_fits')!=0 or any(fit.get(k) is not False for k in ('source_V_fit','novel_fit','target_Q_fit'))):raise ValueError('Frozen inherited PCD density or student identities differ')
    if (set(saved)!={'conditional'} or tau.dtype!=np.float64 or tau.shape!=(6,6) or not np.isfinite(tau).all() or not np.array_equal(tau[:2],old_tau)
            or cal.get('status')!=CALIBRATION_STATUS or cal.get('method_names')!=list(METHODS) or cal.get('conditional_thresholds')!=tau.tolist()
            or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False or cal.get('new_threshold_sets_fitted')!=4 or set(cal.get('methods',{}))!=set(METHODS)
            or any(cal['methods'][n].get('conditional_thresholds')!=tau[j].tolist() for j,n in enumerate(METHODS))):
        raise ValueError('Source-known calibrated native routing thresholds required')
    rows={}
    for key in ROW_KEYS:
        rows[key]={a:_rows(root,'known' if key=='source_V' else 'novel',a) if key.startswith('source_') else _target_rows(root,key[2:],a) for a in ARMS}
        old=_old(_parent_rows(c,key))
        if not _same(rows[key]['OLD_PCD'],old):raise ValueError('Exact frozen PCD control rows changed')
        for j,arm in enumerate(ARMS):
            a=rows[key][arm]
            if any(not np.array_equal(a[k],old[k]) for k in ('ids','baseline_logits','baseline_closed','signal_features')):raise ValueError('Same original baseline and unchanged PCD feature rows required')
            raw=base.parent.fresh._raw(state,a['signal_features'],a['closed']);scores=raw-tau[2*j:2*j+2,a['closed']].T
            if not np.allclose(a['raw_scores'],raw,rtol=0,atol=2e-6) or not np.allclose(a['scores'],scores,rtol=0,atol=2e-6):raise ValueError('Own native identity must route PCD density and source thresholds')
    return done,state,tau,cal,rows


def main():
    print(json.dumps(dict(event='NATIVE_IDENTITY_PREDICT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
