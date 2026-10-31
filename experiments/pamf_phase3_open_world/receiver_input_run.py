"""Frozen receiver IQ adapters, own F0 identity, and inherited PCD readouts."""
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
from . import pcd_cluster_registry_run as parent
from . import star_consensus_score as common

RUN_ID='20261031-phase3-receiver-input-s2026092701-r01'
SCHEMA='pamf_phase3_receiver_input_prediction_v1'
STATUS='RECEIVER_INPUT_PREDICTIONS_COMPLETE'
MODELS_STATUS='RECEIVER_INPUT_MODELS_FROZEN'
CALIBRATION_STATUS='RECEIVER_INPUT_THRESHOLDS_FROZEN'
ARMS=('OLD_PCD','SOURCE_INPUT','TARGET_INPUT')
METHODS=tuple(a+'_'+m for a in ARMS for m in ('pooled_ratio','predicted_ratio'))
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=5
FIELDS={'run_id','code_commit','train_config','reference_config','output_root','batch_size'}
COUNTS=dict(parent.fresh.COUNTS)
SOURCE_COUNT=6300
U_COUNT=576
Q_COUNT=parent.Q_COUNT
CONDITIONS=parent.fresh.CONDITIONS
ROW_KEYS=parent.ROW_KEYS
BASE_FIELDS={'ids','default_logits','closed','baseline_logits','baseline_closed','signal_features','raw_scores','scores'}
_read,_write,_same,_npz=parent._read,parent._write,parent._same,parent._npz
_save=parent.fresh.best._save
_backend,_load_model=parent._backend,parent._load_model
_reference=parent._reference


def _train():
    from . import receiver_input_train
    return receiver_input_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field receiver-input configuration required; no truth or query fitting')
    r=parent.fresh.validate_config(c['reference_config']);t=_train().validate_config(c['train_config'])
    if (r['code_commit']!=parent.REFERENCE_CODE or t['run_id']!=RUN_ID or t['code_commit']!=c['code_commit']
            or t['reference_config']!=r['train_config'] or t['reference_prediction_root']!=r['output_root']):raise ValueError('Exact fresh PCD training and prediction reference required')
    out=Path(c['output_root']).resolve();roots=[r['output_root'],t['output_root'],r['train_config']['output_root'],r['train_config']['U_public_root'],
        r['source_control_config']['output_root'],r['source_control_config']['reference_config']['train_config']['IQ_root']]+[v['public_root'] for v in r['Q_conditions'].values()]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable adapter output overlaps model/source/public inputs')
    return c


def _training(c,pm):
    t=_train();done=_read(Path(c['train_config']['output_root'])/'complete.json')
    if (done.get('schema')!=t.SCHEMA or done.get('status')!=t.STATUS or done.get('run_id')!=RUN_ID or done.get('code_commit')!=c['code_commit']
            or done.get('config')!=c['train_config'] or set(done.get('identities',{}))!=set(ARMS[1:])
            or any(done.get(k) is not True for k in ('source_completed_before_U_open','source_sampling_matched','original_models_unchanged','all_adapters_changed'))
            or done.get('NN_optimizer_updates')!=0 or done.get('total_optimizer_updates')!=2*t.STEPS):raise ValueError('Both completed receiver adapter checkpoints required')
    for arm in ARMS[1:]:
        i=done['identities'][arm];target=arm=='TARGET_INPUT'
        expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,updates=t.STEPS,adapter_parameters=16,adapter_source_only=not target,
            target_U_used=target,target_U_reads=U_COUNT if target else 0,parent_F0_identity=pm['parent_F0_identity'],parent_PCD_identity=pm['model_identities']['TARGET_PCD'],
            dura_identity=pm['dura_identity'],system_PCD_has_target_U_ancestor=True)
        if any(i.get(k)!=v for k,v in expected.items()):raise ValueError('Actual source/target input-adapter ancestry differs')
    return done


def _features(z,n):
    if (z.dtype!=np.float32 or z.shape!=(n,160) or not np.isfinite(z).all()
            or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)):raise ValueError('Finite frozen PCD FP32 unit160 required')


def _validate(a,n,role):
    fields=BASE_FIELDS|({'y','rx_i','day_i','scenario'} if role=='known' else {'rx_i','day_i'} if role=='novel' else set())
    if set(a)!=fields:raise ValueError('Exact source-public/known or public-Q fields required; no target labels')
    parent.core.base._ids(a['ids'],n)
    for key in ('closed','baseline_closed'):parent.core.base._closed(a[key],n)
    for key in ('default_logits','baseline_logits'):
        if a[key].dtype!=np.float32 or a[key].shape!=(n,6) or not np.isfinite(a[key]).all():raise ValueError('Candidate and independent original F0 logits required')
    if not np.array_equal(a['closed'],a['default_logits'].argmax(1)) or not np.array_equal(a['baseline_closed'],a['baseline_logits'].argmax(1)):raise ValueError('Own F0 identity routing or independent baseline differs')
    _features(a['signal_features'],n)
    for key in ('raw_scores','scores'):parent.core.base._float(a[key],(n,2),key)
    if role in ('known','novel'):
        for key in ('rx_i','day_i'):
            if a[key].dtype!=np.int64 or a[key].shape!=(n,) or not np.isin(a[key],common.RECEIVERS if key=='rx_i' else (1,2,3)).all():raise ValueError('Source public RX/day metadata differs')
    if role=='known':
        parent.core.base._closed(a['y'],n)
        if a['scenario'].shape!=(n,) or a['scenario'].dtype.kind!='U' or not np.isin(a['scenario'],common.SCENES).all():raise ValueError('Visible known V metadata required')
    return a


def _old(old):
    a=parent._compact(old);result=dict(ids=a['ids'],closed=a['closed'],baseline_closed=a['baseline_closed'],baseline_logits=a['default_logits'].copy(),default_logits=a['default_logits'],
        signal_features=a['features'],raw_scores=old['raw_scores'][:,4:6].copy(),scores=old['scores'][:,4:6].copy())
    result.update({k:a[k] for k in ('y','rx_i','day_i','scenario') if k in a});return result


def _parent_rows(c,key):return parent._parent_rows(c,key)


def _rows(root,role,arm):
    if arm not in ARMS or role not in ('known','novel'):raise ValueError('Fixed source arm and role required')
    return _validate(_npz(Path(root)/'source'/arm/(role+'_predictions.npz')),COUNTS['V' if role=='known' else 'novel'],role)


def _target_rows(root,condition,arm):
    if arm not in ARMS or condition not in CONDITIONS[1:]:raise ValueError('Fixed future DURA arm/condition required')
    return _validate(_npz(Path(root)/'targets'/condition/arm/'predictions.npz'),Q_COUNT,'Q')


def _forward(b,model,adapters,values,device):
    x=np.asarray(values)
    if x.dtype!=np.float32 or x.shape!=(len(x),2,256) or not len(x) or not np.isfinite(x).all():raise ValueError('Finite canonical public received full256 IQ required')
    result={}
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        iq=torch.tensor(x.tolist(),device=device,dtype=torch.float32);p1=b['dura'][1](iq)[0].detach().clone()
        if p1.shape!=iq.shape or p1.dtype!=torch.float32 or not bool(torch.isfinite(p1).all()):raise ValueError('Single shared frozen DURA output required')
        for arm in ARMS[1:]:
            adapted=adapters[arm](p1);logits=b['reader'].classify_features(b['reader'].features(adapted));z=model(adapted)
            result[arm]=dict(default_logits=np.asarray(logits.detach().cpu().tolist(),np.float32),signal_features=np.asarray(z.detach().cpu().tolist(),np.float32))
            result[arm]['closed']=result[arm]['default_logits'].argmax(1).astype(np.int64);_features(result[arm]['signal_features'],len(x))
    return result


def _smoke(b,model,adapters,device):
    from .star_open_feature_export import _synthetic
    old=parent._smoke(b,model,device);a=_forward(b,model,adapters,np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32),device)
    if old['status']!='PASS' or any(len(v['closed'])!=2 for v in a.values()):raise ValueError('Actual F0/r03/PCD/G and both trained adapter smoke must pass')
    return dict(status='PASS',actual_checkpoints_loaded=True,trained_adapter_names=list(ARMS[1:]),synthetic_F_forward_rows=6,synthetic_signal_forward_rows=8,
        G_synthetic_forward_rows=4,synthetic_adapter_forward_rows=4,real_data_reads=0,optimizer_updates=0)


def _public(c,condition):return parent.fresh._public(c['reference_config'],condition)


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable receiver-input predictions required')
    pm,state,old_tau,old_cal=_reference(c);training=_training(c,pm);b=_backend(c,device);model,identity=_load_model(c,device);adapters,loaded=_train().load_adapters(c['train_config'],device)
    if (loaded!=training or identity!=pm['model_identities']['TARGET_PCD'] or b['reader_identity']!=pm['parent_F0_identity'] or b['model_identity']!=pm['model_identities']['E0']
            or b['dura_identity']!=pm['loaded_dura_identity'] or set(adapters)!=set(ARMS[1:])):raise ValueError('Actual frozen F0/r03/PCD/G/adapter ancestors differ')
    models=(b['reader'],b['model'],b['dura'][0],model,*adapters.values());snapshots=[(m,parent.fresh.best._snapshot_model(m)) for m in models];fixed=deepcopy(state);tau=np.tile(old_tau,(3,1));fixed_tau=tau.copy()
    def check():
        for m,snapshot in snapshots:parent.fresh.best._check_model(m,snapshot)
        if not _same(state,fixed) or not np.array_equal(tau,fixed_tau):raise ValueError('Frozen source density/threshold state changed')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;check();smoke=_smoke(b,model,adapters,device);check()
    if smoke['status']!='PASS':raise ValueError('Actual checkpoint smoke precedes source and Q reads')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);_save(out/'models/density.npz',state);_save(out/'thresholds.npz',dict(conditional=tau))
    theta={arm:adapters[arm].theta.detach().cpu().tolist() for arm in ARMS[1:]}
    if any(theta[arm]!=training['rows'][arm]['theta'] for arm in ARMS[1:]):raise ValueError('Actual adapter parameters differ from completed training')
    _save(out/'models/adapters.npz',{arm:np.asarray(value,np.float32) for arm,value in theta.items()})
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],training_identity=training,model_identity=identity,
        adapter_identities=training['identities'],adapter_theta=theta,models_frozen=True,original_density_refitted=False,new_density_fits=0,new_threshold_sets_fitted=0,source_V_fit=False,novel_fit=False,target_Q_fit=False))
    reports={METHODS[j]:deepcopy(old_cal['methods'][parent.fresh.METHODS[4+j%2]]) for j in range(6)}
    cal=dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),methods=reports,conditional_thresholds=tau.tolist(),
        parent_calibration_identity=old_cal,known_only=True,unknown_inputs_used=False,thresholds_refitted=False,source_V_threshold_fit=False)
    (out/'calibration').mkdir();_write(out/'calibration/complete.json',cal);used=set()
    for key,role in (('source_V','known'),('source_NV','novel')):
        a=_old(_parent_rows(c,key));n=COUNTS['V' if role=='known' else 'novel'];_validate(a,n,role)
        if used&set(a['ids'].tolist()):raise ValueError('Source known/novel physical roles overlap')
        used.update(a['ids'].tolist())
        for arm in ARMS:_save(out/'source'/arm/(role+'_predictions.npz'),a)
        check()
    manifests={};paired=None
    for condition in CONDITIONS[1:]:
        old=_old(_parent_rows(c,'Q_'+condition));manifest,ids,iq=_public(c,condition)
        try:
            if not np.array_equal(ids,old['ids']) or used&set(ids.tolist()):raise ValueError('Future public IDs differ from exact cached baseline or overlap source')
            if paired is None:paired=ids.copy()
            elif not np.array_equal(ids,paired):raise ValueError('Same clean/star future physical order required')
            blocks={arm:[] for arm in ARMS[1:]}
            for start in range(0,len(ids),c['batch_size']):
                stop=min(start+c['batch_size'],len(ids));new=_forward(b,model,adapters,iq[start:stop],device)
                for arm in ARMS[1:]:blocks[arm].append(new[arm])
            for j,arm in enumerate(ARMS):
                a=deepcopy(old)
                if j:
                    a.update({key:np.concatenate([v[key] for v in blocks[arm]]) for key in blocks[arm][0]});a['raw_scores']=parent.fresh._raw(state,a['signal_features'],a['closed']);a['scores']=a['raw_scores']-tau[2*j:2*j+2,a['closed']].T
                _validate(a,Q_COUNT,'Q');_save(out/'targets'/condition/arm/'predictions.npz',a)
            manifests[condition]=manifest;check();print('RECEIVER_INPUT_Q_COMPLETE',condition,len(ids),flush=True)
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,arm_order=list(ARMS),conditions=list(CONDITIONS),
        reference_prediction_identity=pm,training_identity=training,adapter_identities=training['identities'],model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS[1:]},Q_manifests=manifests,
        all_adapters_frozen_before_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,all_thresholds_inherited_unchanged=True,controls_reused_exactly=True,
        original_models_unchanged=True,source_states_unchanged=True,classification_unchanged=False,identity_baseline_path='original_F0_same_DURA',classification_path='same_original_F0_head_after_receiver_adapter',
        new_density_fits=0,new_threshold_sets_fitted=0,optimizer_updates=0,private_truth_reads=0,U_truth_reads=0,target_U_reads=0,target_truth_reads=0,source_novel_truth_reads=0,source_IQ_reads=0,
        target_Q_IQ_reads=2*Q_COUNT,real_F_forward_rows=4*Q_COUNT,real_signal_forward_rows=4*Q_COUNT,G_forward_rows=2*Q_COUNT,
        synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],synthetic_signal_forward_rows=smoke['synthetic_signal_forward_rows'],G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],synthetic_adapter_forward_rows=smoke['synthetic_adapter_forward_rows'],
        threshold_values=36,threshold_bytes=tau.nbytes,new_threshold_values=0,unique_threshold_values=12,adapter_parameters_per_arm=16,new_neural_updates=0,inherited_adapter_optimizer_updates=2*_train().STEPS,
        PCD_U_ancestor=576,source_only=False,new_target_DA=True,target_REG=False,source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,full_goal_status='ACTIVE',final_phase3_complete=False,A_N=None,H_OW=None)
    check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


def _prediction_gate(root):
    root=Path(root);done=_read(root/'complete.json');c=validate_config(done['config']);pm,state,old_tau,old_cal=_reference(c);training=_training(c,pm)
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,arm_order=list(ARMS),conditions=list(CONDITIONS),
        reference_prediction_identity=pm,training_identity=training,adapter_identities=training['identities'],model_identity=pm['model_identities']['TARGET_PCD'],parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS[1:]},all_adapters_frozen_before_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,
        all_thresholds_inherited_unchanged=True,controls_reused_exactly=True,original_models_unchanged=True,source_states_unchanged=True,classification_unchanged=False,new_density_fits=0,new_threshold_sets_fitted=0,
        optimizer_updates=0,private_truth_reads=0,U_truth_reads=0,target_U_reads=0,source_IQ_reads=0,real_F_forward_rows=4*Q_COUNT,real_signal_forward_rows=4*Q_COUNT,G_forward_rows=2*Q_COUNT,final_phase3_complete=False)
    if Path(c['output_root']).resolve()!=root.resolve() or any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Complete frozen three-arm/three-DURA predictions required before truth')
    fit=_read(root/'models/complete.json');cal=_read(root/'calibration/complete.json');saved=_npz(root/'thresholds.npz');tau=saved.get('conditional');theta=_npz(root/'models/adapters.npz')
    if (not _same(_npz(root/'models/density.npz'),state) or fit.get('status')!=MODELS_STATUS or fit.get('schema')!=SCHEMA or fit.get('run_id')!=RUN_ID or fit.get('code_commit')!=c['code_commit']
            or fit.get('training_identity')!=training or fit.get('adapter_identities')!=training['identities'] or fit.get('model_identity')!=done['model_identity']
            or fit.get('models_frozen') is not True or any(fit.get(k)!=0 for k in ('new_density_fits','new_threshold_sets_fitted'))
            or any(fit.get(k) is not False for k in ('source_V_fit','novel_fit','target_Q_fit')) or set(theta)!=set(ARMS[1:])):raise ValueError('Original density and actual frozen adapter provenance changed')
    for arm in ARMS[1:]:
        if (theta[arm].shape!=(16,) or theta[arm].dtype!=np.float32 or not np.isfinite(theta[arm]).all() or theta[arm].tolist()!=fit['adapter_theta'][arm]
                or theta[arm].tolist()!=training['rows'][arm]['theta']):raise ValueError('Frozen receiver theta differs')
    if (set(saved)!={'conditional'} or tau.dtype!=np.float64 or tau.shape!=(6,6) or not np.array_equal(tau,np.tile(old_tau,(3,1)))
            or cal.get('status')!=CALIBRATION_STATUS or cal.get('schema')!=SCHEMA or cal.get('run_id')!=RUN_ID or cal.get('code_commit')!=c['code_commit']
            or cal.get('method_names')!=list(METHODS) or cal.get('conditional_thresholds')!=tau.tolist() or cal.get('parent_calibration_identity')!=old_cal
            or cal.get('known_only') is not True or cal.get('source_V_threshold_fit') is not False or cal.get('thresholds_refitted') is not False or cal.get('unknown_inputs_used') is not False):raise ValueError('Only exact inherited two-by-six thresholds may be used')
    for j,name in enumerate(METHODS):
        if cal['methods'][name]!=old_cal['methods'][parent.fresh.METHODS[4+j%2]]:raise ValueError('Inherited source calibration report changed')
    rows={};used=set();paired=None
    for key in ROW_KEYS:
        old=_old(_parent_rows(c,key));role='known' if key=='source_V' else 'novel' if key=='source_NV' else 'Q';pair={}
        for j,arm in enumerate(ARMS):
            a=_rows(root,role,arm) if role!='Q' else _target_rows(root,key[2:],arm)
            if j==0 or role!='Q':
                if not _same(a,old):raise ValueError('Exact original controls or source identity repetition changed')
            elif any(not np.array_equal(a[k],old[k]) for k in ('ids','baseline_closed','baseline_logits')):raise ValueError('Original same-DURA F0 baseline changed')
            raw=parent.fresh._raw(state,a['signal_features'],a['closed'])
            if not np.allclose(raw,a['raw_scores'],atol=1e-9,rtol=1e-12) or not np.array_equal(a['scores'],a['raw_scores']-tau[2*j:2*j+2,a['closed']].T):raise ValueError('Own candidate identity/density/threshold margins do not reconstruct')
            pair[arm]=a
        ids=old['ids']
        if role!='Q':
            if used&set(ids.tolist()):raise ValueError('Source evaluation roles overlap')
            used.update(ids.tolist())
        else:
            if used&set(ids.tolist()):raise ValueError('Future Q overlaps source evaluation')
            if paired is None:paired=ids.copy()
            elif not np.array_equal(ids,paired):raise ValueError('Future clean/star IDs differ')
        rows[key]=pair
    return done,state,tau,cal,rows


evaluate=run


def main():
    print(json.dumps(dict(event='RECEIVER_INPUT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
