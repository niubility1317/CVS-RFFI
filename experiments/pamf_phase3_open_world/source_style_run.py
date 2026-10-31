"""Source-only native identity/open learning with paired cross-receiver MixStyle controls."""
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
from . import star_density_core as density
from . import star_macro12_risk as risk
from . import star_dura_best_run as best

RUN_ID='20261031-phase3-source-style-s2026092701-r02'
CONTROL_CODE='f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7'
SCHEMA='pamf_phase3_source_style_prediction_v1'
STATUS='SOURCE_STYLE_PREDICTIONS_COMPLETE'
MODELS_STATUS='SOURCE_STYLE_DENSITIES_FROZEN'
CALIBRATION_STATUS='SOURCE_STYLE_THRESHOLDS_FROZEN'
ARMS=('E0','OLD_PCD','SOURCE_NATIVE','STYLE_NATIVE')
NEW_ARMS=ARMS[2:]
METHODS=tuple(a+'_'+m for a in ARMS for m in ('pooled_ratio','predicted_ratio'))
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=7
CONDITIONS=fresh.CONDITIONS
COUNTS=dict(fresh.COUNTS)
Q_COUNT=10238
U_COUNT=0
UPDATES=444
FIELDS={'run_id','code_commit','train_config','reference_config','Q_conditions','output_root','batch_size'}
TARGET_FIELDS={'ids','default_logits','closed','baseline_closed','signal_features','raw_scores','scores'}
KNOWN_FIELDS=TARGET_FIELDS|{'y','rx_i','day_i','scenario'}
NOVEL_FIELDS=TARGET_FIELDS|{'rx_i','day_i'}
_read,_write,_same=best._read,best._write,best._same
_raw=fresh._raw


def _train():
    from . import source_style_train
    return source_style_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact source-style evaluation fields required; no private labels or unknown roster')
    t=_train().validate_config(c['train_config']);p=fresh.validate_config(c['reference_config'])
    if (t.get('run_id')!=RUN_ID or t.get('code_commit')!=c['code_commit'] or t.get('reference_config')!=p
            or p['code_commit']!=CONTROL_CODE or c['Q_conditions']!=p['Q_conditions']):
        raise ValueError('Current joint training and fixed fresh controls/public future handles must match')
    out=Path(c['output_root']).resolve()
    roots=[t['output_root'],p['output_root'],p['train_config']['output_root'],p['train_config']['U_public_root'],p['source_control_config']['reference_config']['train_config']['IQ_root']]
    roots.extend(v['public_root'] for v in c['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('New output must be separate from all training/source/query inputs')
    return c


def _reference(c):
    p=c['reference_config'];root=Path(p['output_root']);m=_read(root/'complete.json')
    required=dict(schema=fresh.SCHEMA,status=fresh.STATUS,run_id=fresh.RUN_ID,code_commit=CONTROL_CODE,config=p,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,source_models_frozen_before_V=True,
        all_thresholds_frozen_before_novel_and_Q=True,source_control_readouts_reused_exactly=True,source_states_unchanged=True,
        frozen_parameters_buffers_unchanged=True,classification_unchanged=True,private_truth_reads=0,target_U_reads=0)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Completed fixed fresh-PCD controls required before any new IQ')
    body,old_states,_,_=fresh._reference(p)
    if m['source_control_prediction_identity']!=body or m['model_identities']['E0']!=body['model_identities']['E0']:
        raise ValueError('Original source/F0/DURA ancestry differs')
    fresh._identity(m['model_identities']['TARGET_PCD'],p,body)
    states={}
    for new,old in (('E0','E0'),('OLD_PCD','TARGET_PCD')):
        with np.load(root/'models'/(old+'.npz'),allow_pickle=False) as z:states[new]={k:z[k].copy() for k in z.files}
        density.validate_state(states[new],'density_ratio')
    if not _same(states['E0'],old_states['E0']):raise ValueError('Original E0 source density changed')
    fit=_read(root/'models/complete.json');cal=_read(root/'calibration/complete.json')
    if (fit.get('status')!=fresh.MODELS_STATUS or fit.get('code_commit')!=CONTROL_CODE or fit.get('identities')!=m['model_identities']
            or fit.get('source_V_fit') is not False or cal.get('status')!=fresh.CALIBRATION_STATUS or cal.get('known_only') is not True
            or cal.get('unknown_inputs_used') is not False or cal.get('code_commit')!=CONTROL_CODE):raise ValueError('Frozen source-fitted control density/calibration identities required')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Exact control threshold state required')
        full=z['conditional'].copy()
    if full.dtype!=np.float64 or full.shape!=(6,6) or not np.isfinite(full).all() or cal.get('conditional_thresholds')!=full.tolist():raise ValueError('Six exact old source threshold vectors required')
    methods={}
    for new,old_index in zip(METHODS[:4],(0,1,4,5)):
        row=cal['methods'][fresh.METHODS[old_index]]
        if row['conditional_thresholds']!=full[old_index].tolist():raise ValueError('Inherited calibration report differs')
        methods[new]=deepcopy(row)
    return m,states,full[[0,1,4,5]],dict(methods=methods)


def _identity(i,arm,c,pm):
    expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,updates=444,signal_parameters=335681,trainable_parameters=335633,
        identity_head_updates=444,projection_updates=444,source_only=True,target_U_used=False,target_U_reads=0,
        identity_head_jointly_trained=True,identity_head_replaced=False,initial_native_logits_equal_original_F0=True,style_enabled=arm=='STYLE_NATIVE')
    if (arm not in NEW_ARMS or any(i.get(k)!=v for k,v in expected.items()) or i.get('parent_F0_identity')!=pm['parent_F0_identity']
            or i.get('parent_signal_identity')!=pm['model_identities']['E0'] or i.get('dura_identity')!=pm['dura_identity']):
        raise ValueError('Complete original F0 native identity and pure-r03 projection source-only lineage required')
    return i


def _training_metadata(c,pm):
    t=_train();root=Path(c['train_config']['output_root']);m=_read(root/'complete.json')
    required=dict(schema=t.SCHEMA,status=t.STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c['train_config'],arms=list(NEW_ARMS),
        total_optimizer_updates=888,source_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True,
        all_student_Sinc_unchanged=True,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,U_truth_reads=0,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Both fixed source-only style-control training arms required')
    identities={}
    for arm in NEW_ARMS:
        row=_read(root/arm/'complete.json');identity=m.get('identities',{}).get(arm,{})
        if (row!=m.get('rows',{}).get(arm) or row.get('status')!=t.ROW_STATUS or row.get('identity')!=identity or row.get('updates')!=444
                or row.get('source_only') is not True or row.get('target_U_reads')!=0):raise ValueError('Complete source-only native/style row differs')
        identities[arm]=_identity(identity,arm,c,pm)
    costs={k:m[k] for k in ('real_parent_signal_forward_rows','real_student_forward_rows','G_forward_rows','real_F_forward_rows')}
    costs.update({k:v for k,v in m.items() if k.startswith('synthetic_') or k=='G_synthetic_forward_rows'})
    return identities,np.asarray([],dtype='<U1'),costs


def _backend(c,device):return fresh._backend(c['reference_config'],device)
def _load_model(c,arm,device):return _train().load_trained_model(c['train_config'],arm,device)
def _iq_role(c,role,pm):return fresh._iq_role(c['reference_config'],role,pm)
def _public(c,condition):return fresh._public(c,condition)


def _slice(old,arm,role):
    index=0 if arm=='E0' else 2
    fields=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS if role=='novel' else TARGET_FIELDS
    a={k:old[k].copy() for k in fields if k not in ('signal_features','raw_scores','scores','baseline_closed')}
    a.update(signal_features=old['signal_features'][:,index].copy(),raw_scores=old['raw_scores'][:,2*index:2*index+2].copy(),scores=old['scores'][:,2*index:2*index+2].copy(),
        baseline_closed=old.get('baseline_closed',old['closed']).copy())
    return a


def _validate(a,role):
    n=COUNTS['V' if role=='known' else 'novel'] if role in ('known','novel') else Q_COUNT
    fields=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS if role=='novel' else TARGET_FIELDS
    if (set(a)!=fields or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n or np.any(np.char.strip(a['ids'])=='')
            or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or any(a[k].shape!=(n,) or a[k].dtype!=np.int64 or not np.isin(a[k],range(6)).all() for k in ('closed','baseline_closed'))
            or not np.array_equal(a['closed'],a['default_logits'].argmax(1))):raise ValueError('Exact complete model-specific default-head identity and original F0 baseline required')
    if (a['signal_features'].shape!=(n,160) or a['signal_features'].dtype!=np.float32 or not np.isfinite(a['signal_features']).all()
            or not np.allclose(np.linalg.norm(a['signal_features'].astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)
            or any(a[k].shape!=(n,2) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):raise ValueError('One unit160 representation and two finite readouts per model required')
    if role in ('known','novel'):
        if any(a[k].shape!=(n,) or a[k].dtype!=np.int64 or not np.isin(a[k],v).all() for k,v in (('rx_i',(1,3,4,6,8)),('day_i',(1,2,3)))):raise ValueError('Legal source RX/day metadata required')
        if role=='known' and (a['y'].shape!=(n,) or a['y'].dtype!=np.int64 or not np.isin(a['y'],range(6)).all()
                or a['scenario'].shape!=(n,) or a['scenario'].dtype.kind!='U' or not np.isin(a['scenario'],risk.SCENES).all()):raise ValueError('Visible source-known V labels/scenarios only')
    return a


def _source_rows(root,arm,role):
    if arm not in ARMS or role not in ('known','novel'):raise ValueError('Fixed arm/source role required')
    with np.load(Path(root)/'source'/arm/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,role)


def _target_rows(root,condition,arm):
    if arm not in ARMS or condition not in CONDITIONS[1:]:raise ValueError('Fixed arm/future condition required')
    with np.load(Path(root)/'targets'/condition/arm/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,'Q')


def _joint(model,values,device):
    blocks=[]
    for start in range(0,len(values),128):
        a=np.asarray(values[start:start+128])
        if a.dtype!=np.float32 or a.shape!=(len(a),2,256) or not np.isfinite(a).all():raise ValueError('Finite cached P1 IQ required')
        with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):z,logits=model(torch.tensor(a.tolist(),device=device,dtype=torch.float32))
        z=np.asarray(z.detach().cpu().tolist(),np.float32);logits=np.asarray(logits.detach().cpu().tolist(),np.float32)
        if z.shape!=(len(a),160) or logits.shape!=(len(a),6) or not np.isfinite(z).all() or not np.isfinite(logits).all():raise ValueError('Actual one-pass joint z160 and native logits6 required')
        blocks.append(dict(signal_features=z,default_logits=logits,closed=logits.argmax(1).astype(np.int64)))
    return {k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]}


def _forward(b,models,values,device):
    a=np.asarray(values)
    if a.dtype!=np.float32 or a.shape!=(len(a),2,256) or not np.isfinite(a).all():raise ValueError('Finite public full256 IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        x=torch.tensor(a.tolist(),device=device,dtype=torch.float32);restored=b['dura'][1](x)[0].detach().clone()
        if restored.shape!=x.shape or restored.dtype!=torch.float32 or not bool(torch.isfinite(restored).all()):raise ValueError('One shared frozen DURA output required')
        logits=b['reader'].classify_features(b['reader'].features(restored))
    original=np.asarray(logits.detach().cpu().tolist(),np.float32)
    if original.shape!=(len(a),6) or not np.isfinite(original).all():raise ValueError('Actual original F0 same-DURA logits required')
    result={arm:_joint(model,np.asarray(restored.cpu().tolist(),np.float32),device) for arm,model in models.items()}
    for v in result.values():v['baseline_closed']=original.argmax(1).astype(np.int64)
    return original,result


def _smoke(b,models,device):
    from .star_open_feature_export import _synthetic
    x=np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32);original,rows=_forward(b,models,x,device)
    if original.shape!=(2,6) or set(rows)!=set(NEW_ARMS):raise ValueError('Actual F0/G and both joint student checkpoints required')
    return dict(status='PASS',actual_F0_DURA_joint_checkpoints=True,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,synthetic_signal_forward_rows=4,real_data_reads=0,optimizer_updates=0)


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable joint identity prediction root required')
    pm,states,old_tau,old_cal=_reference(c);declared,Uids,training_costs=_training_metadata(c,pm)
    b=_backend(c,device);models={};identities={a:pm['model_identities'][old] for a,old in (('E0','E0'),('OLD_PCD','TARGET_PCD'))}
    if b['model_identity']!=identities['E0'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Original F0/r03/G ancestry differs')
    for arm in NEW_ARMS:
        models[arm],identities[arm]=_load_model(c,arm,device);_identity(identities[arm],arm,c,pm)
        if identities[arm]!=declared[arm]:raise ValueError('Loaded student differs from complete training')
        model=models[arm]
        if sum(p.numel() for p in model.parameters())!=335681 or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Frozen complete335681 student including original six-class W required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(model,best._snapshot_model(model)) for model in [b['reader'],b['dura'][0],b['model'],*models.values()]]
    def check():
        for model,before in snapshots:best._check_model(model,before)
    smoke=_smoke(b,models,device);check()
    if smoke.get('status')!='PASS':raise ValueError('Actual checkpoint no-query smoke precedes all source/query IQ')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    used=set(Uids.tolist());source_ids=set();known={};diagnostics={a:{} for a in NEW_ARMS}
    for role in ('L','OE'):
        iq=_iq_role(c,role,pm)
        try:
            ids=set(iq['ids'].tolist())
            if ids&(used|source_ids):raise ValueError('Source fitting and U roles overlap')
            source_ids.update(ids);meta=iq['metadata']
            for arm in NEW_ARMS:
                z=_joint(models[arm],iq['restored'],device)['signal_features']
                if role=='L':known[arm],diagnostics[arm]['known']=density.fit_known(z,meta['y'],meta['rx_i'],meta['day_i'])
                else:states[arm],diagnostics[arm]['background']=density.fit_background(known[arm],z,meta['rx_i'],meta['day_i'])
            check();print('SOURCE_STYLE_SOURCE_PROGRESS',role,len(ids),flush=True)
        finally:fresh.source_eval._close_role(iq)
    frozen=deepcopy(states)
    for arm in ARMS:best._save(out/'models'/(arm+'.npz'),states[arm])
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],identities=identities,diagnostics=diagnostics,
        neural_states_frozen=True,source_fit_inputs_only=True,source_V_fit=False,novel_fit=False,target_Q_fit=False,source_density_models_fitted=2,normalizer_fit_role='L',background_fit_role='OE',STYLE_NATIVE_has_target_U_ancestor=False))
    def frozen_check():
        check()
        if not _same(states,frozen):raise ValueError('Frozen source density state changed')
    def source(role):
        old=fresh._source_rows(c['reference_config']['output_root'],'source_DURA',role);rows={arm:_slice(old,arm,role) for arm in ARMS[:2]};iq=_iq_role(c,'V' if role=='known' else 'novel',pm)
        try:
            fresh.source_eval._align(iq,old,'V' if role=='known' else 'novel');ids=set(old['ids'].tolist())
            if ids&(used|source_ids):raise ValueError('Source L/OE/V/novel/U roles must be disjoint')
            source_ids.update(ids)
            for arm in NEW_ARMS:
                v=_joint(models[arm],iq['restored'],device);raw=_raw(states[arm],v['signal_features'],v['closed'])
                rows[arm]=dict(rows['E0'],**v,raw_scores=raw)
            frozen_check();print('SOURCE_STYLE_SOURCE_PROGRESS',role,len(ids),flush=True);return rows
        finally:fresh.source_eval._close_role(iq)
    V=source('known');tau=np.empty((8,6),np.float64);tau[:4]=old_tau;methods=deepcopy(old_cal['methods'])
    for index,arm in enumerate(NEW_ARMS,2):
        a=V[arm]
        for j in range(2):
            calibrated,report=risk.calibrate_known(a['ids'],a['y'],a['closed'],a['baseline_closed'],a['raw_scores'][:,j],a['rx_i'],a['scenario'])
            tau[2*index+j]=calibrated['conditional_thresholds'];methods[METHODS[2*index+j]]=report
    (out/'calibration').mkdir();best._save(out/'thresholds.npz',dict(conditional=tau))
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        methods=methods,conditional_thresholds=tau.tolist(),known_count=COUNTS['V'],group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),known_only=True,unknown_inputs_used=False,target_reads=0,
        source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,routing='each own frozen native default-head prediction'))
    frozen_tau=tau.copy()
    def save_source(rows,role):
        for index,arm in enumerate(ARMS):
            a=rows[arm];a['scores']=a['raw_scores']-tau[2*index:2*index+2,a['closed']].T;_validate(a,role);best._save(out/'source'/arm/(role+'_predictions.npz'),a)
        frozen_check()
    save_source(V,'known');print(CALIBRATION_STATUS,flush=True);save_source(source('novel'),'novel')
    manifests={};paired_ids=None
    for condition in CONDITIONS[1:]:
        iq=None
        try:
            old=fresh._target_rows(c['reference_config']['output_root'],condition);manifest,ids,iq=_public(c,condition)
            if not np.array_equal(ids,old['ids']) or set(ids.tolist())&(used|source_ids):raise ValueError('Exact future control IDs and disjoint source/U roles required')
            if paired_ids is None:paired_ids=ids.copy()
            elif not np.array_equal(ids,paired_ids):raise ValueError('Both future conditions must share physical ID order')
            rows={arm:_slice(old,arm,'Q') for arm in ARMS[:2]};blocks={arm:[] for arm in NEW_ARMS}
            for start in range(0,len(ids),128):
                original,values=_forward(b,models,iq[start:start+128],device)
                if not np.allclose(original,old['default_logits'][start:start+len(original)],rtol=1e-5,atol=2e-5) or not np.array_equal(original.argmax(1),old['closed'][start:start+len(original)]):raise ValueError('Same-G original F0 baseline differs from frozen fresh controls')
                for arm,v in values.items():
                    v['raw_scores']=_raw(states[arm],v['signal_features'],v['closed']);blocks[arm].append(v)
                print('SOURCE_STYLE_Q_PROGRESS',condition,start+len(original),len(ids),flush=True)
            for arm in NEW_ARMS:rows[arm]=dict(ids=ids.copy(),**{k:np.concatenate([v[k] for v in blocks[arm]]) for k in blocks[arm][0]})
            for index,arm in enumerate(ARMS):
                a=rows[arm];a['scores']=a['raw_scores']-tau[2*index:2*index+2,a['closed']].T;_validate(a,'Q');best._save(out/'targets'/condition/arm/'predictions.npz',a)
            manifests[condition]=manifest;frozen_check()
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Source thresholds changed after novel/query access')
    real_signal=2*sum(COUNTS.values())+4*Q_COUNT
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,arm_order=list(ARMS),conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={k:Q_COUNT for k in CONDITIONS[1:]},Q_manifests=manifests,reference_prediction_identity=pm,model_identities=identities,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=b['dura_identity'],parent_calibration_identity=old_cal,
        classification_path='each own native default head',classification_unchanged=False,identity_baseline_path='same_DURA_output_original_F0',original_F0_unchanged=True,student_heads_jointly_trained=True,identity_head_replaced=False,score_threshold=0.,
        source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,source_control_readouts_reused_exactly=True,
        source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,source_density_models_fitted=2,new_threshold_sets_fitted=4,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,
        source_IQ_reads=sum(COUNTS.values()),target_Q_IQ_reads=2*Q_COUNT,target_U_reads=0,U_ID_metadata_reads=U_COUNT,training_target_U_count=U_COUNT,prototypes_deployed=False,
        source_only=True,SOURCE_NATIVE_density_source_only=True,STYLE_NATIVE_density_source_only=True,target_DA=False,target_REG=False,threshold_values=48,threshold_bytes=384,new_threshold_values=24,reused_threshold_values=24,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,original_U_s_reads=0,new_neural_updates=0,new_candidate_optimizer_updates=888,prediction_optimizer_updates=0,Sinc_optimizer_steps=0,
        real_signal_forward_rows=real_signal,real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,real_signal_forward_rows_by_arm=dict(E0=0,OLD_PCD=0,**{arm:sum(COUNTS.values())+2*Q_COUNT for arm in NEW_ARMS}),
        synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],synthetic_signal_forward_rows=smoke['synthetic_signal_forward_rows'],training_costs=training_costs,
        combined_new_real_signal_forward_rows=training_costs['real_parent_signal_forward_rows']+training_costs['real_student_forward_rows']+real_signal,
        combined_new_G_forward_rows=training_costs['G_forward_rows']+2*Q_COUNT,combined_new_real_F_forward_rows=training_costs['real_F_forward_rows']+2*Q_COUNT,
        original_F0_parameters=232993,joint_student_parameters=335681,primary_deployed_neural_parameters=602632,source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    frozen_check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


run=evaluate


def main():
    print(json.dumps(dict(event='SOURCE_STYLE_EVALUATION_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
