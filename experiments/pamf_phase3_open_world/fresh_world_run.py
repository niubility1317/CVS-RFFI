"""Fresh-RX PCD evaluation with unchanged source-only E0/SOURCE controls."""
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
from . import star_dura_best_run as best
from . import star_density_core as density
from . import star_macro12_risk as risk
from . import star_signal_open_eval as source_eval
from . import star_class_density_run as parent
from . import star_class_density_core as class_density
from .star_local_cov_run import _reference as source_reference

RUN_ID='20261031-phase3-fresh-development-rx3-2-s2026092701-r01'
CONTROL_CODE='4273e88392abe0717c1bd94fca1370d19c67cbb3'
from . import star_body_prototype_run as previous
SCHEMA='pamf_phase3_fresh_world_prediction_v1'
STATUS='FRESH_WORLD_PREDICTIONS_COMPLETE'
MODELS_STATUS='FRESH_WORLD_DENSITIES_FROZEN'
CALIBRATION_STATUS='FRESH_WORLD_THRESHOLDS_FROZEN'
METHODS=('E0_pooled_ratio','E0_predicted_ratio','SOURCE_pooled_ratio','SOURCE_predicted_ratio','PCD_pooled_ratio','PCD_predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=5
MODEL_NAMES=('E0','SOURCE','TARGET_PCD')
ARMS=MODEL_NAMES
PARENT_CODE='4af0668331682636ee212077b38cd912857a1b31'
U_COUNT=576

CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=10238
FIELDS={'run_id','code_commit','train_config','source_control_config','Q_conditions','output_root','batch_size'}
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=parent.KNOWN_FIELDS,parent.NOVEL_FIELDS,parent.TARGET_FIELDS
_read,_write,_same=best._read,best._write,best._same


def _train():
    from . import fresh_pcd_train
    return fresh_pcd_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact seven-field fresh-world evaluation required; no truth or unknown roster')
    t=_train().validate_config(c['train_config']);p=previous.validate_config(c['source_control_config'])
    if t['run_id']!=RUN_ID or t['code_commit']!=c['code_commit'] or t['reference_config']!=p or p['code_commit']!=CONTROL_CODE or t['U_expected_count']!=U_COUNT:
        raise ValueError('Actual fresh PCD training and fixed source-only controls must match')
    q=c['Q_conditions']
    if not isinstance(q,dict) or set(q)!=set(CONDITIONS[1:]):raise ValueError('Both fresh future conditions required')
    for name,family in zip(CONDITIONS[1:],('clean_cross_receiver','practical_cross_receiver')):
        spec=q[name]
        if (not isinstance(spec,dict) or set(spec)!={'public_root','capsule_id','split_id','expected_count','view_family'}
                or type(spec['expected_count']) is not int or spec['expected_count']!=Q_COUNT or spec['view_family']!=family
                or any(not isinstance(spec[k],str) or not spec[k].strip() for k in ('public_root','capsule_id','split_id'))
                or Path(spec['public_root']).name!='public'):
            raise ValueError('Fresh future-Q public handles and declared count required')
    if Path(q['clean_DURA']['public_root']).resolve()==Path(q['star_DURA']['public_root']).resolve():raise ValueError('Separate clean/star future roots required')
    out=Path(c['output_root']).resolve();base=p['reference_config']
    roots=[t['output_root'],t['U_public_root'],p['output_root'],p['train_config']['output_root'],base['train_config']['IQ_root']]+[v['public_root'] for v in q.values()]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable output cannot overlap training, source or public inputs')
    return c


def _reference(c):
    p=c['source_control_config'];root=Path(p['output_root']);m=_read(root/'complete.json')
    required=dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=previous.RUN_ID,code_commit=CONTROL_CODE,config=p,
        all_three_DURA_conditions_complete=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,classification_unchanged=True,
        source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,private_truth_reads=0,target_U_reads=0)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Completed exact body-r02 source controls required')
    base,original,old_tau,old_cal=previous._reference(p)
    if m['reference_prediction_identity']!=base or m['model_identities']['E0']!=base['model_identity']:raise ValueError('Original source r03 ancestry differs')
    _source_identity(m['model_identities']['SOURCE'],c,m)
    fit=_read(root/'models/complete.json')
    if (fit.get('status')!=previous.MODELS_STATUS or fit.get('code_commit')!=CONTROL_CODE or fit.get('source_V_fit') is not False
            or any(fit.get('identities',{}).get(name)!=m['model_identities'][name] for name in ('E0','SOURCE'))):
        raise ValueError('Original source-fitted E0/SOURCE density identities required')
    row=_read(Path(p['train_config']['output_root'])/'SOURCE/complete.json')
    if (row.get('status')!='BODY_PROTOTYPE_STUDENT_FROZEN' or row.get('identity')!=m['model_identities']['SOURCE']
            or row.get('source_only') is not True or row.get('target_U_used') is not False or row.get('target_U_reads')!=0
            or row.get('student_Sinc_unchanged') is not True or row.get('model_buffers_unchanged') is not True):raise ValueError('Only the completed source-only SOURCE control may be reused')
    states={}
    for name in ('E0','SOURCE'):
        with np.load(root/'models'/(name+'.npz'),allow_pickle=False) as z:states[name]={k:z[k].copy() for k in z.files}
        density.validate_state(states[name],'density_ratio')
    if not _same(states['E0'],original):raise ValueError('Source E0 density changed')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:tau=z['conditional'][:4].copy()
    cal=_read(root/'calibration/complete.json')
    if (cal.get('status')!=previous.CALIBRATION_STATUS or cal.get('code_commit')!=CONTROL_CODE or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False
            or tau.dtype!=np.float64 or tau.shape!=(4,6) or not np.isfinite(tau).all() or not np.array_equal(tau[:2],old_tau)):
        raise ValueError('Four original source-control threshold vectors required')
    for j,name in enumerate(METHODS[:4]):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist():raise ValueError('Inherited source-control thresholds differ')
        if j<2 and cal['methods'][name]!=old_cal['methods'][parent.METHODS[(1,3)[j]]]:raise ValueError('Original source E0 calibration differs')
    return m,states,tau,dict(methods={name:cal['methods'][name] for name in METHODS[:4]})


def _source_identity(i,c,pm):
    return previous._identity(i,'SOURCE',c['source_control_config'],pm['reference_prediction_identity'])


def _identity(i,c,pm):
    expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm='TARGET_PCD',updates=444,projection_updates=444,projection_cumulative_updates=2664,
        open_signal_updates=444,signal_encoder_updates=444,native_updates=0,encoder_frozen=False,Sinc_frozen=True,signal_parameters=334721,signal_feature_parameters=232033,
        metric_parameters=102688,signal_feature_cumulative_updates=13764,trainable_parameters=334673,body_trainable_parameters=231985,trainable_projection_parameters=102688,
        new_body_updates=444,new_projection_updates=444,Sinc_optimizer_steps=0,input_path='cached_P1_IQ_only',source_only=False,target_U_used=True,target_U_reads=U_COUNT,
        source_control_source_only=True,CCD_updates=0)
    if (any(i.get(k)!=v for k,v in expected.items()) or i.get('parent_signal_identity')!=pm['model_identities']['E0']
            or i.get('parent_F0_identity')!=pm['parent_F0_identity'] or i.get('dura_identity')!=pm['dura_identity']
            or i.get('source_control_identity')!=pm['model_identities']['SOURCE']):raise ValueError('Independent source-r03 fresh-U576 PCD checkpoint required')
    return i


def _training_metadata(c,pm):
    root=Path(c['train_config']['output_root']);m=_read(root/'complete.json');i=m.get('identity',{})
    required=dict(schema='pamf_phase3_fresh_pcd_train_v1',status='FRESH_PCD_TRAINING_COMPLETE',run_id=RUN_ID,code_commit=c['code_commit'],config=c['train_config'],
        arm='TARGET_PCD',updates=444,total_optimizer_updates=444,source_sampling_matched=True,source_control_source_only=True,source_control_frozen_before_training=True,
        original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True)
    if any(m.get(k)!=v for k,v in required.items()) or m.get('source_control_identity')!=pm['model_identities']['SOURCE']:raise ValueError('One complete fresh-PCD training and exact source-only SOURCE ancestry required')
    _identity(i,c,pm);manifest=m.get('U_manifest',{})
    if any(manifest.get(k)!=v for k,v in dict(count=U_COUNT,capsule_id=c['train_config']['U_capsule_id'],split_id=c['train_config']['U_split_id']).items()):raise ValueError('Fresh discovery capsule/split differs')
    with np.load(root/'U_features.npz',allow_pickle=False) as a:ids=a['ids'].copy()
    if ids.shape!=(U_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=U_COUNT or np.any(np.char.strip(ids)==''):raise ValueError('Complete opaque fresh-U IDs required for disjointness')
    keys=('real_parent_signal_forward_rows','real_student_signal_forward_rows','G_forward_rows','real_F_forward_rows')
    costs={k:m[k] for k in keys}|{k:v for k,v in m.items() if k.startswith('synthetic_') or k=='G_synthetic_forward_rows'}
    return i,ids,costs


def _backend(c,device):return previous._backend(c['source_control_config'],device)
def _load_SOURCE(c,device):return previous._train().load_trained_model(c['source_control_config']['train_config'],'SOURCE',device)
def _load_model(c,device):return _train().load_trained_model(c['train_config'],device)
def _iq_role(c,role,pm):return previous._iq_role(c['source_control_config'],role,pm['model_identities']['E0'])


def _public(c,name):
    from .fresh_development_pool import load_public
    spec=c['Q_conditions'][name];m,ids,iq=load_public(spec['public_root'],spec['capsule_id'],spec['split_id'],spec['expected_count'])
    route='clean' if name=='clean_DURA' else 'residual'
    if m.get('processing_route')!=route:
        if hasattr(iq,'_mmap'):iq._mmap.close()
        raise ValueError('Future clean/star processing route differs')
    return m,ids,iq


def _exact_source_controls(a,old):
    if set(a)!=set(old):raise ValueError('Inherited source fields differ')
    for key,v in old.items():
        actual=a[key][:,:4] if key in ('scores','raw_scores') else a[key][:,:2] if key=='signal_features' else a[key]
        expected=v[:,:4] if key in ('scores','raw_scores') else v[:,:2] if key=='signal_features' else v
        if not np.array_equal(actual,expected):raise ValueError('E0/SOURCE source readouts, features and original identity must remain exact')


def _raw(state,z,closed):return class_density.raw_scores(state,z,closed)[:,[1,3]]



def _validate(a,role):
    n=COUNTS['V' if role=='known' else 'novel'] if role in ('known','novel') else Q_COUNT
    if role in ('known','novel'):
        source_eval._validate_rows({k:(v[:,0] if k=='signal_features' else v) for k,v in a.items() if k!='raw_signal_features'},role,6,True)
        if a['raw_signal_features'].shape!=(n,160) or a['raw_signal_features'].dtype!=np.float32 or not np.isfinite(a['raw_signal_features']).all():raise ValueError('Original raw source feature metadata must remain unchanged')
    elif (set(a)!=TARGET_FIELDS or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n or np.any(np.char.strip(a['ids'])=='')
            or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
            or a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):
        raise ValueError('Complete same-DURA original F0 public query identity required')
    z=a['signal_features']
    if (z.shape!=(n,3,160) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1.,atol=2e-6,rtol=0)
            or any(a[k].shape!=(n,6) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):
        raise ValueError('Three actual IQ-path unit160 representations and six finite readouts required')
    return a


def _source_rows(root,condition,role):
    if condition!='source_DURA' or role not in ('known','novel'):raise ValueError('Fixed source-DURA role required')
    with np.load(Path(root)/'source'/condition/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,role)


def _target_rows(root,condition):
    if condition not in CONDITIONS[1:]:raise ValueError('Fixed target-DURA condition required')
    with np.load(Path(root)/'targets'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,'Q')


def _forward(b,models,values,device):
    x=np.asarray(values)
    if x.dtype!=np.float32 or x.shape!=(len(x),2,256) or not np.isfinite(x).all():raise ValueError('Finite public full256 IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        iq=torch.tensor(x.tolist(),dtype=torch.float32,device=device);restored=b['dura'][1](iq)[0].detach().clone()
        if restored.dtype!=torch.float32 or restored.shape!=iq.shape or not bool(torch.isfinite(restored).all()):raise ValueError('One shared finite DURA output required')
        logits=b['reader'].classify_features(b['reader'].features(restored))
        features=[models[arm](restored) for arm in ARMS]
    logits=np.asarray(logits.detach().cpu().tolist(),np.float32);z=np.stack([np.asarray(a.detach().cpu().tolist(),np.float32) for a in features],axis=1)
    if (logits.shape!=(len(x),6) or z.shape!=(len(x),3,160) or not np.isfinite(logits).all() or not np.isfinite(z).all()
            or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1,rtol=0,atol=2e-6)):raise ValueError('Same-DURA original F0 and three frozen unit160 models required')
    closed=logits.argmax(1).astype(np.int64)
    return dict(default_logits=logits,closed=closed,baseline_closed=closed.copy(),signal_features=z)


def _smoke(b,models,device):
    from .star_open_feature_export import _synthetic
    values=np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32);a=_forward(b,models,values,device)
    if len(a['closed'])!=2:raise ValueError('Actual F0/G and all three open-model smoke required')
    return dict(status='PASS',actual_F0_signal_DURA_checkpoints=True,trained_SOURCE_and_TARGET_PCD=True,
        G_synthetic_forward_rows=2,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=6,real_data_reads=0,optimizer_updates=0)


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable body-prototype predictions required')
    pm,states,old_tau,old_cal=_reference(c);b=_backend(c,device)
    source,source_identity=_load_SOURCE(c,device);model,identity=_load_model(c,device)
    _source_identity(source_identity,c,pm);_identity(identity,c,pm)
    if b['model_identity']!=pm['model_identities']['E0'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:
        raise ValueError('Original F0/r03/G ancestry differs from source controls')
    models={'E0':b['model'],'SOURCE':source,'TARGET_PCD':model};identities=dict(E0=b['model_identity'],SOURCE=source_identity,TARGET_PCD=identity)
    for value in models.values():
        if sum(p.numel() for p in value.parameters())!=334721 or any(p.requires_grad or p.dtype!=torch.float32 for p in value.parameters()) or any(m.training for m in value.modules()):raise ValueError('Three actual frozen SignalOpen models required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,best._snapshot_model(m)) for m in [b['reader'],b['dura'][0],*models.values()]]
    def check():
        for model,before in snapshots:best._check_model(model,before)
    smoke=_smoke(b,models,device);check()
    if smoke['status']!='PASS':raise ValueError('Actual checkpoint smoke precedes source/query values')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    declared,U_ids,training_costs=_training_metadata(c,pm)
    if declared!=identity:raise ValueError('Actual new PCD checkpoint differs from completed training')
    used=set(U_ids.tolist());source_ids=set();known=None;diagnostics={}
    for role in ('L','OE'):
        iq=_iq_role(c,role,pm)
        try:
            ids=set(iq['ids'].tolist())
            if ids&(used|source_ids):raise ValueError('Source fitting and fresh-U roles overlap')
            source_ids.update(ids);z=source_eval._project(model,iq['restored'],device);m=iq['metadata']
            if role=='L':known,diagnostics['known']=density.fit_known(z,m['y'],m['rx_i'],m['day_i'])
            else:states['TARGET_PCD'],diagnostics['background']=density.fit_background(known,z,m['rx_i'],m['day_i'])
            check();print('FRESH_WORLD_SOURCE_PROGRESS',role,len(ids),flush=True)
        finally:source_eval._close_role(iq)
    frozen=deepcopy(states)
    for arm in ARMS:best._save(out/'models'/(arm+'.npz'),states[arm])
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm_order=list(ARMS),identities=identities,
        diagnostics=diagnostics,neural_states_frozen=True,source_fit_inputs_only=True,source_V_fit=False,novel_fit=False,target_Q_fit=False,
        source_density_models_fitted=1,normalizer_fit_role='L',background_fit_role='OE',TARGET_PCD_has_target_U_ancestor=True,E0_state_reused_exactly=True))
    def frozen_check():
        check()
        if not _same(states,frozen):raise ValueError('Frozen source statistics changed')
    def source(role):
        a=previous._source_rows(c['source_control_config']['output_root'],'source_DURA',role);iq=_iq_role(c,'V' if role=='known' else 'novel',pm)
        try:
            source_eval._align(iq,a,'V' if role=='known' else 'novel');ids=set(a['ids'].tolist())
            if ids&(used|source_ids):raise ValueError('Source fitting, validation, novel and U roles must be disjoint')
            source_ids.update(ids);z=source_eval._project(model,iq['restored'],device)
            raw=np.column_stack((a['raw_scores'][:,:4],_raw(states['TARGET_PCD'],z,a['closed'])))
            print('FRESH_WORLD_SOURCE_PROGRESS',role,len(ids),flush=True)
            return dict(a,signal_features=np.concatenate((a['signal_features'][:,:2],z[:,None]),axis=1),raw_scores=raw)
        finally:source_eval._close_role(iq)
    V=source('known');tau=np.empty((6,6),np.float64);tau[:4]=old_tau
    methods=deepcopy(old_cal['methods'])
    for j in (4,5):
        calibrated,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],V['raw_scores'][:,j],V['rx_i'],V['scenario'])
        tau[j]=calibrated['conditional_thresholds'];methods[METHODS[j]]=report
    (out/'calibration').mkdir();best._save(out/'thresholds.npz',dict(conditional=tau));_write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,known_count=COUNTS['V'],group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),methods=methods,
        conditional_thresholds=tau.tolist(),known_only=True,unknown_inputs_used=False,target_reads=0,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,
        TARGET_PCD_neural_ancestor_U_count=_train().U_COUNT,all_shared_source_thresholds=True))
    V['scores']=V['raw_scores']-tau[:,V['closed']].T;_validate(V,'known');best._save(out/'source/source_DURA/known_predictions.npz',V);frozen_tau=tau.copy();frozen_check()
    print(CALIBRATION_STATUS,flush=True)
    N=source('novel');N['scores']=N['raw_scores']-tau[:,N['closed']].T;_validate(N,'novel');best._save(out/'source/source_DURA/novel_predictions.npz',N);frozen_check()
    manifests={};paired_ids=None
    for condition in CONDITIONS[1:]:
        iq=None
        try:
            manifest,ids,iq=_public(c,condition)
            if set(ids.tolist())&(used|source_ids):raise ValueError('Future Q cannot overlap any fitting/training/source role')
            blocks=[]
            for start in range(0,len(ids),128):
                a=_forward(b,models,iq[start:start+128],device)
                raw=np.concatenate([_raw(states[arm],a['signal_features'][:,j],a['closed']) for j,arm in enumerate(ARMS)],axis=1)
                blocks.append(dict(a,raw_scores=raw,scores=raw-tau[:,a['closed']].T));print('FRESH_WORLD_Q_PROGRESS',condition,start+len(a['closed']),len(ids),flush=True)
            a={k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]};a['ids']=ids.copy()
            if paired_ids is None:paired_ids=ids.copy()
            elif not np.array_equal(ids,paired_ids):raise ValueError('Clean and star must share the exact future physical ID order')
            _validate(a,'Q');best._save(out/'targets'/condition/'predictions.npz',a);manifests[condition]=manifest;frozen_check()
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Source thresholds changed after novel/query access')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,arm_order=list(ARMS),conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS[1:]},Q_manifests=manifests,source_control_prediction_identity=pm,
        model_identities=identities,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=b['dura_identity'],parent_calibration_identity=old_cal,
        classification_path='same_DURA_output_original_F0',identity_baseline_path='same_DURA_output_original_F0',classification_unchanged=True,score_threshold=0.,
        source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,
        source_control_readouts_reused_exactly=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,
        source_density_models_fitted=1,new_threshold_sets_fitted=2,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,
        source_IQ_reads=sum(COUNTS.values()),target_Q_IQ_reads=2*Q_COUNT,target_U_reads=0,U_ID_metadata_reads=_train().U_COUNT,training_target_U_count=_train().U_COUNT,prototypes_deployed=False,projection_updates_per_arm=444,encoder_updates_per_arm=444,Sinc_optimizer_steps=0,new_body_updates=444,new_projection_updates=444,
        SOURCE_density_source_only=True,TARGET_PCD_density_source_only=False,source_only=False,target_DA=True,target_REG=False,
        threshold_values=36,threshold_bytes=288,new_threshold_values=12,reused_threshold_values=24,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,original_U_s_reads=0,
        new_neural_updates=0,inherited_SOURCE_updates=444,new_candidate_optimizer_updates=444,
        prediction_optimizer_updates=0,inherited_SOURCE_training_updates=444,
        real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,real_signal_forward_rows=sum(COUNTS.values())+6*Q_COUNT,
        real_signal_forward_rows_by_arm=dict(E0=2*Q_COUNT,SOURCE=2*Q_COUNT,TARGET_PCD=sum(COUNTS.values())+2*Q_COUNT),
        synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,synthetic_signal_forward_rows=6,
        training_costs=training_costs,combined_new_real_signal_forward_rows=training_costs['real_parent_signal_forward_rows']+training_costs['real_student_signal_forward_rows']+sum(COUNTS.values())+6*Q_COUNT,
        combined_new_G_forward_rows=training_costs['G_forward_rows']+2*Q_COUNT,combined_new_real_F_forward_rows=training_costs['real_F_forward_rows']+2*Q_COUNT,
        source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    frozen_check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


run=evaluate


def main():
    print(json.dumps(dict(event='FRESH_WORLD_EVALUATION_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
