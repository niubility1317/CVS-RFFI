"""Frozen projection-only prototype students with predicted-class source densities."""
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

RUN_ID='20261031-phase3-target-prototype-s2026092701-r01'
SCHEMA='pamf_phase3_target_prototype_prediction_v1'
STATUS='TARGET_PROTOTYPE_PREDICTIONS_COMPLETE'
MODELS_STATUS='TARGET_PROTOTYPE_DENSITIES_FROZEN'
CALIBRATION_STATUS='TARGET_PROTOTYPE_THRESHOLDS_FROZEN'
METHODS=('E0_pooled_ratio','E0_predicted_ratio','SOURCE_pooled_ratio','SOURCE_predicted_ratio','PCD_pooled_ratio','PCD_predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=5
MODEL_NAMES=('E0','SOURCE','TARGET_PCD')
ARMS=MODEL_NAMES
PARENT_CODE='4af0668331682636ee212077b38cd912857a1b31'
U_COUNT=384
from .star_mixture_uda_run import U_CAPSULE_ID,U_SPLIT_ID
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
FIELDS={'run_id','code_commit','train_config','reference_config','output_root','batch_size'}
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=parent.KNOWN_FIELDS,parent.NOVEL_FIELDS,parent.TARGET_FIELDS
_read,_write,_same=best._read,best._write,best._same


def _train():
    from . import star_target_prototype_train
    return star_target_prototype_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field frozen TARGET-PROTOTYPE evaluation required; no truth or role overrides')
    t=_train().validate_config(c['train_config']);p=parent.validate_config(c['reference_config'])
    if t['run_id']!=RUN_ID or t['code_commit']!=c['code_commit'] or t['reference_config']!=p or p['code_commit']!=PARENT_CODE:
        raise ValueError('Actual training and original selected 4af identity must match exactly')
    out=Path(c['output_root']).resolve()
    roots=[t['output_root'],t['U_public_root'],p['output_root'],p['reference_prediction_root'],p['train_config']['IQ_root'],p['train_config']['output_root']]+[a['public_root'] for a in p['Q_conditions'].values()]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable prediction output cannot overlap model/source/public inputs')
    return c


def _reference(c):return source_reference(c)
def _backend(c,device):return parent._backend(c['reference_config'],device)
def _load_model(c,arm,device):return _train().load_trained_model(c['train_config'],arm,device)
def _iq_role(c,role,identity):return source_eval._iq_role(dict(IQ_root=c['reference_config']['train_config']['IQ_root']),role,identity)
def _public(c,name):return parent._public(c['reference_config'],name)


def _identity(i,arm,c,pm):
    expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,updates=444,projection_updates=444,projection_cumulative_updates=2664,
        open_signal_updates=0,native_updates=0,encoder_frozen=True,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,
        signal_feature_cumulative_updates=13320,source_only=arm=='SOURCE',target_U_used=arm=='TARGET_PCD',target_U_reads=U_COUNT if arm=='TARGET_PCD' else 0)
    if (any(i.get(k)!=v for k,v in expected.items()) or i.get('parent_signal_identity')!=pm['model_identity']
            or i.get('parent_F0_identity')!=pm['parent_F0_identity'] or i.get('dura_identity')!=pm['dura_identity']):
        raise ValueError('Strict frozen projection-only SOURCE/TARGET_PCD ancestry required')
    return i


def _training_metadata(c,pm):
    root=Path(c['train_config']['output_root']);m=_read(root/'complete.json')
    expected=dict(schema='pamf_phase3_target_prototype_train_v1',status='TARGET_PROTOTYPE_TRAINING_COMPLETE',run_id=RUN_ID,code_commit=c['code_commit'],
        config=c['train_config'],arms=['SOURCE','TARGET_PCD'],updates_per_arm=444,total_optimizer_updates=888,
        source_sampling_matched=True,SOURCE_completed_before_U_open=True,original_models_unchanged=True,
        all_student_buffers_unchanged=True,all_student_signal_encoders_unchanged=True)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Both complete matched projection training arms are required')
    manifest=m.get('U_manifest',{})
    if any(manifest.get(k)!=v for k,v in dict(count=U_COUNT,capsule_id=c['train_config']['U_capsule_id'],split_id=c['train_config']['U_split_id']).items()):
        raise ValueError('Fixed label-free star-U capsule and split ancestry required')
    identities={'E0':pm['model_identity']}
    for arm in MODEL_NAMES[1:]:
        row=_read(root/arm/'complete.json');identity=row.get('identity',{})
        _identity(identity,arm,c,pm)
        required=dict(schema='pamf_phase3_target_prototype_train_v1',status='TARGET_PROTOTYPE_STUDENT_FROZEN',updates=444,
            source_only=arm=='SOURCE',target_U_used=arm=='TARGET_PCD',target_U_reads=U_COUNT if arm=='TARGET_PCD' else 0,
            initial_state_equal_to_parent=True,student_encoder_unchanged=True,model_buffers_unchanged=True)
        if (any(row.get(k)!=v for k,v in required.items()) or m.get('rows',{}).get(arm,{}).get('identity')!=identity
                or m.get('identities',{}).get(arm)!=identity or row.get('U_manifest')!=(manifest if arm=='TARGET_PCD' else None)):
            raise ValueError('Global and frozen row training identities differ')
        identities[arm]=identity
    with np.load(root/'U_features.npz',allow_pickle=False) as a:ids=a['ids'].copy()
    if ids.shape!=(U_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=U_COUNT or np.any(np.char.strip(ids)==''):raise ValueError('Fixed complete opaque training U IDs required for role isolation')
    return identities,ids


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
    if out.exists():raise FileExistsError('New immutable target-prototype predictions required')
    pm,old_state,old_tau,old_cal=_reference(c);b=_backend(c,device)
    models={'E0':b['model']};identities={'E0':pm['model_identity']}
    for arm in ARMS[1:]:models[arm],identities[arm]=_load_model(c,arm,device)
    if b['model_identity']!=pm['model_identity'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:
        raise ValueError('Original F0/P1/DURA input computation differs from the fixed reference')
    for arm in ARMS[1:]:_identity(identities[arm],arm,c,pm)
    for model in models.values():
        if sum(p.numel() for p in model.parameters())!=334721 or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('All trained models must freeze before source statistics and queries')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,best._snapshot_model(m)) for m in [b['reader'],b['dura'][0],*models.values()]]
    def check():
        for model,before in snapshots:best._check_model(model,before)
    smoke=_smoke(b,models,device);check()
    if smoke['status']!='PASS':raise ValueError('Actual checkpoint smoke precedes source/query values')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    declared,U_ids=_training_metadata(c,pm)
    if declared!=identities:raise ValueError('Actual loaded students differ from complete training metadata')
    used=set(U_ids.tolist());source_ids=set();states={'E0':old_state};diagnostics={}
    knowns={};features={};metadata={}
    for role in ('L','OE'):
        iq=_iq_role(c,role,pm['model_identity'])
        try:
            ids=set(iq['ids'].tolist())
            if ids&(used|source_ids):raise ValueError('Source fitting and target U physical roles overlap')
            source_ids.update(ids);metadata[role]=iq['metadata'];features[role]={}
            for arm in ARMS[1:]:
                features[role][arm]=source_eval._project(models[arm],iq['restored'],device)
                print('TARGET_PROTOTYPE_SOURCE_PROGRESS',role,arm,len(iq['ids']),flush=True)
            check()
            for arm in ARMS[1:]:
                m=iq['metadata']
                if role=='L':knowns[arm],d=density.fit_known(features[role][arm],m['y'],m['rx_i'],m['day_i']);diagnostics[arm]={'known':d}
                else:states[arm],d=density.fit_background(knowns[arm],features[role][arm],m['rx_i'],m['day_i']);diagnostics[arm]['background']=d
        finally:source_eval._close_role(iq)
    del features,metadata
    frozen=deepcopy(states)
    for arm in ARMS:best._save(out/'models'/(arm+'.npz'),states[arm])
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm_order=list(ARMS),identities=identities,
        diagnostics=diagnostics,neural_states_frozen=True,source_fit_inputs_only=True,source_V_fit=False,novel_fit=False,target_Q_fit=False,
        source_density_models_fitted=2,normalizer_fit_role='L',background_fit_role='OE',TARGET_PCD_has_target_U_ancestor=True,E0_state_reused_exactly=True))
    def frozen_check():
        check()
        if not _same(states,frozen):raise ValueError('Frozen source statistics changed')
    def source(role):
        a=parent._rows(c['reference_config']['output_root'],role);iq=_iq_role(c,'V' if role=='known' else 'novel',pm['model_identity'])
        try:
            source_eval._align(iq,a,'V' if role=='known' else 'novel');ids=set(a['ids'].tolist())
            if ids&(used|source_ids):raise ValueError('Source fitting, validation, novel and U roles must be disjoint')
            source_ids.update(ids);z=[a['signal_features']]
            for arm in ARMS[1:]:
                z.append(source_eval._project(models[arm],iq['restored'],device))
                print('TARGET_PROTOTYPE_SOURCE_PROGRESS',role,arm,len(iq['ids']),flush=True)
            raw=np.concatenate([a['raw_scores'][:,[1,3]]]+[_raw(states[arm],z[j],a['closed']) for j,arm in enumerate(ARMS[1:],1)],axis=1)
            return dict(a,signal_features=np.stack(z,axis=1),raw_scores=raw)
        finally:source_eval._close_role(iq)
    V=source('known');tau=np.empty((6,6),np.float64);tau[:2]=old_tau
    methods={METHODS[j]:deepcopy(old_cal['methods'][parent.METHODS[(1,3)[j]]]) for j in range(2)}
    for j in range(2,6):
        calibrated,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],V['raw_scores'][:,j],V['rx_i'],V['scenario'])
        tau[j]=calibrated['conditional_thresholds'];methods[METHODS[j]]=report
    (out/'calibration').mkdir();best._save(out/'thresholds.npz',dict(conditional=tau));_write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,known_count=COUNTS['V'],group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),methods=methods,
        conditional_thresholds=tau.tolist(),known_only=True,unknown_inputs_used=False,target_reads=0,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,
        TARGET_PCD_neural_ancestor_U_count=_train().U_COUNT,all_shared_source_thresholds=True))
    V['scores']=V['raw_scores']-tau[:,V['closed']].T;_validate(V,'known');best._save(out/'source/source_DURA/known_predictions.npz',V);frozen_tau=tau.copy();frozen_check()
    print(CALIBRATION_STATUS,flush=True)
    N=source('novel');N['scores']=N['raw_scores']-tau[:,N['closed']].T;_validate(N,'novel');best._save(out/'source/source_DURA/novel_predictions.npz',N);frozen_check()
    manifests={}
    for condition in CONDITIONS[1:]:
        iq=None
        try:
            manifest,ids,iq=_public(c,condition)
            if set(ids.tolist())&(used|source_ids):raise ValueError('Future Q cannot overlap any fitting/training/source role')
            blocks=[]
            for start in range(0,len(ids),128):
                a=_forward(b,models,iq[start:start+128],device)
                raw=np.concatenate([_raw(states[arm],a['signal_features'][:,j],a['closed']) for j,arm in enumerate(ARMS)],axis=1)
                blocks.append(dict(a,raw_scores=raw,scores=raw-tau[:,a['closed']].T));print('TARGET_PROTOTYPE_Q_PROGRESS',condition,start+len(a['closed']),len(ids),flush=True)
            a={k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]};a['ids']=ids.copy()
            old=parent._target_rows(c['reference_config']['output_root'],condition)
            if (not np.array_equal(ids,old['ids']) or not np.array_equal(a['closed'],old['closed'])
                    or not np.allclose(a['default_logits'],old['default_logits'],rtol=2e-5,atol=1e-5)
                    or not np.allclose(a['signal_features'][:,0],old['signal_features'],rtol=2e-5,atol=1e-5)
                    or not np.allclose(a['raw_scores'][:,:2],old['raw_scores'][:,[1,3]],rtol=1e-12,atol=1e-9)):
                raise ValueError('Actual same-G original F0/r03 must reproduce the cached 4af baseline and controls')
            a['default_logits']=old['default_logits'].copy();a['closed']=old['closed'].copy();a['baseline_closed']=old['baseline_closed'].copy()
            a['signal_features'][:,0]=old['signal_features'];a['raw_scores'][:,:2]=old['raw_scores'][:,[1,3]];a['scores'][:,:2]=old['scores'][:,[1,3]]
            _validate(a,'Q');best._save(out/'targets'/condition/'predictions.npz',a);manifests[condition]=manifest;frozen_check()
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Source thresholds changed after novel/query access')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,arm_order=list(ARMS),conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS[1:]},Q_manifests=manifests,reference_prediction_identity=pm,
        model_identities=identities,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=b['dura_identity'],parent_calibration_identity=old_cal,
        classification_path='same_DURA_output_original_F0',identity_baseline_path='same_DURA_output_original_F0',classification_unchanged=True,score_threshold=0.,
        source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,
        source_E0_controls_reused_exactly=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,
        source_density_models_fitted=2,new_threshold_sets_fitted=4,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,
        source_IQ_reads=sum(COUNTS.values()),target_Q_IQ_reads=2*Q_COUNT,target_U_reads=0,U_ID_metadata_reads=_train().U_COUNT,training_target_U_count=_train().U_COUNT,prototypes_deployed=False,projection_updates_per_arm=444,encoder_updates_per_arm=0,
        SOURCE_density_source_only=True,TARGET_PCD_density_source_only=False,source_only=False,target_DA=True,target_REG=False,
        threshold_values=36,threshold_bytes=288,new_threshold_values=24,reused_threshold_values=12,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,original_U_s_reads=0,
        new_neural_updates=0,inherited_new_SOURCE_updates=_train().STEPS,inherited_new_TARGET_PCD_updates=_train().STEPS,
        real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,real_signal_forward_rows=2*sum(COUNTS.values())+6*Q_COUNT,
        real_signal_forward_rows_by_arm=dict(E0=2*Q_COUNT,SOURCE=sum(COUNTS.values())+2*Q_COUNT,TARGET_PCD=sum(COUNTS.values())+2*Q_COUNT),
        synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,synthetic_signal_forward_rows=6,
        source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    frozen_check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


run=evaluate


def main():
    print(json.dumps(dict(event='TARGET_PROTOTYPE_EVALUATION_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
