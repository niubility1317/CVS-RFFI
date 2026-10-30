"""Class-domain target latent adaptation from cached star-U features only."""
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
from . import star_mixture_uda_run as mix

best=mix.best
RUN_ID='20261031-phase3-class-shift-s2026092701-r01'
SCHEMA='pamf_phase3_class_shift_prediction_v1'
SOURCE_STATUS='CLASS_SHIFT_SOURCE_FROZEN'
ADAPTER_STATUS='CLASS_SHIFT_ADAPTER_FROZEN'
STATUS='CLASS_SHIFT_PREDICTIONS_COMPLETE'
ADAPTATION_RUN='20261031-phase3-mixture-uda-s2026092701-r01'
ADAPTATION_CODE='d72a7b7c301fec2ff3736e4221dfe5047ab78efc'
METHODS=('DA0_known','DA0_ratio','SHARED_known','SHARED_ratio','CLASS_known','CLASS_ratio')
PRIMARY_METHOD='CLASS_ratio'
PRIMARY_INDEX=5
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
U_COUNT=384
Q_COUNT=864
U_CAPSULE_ID=mix.U_CAPSULE_ID
U_SPLIT_ID=mix.U_SPLIT_ID
FIELDS={'run_id','code_commit','reference_prediction_root','reference_adaptation_root','U_capsule_id','U_split_id','output_root','batch_size'}
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=best.KNOWN_FIELDS,best.NOVEL_FIELDS,best.TARGET_FIELDS
_read,_write,_same,_reference=best._read,best._write,best._same,mix._reference


def _core():
    from . import star_class_shift_core
    return star_class_shift_core


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in FIELDS-{'batch_size'})):
        raise ValueError('Exact eight-field cached-star-U class-shift configuration required; no truth/new IQ/Q input')
    if c['U_capsule_id']!=U_CAPSULE_ID or c['U_split_id']!=U_SPLIT_ID:raise ValueError('Only the original star-G10 discovery U may fit; clean is test only')
    output=Path(c['output_root']).resolve()
    for name in ('reference_prediction_root','reference_adaptation_root'):
        p=Path(c[name]).resolve()
        if output.is_relative_to(p) or p.is_relative_to(output):raise ValueError('Immutable output cannot overlap parent inputs')
    return c


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _shared(root,parent,state,tau,cal):
    """Old mixture metadata and aggregate SHARED state; no U/Q row values."""
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=mix.SCHEMA,status=mix.STATUS,run_id=ADAPTATION_RUN,code_commit=ADAPTATION_CODE,method_names=list(mix.METHODS),primary_method=mix.PRIMARY_METHOD,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,
        all_three_DURA_conditions_complete=True,source_controls_reused_exactly=True,DA0_controls_reused_exactly=True,all_thresholds_reused_exactly=True,
        source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,classification_unchanged=True,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,new_neural_updates=0,density_fit_updates=0,threshold_fit_updates=0,
        source_IQ_reads=0,target_Q_IQ_reads=0,formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Exact completed source-only-model mixture-UDA parent required')
    c=mix.validate_config(m['config']);pm,s,t,pc=mix._reference(c['reference_prediction_root'])
    if (c['run_id']!=m['run_id'] or c['code_commit']!=m['code_commit'] or pm!=parent or m['reference_prediction_identity']!=parent
            or not _same(s,state) or not np.array_equal(t,tau) or pc!=cal or m['parent_calibration_identity']!=cal
            or m['model_identity']!=parent['model_identity'] or m['parent_F0_identity']!=parent['parent_F0_identity'] or m['dura_identity']!=parent['dura_identity']):
        raise ValueError('SHARED must inherit exactly the same best/P1 signal/F0/DURA/density/threshold identity')
    if not _same(_arrays(root/'models/density.npz'),state):raise ValueError('Inherited source density changed')
    limits=_arrays(root/'thresholds.npz')
    if set(limits)!={'conditional'} or limits['conditional'].dtype!=np.float64 or not np.array_equal(limits['conditional'],tau):raise ValueError('Inherited source thresholds changed')
    fit=_read(root/'models/complete.json');done=_read(root/'adaptation/complete.json');diag=_read(root/'adaptation/fit_diagnostics.json')
    if (any(fit.get(k)!=v for k,v in dict(schema=mix.SCHEMA,status=mix.SOURCE_STATUS,run_id=ADAPTATION_RUN,code_commit=ADAPTATION_CODE,
                models_frozen=True,models_refitted=False,thresholds_refitted=False,source_prior_frozen_before_U=True).items()) or fit.get('parent_calibration_identity')!=cal
            or _read(root/'calibration/complete.json')!=cal
            or any(done.get(k)!=v for k,v in dict(schema=mix.SCHEMA,status=mix.ADAPTER_STATUS,run_id=ADAPTATION_RUN,code_commit=ADAPTATION_CODE,
                U_count=U_COUNT,adapter_names=['DA_K','DA_M'],fit_input='U_signal_features_only',fit_diagnostics_ref='fit_diagnostics.json',iterations_per_adapter=20,
                source_prior_frozen_before_U=True,query_fit=False,true_U_labels_used=False,true_unknown_count_used=False,source_density_refitted=False,thresholds_refitted=False,
                adapters_frozen=True,Q_reads=0,source_prediction_reads=0,U_bank_deployed=False,private_U_features_ref='U_features.npz').items())):
        raise ValueError('Complete frozen inherited SHARED adapter/source ancestry required')
    manifest=done.get('U_manifest',{})
    if (manifest!=m.get('U_manifest') or manifest.get('capsule_id')!=U_CAPSULE_ID or manifest.get('split_id')!=U_SPLIT_ID or manifest.get('count')!=U_COUNT
            or manifest.get('registered_classes')!=best.signal_train.native.CLASSES
            or diag.get('iterations')!=20 or diag.get('U_count')!=U_COUNT or diag.get('U_only') is not True
            or any(diag.get(k) is not False for k in ('Q_inputs_used','true_U_labels_used','true_unknown_count_used'))
            or diag.get('arms',{}).get('DA_M',{}).get('group_prior')!=[.5,.5]
            or [row.get('iteration') for row in diag.get('arms',{}).get('DA_M',{}).get('trace',[])]!=list(range(21))):raise ValueError('Inherited star-U identity and fixed twenty-step SHARED diagnostics required')
    prior=_arrays(root/'adaptation/source_prior.npz');adapter=_arrays(root/'adaptation/DA_M.npz')
    mix._core().validate_prior(prior);mix._core().validate_adapter(adapter,prior)
    for a in [*prior.values(),*adapter.values()]:a.setflags(write=False)
    return m,prior,adapter,diag


def _backend(parent,device):return mix._backend(parent,device)


def _U_features(c,parent):
    """Only called after actual-checkpoint smoke and source-class prior freeze."""
    a=_arrays(Path(c['reference_adaptation_root'])/'adaptation/U_features.npz')
    if (set(a)!={'ids','default_logits','closed','signal_features'} or a['ids'].shape!=(U_COUNT,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=U_COUNT
            or np.any(np.char.strip(a['ids'])=='') or a['default_logits'].shape!=(U_COUNT,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(U_COUNT,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
            or a['signal_features'].shape!=(U_COUNT,160) or a['signal_features'].dtype!=np.float32 or not np.isfinite(a['signal_features']).all()
            or not np.allclose(np.linalg.norm(a['signal_features'].astype(np.float64),axis=1),1,rtol=0,atol=2e-6)
            or parent['U_manifest']['capsule_id']!=c['U_capsule_id'] or parent['U_manifest']['split_id']!=c['U_split_id']):raise ValueError('Only complete aligned label-free original star-U signal features may fit')
    return a


def _parent_rows(root,role):return best._source_rows(root,'source_DURA',role)
def _parent_target_rows(root,condition):return best._target_rows(root,condition)
def _shared_target_rows(root,condition):return mix._target_rows(root,condition)
_source_rows,_target_rows=mix._source_rows,mix._target_rows


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable class-shift output root required')
    pm,state,tau,cal=_reference(c['reference_prediction_root']);am,shared_prior,shared,shared_diag=_shared(c['reference_adaptation_root'],pm,state,tau,cal)
    core=_core();identity=pm['model_identity']
    for p in (pm['config']['train_config']['output_root'],pm['config']['source_IQ_root'],pm['config']['dura_release'],pm['config']['dura_checkpoint'],am['config']['U_public_root']):
        p=Path(p).resolve();o=out.resolve()
        if o.is_relative_to(p) or p.is_relative_to(o):raise ValueError('Output cannot overlap inherited model/IQ/public inputs')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=_backend(pm,device)
    if b['model_identity']!=identity or b['reader_identity']!=pm['parent_F0_identity']:raise ValueError('Actual P1 signal and original F0 differ from both frozen controls')
    actual=b['dura_identity'];expected=pm['dura_identity']
    if any(actual.get(k)!=v or expected.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()) or any(actual.get(k)!=expected[k] for k in ('checkpoint','arm','training_commit','reuse','graph','native','precision','TF32') if k in expected):raise ValueError('Original DURA v1 computation required')
    models=[(m,best._snapshot_model(m)) for m in (b['reader'],b['model'],b['dura'][0])]
    frozen=deepcopy(state);frozen_tau=tau.copy();tau.setflags(write=False);shared_before=deepcopy(shared);shared_prior_before=deepcopy(shared_prior)
    def check():
        for m,before in models:best._check_model(m,before)
        if (not _same(state,frozen) or not np.array_equal(tau,frozen_tau) or not _same(shared,shared_before)
                or not _same(shared_prior,shared_prior_before)):raise ValueError('Frozen source/shared state changed')
    smoke=best._smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint smoke must pass before cached U access')
    prior=core.build_prior(state);prior_before=deepcopy(prior)
    out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);best._save(out/'models/density.npz',state);best._save(out/'thresholds.npz',dict(conditional=tau))
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=SOURCE_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],model_identity=identity,models_frozen=True,models_refitted=False,thresholds_refitted=False,
        parent_calibration_identity=cal,source_prior_frozen_before_U=True,U_reads=0,Q_reads=0))
    (out/'calibration').mkdir();_write(out/'calibration/complete.json',cal);best._save(out/'adaptation/source_prior.npz',prior);best._save(out/'adaptation/shared_prior.npz',shared_prior)
    best._save(out/'adaptation/SHARED.npz',shared);_write(out/'adaptation/shared_diagnostics.json',shared_diag)
    U=_U_features(c,am);adapter,diag=core.fit_adapter(state,U['signal_features'],prior);core.validate_adapter(adapter,prior);check()
    if not _same(prior,prior_before):raise ValueError('Source class-domain prior changed during U fit')
    for value in adapter.values():value.setflags(write=False)
    adapter_before=deepcopy(adapter);best._save(out/'adaptation/CLASS.npz',adapter);best._save(out/'adaptation/U_features.npz',U);_write(out/'adaptation/fit_diagnostics.json',diag)
    _write(out/'adaptation/complete.json',dict(schema=SCHEMA,status=ADAPTER_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],U_count=U_COUNT,U_manifest=am['U_manifest'],
        adapter_names=['SHARED','CLASS'],fit_input='cached_star_U_signal_features_only',fit_diagnostics_ref='fit_diagnostics.json',shared_diagnostics_ref='shared_diagnostics.json',
        iterations=20,inherited_shared_iterations=20,source_prior_frozen_before_U=True,query_fit=False,true_U_labels_used=False,true_unknown_count_used=False,source_density_refitted=False,thresholds_refitted=False,
        adapters_frozen=True,Q_reads=0,source_prediction_reads=0,U_bank_deployed=False,private_U_features_ref='U_features.npz'))
    print(ADAPTER_STATUS,flush=True)
    def frozen_check():
        check()
        if not _same(prior,prior_before) or not _same(adapter,adapter_before):raise ValueError('Frozen CLASS target state changed on source/Q')
    U_ids=set(U['ids'].tolist());source_ids=set();six_tau=np.tile(tau,(3,1))
    for role in ('known','novel'):
        a=_parent_rows(c['reference_prediction_root'],role);ids=set(a['ids'].tolist())
        if ids&(U_ids|source_ids):raise ValueError('Source known/novel and fit U roles overlap')
        if not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('Source controls changed')
        source_ids.update(ids);result=dict(a,raw_scores=np.tile(a['raw_scores'],(1,3)),scores=np.tile(a['scores'],(1,3)))
        mix._validate(result,role);best._save(out/'source/source_DURA'/(role+'_predictions.npz'),result);frozen_check()
    for condition in CONDITIONS[1:]:
        a=_parent_target_rows(c['reference_prediction_root'],condition);old=_shared_target_rows(c['reference_adaptation_root'],condition)
        if set(a['ids'].tolist())&(U_ids|source_ids):raise ValueError('Future Q cannot overlap source or U fit records')
        if (any(not np.array_equal(old[k],v) for k,v in a.items() if k not in ('raw_scores','scores'))
                or not np.array_equal(old['raw_scores'][:,:2],a['raw_scores']) or not np.array_equal(old['scores'][:,:2],a['scores'])
                or not np.allclose(mix._core().scores(state,a['signal_features'],shared['delta']),old['raw_scores'][:,4:6],rtol=1e-12,atol=1e-9)
                or not np.array_equal(old['scores'][:,4:6],old['raw_scores'][:,4:6]-tau[:,a['closed']].T)):
            raise ValueError('Exact same-record DA0/SHARED controls must preserve identity and reconstruct')
        raw=np.concatenate((a['raw_scores'],old['raw_scores'][:,4:6],core.scores(state,a['signal_features'],adapter['delta'])),axis=1)
        margin=np.concatenate((a['scores'],old['scores'][:,4:6],raw[:,4:6]-tau[:,a['closed']].T),axis=1)
        result=dict(a,raw_scores=raw,scores=margin)
        if not np.array_equal(margin,raw-six_tau[:,a['closed']].T):raise ValueError('Original two threshold vectors must route all six methods')
        mix._validate(result,'Q');best._save(out/'targets'/condition/'predictions.npz',result);frozen_check()
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,Q_counts={name:Q_COUNT for name in CONDITIONS[1:]},U_manifest=am['U_manifest'],Q_manifests=pm['Q_manifests'],
        reference_prediction_identity=pm,reference_adaptation_identity=am,parent_calibration_identity=cal,model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=expected,loaded_dura_identity=actual,
        classification_path='parent_original_F0_same_DURA_input',identity_baseline_path='parent_original_F0_same_DURA_input',classification_unchanged=True,score_threshold=0.,
        source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,source_controls_reused_exactly=True,DA0_controls_reused_exactly=True,shared_controls_reused_exactly=True,
        all_thresholds_reused_exactly=True,source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,
        source_IQ_reads=0,source_L_reads=0,source_OE_reads=0,target_Q_IQ_reads=0,target_U_reads=U_COUNT,U_feature_rows_read=U_COUNT,original_U_s_reads=0,private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,
        new_neural_updates=0,native_updates=0,density_fit_updates=0,source_density_fit_updates=0,normalizer_fit_updates=0,threshold_fit_updates=0,adapter_iterations=20,new_target_statistical_iterations=20,inherited_shared_adapter_iterations=20,
        real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,
        source_bank_deployed=False,U_bank_deployed=False,per_record_source_rows_deployed=0,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    frozen_check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


evaluate=run


def main(argv=None):
    print(json.dumps(dict(event='CLASS_SHIFT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv);run(_read(a.config),a.device)


if __name__=='__main__':main()
