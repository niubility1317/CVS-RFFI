"""Cached star-U affine EM, with frozen source Gaussians and original thresholds."""
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
from . import star_class_density_run as parent
from . import star_mixture_uda_run as mix
from . import star_class_shift_run as cached
from . import star_diagonal_uda_core as core
from .star_local_cov_run import _reference
from .development_transfer import _snapshot_model,_check_model

RUN_ID='20261031-phase3-diagonal-uda-s2026092701-r01'
SCHEMA='pamf_phase3_diagonal_uda_prediction_v1'
STATUS='DIAGONAL_UDA_PREDICTIONS_COMPLETE'
SOURCE_STATUS='DIAGONAL_UDA_SOURCE_FROZEN'
ADAPTER_STATUS='DIAGONAL_UDA_ADAPTERS_FROZEN'
PARENT_CODE='4af0668331682636ee212077b38cd912857a1b31'
ADAPTATION_CODE='d72a7b7c301fec2ff3736e4221dfe5047ab78efc'
METHODS=('pooled_ratio','predicted_ratio','shift_pooled_ratio','shift_predicted_ratio','diagonal_pooled_ratio','diagonal_predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=5
ADAPTERS=core.ARMS
CONDITIONS,COUNTS,Q_COUNT=parent.CONDITIONS,parent.COUNTS,parent.Q_COUNT
U_COUNT,U_CAPSULE_ID,U_SPLIT_ID=mix.U_COUNT,mix.U_CAPSULE_ID,mix.U_SPLIT_ID
FIELDS={'run_id','code_commit','reference_config','reference_adaptation_root','U_capsule_id','U_split_id','output_root','batch_size'}
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=parent.KNOWN_FIELDS,parent.NOVEL_FIELDS,parent.TARGET_FIELDS
_read,_write,_npz,_same=parent._read,parent._write,parent._npz,parent._same
_parent_rows,_parent_target_rows=parent._rows,parent._target_rows


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in FIELDS-{'batch_size','reference_config'})):
        raise ValueError('Exact cached-star-U affine configuration required; no truth/new IQ/K or query fitting inputs')
    p=parent.validate_config(c['reference_config'])
    if p['code_commit']!=PARENT_CODE or c['U_capsule_id']!=U_CAPSULE_ID or c['U_split_id']!=U_SPLIT_ID:
        raise ValueError('Only original selected source-only density and fixed star-G10 discovery U are allowed')
    output=Path(c['output_root']).resolve()
    paths=[c['reference_adaptation_root'],p['output_root'],p['reference_prediction_root'],p['train_config']['IQ_root'],p['train_config']['output_root']]
    if any(output.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(output) for v in paths):
        raise ValueError('New immutable output cannot overlap inherited source/model/cache inputs')
    return c


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _cached_U_header(c,pm):
    """Original mixture completion metadata only: no old adapters, prior, or row values."""
    done=_read(Path(c['reference_adaptation_root'])/'complete.json')
    expected=dict(schema=mix.SCHEMA,status=mix.STATUS,run_id=mix.RUN_ID,code_commit=ADAPTATION_CODE,
        method_names=list(mix.METHODS),primary_method=mix.PRIMARY_METHOD,known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,
        source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,
        source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,classification_unchanged=True,
        new_neural_updates=0,native_updates=0,density_fit_updates=0,threshold_fit_updates=0,private_truth_reads=0,U_truth_reads=0,
        source_novel_truth_reads=0,target_truth_reads=0,formal_target_accessed=False,final_phase3_complete=False)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Exact completed original mixture-UDA cache ancestry required')
    old=mix.validate_config(done['config']);identity=pm['model_identity'];prior=done.get('reference_prediction_identity',{})
    if (old['run_id']!=mix.RUN_ID or old['code_commit']!=ADAPTATION_CODE or Path(old['output_root']).resolve()!=Path(c['reference_adaptation_root']).resolve()
            or any(old[k]!=c[k] for k in ('U_capsule_id','U_split_id'))
            or done.get('model_identity')!=identity or done.get('parent_F0_identity')!=pm['parent_F0_identity'] or done.get('dura_identity')!=pm['dura_identity']
            or prior.get('run_id')!=mix.best.RUN_ID or prior.get('code_commit')!=mix.REFERENCE_CODE
            or prior.get('model_identity')!=identity or prior.get('parent_F0_identity')!=pm['parent_F0_identity'] or prior.get('dura_identity')!=pm['dura_identity']
            or prior.get('config',{}).get('train_config')!=c['reference_config']['train_config'] or prior.get('source_only') is not True
            or prior.get('target_U_reads')!=0 or prior.get('private_truth_reads')!=0):
        raise ValueError('Cached pre-adaptation U must use exactly the original source r03, F0 and DURA identity')
    manifest=done.get('U_manifest',{})
    if (manifest.get('count')!=U_COUNT or manifest.get('capsule_id')!=U_CAPSULE_ID or manifest.get('split_id')!=U_SPLIT_ID
            or manifest.get('registered_classes')!=parent.signal_train.native.CLASSES):raise ValueError('Fixed complete label-free star-G10 U capsule/split required')
    output=Path(c['output_root']).resolve();public=Path(old['U_public_root']).resolve()
    if output.is_relative_to(public) or public.is_relative_to(output):raise ValueError('Output cannot overlap inherited public U')
    return done


def _U_features(c,header):return cached._U_features(c,header)


def _rows(root,role):
    if role not in ('known','novel'):raise ValueError('Only fixed source known/novel roles allowed')
    a=_arrays(Path(root)/(role+'_predictions.npz'))
    if set(a)!=(KNOWN_FIELDS if role=='known' else NOVEL_FIELDS):raise ValueError('Exact source prediction fields required; novel private labels forbidden')
    parent.signal_eval._validate_rows({k:v for k,v in a.items() if k!='raw_signal_features'},role,6,True)
    value=a['raw_signal_features']
    if value.shape!=(len(a['ids']),160) or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Exact finite inherited raw signal features required')
    return a


def _target_rows(root,condition):
    if condition not in CONDITIONS:raise ValueError('Only the fixed clean/star DURA future conditions allowed')
    a=_arrays(Path(root)/'targets'/condition/'predictions.npz')
    if (set(a)!=TARGET_FIELDS or a['ids'].shape!=(Q_COUNT,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=Q_COUNT
            or np.any(np.char.strip(a['ids'])=='') or a['default_logits'].shape!=(Q_COUNT,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(Q_COUNT,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
            or a['baseline_closed'].shape!=(Q_COUNT,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])
            or a['signal_features'].shape!=(Q_COUNT,160) or a['signal_features'].dtype!=np.float32 or not np.isfinite(a['signal_features']).all()
            or not np.allclose(np.linalg.norm(a['signal_features'].astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)
            or any(a[key].shape!=(Q_COUNT,6) or a[key].dtype!=np.float64 or not np.isfinite(a[key]).all() for key in ('raw_scores','scores'))):
        raise ValueError('Complete label-free six-column predictions with unchanged original F0 identity required')
    return a


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable affine-UDA output required; never replace an existing run')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=parent._backend(c['reference_config'],device)
    snapshots=[(model,_snapshot_model(model)) for model in (b['reader'],b['model'],b['dura'][0])]
    def models_unchanged():
        for model,before in snapshots:_check_model(model,before)
    smoke=parent._smoke(b,device);models_unchanged()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint smoke must pass before cached U access')
    pm,state,tau,cal=_reference(c);identity=pm['model_identity'];header=_cached_U_header(c,pm)
    if b['model_identity']!=identity or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:
        raise ValueError('Actual original F0/r03/DURA must match the frozen source and cached U')
    frozen=deepcopy(state);six_tau=np.tile(tau,(3,1));frozen_tau=six_tau.copy()
    def check():
        models_unchanged()
        if not _same(state,frozen) or not np.array_equal(six_tau,frozen_tau):raise ValueError('Frozen source density or operating points changed')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);(out/'models').mkdir()
    _npz(out/'models/density.npz',state);_npz(out/'thresholds.npz',dict(conditional=six_tau));(out/'calibration').mkdir();_write(out/'calibration/complete.json',cal)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=SOURCE_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        models_frozen=True,models_refitted=False,thresholds_refitted=False,source_models_frozen_before_U=True,model_identity=identity,parent_calibration_identity=cal,U_reads=0,Q_reads=0))
    U=_U_features(c,header);adapters,diag=core.fit_adapters(state,U['signal_features']);check()
    if set(adapters)!=set(ADAPTERS):raise ValueError('Both preregistered affine adapters must complete')
    (out/'adaptation').mkdir()
    for arm in ADAPTERS:
        core.validate_adapter(adapters[arm])
        if adapters[arm]['kind'].item()!=arm:raise ValueError('Adapter kind must match the named arm')
        for value in adapters[arm].values():value.setflags(write=False)
        _npz(out/'adaptation'/(arm+'.npz'),adapters[arm])
    before=deepcopy(adapters);_npz(out/'adaptation/U_features.npz',U);_write(out/'adaptation/fit_diagnostics.json',diag)
    _write(out/'adaptation/complete.json',dict(schema=SCHEMA,status=ADAPTER_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],U_count=U_COUNT,U_manifest=header['U_manifest'],
        adapter_names=list(ADAPTERS),fit_input='cached_star_U_signal_features_only',iterations_per_adapter=20,ridge=160.,group_prior=[.5,.5],
        query_fit=False,true_U_labels_used=False,true_unknown_count_used=False,source_density_refitted=False,thresholds_refitted=False,
        adapters_frozen=True,Q_reads=0,source_prediction_reads=0,U_bank_deployed=False,private_U_features_ref='U_features.npz'))
    print(ADAPTER_STATUS,flush=True)
    def frozen_check():
        check()
        if not _same(adapters,before):raise ValueError('Frozen U adapters changed while predicting future records')
    seen=set(U['ids'].tolist())
    def controls(a):
        raw=a['raw_scores'][:,[1,3]];margin=a['scores'][:,[1,3]]
        if (not np.allclose(parent.core.raw_scores(state,a['signal_features'],a['closed'])[:,[1,3]],raw,rtol=1e-12,atol=1e-9)
                or not np.array_equal(margin,raw-tau[:,a['closed']].T)):raise ValueError('Original pooled/predicted ratio controls must reconstruct exactly')
        return raw,margin
    for role in ('known','novel'):
        a=_parent_rows(c['reference_config']['output_root'],role);ids=set(a['ids'].tolist())
        if ids&seen:raise ValueError('Source known/novel and U fitting records must be physically disjoint')
        seen.update(ids);raw,margin=controls(a);result=dict(a,raw_scores=np.tile(raw,(1,3)),scores=np.tile(margin,(1,3)))
        _npz(out/(role+'_predictions.npz'),result);_rows(out,role);frozen_check()
    for condition in CONDITIONS:
        a=_parent_target_rows(c['reference_config']['output_root'],condition)
        if seen&set(a['ids'].tolist()):raise ValueError('Future Q cannot overlap source or U fitting records')
        raw,margin=controls(a)
        six=np.concatenate([raw]+[core.raw_scores(state,a['signal_features'],a['closed'],adapters[arm]) for arm in ADAPTERS],axis=1)
        result=dict(a,raw_scores=six,scores=six-six_tau[:,a['closed']].T)
        if not np.array_equal(result['scores'][:,:2],margin):raise ValueError('Original target control margins changed')
        folder=out/'targets'/condition;folder.mkdir(parents=True);_npz(folder/'predictions.npz',result);_target_rows(out,condition);frozen_check()
        print('DIAGONAL_UDA_Q_COMPLETE',condition,len(a['ids']),flush=True)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        reference_prediction_identity=pm,reference_adaptation_identity=header,parent_calibration_identity=cal,model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,U_manifest=header['U_manifest'],Q_counts=pm['Q_counts'],Q_manifests=pm['Q_manifests'],
        classification_path='parent_original_F0_same_DURA_input',identity_baseline_path='parent_original_F0_same_DURA_input',classification_unchanged=True,score_threshold=0.,
        source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,all_thresholds_reused_exactly=True,
        source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,source_controls_reused_exactly=True,DA0_controls_reused_exactly=True,
        new_neural_updates=0,native_updates=0,density_fit_updates=0,threshold_fit_updates=0,adapter_iterations_per_arm=20,new_target_statistical_iterations=40,
        target_U_reads=U_COUNT,U_feature_rows_read=U_COUNT,U_features_reused=True,source_L_reads=0,source_OE_reads=0,source_IQ_reads=0,target_Q_IQ_reads=0,original_U_s_reads=0,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,
        synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,source_only=False,original_neural_models_source_only=True,target_DA=True,target_REG=False,
        source_bank_deployed=False,U_bank_deployed=False,per_record_source_rows_deployed=0,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    frozen_check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


evaluate=run


def main():
    print(json.dumps(dict(event='DIAGONAL_UDA_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0');a=parser.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
