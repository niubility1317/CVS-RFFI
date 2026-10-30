"""Source-radius discovery and joint anonymous registration on frozen r03."""
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
from . import star_signal_register_core as core
from . import star_mixture_uda_run as U_reader
from .star_local_cov_run import _reference, _source_L
from .development_transfer import _snapshot_model, _check_model

RUN_ID='20261031-phase3-signal-register-s2026092701-r01'
SCHEMA='pamf_phase3_signal_register_prediction_v1'
STATUS='SIGNAL_REGISTER_PREDICTIONS_COMPLETE'
MODELS_STATUS='SIGNAL_REGISTER_SOURCE_FROZEN'
REGISTRATION_STATUS='SIGNAL_REGISTER_ANONYMOUS_FROZEN'
CALIBRATION_STATUS='SIGNAL_REGISTER_THRESHOLDS_FROZEN'
METHODS=('pooled_REG0','predicted_REG0','pooled_REG1','predicted_REG1')
PRIMARY_METHOD=METHODS[3]
BASE_METHODS=('pooled_ratio','predicted_ratio')
CONDITIONS=parent.CONDITIONS
COUNTS=parent.COUNTS
Q_COUNT=parent.Q_COUNT
U_COUNT=U_reader.U_COUNT
CACHED_U_FEATURES=False
PARENT_CODE='4af0668331682636ee212077b38cd912857a1b31'
FIELDS={'run_id','code_commit','reference_config','output_root','batch_size','U_public_root','U_capsule_id','U_split_id'}
_read,_write,_npz,_same=parent._read,parent._write,parent._npz,parent._same
_rows,_target_rows=parent._rows,parent._target_rows


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('output_root','U_public_root','U_capsule_id','U_split_id'))):
        raise ValueError('Exact frozen source/U-only registration configuration; no truth/K/Q fit inputs')
    p=parent.validate_config(c['reference_config'])
    if p['code_commit']!=PARENT_CODE:raise ValueError('Only the original selected source-only class-density model is allowed')
    u=Path(c['U_public_root'])
    if (u.name!='public' or u.parent.name!='discovery' or c['U_capsule_id']!=U_reader.U_CAPSULE_ID
            or c['U_split_id']!=U_reader.U_SPLIT_ID):raise ValueError('Only the fixed star-G10 unlabeled U capsule/split may register')
    out=Path(c['output_root']).resolve()
    paths=[p['output_root'],p['reference_prediction_root'],p['train_config']['IQ_root'],p['train_config']['output_root'],
           p['dura_release'],p['dura_checkpoint'],c['U_public_root']]+[q['public_root'] for q in p['Q_conditions'].values()]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in paths):
        raise ValueError('Immutable output cannot overlap source, checkpoint, U or future-Q inputs')
    return c


def _U(c,b,device):
    iq=None
    try:
        manifest,ids,iq=U_reader._public(c);blocks=[]
        for start in range(0,len(ids),128):
            value=parent._current(b,iq[start:start+128],device);value.pop('baseline_closed');blocks.append(value)
            print('SIGNAL_REGISTER_U_PROGRESS',start+len(value['closed']),len(ids),flush=True)
        result={k:np.concatenate([a[k] for a in blocks]) for k in blocks[0]};result['ids']=ids.copy()
        return result,deepcopy(manifest)
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()


def _base(state,features,closed,tau):
    raw=parent.core.raw_scores(state,features,closed)[:,[1,3]]
    return raw,raw-tau[:,closed].T


def _produce(a,state,anonymous,tau,registration_tau):
    raw,base=_base(state,a['signal_features'],a['closed'],tau)
    if not np.allclose(raw,a['raw_scores'][:,[1,3]],rtol=1e-12,atol=1e-9):raise ValueError('Frozen selected controls do not reconstruct')
    if not np.array_equal(a['scores'][:,[1,3]],a['raw_scores'][:,[1,3]]-tau[:,a['closed']].T):
        raise ValueError('Parent source operating points differ')
    raw=a['raw_scores'][:,[1,3]].copy();base=a['scores'][:,[1,3]].copy()
    new=core.anonymous_ratios(state,anonymous,a['signal_features'])
    predictions,decision=core.predict(a['closed'],base,new,registration_tau)
    return dict(a,raw_scores=raw,scores=base,predictions=predictions,decision_scores=decision,anonymous_raw_scores=new)


def _source(c,b,identity,device,state):
    features,meta,ids,manifest=_source_L(c,b,identity,device)
    source,diag=core.source_radius(state,features,meta['y'],meta['rx'],meta['day'])
    return source,diag,ids,manifest,dict(source_L_reads=COUNTS['L'],source_signal_rows=COUNTS['L'],source_radius_new_fits=1)


def run(c,device='cuda:0',*,policy=None):
    policy=sys.modules[__name__] if policy is None else policy
    RUN_ID,SCHEMA,STATUS=policy.RUN_ID,policy.SCHEMA,policy.STATUS
    MODELS_STATUS,REGISTRATION_STATUS,CALIBRATION_STATUS=policy.MODELS_STATUS,policy.REGISTRATION_STATUS,policy.CALIBRATION_STATUS
    COUNTS,U_COUNT,core=policy.COUNTS,policy.U_COUNT,policy.core
    validate_config,_source,_U=policy.validate_config,policy._source,policy._U
    cached_U=getattr(policy,'CACHED_U_FEATURES',False)
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('Observe original immutable run; never replace or restart it')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=c['reference_config'];b=parent._backend(p,device)
    snapshots=[(model,_snapshot_model(model)) for model in (b['reader'],b['model'],b['dura'][0])]
    def unchanged():
        for model,before in snapshots:_check_model(model,before)
    smoke=parent._smoke(b,device);unchanged()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:
        raise ValueError('Actual source checkpoints must pass a no-query smoke before source L or U')
    pm,state,tau,old_cal=_reference(c);identity=pm['model_identity']
    if (b['model_identity']!=identity or b['reader_identity']!=identity['parent_F0_identity']
            or b['dura_identity']!=pm['loaded_dura_identity']):raise ValueError('Same source r03, original F0 and DURA v1 required')
    original=deepcopy(state);old_tau=tau.copy();out.mkdir(parents=True)
    for name in ('models','registration','calibration'):(out/name).mkdir()
    _write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    source,source_diag,L_ids,L_manifest,source_cost=_source(c,b,identity,device,state);unchanged()
    source_before=deepcopy(source)
    _npz(out/'models/density.npz',state);_npz(out/'models/source_radius.npz',source)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        model_identity=identity,source_L_count=COUNTS['L'],source_L_manifest=L_manifest,source_radius_fitted=True,
        source_radius_frozen_before_U=True,source_radius_new_fits=source_cost['source_radius_new_fits'],current_source_L_reads=source_cost['source_L_reads'],
        U_features_reused=cached_U,original_density_refitted=False,source_V_fit=False,target_U_reads=0,diagnostics=source_diag))
    U,U_manifest=_U(c,b,device);unchanged()
    if set(L_ids.tolist())&set(U['ids'].tolist()):raise ValueError('Target U cannot overlap source L')
    _,base=_base(state,U['signal_features'],U['closed'],tau);U['base_scores']=base
    anonymous,assignments,diagnostics=core.discover(state,source,U['signal_features'],base,U['ids'])
    if not np.array_equal(assignments['ids'],U['ids']):raise ValueError('Discovery must preserve the exact opaque U ID order')
    anonymous_before=deepcopy(anonymous);U.update({k:v for k,v in assignments.items() if k!='ids'})
    _npz(out/'models/anonymous.npz',anonymous);_npz(out/'registration/U.npz',U)
    _write(out/'registration/complete.json',dict(schema=SCHEMA,status=REGISTRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        U_count=U_COUNT,U_manifest=U_manifest,anonymous_count=len(anonymous['means']),anonymous_ids=anonymous['anonymous_ids'].tolist(),
        diagnostics=diagnostics,anonymous_frozen_before_V_and_Q=True,true_U_labels_used=False,true_unknown_count_used=False,
        query_fit=False,source_prediction_reads=0,source_radius_frozen_before_U=True,U_bank_deployed=False))
    print(REGISTRATION_STATUS,len(anonymous['means']),flush=True)
    def check():
        unchanged()
        if not _same(state,original) or not _same(source,source_before) or not _same(anonymous,anonymous_before) or not np.array_equal(tau,old_tau):
            raise ValueError('Original density, thresholds or frozen registration state changed')
    check();seen=set(L_ids.tolist())|set(U['ids'].tolist());V=_rows(p['output_root'],'known')
    if seen&set(V['ids'].tolist()):raise ValueError('Known V cannot overlap source training or U discovery')
    _,V_base=_base(state,V['signal_features'],V['closed'],tau)
    if not np.allclose(V_base,V['scores'][:,[1,3]],rtol=1e-12,atol=1e-9):raise ValueError('Original V margins differ')
    new=core.anonymous_ratios(state,anonymous,V['signal_features'])
    registration_tau,report=core.calibrate(V['ids'],V['y'],V['closed'],V['baseline_closed'],V['scores'][:,[1,3]],new,V['rx_i'],V['scenario'])
    frozen_registration_tau=registration_tau.copy();_npz(out/'thresholds.npz',dict(conditional=tau,registration=registration_tau))
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,base_method_names=list(BASE_METHODS),known_count=COUNTS['V'],
        conditional_thresholds=tau.tolist(),registration_thresholds=registration_tau.tolist(),report=report,
        known_only=True,unknown_inputs_used=False,target_label_reads=0,U_state_frozen=True,source_V_neural_fit=False,
        source_V_density_fit=False,source_V_threshold_fit=True,shared_thresholds_for_all_conditions=True))
    check();_npz(out/'known_predictions.npz',_produce(V,state,anonymous,tau,registration_tau));seen.update(V['ids'].tolist())
    print(CALIBRATION_STATUS,flush=True)
    N=_rows(p['output_root'],'novel')
    if seen&set(N['ids'].tolist()):raise ValueError('Source novel cannot overlap L/V/U')
    _npz(out/'novel_predictions.npz',_produce(N,state,anonymous,tau,registration_tau));seen.update(N['ids'].tolist());check()
    for condition in CONDITIONS:
        a=_target_rows(p['output_root'],condition)
        if seen&set(a['ids'].tolist()):raise ValueError('Future Q must be disjoint from all fitting/calibration/source roles')
        folder=out/'targets'/condition;folder.mkdir(parents=True)
        _npz(folder/'predictions.npz',_produce(a,state,anonymous,tau,registration_tau));check()
        print('SIGNAL_REGISTER_Q_COMPLETE',condition,len(a['ids']),flush=True)
    if not np.array_equal(registration_tau,frozen_registration_tau):raise ValueError('Registration threshold changed after novel/query access')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        reference_prediction_identity=pm,model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,U_manifest=U_manifest,Q_counts=pm['Q_counts'],Q_manifests=pm['Q_manifests'],
        anonymous_count=len(anonymous['means']),source_radius_frozen_before_U=True,anonymous_frozen_before_V_and_Q=True,
        all_thresholds_frozen_before_novel_and_Q=True,all_three_conditions_complete=True,all_conditions_through_DURA=True,
        original_F0_default_identity_unchanged=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,REG0_controls_reused_exactly=True,
        original_density_refitted=False,source_radius_fitted=True,source_radius_new_fits=source_cost['source_radius_new_fits'],
        current_source_L_reads=source_cost['source_L_reads'],U_features_reused=cached_U,registration_threshold_sets_fitted=2,neural_updates=0,original_U_s_reads=0,
        source_L_reads=source_cost['source_L_reads'],source_OE_reads=0,target_U_reads=U_COUNT,target_Q_IQ_reads=0,private_truth_reads=0,U_truth_reads=0,
        source_novel_truth_reads=0,target_truth_reads=0,formal_target_accessed=False,real_signal_forward_rows=source_cost['source_signal_rows']+(0 if cached_U else U_COUNT),
        real_F_forward_rows=0 if cached_U else U_COUNT,G_forward_rows=0 if cached_U else U_COUNT,
        synthetic_signal_forward_rows=2,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,
        source_bank_deployed=False,U_bank_deployed=False,per_record_source_rows_deployed=0,target_DA=False,target_REG=True,
        classification_path='same_DURA_F0_old_identity_plus_anonymous_joint_competition',identity_baseline_path='same_DURA_output_original_F0',
        queries_reused_without_new_neural_forward=True,source_only=False,original_neural_models_source_only=True,
        final_phase3_complete=False,full_goal_status='ACTIVE')
    check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


def main():
    print(json.dumps(dict(event='SIGNAL_REGISTER_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    run(_read(a.config),a.device)


if __name__=='__main__':main()
