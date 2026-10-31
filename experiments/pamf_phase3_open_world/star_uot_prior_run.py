"""Unlabelled star-U mixture-prior transport with frozen source distributions."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from . import star_diagonal_uda_run as diagonal
from . import star_class_density_run as parent
from . import star_uot_prior_core as core
from .star_local_cov_run import _reference
from .development_transfer import _snapshot_model,_check_model

RUN_ID='20261031-phase3-uot-prior-s2026092701-r01'
SCHEMA='pamf_phase3_uot_prior_prediction_v1'
STATUS='UOT_PRIOR_PREDICTIONS_COMPLETE'
SOURCE_STATUS='UOT_PRIOR_SOURCE_FROZEN'
ADAPTER_STATUS='UOT_PRIOR_ADAPTER_FROZEN'
METHODS=('pooled_ratio','predicted_ratio','UOT_pooled_ratio','UOT_predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=3
PARENT_CODE,ADAPTATION_CODE=diagonal.PARENT_CODE,diagonal.ADAPTATION_CODE
CONDITIONS,COUNTS,Q_COUNT=parent.CONDITIONS,parent.COUNTS,parent.Q_COUNT
U_COUNT,U_CAPSULE_ID,U_SPLIT_ID=diagonal.U_COUNT,diagonal.U_CAPSULE_ID,diagonal.U_SPLIT_ID
FIELDS=diagonal.FIELDS
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=parent.KNOWN_FIELDS,parent.NOVEL_FIELDS,parent.TARGET_FIELDS
mix=diagonal.mix
_read,_write,_npz,_same=parent._read,parent._write,parent._npz,parent._same
_arrays,_cached_U_header,_U_features=diagonal._arrays,diagonal._cached_U_header,diagonal._U_features
_rows,_target_rows=parent._rows,parent._target_rows
_parent_rows,_parent_target_rows=parent._rows,parent._target_rows


def validate_config(c):
    if not isinstance(c,dict) or c.get('run_id')!=RUN_ID:raise ValueError('Exact UOT-PRIOR run configuration required')
    diagonal.validate_config(dict(c,run_id=diagonal.RUN_ID))
    return c


def _validate_diagnostics(diag,adapter):
    expected=dict(iterations=200,epsilon=.05,rho_source=1.,rho_target=1.,theta_source=1/1.05,theta_target=1/1.05,
        blend=.5,U_count=U_COUNT,U_only=True,Q_inputs_used=False,true_U_labels_used=False,true_unknown_count_used=False,
        default_logits_used=False,source_state_unchanged_by_fit=True,U_bank_deployed=False)
    if not isinstance(diag,dict) or any(diag.get(k)!=v for k,v in expected.items()):raise ValueError('Complete fixed 200-step U-only transport diagnostics required')
    trace=diag.get('trace',[])
    if (len(trace)!=200 or [r.get('iteration') for r in trace]!=list(range(1,201))
            or any(set(r)!={'iteration','max_log_scaling_change'} for r in trace)):
        raise ValueError('Complete finite fixed-iteration transport trace required')
    values=np.asarray([diag.get(k,np.nan) for k in ('log_total_mass','kkt_stationarity_residual','max_log_scaling_change','known_transport_fraction')]+
        [r['max_log_scaling_change'] for r in trace],np.float64)
    learned_d=np.asarray(diag.get('learned_domain_weights'),np.float64);learned_b=np.asarray(diag.get('learned_background_weights'),np.float64)
    if (not np.isfinite(values).all() or not 0<=values[3]<=1 or np.any(values[1:3]<0) or np.any(values[4:]<0)
            or learned_d.shape!=(15,) or learned_b.shape!=adapter['background_weights'].shape
            or not np.isfinite(learned_d).all() or not np.isfinite(learned_b).all()
            or np.any(learned_d<0) or np.any(learned_b<0)
            or not np.isclose(learned_d.sum(),1.,rtol=0,atol=1e-12) or not np.isclose(learned_b.sum(),1.,rtol=0,atol=1e-12)):
        raise ValueError('Finite aggregate transport diagnostics and normalized learned group priors required')


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable UOT-PRIOR output required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    b=parent._backend(c['reference_config'],device)
    snapshots=[(m,_snapshot_model(m)) for m in (b['reader'],b['model'],b['dura'][0])]
    def models_unchanged():
        for model,before in snapshots:_check_model(model,before)
    smoke=parent._smoke(b,device);models_unchanged()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint smoke must precede cached U access')
    pm,state,tau,cal=_reference(c);header=_cached_U_header(c,pm);identity=pm['model_identity']
    if b['model_identity']!=identity or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:
        raise ValueError('Actual source F0/r03/DURA must match the original source and pre-adaptation U')
    frozen=deepcopy(state);four_tau=np.tile(tau,(2,1));frozen_tau=four_tau.copy()
    def check():
        models_unchanged()
        if not _same(state,frozen) or not np.array_equal(four_tau,frozen_tau):raise ValueError('Frozen source distributions or original thresholds changed')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);(out/'models').mkdir()
    _npz(out/'models/density.npz',state);_npz(out/'thresholds.npz',dict(conditional=four_tau));(out/'calibration').mkdir();_write(out/'calibration/complete.json',cal)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=SOURCE_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],models_frozen=True,
        models_refitted=False,thresholds_refitted=False,source_models_frozen_before_U=True,model_identity=identity,parent_calibration_identity=cal,U_reads=0,Q_reads=0))
    U=_U_features(c,header);adapter,diag=core.fit_adapter(state,U['signal_features']);check();core.validate_adapter(adapter,state);_validate_diagnostics(diag,adapter)
    for value in adapter.values():value.setflags(write=False)
    before=deepcopy(adapter);(out/'adaptation').mkdir();_npz(out/'adaptation/UOT.npz',adapter);_npz(out/'adaptation/U_features.npz',U);_write(out/'adaptation/fit_diagnostics.json',diag)
    _write(out/'adaptation/complete.json',dict(schema=SCHEMA,status=ADAPTER_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],U_count=U_COUNT,U_manifest=header['U_manifest'],
        adapter_names=['UOT'],fit_input='cached_star_U_signal_features_only',iterations=200,epsilon=.05,rho_source=1.,rho_target=1.,blend=.5,
        query_fit=False,true_U_labels_used=False,true_unknown_count_used=False,source_density_refitted=False,thresholds_refitted=False,
        adapters_frozen=True,Q_reads=0,source_prediction_reads=0,U_bank_deployed=False,private_U_features_ref='U_features.npz'))
    print(ADAPTER_STATUS,flush=True)
    def frozen_check():
        check()
        if not _same(adapter,before):raise ValueError('Frozen target mixture prior changed during source/future inference')
    seen=set(U['ids'].tolist())
    def controls(a):
        raw=a['raw_scores'][:,[1,3]];margin=a['scores'][:,[1,3]]
        if (not np.allclose(parent.core.raw_scores(state,a['signal_features'],a['closed'])[:,[1,3]],raw,rtol=1e-12,atol=1e-9)
                or not np.array_equal(margin,raw-tau[:,a['closed']].T)):raise ValueError('Both original ratio controls must reconstruct without threshold changes')
        return raw,margin
    for role in ('known','novel'):
        a=_parent_rows(c['reference_config']['output_root'],role);ids=set(a['ids'].tolist())
        if ids&seen:raise ValueError('Source roles and U must be physically disjoint')
        seen.update(ids);raw,margin=controls(a);result=dict(a,raw_scores=np.tile(raw,(1,2)),scores=np.tile(margin,(1,2)))
        _npz(out/(role+'_predictions.npz'),result);_rows(out,role);frozen_check()
    for condition in CONDITIONS:
        a=_parent_target_rows(c['reference_config']['output_root'],condition)
        if seen&set(a['ids'].tolist()):raise ValueError('Future Q cannot overlap source or U fitting records')
        raw,margin=controls(a);four=np.column_stack((raw,core.raw_scores(state,a['signal_features'],a['closed'],adapter)))
        result=dict(a,raw_scores=four,scores=four-four_tau[:,a['closed']].T)
        if not np.array_equal(result['scores'][:,:2],margin):raise ValueError('Original target operating point changed')
        folder=out/'targets'/condition;folder.mkdir(parents=True);_npz(folder/'predictions.npz',result);_target_rows(out,condition);frozen_check()
        print('UOT_PRIOR_Q_COMPLETE',condition,len(a['ids']),flush=True)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        reference_prediction_identity=pm,reference_adaptation_identity=header,parent_calibration_identity=cal,model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,U_manifest=header['U_manifest'],Q_counts=pm['Q_counts'],Q_manifests=pm['Q_manifests'],
        classification_path='parent_original_F0_same_DURA_input',identity_baseline_path='parent_original_F0_same_DURA_input',classification_unchanged=True,score_threshold=0.,
        source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,all_thresholds_reused_exactly=True,
        source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,source_controls_reused_exactly=True,DA0_controls_reused_exactly=True,
        new_neural_updates=0,native_updates=0,density_fit_updates=0,threshold_fit_updates=0,adapter_iterations=200,new_target_statistical_iterations=200,
        new_target_statistical_parameters=47,new_target_statistical_degrees_of_freedom=45,original_density_arrays_unchanged=True,normalizer_fit_updates=0,
        target_U_reads=U_COUNT,U_feature_rows_read=U_COUNT,U_features_reused=True,source_L_reads=0,source_OE_reads=0,source_IQ_reads=0,target_Q_IQ_reads=0,original_U_s_reads=0,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,
        synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,source_only=False,original_neural_models_source_only=True,target_DA=True,target_REG=False,
        unique_source_threshold_values=12,unique_source_threshold_bytes=96,repeated_threshold_values=24,repeated_threshold_bytes=192,primary_threshold_bytes=48,
        source_bank_deployed=False,U_bank_deployed=False,per_record_source_rows_deployed=0,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    frozen_check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


evaluate=run


def main():
    print(json.dumps(dict(event='UOT_PRIOR_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
