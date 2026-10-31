"""Known-only U adaptation, retaining the original frozen shared-DIAG control."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from . import star_diagonal_uda_run as shared_evaluation
from . import star_diagonal_uda_core as shared_core
from . import star_selective_uda_core as core

RUN_ID='20261031-phase3-selective-uda-s2026092701-r01'
SCHEMA='pamf_phase3_selective_uda_prediction_v1'
STATUS='SELECTIVE_UDA_PREDICTIONS_COMPLETE'
SOURCE_STATUS='SELECTIVE_UDA_SOURCE_FROZEN'
ADAPTER_STATUS='SELECTIVE_UDA_ADAPTERS_FROZEN'
ADAPTATION_CODE='73f460c2e0c89b192e52af2e1b6ef01a576ca686'
PARENT_CODE=shared_evaluation.PARENT_CODE
METHODS=('pooled_ratio','predicted_ratio','shared_pooled_ratio','shared_predicted_ratio','selective_pooled_ratio','selective_predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=5
parent=shared_evaluation.parent
CONDITIONS,COUNTS,Q_COUNT=shared_evaluation.CONDITIONS,shared_evaluation.COUNTS,shared_evaluation.Q_COUNT
U_COUNT,U_CAPSULE_ID,U_SPLIT_ID=shared_evaluation.U_COUNT,shared_evaluation.U_CAPSULE_ID,shared_evaluation.U_SPLIT_ID
FIELDS=shared_evaluation.FIELDS
_read,_write,_npz,_same=shared_evaluation._read,shared_evaluation._write,shared_evaluation._npz,shared_evaluation._same
_reference,_arrays=shared_evaluation._reference,shared_evaluation._arrays
_rows,_target_rows=shared_evaluation._rows,shared_evaluation._target_rows
_parent_rows,_parent_target_rows=parent._rows,parent._target_rows
_shared_target_rows=shared_evaluation._target_rows


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID:
        raise ValueError('Exact selective known-only UDA configuration required; no truth/K/new IQ inputs')
    shared_evaluation.validate_config(dict(c,run_id=shared_evaluation.RUN_ID))
    return c


def _cached_U_header(c,pm):
    """Completed DIAG metadata only; no prior, source/Q rows, or U feature values."""
    header=_read(Path(c['reference_adaptation_root'])/'complete.json')
    expected=dict(schema=shared_evaluation.SCHEMA,status=shared_evaluation.STATUS,run_id=shared_evaluation.RUN_ID,code_commit=ADAPTATION_CODE,
        method_names=list(shared_evaluation.METHODS),primary_method=shared_evaluation.PRIMARY_METHOD,known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,
        source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,all_thresholds_reused_exactly=True,
        source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,classification_unchanged=True,
        new_neural_updates=0,native_updates=0,density_fit_updates=0,threshold_fit_updates=0,private_truth_reads=0,U_truth_reads=0,
        source_novel_truth_reads=0,target_truth_reads=0,real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,
        U_features_reused=True,original_neural_models_source_only=True,source_only=False,formal_target_accessed=False,final_phase3_complete=False)
    if any(header.get(k)!=v for k,v in expected.items()):raise ValueError('Exact completed frozen shared-DIAG/cache ancestry required')
    old=shared_evaluation.validate_config(header['config'])
    if (old['code_commit']!=ADAPTATION_CODE or Path(old['output_root']).resolve()!=Path(c['reference_adaptation_root']).resolve()
            or old['reference_config']!=c['reference_config'] or any(old[k]!=c[k] for k in ('U_capsule_id','U_split_id'))
            or header.get('reference_prediction_identity')!=pm or header.get('model_identity')!=pm['model_identity']
            or header.get('parent_F0_identity')!=pm['parent_F0_identity'] or header.get('dura_identity')!=pm['dura_identity']):
        raise ValueError('Shared control and original U must retain exactly the current source F0/r03/Gaussians/DURA identity')
    manifest=header.get('U_manifest',{})
    if (manifest.get('count')!=U_COUNT or manifest.get('capsule_id')!=U_CAPSULE_ID or manifest.get('split_id')!=U_SPLIT_ID
            or manifest.get('registered_classes')!=parent.signal_train.native.CLASSES):raise ValueError('Only the fixed complete pre-adaptation star-G10 U capsule/split may fit')
    output=Path(c['output_root']).resolve();old_input=Path(old['reference_adaptation_root']).resolve()
    if output.is_relative_to(old_input) or old_input.is_relative_to(output):raise ValueError('Output cannot overlap inherited U/cache inputs')
    return header


def _shared(c,pm,state,tau,cal):
    header=_cached_U_header(c,pm);root=Path(c['reference_adaptation_root'])
    if not _same(_arrays(root/'models/density.npz'),state):raise ValueError('Original source Gaussian parameters must remain exact')
    limits=_arrays(root/'thresholds.npz')
    if (set(limits)!={'conditional'} or not np.array_equal(limits['conditional'],np.tile(tau,(3,1)))
            or _read(root/'calibration/complete.json')!=cal):raise ValueError('Only unchanged original source thresholds/calibration may be inherited')
    fit=_read(root/'models/complete.json');done=_read(root/'adaptation/complete.json')
    expected=dict(schema=shared_evaluation.SCHEMA,status=shared_evaluation.SOURCE_STATUS,run_id=shared_evaluation.RUN_ID,code_commit=ADAPTATION_CODE,
        models_frozen=True,models_refitted=False,thresholds_refitted=False,source_models_frozen_before_U=True,model_identity=pm['model_identity'],parent_calibration_identity=cal,U_reads=0,Q_reads=0)
    required=dict(schema=shared_evaluation.SCHEMA,status=shared_evaluation.ADAPTER_STATUS,run_id=shared_evaluation.RUN_ID,code_commit=ADAPTATION_CODE,
        U_count=U_COUNT,U_manifest=header['U_manifest'],adapter_names=['SHIFT','DIAG'],fit_input='cached_star_U_signal_features_only',iterations_per_adapter=20,ridge=160.,group_prior=[.5,.5],
        query_fit=False,true_U_labels_used=False,true_unknown_count_used=False,source_density_refitted=False,thresholds_refitted=False,adapters_frozen=True,
        Q_reads=0,source_prediction_reads=0,U_bank_deployed=False,private_U_features_ref='U_features.npz')
    if any(fit.get(k)!=v for k,v in expected.items()) or any(done.get(k)!=v for k,v in required.items()):raise ValueError('Frozen source and matched control provenance must precede all new U access')
    shared=_arrays(root/'adaptation/DIAG.npz');shared_core.validate_adapter(shared)
    if shared['kind'].item()!='DIAG':raise ValueError('Only the matched full shared-DIAG control may be inherited')
    diag=_read(root/'adaptation/fit_diagnostics.json');report=diag.get('arms',{}).get('DIAG',{});trace=report.get('trace',[])
    if (diag.get('iterations')!=20 or diag.get('ridge')!=160. or diag.get('U_count')!=U_COUNT or report.get('iterations')!=20
            or report.get('group_prior')!=[.5,.5] or len(trace)!=21 or [r.get('iteration') for r in trace]!=list(range(21))
            or any(set(r)!={'iteration','objective','scale_min','scale_max','offset_norm','known_posterior_mass'} for r in trace)):
        raise ValueError('Complete original twenty-step DIAG control diagnostics required')
    values=np.asarray([[r[k] for k in ('objective','scale_min','scale_max','offset_norm','known_posterior_mass')] for r in trace],np.float64)
    if (not np.isfinite(values).all() or np.any(np.diff(values[:,0])<-1e-7) or np.any(values[:,1]<=0) or np.any(values[:,2]<values[:,1])
            or np.any(values[:,3]<0) or np.any(values[:,4]<-1e-12) or np.any(values[:,4]>1+1e-12)
            or not np.allclose(values[-1,1:4],[shared['inverse_scale'].min(),shared['inverse_scale'].max(),np.linalg.norm(shared['offset'])],rtol=1e-12,atol=1e-12)):
        raise ValueError('Inherited positive-scale DIAG state must match its actual frozen terminal trace')
    return header,shared,report


def _U_features(c,header):return shared_evaluation._U_features(c,header)


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable selective-UDA output required; never replace an existing run')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=parent._backend(c['reference_config'],device)
    snapshots=[(model,shared_evaluation._snapshot_model(model)) for model in (b['reader'],b['model'],b['dura'][0])]
    def models_unchanged():
        for model,before in snapshots:shared_evaluation._check_model(model,before)
    smoke=parent._smoke(b,device);models_unchanged()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint smoke must pass before cache/U access')
    pm,state,tau,cal=_reference(c);identity=pm['model_identity'];header,shared,shared_diag=_shared(c,pm,state,tau,cal)
    if b['model_identity']!=identity or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:
        raise ValueError('Actual frozen F0/r03/DURA must match the source and inherited shared control')
    frozen=deepcopy(state);shared_before=deepcopy(shared);six_tau=np.tile(tau,(3,1));tau_before=tau.copy();frozen_tau=six_tau.copy()
    def check():
        models_unchanged()
        if not _same(state,frozen) or not _same(shared,shared_before) or not np.array_equal(six_tau,frozen_tau) or not np.array_equal(tau,tau_before):
            raise ValueError('Original source Gaussian/thresholds or matched shared adapter changed')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);(out/'models').mkdir()
    _npz(out/'models/density.npz',state);_npz(out/'thresholds.npz',dict(conditional=six_tau));(out/'calibration').mkdir();_write(out/'calibration/complete.json',cal)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=SOURCE_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        models_frozen=True,models_refitted=False,thresholds_refitted=False,source_models_frozen_before_U=True,model_identity=identity,parent_calibration_identity=cal,U_reads=0,Q_reads=0))
    U=_U_features(c,header);adapter,diag=core.fit_adapter(state,U['signal_features']);core.validate_adapter(adapter);check()
    for value in adapter.values():value.setflags(write=False)
    before=deepcopy(adapter);(out/'adaptation').mkdir();_npz(out/'adaptation/SELECTIVE.npz',adapter);_npz(out/'adaptation/SHARED.npz',shared)
    _npz(out/'adaptation/U_features.npz',U);_write(out/'adaptation/fit_diagnostics.json',diag);_write(out/'adaptation/shared_diagnostics.json',shared_diag)
    _write(out/'adaptation/complete.json',dict(schema=SCHEMA,status=ADAPTER_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],U_count=U_COUNT,U_manifest=header['U_manifest'],
        adapter_names=['SHARED','SELECTIVE'],fit_input='cached_star_U_signal_features_only',iterations=20,inherited_shared_iterations=20,ridge=160.,group_prior=[.5,.5],
        known_only_transform=True,background_frozen=True,shared_adapter_inherited=True,query_fit=False,true_U_labels_used=False,true_unknown_count_used=False,
        source_density_refitted=False,thresholds_refitted=False,adapters_frozen=True,Q_reads=0,source_prediction_reads=0,U_bank_deployed=False,private_U_features_ref='U_features.npz'))
    print(ADAPTER_STATUS,flush=True)
    def frozen_check():
        check()
        if not _same(adapter,before):raise ValueError('Frozen selective adapter changed while predicting future records')
    seen=set(U['ids'].tolist())
    def controls(a):
        raw=a['raw_scores'][:,[1,3]];margin=a['scores'][:,[1,3]]
        if (not np.allclose(parent.core.raw_scores(state,a['signal_features'],a['closed'])[:,[1,3]],raw,rtol=1e-12,atol=1e-9)
                or not np.array_equal(margin,raw-tau[:,a['closed']].T)):raise ValueError('Original pooled/predicted controls must reconstruct unchanged')
        return raw,margin
    for role in ('known','novel'):
        a=_parent_rows(c['reference_config']['output_root'],role);ids=set(a['ids'].tolist())
        if seen&ids:raise ValueError('Source known/novel and fitting U records must be physically disjoint')
        seen.update(ids);raw,margin=controls(a);result=dict(a,raw_scores=np.tile(raw,(1,3)),scores=np.tile(margin,(1,3)))
        _npz(out/(role+'_predictions.npz'),result);_rows(out,role);frozen_check()
    for condition in CONDITIONS:
        a=_parent_target_rows(c['reference_config']['output_root'],condition);old=_shared_target_rows(c['reference_adaptation_root'],condition)
        if seen&set(a['ids'].tolist()):raise ValueError('Future Q cannot overlap any source or U fitting record')
        raw,margin=controls(a)
        if (set(old)!=set(a) or any(not np.array_equal(old[k],v) for k,v in a.items() if k not in ('raw_scores','scores'))
                or not np.array_equal(old['raw_scores'][:,:2],raw) or not np.array_equal(old['scores'][:,:2],margin)
                or not np.allclose(shared_core.raw_scores(state,a['signal_features'],a['closed'],shared),old['raw_scores'][:,4:6],rtol=1e-12,atol=1e-9)
                or not np.array_equal(old['scores'][:,4:6],old['raw_scores'][:,4:6]-tau[:,a['closed']].T)):
            raise ValueError('Matched shared-DIAG control must retain the same records, source state and fixed operating points')
        six=np.concatenate((raw,old['raw_scores'][:,4:6],core.raw_scores(state,a['signal_features'],a['closed'],adapter)),axis=1)
        scores=np.concatenate((margin,old['scores'][:,4:6],six[:,4:6]-tau[:,a['closed']].T),axis=1)
        if not np.array_equal(scores,six-six_tau[:,a['closed']].T):raise ValueError('All six methods must use the original two routed threshold vectors')
        result=dict(a,raw_scores=six,scores=scores);folder=out/'targets'/condition;folder.mkdir(parents=True)
        _npz(folder/'predictions.npz',result);_target_rows(out,condition);frozen_check();print('SELECTIVE_UDA_Q_COMPLETE',condition,len(a['ids']),flush=True)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        reference_prediction_identity=pm,reference_adaptation_identity=header,parent_calibration_identity=cal,model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,U_manifest=header['U_manifest'],Q_counts=pm['Q_counts'],Q_manifests=pm['Q_manifests'],
        classification_path='parent_original_F0_same_DURA_input',identity_baseline_path='parent_original_F0_same_DURA_input',classification_unchanged=True,score_threshold=0.,
        source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,all_thresholds_reused_exactly=True,
        source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,source_controls_reused_exactly=True,DA0_controls_reused_exactly=True,shared_controls_reused_exactly=True,
        new_neural_updates=0,native_updates=0,density_fit_updates=0,threshold_fit_updates=0,adapter_iterations=20,new_target_statistical_iterations=20,inherited_shared_iterations=20,
        known_only_transform=True,selective_background_input_untransformed=True,target_U_reads=U_COUNT,U_feature_rows_read=U_COUNT,U_features_reused=True,
        source_L_reads=0,source_OE_reads=0,source_IQ_reads=0,target_Q_IQ_reads=0,original_U_s_reads=0,private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,
        real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,
        source_only=False,original_neural_models_source_only=True,target_DA=True,target_REG=False,source_bank_deployed=False,U_bank_deployed=False,per_record_source_rows_deployed=0,
        formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    frozen_check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


evaluate=run


def main():
    print(json.dumps(dict(event='SELECTIVE_UDA_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0');a=parser.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
