"""Independent truth-last source falsification for frozen aggregate densities."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from . import star_density_run as evaluation
from . import star_consensus_score as common
from .star_physical_open_score import _metrics, _known_pass

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_star_density_scores_v1'
STATUS='STAR_DENSITY_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=evaluation.PRIMARY_METHOD
COUNTS=evaluation.COUNTS
_read,_write,_truth=common._read,common._write,common._truth


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,
        classification_path='P1',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],source_only=True,private_truth_reads=0,source_novel_truth_reads=0,
        target_reads=0,original_U_s_reads=0,real_IQ_reads=0,real_F_forward_rows=0,synthetic_F_forward_rows=2,G_forward_rows=0,
        G_checkpoint_loaded=False,neural_updates=0,native_updates=0,identity_head_updates=0,G_updates=0,default_head_replaced=False,
        identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,
        source_V_model_fit=False,source_V_threshold_fit=True,source_novel_fit_reads=0,OE_TX_or_K_used=False,
        known_only_uses_OE=False,density_ratio_uses_OE_membership=True,old_scores_or_thresholds_read=False,
        reference_run_id=evaluation.REFERENCE_RUN,reference_code_commit=evaluation.REFERENCE_CODE,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):
        raise ValueError('Both frozen source policies and complete legal predictions must precede private truth')
    c=evaluation.validate_config(m['config'])
    if c['code_commit']!=m['code_commit']:raise ValueError('Current source code identity differs')
    evaluation.reference._identity(m['model_identity'],c['source_config'])
    fm=m['source_feature_manifest'];ref=m['reference_prediction_identity']
    if (fm.get('parent_F0_identity')!=m['model_identity'] or fm.get('dura_identity')!=m['dura_identity']
            or ref.get('model_identity')!=m['model_identity'] or ref.get('dura_identity')!=m['dura_identity']
            or ref.get('run_id')!=evaluation.REFERENCE_RUN or ref.get('code_commit')!=evaluation.REFERENCE_CODE
            or ref.get('schema')!=evaluation.REFERENCE_SCHEMA or ref.get('status')!=evaluation.REFERENCE_STATUS):
        raise ValueError('Only matching original F0/DURA source features and completed reference evidence are legal')
    smoke=_read(root/'smoke.json')
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_checkpoint') is not True:
        raise ValueError('Actual original-F0 checkpoint smoke required')
    fit=_read(root/'models/complete.json')
    if (fit.get('status')!='STAR_DENSITY_MODELS_FROZEN' or fit.get('run_id')!=RUN_ID or fit.get('code_commit')!=m['code_commit']
            or fit.get('method_names')!=list(METHODS) or fit.get('fit_roles')!={'known_only':['L'],'density_ratio':['L','OE']}
            or fit.get('source_V_fit') is not False or fit.get('novel_fit') is not False
            or fit.get('source_bank_deployed') is not False or fit.get('per_record_source_rows_deployed')!=0):
        raise ValueError('L-only control and L/OE aggregate fit must both be frozen before V')
    states={}
    for name in METHODS:
        with np.load(root/'models'/(name+'.npz'),allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
        evaluation._core().validate_state(state,name);states[name]=state
    if any(not np.array_equal(value,states[PRIMARY][key]) for key,value in states['known_only'].items()):
        raise ValueError('Density ratio must preserve the identical L-fitted known model')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional','global'}:raise ValueError('Exactly both source-known calibration policies required')
        tau=z['conditional'].copy();global_tau=z['global'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(2,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or global_tau.shape!=(2,) or global_tau.dtype!=np.float64 or not np.isfinite(global_tau).all()
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,
                code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,known_count=COUNTS['V'],
                group_count=42,known_only=True,unknown_inputs_used=False,source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,
                conditional_thresholds=tau.tolist(),global_thresholds=global_tau.tolist()).items())
            or set(cal.get('methods',{}))!=set(METHODS)):
        raise ValueError('Twelve frozen predicted-class thresholds from known V only required')
    for j,name in enumerate(METHODS):
        row=cal['methods'][name]
        if (set(row.get('groups',{}))!=set(common.GROUP_NAMES) or row.get('conditional_thresholds')!=tau[j].tolist()
                or type(row.get('constraints_passed')) is not bool):
            raise ValueError('Each source policy must retain all 42 known-risk groups')
    rows={}
    for role in ('known','novel'):
        fields=evaluation.KNOWN_FIELDS if role=='known' else evaluation.NOVEL_FIELDS
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as z:
            if set(z.files)!=fields:raise ValueError('Exact completed public prediction fields required; novel TX forbidden')
            a={k:z[k].copy() for k in z.files}
        evaluation.validate_rows(a,role)
        raw=np.column_stack([evaluation._density_score(states[name],a['unit_features'],name) for name in METHODS])
        if (not np.allclose(a['raw_scores'],raw,rtol=0.,atol=1e-10)
                or not np.allclose(a['scores'],raw-tau[:,a['closed']].T,rtol=0.,atol=1e-10)):
            raise ValueError('Both densities and class-routed margins must reconstruct from frozen aggregates')
        rows[role]=a
    if set(rows['known']['ids'].tolist())&set(rows['novel']['ids'].tolist()):
        raise ValueError('Known V and source novel IDs must be physically disjoint')
    return m,cal,tau,states,rows['known'],rows['novel']


def _paired_counts(known,novel,k,n):
    left=known['scores'][k,1]>=0;right=known['scores'][k,0]>=0
    correct=known['closed'][k]==known['y'][k]
    ul=novel['scores'][n,1]<0;ur=novel['scores'][n,0]<0
    result={}
    for name,a,b in (('known_accepted',left,right),('known_correct_accepted',left&correct,right&correct),('unknown_rejected',ul,ur)):
        gain=int(np.count_nonzero(a&~b));loss=int(np.count_nonzero(b&~a))
        result.update({name+'_gain':gain,name+'_loss':loss,name+'_net':gain-loss})
    return result


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable independent density score root required')
    marker,cal,tau,states,known,novel=_prediction(prediction_root)
    truth=_truth(novel_truth,novel)  # Sole private read, after both complete policies and all public evidence.
    km=dict(common._masks(known['rx_i'],known['scenario']));nm=dict(common._masks(novel['rx_i'],truth['scenario']))
    allk=np.ones(len(known['ids']),bool);alln=np.ones(len(novel['ids']),bool);methods={}
    for j,name in enumerate(METHODS):
        def metric(k,n,kr=tuple(range(6)),ur=common.UNKNOWN_TX):
            return _metrics(known['y'][k],known['closed'][k],known['scores'][k,j],truth['tx'][n],novel['scores'][n,j],known['baseline_closed'][k],kr,ur)
        groups={key:metric(mask,nm[key]) for key,mask in km.items()};overall=groups['overall']
        known_pass=all(_known_pass(row) for row in groups.values())
        unknown_pass=bool(overall['coverage']['unknown_complete'] and overall['R_U'] is not None and overall['R_U']>=.90-1e-12)
        calibrated=cal['methods'][name]['constraints_passed']
        goals=dict(known_42_groups_passed=known_pass,source_known_calibration_passed=calibrated,
            source_overall_unknown_macro_90_passed=unknown_pass,passed=bool(known_pass and calibrated and unknown_pass),
            gate_scope='source_overall_falsification_only',RX_scene_unknown_diagnostic_only=True)
        methods[name]=dict(overall=overall,groups=groups,
            by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},by_scene={s:groups['scene/'+s] for s in common.SCENES},
            by_day={str(d):metric(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},
            by_TX={str(c):metric(known['y']==c,alln,(c,)) for c in range(6)}|
                {tx:metric(allk,truth['tx']==tx,ur=(tx,)) for tx in common.UNKNOWN_TX},
            goals=goals,diagnostic_only=name!=PRIMARY)
    main=methods[PRIMARY];control=methods['known_only'];passed=main['goals']['passed']
    delta_keys=('A_K','C_K','R_U','H_score','auroc_known_macro','oscr_auc_macro','original_closed_minus_new_open')
    def delta(a,b):return {k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in delta_keys}
    paired=dict(primary=PRIMARY,control='known_only',same_physical_IDs_and_original_F0_head=True,
        thresholds_scope='Each policy independently uses the same known-V-only class-risk calibration rule',
        counts=_paired_counts(known,novel,allk,alln),overall=delta(main['overall'],control['overall']),
        groups={key:dict(delta=delta(main['groups'][key],control['groups'][key]),counts=_paired_counts(known,novel,km[key],nm[key])) for key in km},
        by_day={str(d):_paired_counts(known,novel,known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},
        by_known_TX={str(c):_paired_counts(known,novel,known['y']==c,alln) for c in range(6)},
        by_unknown_TX={tx:_paired_counts(known,novel,allk,truth['tx']==tx) for tx in common.UNKNOWN_TX})
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,
        primary_summary=main['overall'],methods=methods,paired_primary_vs_control=paired,source_gate_passed=passed,eligible_primary=passed,
        gate_scope='source_overall_falsification_only',next_step='SOURCE_ELIGIBLE_FOR_SEPARATE_TARGET_PREREGISTRATION' if passed else 'NO_PROMOTION',
        truth_last=True,source_only=True,target_reads=0,target_U_reads=0,target_Q_reads=0,formal_target_accessed=False,target_DA=False,target_REG=False,
        source_RX_seen=True,source_novel_historically_exposed=True,new_blind_confirmation=False,A_N=None,H_OW=None,
        final_phase3_complete=False,full_goal_passed=False,full_goal_status='ACTIVE',native_updates=0,G_updates=0,
        identifier_parameters=232993,restorer_parameters=266951,new_neural_parameters=0,source_bank_deployed=False,
        costs={key:marker[key] for key in ('real_F_forward_rows','synthetic_F_forward_rows','G_forward_rows','source_aggregate_bytes_by_method',
            'threshold_bytes','cpu_fit_seconds','cpu_readout_seconds','cpu_readout_rows','cpu_readout_policy_rows')},
        scope='Seen-source-RX historical novel falsification only; eligibility does not establish cross-RX target, adaptation or registration success')
    output.mkdir(parents=True);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_control')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
