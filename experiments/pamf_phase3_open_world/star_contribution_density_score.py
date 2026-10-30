"""Independent source-only contribution-density scoring; no fitting or selection."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_contribution_density_run as evaluation
from . import star_consensus_score as common
from .star_physical_open_score import _metrics,_known_pass
from .star_density_score import _paired_counts

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_star_contribution_density_scores_v1'
STATUS='STAR_CONTRIBUTION_DENSITY_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=evaluation.PRIMARY_METHOD
COUNTS=evaluation.COUNTS
_read,_write,_truth=common._read,common._write,common._truth


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,
        classification_path='P1',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,source_novel_fit_reads=0,OE_TX_or_K_used=False,
        real_F_forward_rows=COUNTS['V']+COUNTS['novel'],synthetic_F_forward_rows=2,G_forward_rows=0,G_checkpoint_loaded=False,
        source_L_forward_rows=0,source_OE_forward_rows=0,received_P0_values_read=0,neural_updates=0,native_updates=0,identity_head_updates=0,G_updates=0,
        default_head_replaced=False,open_parts_installed=False,identity_predictions_unchanged=True,control_reused_exactly=True,control_refit=False,control_recalibrated=False,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,
        source_bank_deployed=False,per_record_source_rows_deployed=0,source_V_model_fit=False,source_V_threshold_fit=True,
        reference_run_id=evaluation.PARENT_RUN,reference_code_commit=evaluation.PARENT_CODE,dimensions=dict(zip(METHODS,(160,640))),final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Both fixed policies must complete legal frozen source predictions before truth')
    c=evaluation.validate_config(m['config'])
    if c['code_commit']!=m['code_commit']:raise ValueError('Prediction code identity differs')
    evaluation.parent.reference._identity(m['model_identity'],c['source_config'])
    fm=m['source_feature_manifest'];pm,parent_state,parent_tau,parent_global,pc=evaluation._parent(c['reference_prediction_root'],m['model_identity'],fm)
    if pm!=m['reference_prediction_identity'] or m['dura_identity']!=fm['dura_identity']:raise ValueError('Fixed parent and source G10 handles differ')
    smoke=_read(root/'smoke.json')
    if any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_F0_checkpoint=True,contributions_checked=True).items()):
        raise ValueError('Actual frozen F0 same-pass contribution smoke required')
    fit=_read(root/'models/complete.json')
    if any(fit.get(k)!=v for k,v in dict(run_id=RUN_ID,code_commit=m['code_commit'],status='STAR_CONTRIBUTION_DENSITY_MODELS_FROZEN',
            method_names=list(METHODS),control_reused=True,source_V_fit=False,novel_fit=False).items()):
        raise ValueError('Source contribution models must fit only L/OE and freeze before V')
    states={}
    for name,d in zip(METHODS,(160,640)):
        with np.load(root/'models'/(name+'.npz'),allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
        evaluation._core().validate_state(state,'density_ratio')
        if state['whitener'].shape!=(d,d):raise ValueError('Explicit 160/640 aggregate representations required')
        states[name]=state
    if not evaluation._same(states[METHODS[0]],parent_state):raise ValueError('Frozen control aggregate was altered')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional','global'}:raise ValueError('Exact frozen density threshold arrays required')
        tau=z['conditional'].copy();global_tau=z['global'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(2,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or global_tau.shape!=(2,) or global_tau.dtype!=np.float64 or not np.isfinite(global_tau).all()
            or not np.array_equal(tau[0],parent_tau) or global_tau[0]!=parent_global
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],
                method_names=list(METHODS),primary_method=PRIMARY,known_count=COUNTS['V'],group_count=42,known_only=True,unknown_inputs_used=False,
                source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,conditional_thresholds=tau.tolist(),global_thresholds=global_tau.tolist(),control_calibration_reused=True).items())
            or cal.get('methods',{}).get(METHODS[0])!=pc['methods']['density_ratio']):
        raise ValueError('Use the exact parent primary thresholds and completed known-only candidate calibration')
    for j,name in enumerate(METHODS):
        row=cal['methods'][name]
        if set(row.get('groups',{}))!=set(common.GROUP_NAMES) or row.get('conditional_thresholds')!=tau[j].tolist():
            raise ValueError('All 42 calibrated known-risk groups required for each fixed policy')
    rows={}
    for role in ('known','novel'):
        fields=evaluation.KNOWN_FIELDS if role=='known' else evaluation.NOVEL_FIELDS
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as z:
            if set(z.files)!=fields:raise ValueError('Only exact completed public source fields allowed; novel labels forbidden')
            a={k:z[k].copy() for k in z.files}
        evaluation.parent.validate_rows({k:v for k,v in a.items() if k!='contribution_features'},role)
        n=len(a['ids']);u=a['contribution_features']
        if (u.shape!=(n,640) or u.dtype!=np.float32 or not np.isfinite(u).all()
                or not np.allclose(np.linalg.norm(u.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)):
            raise ValueError('Finite same-pass contribution unit640 required')
        control=evaluation._control_rows(c['reference_prediction_root'],role)
        for key,value in control.items():
            actual=a[key][:,0] if key in ('scores','raw_scores') else a[key]
            if not np.array_equal(actual,value):raise ValueError('All original F0 control fields must be reused exactly')
        raw=evaluation._core().score(states[PRIMARY],u,'density_ratio')
        # Runner batches and full-array CPU verification can differ by FP64
        # BLAS rounding. The reused control and saved margin arithmetic stay exact.
        if (not np.allclose(a['raw_scores'][:,1],raw,rtol=1e-12,atol=1e-9)
                or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T)):
            raise ValueError('Candidate density and both routed margins must reconstruct from frozen source state')
        rows[role]=a
    if set(rows['known']['ids'].tolist())&set(rows['novel']['ids'].tolist()):raise ValueError('Known/novel roles must remain physically disjoint')
    return m,cal,states,rows['known'],rows['novel']


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable independent contribution-density scoring root required')
    marker,cal,states,known,novel=_prediction(prediction_root)
    truth=_truth(novel_truth,novel)  # Sole private read, after both complete frozen predictions and all checks.
    km=dict(common._masks(known['rx_i'],known['scenario']));nm=dict(common._masks(novel['rx_i'],truth['scenario']))
    allk=np.ones(len(known['ids']),bool);alln=np.ones(len(novel['ids']),bool);methods={}
    for j,name in enumerate(METHODS):
        def metric(k,n,kr=tuple(range(6)),ur=common.UNKNOWN_TX):
            return _metrics(known['y'][k],known['closed'][k],known['scores'][k,j],truth['tx'][n],novel['scores'][n,j],known['baseline_closed'][k],kr,ur)
        groups={key:metric(mask,nm[key]) for key,mask in km.items()};overall=groups['overall']
        kp=all(_known_pass(row) for row in groups.values());cp=cal['methods'][name]['constraints_passed']
        up=bool(overall['coverage']['unknown_complete'] and overall['R_U'] is not None and overall['R_U']>=.90-1e-12)
        goals=dict(known_42_groups_passed=kp,source_known_calibration_passed=cp,source_overall_unknown_macro_90_passed=up,
            passed=bool(kp and cp and up),gate_scope='source_overall_falsification_only',RX_scene_unknown_diagnostic_only=True)
        methods[name]=dict(overall=overall,groups=groups,goals=goals,diagnostic_only=name!=PRIMARY,
            by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},by_scene={s:groups['scene/'+s] for s in common.SCENES},
            by_day={str(d):metric(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},
            by_TX={str(c):metric(known['y']==c,alln,(c,)) for c in range(6)}|{tx:metric(allk,truth['tx']==tx,ur=(tx,)) for tx in common.UNKNOWN_TX})
    main=methods[PRIMARY];control=methods[METHODS[0]];passed=main['goals']['passed'];keys=('A_K','C_K','R_U','H_score','auroc_known_macro','oscr_auc_macro','original_closed_minus_new_open')
    def delta(a,b):return {k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in keys}
    paired=dict(primary=PRIMARY,control=METHODS[0],same_physical_IDs_and_original_F0_head=True,
        thresholds_scope='Frozen parent control exactly reused; candidate independently known-V calibrated before novel',
        counts=_paired_counts(known,novel,allk,alln),overall=delta(main['overall'],control['overall']),
        groups={key:dict(delta=delta(main['groups'][key],control['groups'][key]),counts=_paired_counts(known,novel,km[key],nm[key])) for key in km},
        by_day={str(d):_paired_counts(known,novel,known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},
        by_known_TX={str(c):_paired_counts(known,novel,known['y']==c,alln) for c in range(6)},
        by_unknown_TX={tx:_paired_counts(known,novel,allk,truth['tx']==tx) for tx in common.UNKNOWN_TX})
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,
        methods=methods,primary_summary=main['overall'],paired_primary_vs_control=paired,control_reused_exactly=True,source_gate_passed=passed,eligible_primary=passed,
        gate_scope='source_overall_falsification_only',next_step='SOURCE_ELIGIBLE_FOR_SEPARATE_TARGET_PREREGISTRATION' if passed else 'NO_PROMOTION',
        source_only=True,truth_last=True,target_reads=0,target_U_reads=0,target_Q_reads=0,formal_target_accessed=False,target_DA=False,target_REG=False,
        source_RX_seen=True,source_novel_historically_exposed=True,new_blind_confirmation=False,A_N=None,H_OW=None,final_phase3_complete=False,full_goal_passed=False,full_goal_status='ACTIVE',
        identifier_parameters=232993,restorer_parameters=266951,new_neural_parameters=0,native_updates=0,G_updates=0,source_bank_deployed=False,
        dimension_dependent_covariance_volume_and_shrinkage=True,
        costs={k:marker[k] for k in ('real_F_forward_rows','synthetic_F_forward_rows','G_forward_rows','source_aggregate_bytes_by_method','threshold_bytes','cpu_fit_seconds','cpu_readout_seconds','cpu_readout_rows')},
        scope='Source seen-RX historical-novel falsification only; eligibility is not cross-RX target, DA or registration success')
    output.mkdir(parents=True);_write(output/'results.json',result);_write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_control')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
