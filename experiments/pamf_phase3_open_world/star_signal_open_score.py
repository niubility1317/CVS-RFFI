"""Independent truth-last four-readout source screening of one signal branch."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_signal_open_eval as evaluation
from . import star_consensus_score as common
from .star_crossset_metric_score import paired_counts
from .star_physical_open_score import _metrics,_known_pass

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_star_signal_open_scores_v1'
STATUS='STAR_SIGNAL_OPEN_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=evaluation.PRIMARY_METHOD
COUNTS=evaluation.COUNTS
_read,_write,_truth=common._read,common._write,common._truth


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,arm=evaluation.PRIMARY_ARM,primary_arm=evaluation.PRIMARY_ARM,
        method_names=list(METHODS),primary_method=PRIMARY,classification_path='P1',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,source_novel_fit_reads=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,G_checkpoint_loaded=False,
        real_signal_forward_rows=sum(COUNTS.values()),synthetic_signal_forward_rows=2,signal_forward_rows_by_role=dict(COUNTS),eval_neural_updates=0,native_updates=0,open_signal_updates=2220,
        default_head_replaced=False,identity_predictions_unchanged=True,controls_reused_exactly=True,control_refit=False,control_recalibrated=False,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,source_V_model_fit=False,source_V_threshold_fit=True,
        signal_known_statistics_shared=True,same_projection_for_signal_methods=True,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,original_F0_parameters=232993,combined_active_parameters=567714,
        reference_run_id=evaluation.REFERENCE_RUN,reference_code_commit=evaluation.REFERENCE_CODE,received_P0_values_read=0,channel_regenerations=0,second_RMS_applied=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('All four completed legal source signal readouts required before truth')
    c=evaluation.validate_config(m['config'])
    if any(c.get(k)!=m.get(k) for k in ('run_id','code_commit','arm')) or evaluation._identity(m['model_identity'],c)!=m['parent_F0_identity']:raise ValueError('Actual frozen signal checkpoint identity differs')
    pm,controls,old_tau,old_global,pc=evaluation._reference(c['reference_prediction_root'],m['model_identity'],m['source_feature_manifest'])
    if pm!=m['reference_prediction_identity'] or m['dura_identity']!=m['model_identity']['dura_identity']:raise ValueError('Original frozen control/source ancestry differs')
    for role,n in COUNTS.items():
        iq=m.get('IQ_manifests',{}).get(role,{})
        if (iq.get('schema')!=evaluation.IQ.ROLE_SCHEMA or iq.get('status')!='STAR_REPRESENTATION_ROLE_COMPLETE' or iq.get('role')!=role or iq.get('count')!=n
                or iq.get('source_only') is not True or iq.get('restored_input_path')!='P1' or iq.get('parent_F0_identity')!=m['parent_F0_identity'] or iq.get('dura_identity')!=m['dura_identity']):raise ValueError('All original source P1 role handles required')
    smoke=_read(root/'smoke.json');fit=_read(root/'models/complete.json')
    if any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_checkpoint=True,synthetic_signal_forward_rows=2,real_signal_forward_rows=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0).items()):raise ValueError('Actual signal checkpoint no-query smoke required')
    if any(fit.get(k)!=v for k,v in dict(status=evaluation.DENSITY_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],method_names=list(METHODS),controls_reused=True,signal_known_statistics_shared=True,source_V_fit=False,novel_fit=False).items()):raise ValueError('All source density statistics must freeze before V')
    states={}
    for j,name in enumerate(METHODS):
        with np.load(root/'models'/(name+'.npz'),allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
        evaluation._core().validate_state(state,'known_only' if j in (0,1) else 'density_ratio')
        if state['whitener'].shape!=(160,160):raise ValueError('Fixed D160 Gaussian family required')
        states[name]=state
    if any(not evaluation._same(states[name],controls[name]) for name in (METHODS[0],METHODS[2])):raise ValueError('Frozen source control model changed')
    if any(not np.array_equal(v,states[METHODS[3]][k]) for k,v in states[PRIMARY].items()):raise ValueError('New signal readouts must share exactly the same known statistics')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional','global'}:raise ValueError('Exactly four source threshold sets required')
        tau=z['conditional'].copy();glob=z['global'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or glob.shape!=(4,) or glob.dtype!=np.float64 or not np.isfinite(glob).all()
            or not np.array_equal(tau[[0,2]],old_tau) or not np.array_equal(glob[[0,2]],old_global)
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],method_names=list(METHODS),primary_method=PRIMARY,
                known_count=COUNTS['V'],group_count=42,known_only=True,unknown_inputs_used=False,source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,
                conditional_thresholds=tau.tolist(),global_thresholds=glob.tolist(),control_calibration_reused=True).items()) or set(cal.get('methods',{}))!=set(METHODS)):
        raise ValueError('Both controls and two independent known-only calibrations must be frozen')
    for j,name in enumerate(METHODS):
        r=cal['methods'][name]
        if set(r.get('groups',{}))!=set(common.GROUP_NAMES) or r.get('conditional_thresholds')!=tau[j].tolist():raise ValueError('All 42 known-risk groups required for each readout')
    for new,old in ((METHODS[0],evaluation.parent.METHODS[1]),(METHODS[2],evaluation.parent.METHODS[2])):
        if cal['methods'][new]!=pc['methods'][old] or cal['global_controls'][new]!=pc['global_controls'][old]:raise ValueError('Frozen control calibration changed')
    rows={}
    for role in ('known','novel'):
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        evaluation._validate_rows(a,role,4,True)
        old=evaluation._rows(c['reference_prediction_root'],role)
        for k,v in old.items():
            actual=a[k][:,[0,2]] if k in ('scores','raw_scores') else a[k];expected=v[:,[1,2]] if k in ('scores','raw_scores') else v
            if not np.array_equal(actual,expected):raise ValueError('Frozen control fields, original identity or frozen MLP features changed')
        reconstructed=np.column_stack([evaluation._core().score(states[name],a['learned_features'] if j in (0,2) else a['signal_features'],'known_only' if j in (0,1) else 'density_ratio') for j,name in enumerate(METHODS)])
        if not np.allclose(a['raw_scores'],reconstructed,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('All four density raw scores and predicted-class margins must reconstruct')
        rows[role]=a
    if set(rows['known']['ids'].tolist())&set(rows['novel']['ids'].tolist()):raise ValueError('Source calibration and novel roles overlap')
    return m,cal,rows['known'],rows['novel']


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable independent signal score root required')
    m,cal,known,novel=_prediction(prediction_root)
    truth=_truth(novel_truth,novel)  # Sole private read after four predictions and every frozen state.
    km=dict(common._masks(known['rx_i'],known['scenario']));nm=dict(common._masks(novel['rx_i'],truth['scenario']))
    allk=np.ones(len(known['ids']),bool);alln=np.ones(len(novel['ids']),bool);methods={}
    for j,name in enumerate(METHODS):
        def metric(k,n,kr=tuple(range(6)),ur=common.UNKNOWN_TX):return _metrics(known['y'][k],known['closed'][k],known['scores'][k,j],truth['tx'][n],novel['scores'][n,j],known['baseline_closed'][k],kr,ur)
        groups={key:metric(mask,nm[key]) for key,mask in km.items()};overall=groups['overall'];kp=all(_known_pass(v) for v in groups.values());cp=cal['methods'][name]['constraints_passed']
        up=bool(overall['coverage']['unknown_complete'] and overall['R_U'] is not None and overall['R_U']>=.90-1e-12)
        goals=dict(known_42_groups_passed=kp,source_known_calibration_passed=cp,source_overall_unknown_macro_90_passed=up,passed=bool(kp and cp and up),gate_scope='source_overall_falsification_only',RX_scene_unknown_diagnostic_only=True)
        methods[name]=dict(overall=overall,groups=groups,goals=goals,diagnostic_only=name!=PRIMARY,
            by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},by_scene={s:groups['scene/'+s] for s in common.SCENES},by_day={str(d):metric(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},
            by_TX={str(c):metric(known['y']==c,alln,(c,)) for c in range(6)}|{tx:metric(allk,truth['tx']==tx,ur=(tx,)) for tx in common.UNKNOWN_TX})
    keys=('A_K','C_K','R_U','H_score','auroc_known_macro','oscr_auc_macro','original_closed_minus_new_open');main=methods[PRIMARY]
    def delta(a,b):return {k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in keys}
    def paired(index):
        other=METHODS[index];pc=lambda k,n:paired_counts(known,novel,k,n,primary_index=1,reference_index=index)
        return dict(primary=PRIMARY,reference=other,primary_index=1,reference_index=index,same_physical_IDs_and_original_F0_head=True,thresholds_scope='Frozen controls or independently known-V calibrated signal readouts; no novel tuning',
            counts=pc(allk,alln),overall=delta(main['overall'],methods[other]['overall']),groups={key:dict(delta=delta(main['groups'][key],methods[other]['groups'][key]),counts=pc(km[key],nm[key])) for key in km},
            by_day={str(d):pc(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},by_known_TX={str(c):pc(known['y']==c,alln) for c in range(6)},by_unknown_TX={tx:pc(allk,truth['tx']==tx) for tx in common.UNKNOWN_TX})
    passed=main['goals']['passed']
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],primary_arm=evaluation.PRIMARY_ARM,method_names=list(METHODS),primary_method=PRIMARY,
        methods=methods,primary_summary=main['overall'],paired_primary_vs_frozen_known=paired(0),paired_primary_vs_signal_ratio=paired(3),controls_reused_exactly=True,signal_known_statistics_shared=True,
        source_gate_passed=passed,eligible_for_target_prereg=passed,eligible_primary=passed,gate_scope='source_overall_falsification_only',next_step='SOURCE_ELIGIBLE_FOR_SEPARATE_TARGET_PREREGISTRATION' if passed else 'NO_PROMOTION',
        source_only=True,truth_last=True,target_reads=0,target_U_reads=0,target_Q_reads=0,formal_target_accessed=False,target_DA=False,target_REG=False,source_RX_seen=True,source_novel_historically_exposed=True,new_blind_confirmation=False,
        A_N=None,H_OW=None,final_phase3_complete=False,full_goal_passed=False,full_goal_status='ACTIVE',original_F0_parameters=232993,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,combined_active_parameters=567714,existing_restorer_parameters=266951,native_updates=0,open_signal_updates=2220,
        embedding_training_uses_OE=True,density_background_uses_OE=True,source_bank_deployed=False,costs={k:m[k] for k in ('real_F_forward_rows','synthetic_F_forward_rows','G_forward_rows','synthetic_signal_forward_rows','real_signal_forward_rows','signal_forward_rows_by_role',
            'source_aggregate_bytes_by_method','threshold_bytes','cpu_fit_seconds','cpu_readout_seconds','cpu_readout_rows','cpu_readout_score_rows')},scope='Source seen-RX historical-novel screening; no target, DA, discovery or registration success')
    output.mkdir(parents=True);_write(output/'results.json',result);_write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_frozen_known','paired_primary_vs_signal_ratio')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
