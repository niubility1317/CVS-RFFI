"""Independent P0-open/P1-identity source screening; no target or neural fitting."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_raw_open_eval as evaluation
from . import star_consensus_score as common
from .star_crossset_metric_score import paired_counts
from .star_physical_open_score import _metrics,_known_pass
from .star_macro12_score import _cross_report
from .star_ova_score import _ci

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_star_raw_open_scores_v1'
STATUS='STAR_RAW_OPEN_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=evaluation.PRIMARY_METHOD
COUNTS=evaluation.COUNTS
HARD_NAMES=evaluation.risk.HARD_NAMES
DIAGNOSTIC_NAMES=evaluation.risk.DIAGNOSTIC_NAMES
_read,_write,_truth=common._read,common._write,common._truth


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,arm=evaluation.PRIMARY_ARM,primary_arm=evaluation.PRIMARY_ARM,method_names=list(METHODS),primary_method=PRIMARY,
        classification_path='P1',identity_baseline_path='P1',rejection_input_path='P0',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,source_novel_fit_reads=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,G_checkpoint_loaded=False,
        real_signal_forward_rows=sum(COUNTS.values()),synthetic_signal_forward_rows=2,signal_forward_rows_by_role=dict(COUNTS),native_updates=0,open_signal_updates=2220,eval_neural_updates=0,
        teacher_input_path='P0',teacher_real_F_forward_rows_in_eval=0,restored_P1_values_read=0,received_P0_values_read=sum(COUNTS.values()),channel_regenerations=0,second_RMS_applied=False,
        identity_predictions_unchanged=True,default_head_replaced=False,P1_known12_control_reused_exactly=True,P1_models_reused=True,P1_known_statistics_shared=True,P0_known_statistics_shared=True,same_projection_for_P0_methods=True,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,
        source_V_model_fit=False,source_V_threshold_fit=True,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,original_F0_parameters=232993,combined_active_parameters=567714,
        reference_run_id=evaluation.REFERENCE_RUN,reference_code_commit=evaluation.REFERENCE_CODE,signal_reference_run_id=evaluation.SIGNAL_REFERENCE_RUN,signal_reference_code_commit=evaluation.SIGNAL_REFERENCE_CODE,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Complete P0-open and immutable original-P1 identity predictions required before truth')
    c=evaluation.validate_config(m['config'])
    if any(c[k]!=m[k] for k in ('run_id','code_commit','arm')) or evaluation._identity(m['model_identity'],c)!=m['parent_F0_identity']:raise ValueError('Actual frozen P0 checkpoint identity differs')
    pm,sm,p1known,p1ratio,old_tau,old_cal=evaluation._references(c,m['model_identity'])
    if m['reference_prediction_identity']!=pm or m['signal_reference_identity']!=sm or m['dura_identity']!=m['model_identity']['dura_identity']:raise ValueError('Two frozen P1 parent sources differ')
    for role,n in COUNTS.items():
        a=m['IQ_manifests'][role]
        if a!=pm['source_IQ_manifests'][role] or a['count']!=n or a['role']!=role or a['received_input_path']!='P0':raise ValueError('P0 public IQ role handles must remain identical to the P1 parents')
    smoke=_read(root/'smoke.json');fit=_read(root/'models/complete.json')
    if any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_checkpoint=True,synthetic_signal_forward_rows=2,real_signal_forward_rows=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0).items()):raise ValueError('Actual P0 checkpoint synthetic smoke missing')
    if any(fit.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],method_names=list(METHODS),
            P1_models_reused=True,P0_known_statistics_shared=True,P1_known_statistics_shared=True,source_V_fit=False,novel_fit=False).items()):raise ValueError('Four source known/background states must freeze before V')
    states={}
    for j,name in enumerate(METHODS):
        with np.load(root/'models'/(name+'.npz'),allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
        evaluation._core().validate_state(state,'known_only' if j in (0,2) else 'density_ratio')
        if state['whitener'].shape!=(160,160):raise ValueError('Fixed D160 density family required')
        states[name]=state
    if not evaluation._same(states[METHODS[0]],p1known) or not evaluation._same(states[METHODS[1]],p1ratio):raise ValueError('Frozen P1 known/ratio model changed')
    for known,ratio in ((0,1),(2,3)):
        if any(not np.array_equal(v,states[METHODS[ratio]][k]) for k,v in states[METHODS[known]].items()):raise ValueError('Each path must share its own known8 between known and ratio readouts')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Four frozen six-class threshold sets required')
        tau=z['conditional'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or not np.array_equal(tau[0],old_tau)
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],method_names=list(METHODS),primary_method=PRIMARY,
                known_count=COUNTS['V'],group_count=42,hard_names=list(HARD_NAMES),diagnostic_names=list(DIAGNOSTIC_NAMES),known_only=True,unknown_inputs_used=False,target_reads=0,
                conditional_thresholds=tau.tolist(),P1_known12_calibration_reused=True,source_V_neural_fit=False,source_V_model_fit=False,source_V_threshold_fit=True).items()) or set(cal.get('methods',{}))!=set(METHODS)):
        raise ValueError('Exact P1 known12 control and three independent V-only hard12 calibrations required')
    for j,name in enumerate(METHODS):
        r=cal['methods'][name]
        if (set(r.get('groups',{}))!=set(common.GROUP_NAMES) or r.get('conditional_thresholds')!=tau[j].tolist()
                or r.get('hard_names')!=list(HARD_NAMES) or r.get('diagnostic_names')!=list(DIAGNOSTIC_NAMES)):raise ValueError('Every method must retain hard12 and all thirty cross diagnostics')
    if cal['methods'][METHODS[0]]!=old_cal['methods']['signal_known12']:raise ValueError('Exact previous P1 known12 calibration report changed')
    rows={}
    for role in ('known','novel'):
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        if set(a)!=(evaluation.KNOWN_FIELDS if role=='known' else evaluation.NOVEL_FIELDS):raise ValueError('Exact public predictions required; novel TX forbidden')
        evaluation.previous._validate_rows({k:v for k,v in a.items() if k!='raw_signal_features'},role,4,True)
        n=len(a['ids']);z=a['raw_signal_features']
        if z.shape!=(n,160) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,atol=2e-6,rtol=0):raise ValueError('Saved same-row P0 unit160 required')
        old,sig=evaluation._reference_rows(c,role)
        for k,v in old.items():
            if k not in ('raw_scores','scores') and not np.array_equal(a[k],v):raise ValueError('Original P1 identity/baseline/metadata/frozen representations changed')
        if not np.array_equal(a['raw_scores'][:,0],old['raw_scores'][:,3]) or not np.array_equal(a['scores'][:,0],old['scores'][:,3]) or not np.array_equal(a['raw_scores'][:,1],sig['raw_scores'][:,3]):raise ValueError('Frozen P1 known12 operating point or P1 ratio evidence changed')
        reconstructed=np.column_stack([evaluation._core().score(states[name],a['signal_features'] if j<2 else a['raw_signal_features'],'known_only' if j in (0,2) else 'density_ratio') for j,name in enumerate(METHODS)])
        if not np.allclose(a['raw_scores'],reconstructed,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('P1/P0 four raw scores and original-P1-class routed margins must reconstruct')
        rows[role]=a
    if set(rows['known']['ids'].tolist())&set(rows['novel']['ids'].tolist()):raise ValueError('Source fitting/calibration and novel physical roles overlap')
    return m,cal,rows['known'],rows['novel']


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable independent raw-open score root required')
    m,cal,known,novel=_prediction(prediction_root)
    truth=_truth(novel_truth,novel)  # Only private read after all four source predictions and states.
    km=dict(common._masks(known['rx_i'],known['scenario']));nm=dict(common._masks(novel['rx_i'],truth['scenario']))
    allk=np.ones(len(known['ids']),bool);alln=np.ones(len(novel['ids']),bool);methods={}
    for j,name in enumerate(METHODS):
        def metric(k,n,kr=tuple(range(6)),ur=common.UNKNOWN_TX):return _metrics(known['y'][k],known['closed'][k],known['scores'][k,j],truth['tx'][n],novel['scores'][n,j],known['baseline_closed'][k],kr,ur)
        groups={key:metric(mask,nm[key]) for key,mask in km.items()};overall=groups['overall'];k12=all(_known_pass(groups[k]) for k in HARD_NAMES);k42=all(_known_pass(v) for v in groups.values());cp=cal['methods'][name]['constraints_passed']
        up=bool(overall['coverage']['unknown_complete'] and overall['R_U'] is not None and overall['R_U']>=.90-1e-12)
        methods[name]=dict(overall=overall,groups=groups,hard_names=list(HARD_NAMES),diagnostic_names=list(DIAGNOSTIC_NAMES),diagnostic_only=name!=PRIMARY,
            goals=dict(known_12_groups_passed=k12,known_42_groups_passed=k42,source_known_calibration_passed=cp,source_overall_unknown_macro_90_passed=up,passed=bool(k12 and cp and up),gate_scope='source_overall_falsification_only',cross30_diagnostic=True),
            cross_guard_violations_or_missing=_cross_report(groups),by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},by_scene={s:groups['scene/'+s] for s in common.SCENES},
            by_day={str(d):metric(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},by_TX={str(c):metric(known['y']==c,alln,(c,)) for c in range(6)}|{tx:metric(allk,truth['tx']==tx,ur=(tx,)) for tx in common.UNKNOWN_TX},
            confidence_intervals=dict(level=.95,method='Per true-class Wilson; repeated source development pool, not independent-RX macro confidence',A_K_per_class=_ci(overall['known_per_class'],'correct_accepted'),C_K_per_class=_ci(overall['known_per_class'],'accepted'),R_U_per_class=_ci(overall['unknown_per_class'],'rejected')))
    main=methods[PRIMARY];keys=('A_K','C_K','R_U','H_score','auroc_known_macro','oscr_auc_macro','original_closed_minus_new_open')
    def delta(a,b):return {k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in keys}
    def paired(index):
        reference=METHODS[index];pc=lambda k,n:paired_counts(known,novel,k,n,primary_index=2,reference_index=index)
        return dict(primary=PRIMARY,reference=reference,primary_index=2,reference_index=index,same_physical_IDs_and_original_F0_P1_head=True,thresholds_scope='Exact historical P1 known12 or independent same known-V hard12 calibrations; no novel tuning',
            counts=pc(allk,alln),overall=delta(main['overall'],methods[reference]['overall']),groups={key:dict(delta=delta(main['groups'][key],methods[reference]['groups'][key]),counts=pc(km[key],nm[key])) for key in km},
            by_day={str(d):pc(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},by_known_TX={str(c):pc(known['y']==c,alln) for c in range(6)},by_unknown_TX={tx:pc(allk,truth['tx']==tx) for tx in common.UNKNOWN_TX})
    passed=main['goals']['passed']
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],primary_arm=evaluation.PRIMARY_ARM,method_names=list(METHODS),primary_method=PRIMARY,methods=methods,primary_summary=main['overall'],
        paired_primary_vs_P1_known=paired(0),paired_primary_vs_P0_ratio=paired(3),source_gate_passed=passed,eligible_for_target_prereg=passed,eligible_primary=passed,gate_scope='source_overall_falsification_only',
        next_step='SOURCE_ELIGIBLE_FOR_SEPARATE_TARGET_PREREGISTRATION' if passed else 'NO_PROMOTION',source_only=True,truth_last=True,target_reads=0,target_U_reads=0,target_Q_reads=0,formal_target_accessed=False,target_DA=False,target_REG=False,
        source_RX_seen=True,source_novel_historically_exposed=True,new_blind_confirmation=False,A_N=None,H_OW=None,final_phase3_complete=False,full_goal_passed=False,full_goal_status='ACTIVE',native_updates=0,open_signal_updates=2220,
        classification_path='P1',identity_baseline_path='P1',new_rejection_input_path='P0',teacher_input_path='P0',original_F0_parameters=232993,signal_parameters=334721,combined_active_parameters=567714,existing_restorer_parameters=266951,
        costs={k:m[k] for k in ('real_signal_forward_rows','synthetic_signal_forward_rows','real_F_forward_rows','synthetic_F_forward_rows','G_forward_rows','source_aggregate_bytes_by_method','threshold_bytes','cpu_fit_seconds','cpu_readout_seconds','cpu_readout_rows','cpu_readout_score_rows')},
        inherited_training_cost_scope=m['training_teacher_cost_scope'],scope='Source seen-RX historical-novel P0/P1 comparison; original P1 identity protection, hard12 source screening only, no target/DA/registration claim')
    output.mkdir(parents=True);_write(output/'results.json',result);_write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_P1_known','paired_primary_vs_P0_ratio')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
