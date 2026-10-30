"""Truth-last source comparison of fixed-score hard12 and historical hard42 risk."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_macro12_run as evaluation
from . import star_consensus_score as common
from .star_crossset_metric_score import paired_counts
from .star_physical_open_score import _metrics,_known_pass
from .star_ova_score import _ci

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_star_macro12_scores_v1'
STATUS='STAR_MACRO12_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=evaluation.PRIMARY_METHOD
COUNTS=evaluation.COUNTS
HARD_NAMES=tuple(n for n in common.GROUP_NAMES if not n.startswith('rxscene/'))
DIAGNOSTIC_NAMES=tuple(n for n in common.GROUP_NAMES if n.startswith('rxscene/'))
_read,_write,_truth=common._read,common._write,common._truth


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,classification_path='P1',score_threshold=0.,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_only=True,target_reads=0,private_truth_reads=0,all_thresholds_frozen_before_novel=True,
        identity_predictions_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,
        synthetic_signal_forward_rows=2,native_updates=0,new_neural_updates=0,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('All four completed frozen-score predictions required before truth')
    c=evaluation.validate_config(m['config'])
    if c['run_id']!=m['run_id'] or c['code_commit']!=m['code_commit']:raise ValueError('Actual calibration run/code identity differs')
    pm,old_states,old_tau,old_cal=evaluation._reference(c['reference_prediction_root'],c['train_config'])
    if m.get('reference_prediction_identity')!=pm:raise ValueError('Fixed original signal prediction identity differs')
    smoke=_read(root/'smoke.json');fit=_read(root/'models/complete.json')
    if any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_checkpoint=True,synthetic_signal_forward_rows=2,real_signal_forward_rows=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0).items()):raise ValueError('Actual frozen parent-checkpoint synthetic smoke required')
    if any(fit.get(k)!=v for k,v in dict(status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],models_frozen=True,models_refitted=False).items()):raise ValueError('Both known-only models must be copied and frozen before V calibration')
    states={}
    for name in ('frozen_known','signal_known'):
        with np.load(root/'models'/(name+'.npz'),allow_pickle=False) as z:states[name]={k:z[k].copy() for k in z.files}
        evaluation._core().validate_state(states[name],'known_only')
        if not evaluation._same(states[name],old_states[name]):raise ValueError('Frozen source known-support model changed')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Only four six-class frozen threshold sets required')
        tau=z['conditional'].copy()
    if tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or not np.array_equal(tau[[0,2]],old_tau):raise ValueError('Historical known42 thresholds must be exact')
    if np.any(tau[1]<tau[0]) or np.any(tau[3]<tau[2]):raise ValueError('Same-score known12 thresholds must not decrease from known42')
    cal=_read(root/'calibration/complete.json');hard={name:list(common.GROUP_NAMES if j in (0,2) else HARD_NAMES) for j,name in enumerate(METHODS)}
    expected_cal=dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,
        known_count=COUNTS['V'],known_only=True,unknown_inputs_used=False,target_reads=0,conditional_thresholds=tau.tolist(),method_hard_groups=hard)
    if any(cal.get(k)!=v for k,v in expected_cal.items()) or set(cal.get('methods',{}))!=set(METHODS):raise ValueError('Four frozen known-only calibrations with explicit hard scopes required')
    for j,name in enumerate(METHODS):
        report=cal['methods'][name]
        if set(report.get('groups',{}))!=set(common.GROUP_NAMES) or report.get('conditional_thresholds')!=tau[j].tolist():raise ValueError('All 42 selected-risk group reports required')
        if j in (1,3) and (report.get('hard_names')!=list(HARD_NAMES) or report.get('diagnostic_names')!=list(DIAGNOSTIC_NAMES)):raise ValueError('Hard12 and diagnostic30 scopes must remain distinct')
    for new,old in ((METHODS[0],'frozen_known_support'),(METHODS[2],'signal_known_support')):
        if cal['methods'][new]!=old_cal['methods'][old]:raise ValueError('Historical known42 report changed')
    rows={}
    for role in ('known','novel'):
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        evaluation.source._validate_rows(a,role,4,True)
        old=evaluation._rows(c['reference_prediction_root'],role)
        for key,value in old.items():
            actual=a[key][:,[0,2]] if key in ('raw_scores','scores') else a[key];expected=value[:,:2] if key in ('raw_scores','scores') else value
            if not np.array_equal(actual,expected):raise ValueError('Historical controls/identity/metadata/feature rows changed')
        if not np.array_equal(a['raw_scores'],old['raw_scores'][:,[0,0,1,1]]):raise ValueError('All four same-score columns must retain exact original raw evidence')
        raw0=evaluation._core().score(states['frozen_known'],a['learned_features'],'known_only');raw1=evaluation._core().score(states['signal_known'],a['signal_features'],'known_only')
        if not np.allclose(a['raw_scores'],np.column_stack((raw0,raw0,raw1,raw1)),rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('Four model scores and predicted-class threshold routing must reconstruct')
        if np.any((a['scores'][:,1]>=0)&(a['scores'][:,0]<0)) or np.any((a['scores'][:,3]>=0)&(a['scores'][:,2]<0)):raise ValueError('Known12 acceptance must be a subset of its same-score known42 control')
        rows[role]=a
    if set(rows['known']['ids'].tolist())&set(rows['novel']['ids'].tolist()):raise ValueError('Source calibration and novel roles overlap')
    return m,cal,rows['known'],rows['novel']


def _cross_report(groups):
    result={}
    for name in DIAGNOSTIC_NAMES:
        row=groups[name]
        if _known_pass(row):continue
        reasons=[]
        if not row['coverage']['known_complete']:reasons.append('missing_known_class')
        if row['C_K'] is not None and row['C_K']<.90-1e-12:reasons.append('C_K_below_90')
        if row['original_closed_minus_new_open'] is not None and row['original_closed_minus_new_open']>.02+1e-12:reasons.append('original_F0_loss_above_2pp')
        result[name]=dict(known_count=row['known_count'],coverage=row['coverage'],C_K=row['C_K'],A_K=row['A_K'],original_closed_minus_new_open=row['original_closed_minus_new_open'],reasons=reasons)
    return result


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable independent macro12 score output required')
    m,cal,known,novel=_prediction(prediction_root)
    truth=_truth(novel_truth,novel)  # Only private access, after all four predictions/state reconstructions.
    km=dict(common._masks(known['rx_i'],known['scenario']));nm=dict(common._masks(novel['rx_i'],truth['scenario']))
    allk=np.ones(len(known['ids']),bool);alln=np.ones(len(novel['ids']),bool);methods={}
    for j,name in enumerate(METHODS):
        def metric(k,n,kr=tuple(range(6)),ur=common.UNKNOWN_TX):return _metrics(known['y'][k],known['closed'][k],known['scores'][k,j],truth['tx'][n],novel['scores'][n,j],known['baseline_closed'][k],kr,ur)
        groups={key:metric(mask,nm[key]) for key,mask in km.items()};overall=groups['overall'];hard_names=tuple(common.GROUP_NAMES) if j in (0,2) else HARD_NAMES
        k12=all(_known_pass(groups[k]) for k in HARD_NAMES);k42=all(_known_pass(v) for v in groups.values());kp=all(_known_pass(groups[k]) for k in hard_names);cp=cal['methods'][name]['constraints_passed']
        up=bool(overall['coverage']['unknown_complete'] and overall['R_U'] is not None and overall['R_U']>=.90-1e-12)
        goals=dict(known_hard_group_count=len(hard_names),known_hard_groups_passed=kp,known_12_groups_passed=k12,known_42_groups_passed=k42,source_known_calibration_passed=cp,
            source_overall_unknown_macro_90_passed=up,passed=bool(kp and cp and up),gate_scope='source_overall_falsification_only',RX_scene_unknown_diagnostic_only=True,cross_groups_hard=j in (0,2))
        methods[name]=dict(overall=overall,groups=groups,goals=goals,hard_names=list(hard_names),diagnostic_names=list(DIAGNOSTIC_NAMES) if j in (1,3) else [],diagnostic_only=name!=PRIMARY,
            cross_guard_violations_or_missing=_cross_report(groups),by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},by_scene={s:groups['scene/'+s] for s in common.SCENES},
            by_day={str(d):metric(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},by_TX={str(c):metric(known['y']==c,alln,(c,)) for c in range(6)}|{tx:metric(allk,truth['tx']==tx,ur=(tx,)) for tx in common.UNKNOWN_TX})
        methods[name]['confidence_intervals']=dict(level=.95,method='Per true-class Wilson; descriptive repeated source pool, not independent-RX macro confidence',
            A_K_per_class=_ci(overall['known_per_class'],'correct_accepted'),C_K_per_class=_ci(overall['known_per_class'],'accepted'),R_U_per_class=_ci(overall['unknown_per_class'],'rejected'))
    main=methods[PRIMARY];keys=('A_K','C_K','R_U','H_score','auroc_known_macro','oscr_auc_macro','original_closed_minus_new_open')
    def delta(a,b):return {k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in keys}
    def paired(index):
        reference=METHODS[index];pc=lambda k,n:paired_counts(known,novel,k,n,primary_index=3,reference_index=index)
        return dict(primary=PRIMARY,reference=reference,primary_index=3,reference_index=index,same_physical_IDs_and_original_F0_head=True,
            calibration_scope='Historical hard42 or new independently V-only hard12; exact frozen raw scores, no novel tuning',counts=pc(allk,alln),overall=delta(main['overall'],methods[reference]['overall']),
            groups={key:dict(delta=delta(main['groups'][key],methods[reference]['groups'][key]),counts=pc(km[key],nm[key])) for key in km},
            by_day={str(d):pc(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},by_known_TX={str(c):pc(known['y']==c,alln) for c in range(6)},by_unknown_TX={tx:pc(allk,truth['tx']==tx) for tx in common.UNKNOWN_TX})
    passed=main['goals']['passed']
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,methods=methods,primary_summary=main['overall'],
        paired_primary_vs_signal42=paired(2),paired_primary_vs_frozen12=paired(1),source_gate_passed=passed,eligible_for_target_prereg=passed,eligible_primary=passed,
        gate_scope='source_overall_falsification_only',hard_known_group_names=list(HARD_NAMES),diagnostic_cross_group_names=list(DIAGNOSTIC_NAMES),
        next_step='SOURCE_ELIGIBLE_FOR_SEPARATE_TARGET_PREREGISTRATION' if passed else 'NO_PROMOTION',source_only=True,truth_last=True,target_reads=0,target_U_reads=0,target_Q_reads=0,
        formal_target_accessed=False,target_DA=False,target_REG=False,source_RX_seen=True,source_novel_historically_exposed=True,new_blind_confirmation=False,A_N=None,H_OW=None,
        final_phase3_complete=False,full_goal_passed=False,full_goal_status='ACTIVE',native_updates=0,new_neural_updates=0,new_signal_updates=0,new_projection_updates=0,
        real_IQ_reads=0,real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,synthetic_signal_forward_rows=2,source_L_fit_reads=0,source_OE_fit_reads=0,
        parent_prediction_identity=m['reference_prediction_identity'],parent_training_cost_scope='Original F0 ancestor11100, copied signal+2220/projection2220 inherited unchanged; not new updates in this calibration run',
        original_F0_parameters=232993,signal_parameters=334721,combined_active_parameters=567714,existing_restorer_parameters=266951,
        costs={k:m[k] for k in ('real_F_forward_rows','real_signal_forward_rows','G_forward_rows','synthetic_signal_forward_rows')},
        scope='Source seen-RX historical-novel calibration comparison. Hard12 pass is not hard42 pass, final main-scene certification, cross-RX target success or registration.')
    output.mkdir(parents=True);_write(output/'results.json',result);_write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_signal42','paired_primary_vs_frozen12')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
