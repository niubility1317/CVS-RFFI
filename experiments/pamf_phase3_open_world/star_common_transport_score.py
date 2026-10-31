"""Independent eight-readout scoring after completed CCD predictions only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_common_transport_run as evaluation
from . import star_density_core as density
from . import star_dura_best_score as previous
from . import star_consensus_score as common
from .star_crossset_metric_score import paired_counts
from .scoring import _truth as _target_truth

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_common_transport_scores_v1'
STATUS='COMMON_TRANSPORT_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY_INDEX=7
PRIMARY=METHODS[PRIMARY_INDEX]
CONDITIONS=evaluation.CONDITIONS
COUNTS=dict(evaluation.COUNTS)
Q_COUNT=864
_read,_write,_source_truth=common._read,common._write,common._truth


def _control_gate(main,controls):
    gain=all(main['A_K'] is not None and row['A_K'] is not None and main['A_K']>row['A_K']+1e-12 for row in controls)
    keep=all(main['R_U'] is not None and row['R_U'] is not None and main['R_U']>=row['R_U']-1e-12 for row in controls)
    return bool(gain),bool(keep)


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,classification_unchanged=True,control_readouts_reused_exactly=True,
        source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,source_density_models_fitted=1,new_threshold_sets_fitted=2,
        source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,private_truth_reads=0,target_U_reads=0,U_truth_reads=0,
        source_novel_truth_reads=0,target_truth_reads=0,original_U_s_reads=0,new_neural_updates=0,new_candidate_optimizer_updates=444,
        source_IQ_reads=sum(COUNTS.values()),target_Q_IQ_reads=2*Q_COUNT,real_signal_forward_rows=sum(COUNTS.values())+2*Q_COUNT,
        real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,old_control_real_signal_forward_rows=0,threshold_values=48,threshold_bytes=384,
        target_DA=True,target_REG=False,formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('All eight-column three-condition predictions must complete before truth')
    c=evaluation.validate_config(m['config'])
    if c['code_commit']!=m['code_commit']:raise ValueError('Actual prediction code identity differs')
    # The established parent validator reads no truth or neural checkpoint.
    old,old_cal,old_rows=evaluation.previous_score._prediction(c['reference_config']['output_root'])
    pm,parent_states,old_tau,_,old_Uids=evaluation._reference(c)
    if (old!=pm or m['reference_prediction_identity']!=pm or m['parent_F0_identity']!=pm['parent_F0_identity']
            or m['dura_identity']!=pm['dura_identity'] or m['loaded_dura_identity']!=pm['loaded_dura_identity']):raise ValueError('Completed control and original F0/r03/G ancestry differ')
    identity,Uids,_=evaluation._training_metadata(c,pm);evaluation._identity(identity,c,pm)
    if not np.array_equal(Uids,old_Uids):raise ValueError('The exact fixed star-U role must be preserved')
    fit=_read(root/'models/complete.json');cal=_read(root/'calibration/complete.json')
    expected_fit=dict(schema=evaluation.SCHEMA,status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],neural_states_frozen=True,source_fit_inputs_only=True,
        source_V_fit=False,novel_fit=False,target_Q_fit=False,source_density_models_fitted=1,normalizer_fit_role='L',background_fit_role='OE',control_readouts_reused_exactly=True,TARGET_CCD_has_target_U_ancestor=True)
    if any(fit.get(k)!=v for k,v in expected_fit.items()):raise ValueError('One new source-only fit and all inherited densities must freeze before V')
    identities=dict(pm['model_identities'],TARGET_CCD=identity)
    if fit.get('identities')!=identities or m['model_identities']!=identities:raise ValueError('Actual checkpoint identities must match all four feature axes')
    states={name:evaluation.previous_score._arrays(root/'models'/(name+'.npz')) for name in evaluation.MODEL_NAMES}
    for state in states.values():density.validate_state(state,'density_ratio')
    if any(not evaluation._same(states[name],parent_states[name]) for name in evaluation.previous.MODEL_NAMES):raise ValueError('Inherited three densities changed')
    expected_cal=dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,
        group_count=42,hard_names=list(evaluation.risk.HARD_NAMES),diagnostic_names=list(evaluation.risk.DIAGNOSTIC_NAMES),known_only=True,unknown_inputs_used=False,target_reads=0,
        source_V_neural_fit=False,source_V_density_fit=False,new_threshold_sets_fitted=2,control_readouts_reused_exactly=True)
    if any(cal.get(k)!=v for k,v in expected_cal.items()):raise ValueError('All old thresholds and two known-only CCD threshold vectors must freeze')
    saved=evaluation.previous_score._arrays(root/'thresholds.npz');tau=saved.get('conditional')
    if (set(saved)!={'conditional'} or tau.dtype!=np.float64 or tau.shape!=(8,6) or not np.isfinite(tau).all()
            or not np.array_equal(tau[:6],old_tau) or not np.array_equal(tau,np.asarray(cal['conditional_thresholds'])) or set(cal.get('methods',{}))!=set(METHODS)):
        raise ValueError('Exact inherited thirty-six plus twelve new thresholds required')
    for j,name in enumerate(METHODS):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist() or (j<6 and cal['methods'][name]!=old_cal['methods'][name]):raise ValueError('Frozen class-threshold reports or inherited controls differ')
    used=set(Uids.tolist());rows={'source_DURA':{}}
    def verify(a,old):
        evaluation._exact_controls(a,old)
        raw=evaluation._raw(states['TARGET_CCD'],a['signal_features'][:,3],a['closed'])
        if not np.allclose(a['raw_scores'][:,6:],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('Actual CCD geometry/readouts and predicted-class threshold routing must reconstruct')
    for role in ('known','novel'):
        a=evaluation._source_rows(root,'source_DURA',role);verify(a,old_rows['source_DURA'][role]);ids=set(a['ids'].tolist())
        if used&ids:raise ValueError('Source and U fitting/evaluation records overlap')
        used.update(ids);rows['source_DURA'][role]=a
    for condition in CONDITIONS[1:]:
        a=evaluation._target_rows(root,condition);verify(a,old_rows[condition])
        if used&set(a['ids'].tolist()):raise ValueError('Future Q overlaps source or discovery fitting records')
        rows[condition]=a
    return m,cal,rows


def _pair(k,n,main,reference,km=None,nm=None):
    return paired_counts(k,n,np.ones(len(k['y']),bool) if km is None else km,np.ones(len(n['scores']),bool) if nm is None else nm,primary_index=main,reference_index=reference)


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent UDA score output required')
    m,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['source_DURA']['novel']);truths={base:_target_truth(path,rows[base+'_DURA']['ids']) for base,path in (('clean',clean_truth),('star',star_truth))}
    for base,t in truths.items():
        if not np.isin(t['y'],range(9)).all() or not np.all(t['receiver']==26) or not np.isin(t['scenario'],('clean',) if base=='clean' else common.SCENES).all():raise ValueError('Fixed cross-RX development truth and private scorer roster required')
    methods={}
    for j,name in enumerate(METHODS):
        paths={'source_DURA':previous._source(rows['source_DURA'],nt,j)};paths.update({base+'_DURA':previous._target(rows[base+'_DURA'],truths[base],base+'_DURA',j) for base in ('clean','star')});methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY)
    main=methods[PRIMARY]['paths'];paired={}
    for reference in (1,3,5):
        by_condition={}
        for condition in CONDITIONS:
            if condition=='source_DURA':
                k,n=rows[condition]['known'],rows[condition]['novel'];km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],nt['scenario']));groups={g:_pair(k,n,7,reference,mask,nm[g]) for g,mask in km.items()};uy=nt['tx'];unknown_roster=common.UNKNOWN_TX
            else:
                t=truths[condition.split('_')[0]];y=t['y'];mask=y<6;a=rows[condition];k=dict(y=y[mask],closed=a['closed'][mask],scores=a['scores'][mask]);n=dict(scores=a['scores'][~mask]);scenes=('clean',) if condition=='clean_DURA' else common.SCENES
                groups={g:_pair(k,n,7,reference,t['scenario'][mask]==g,t['scenario'][~mask]==g) for g in scenes};uy=y[~mask];unknown_roster=(6,7,8)
            by_condition[condition]=dict(counts=_pair(k,n,7,reference),groups=groups,delta=previous._delta(main[condition]['overall'],methods[METHODS[reference]]['paths'][condition]['overall']),
                by_known_TX={str(c):_pair(k,n,7,reference,k['y']==c) for c in range(6)},by_unknown_TX={str(c):_pair(k,n,7,reference,nm=uy==c) for c in unknown_roster})
        paired[METHODS[reference]]=by_condition
    gains={reference:paired[reference]['star_DURA']['delta']['A_K'] for reference in ('E0_predicted_ratio','SOURCE_predicted_ratio','PCD_predicted_ratio')}
    positive,rejection_kept=_control_gate(main['star_DURA']['overall'],[methods[name]['paths']['star_DURA']['overall'] for name in ('E0_predicted_ratio','SOURCE_predicted_ratio','PCD_predicted_ratio')])
    source_known=main['source_DURA']['known_12_groups_passed'] and bool(cal['methods'][PRIMARY]['constraints_passed'])
    protection=all(main[c]['goals']['passed'] for c in CONDITIONS[1:]);success=bool(source_known and positive and rejection_kept and protection)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,primary_summary={c:main[c]['overall'] for c in CONDITIONS},
        paired_primary_vs_E0_predicted_ratio=paired['E0_predicted_ratio'],paired_primary_vs_SOURCE_predicted_ratio=paired['SOURCE_predicted_ratio'],star_A_K_gain_vs_E0=gains['E0_predicted_ratio'],star_A_K_gain_vs_SOURCE=gains['SOURCE_predicted_ratio'],star_A_K_gain_vs_PCD=gains['PCD_predicted_ratio'],paired_primary_vs_PCD_predicted_ratio=paired['PCD_predicted_ratio'],star_A_K_gain_over_all_three_controls=bool(positive),star_R_U_not_lower_than_all_three_controls=bool(rejection_kept),source_known_12_passed=bool(source_known),target_protection_passed=bool(protection),adaptation_success=success,development_gate_passed=success,eligible_primary=success,
        next_step='DEVELOPMENT_ADAPTATION_PASSED_FOR_SEPARATE_REGISTRATION_PREREGISTRATION' if success else 'NO_PROMOTION',truth_last=True,all_three_predictions_before_truth=True,U_truth_reads=0,source_truth_reads=1,target_truth_reads=2,
        classification_unchanged=True,control_readouts_reused_exactly=True,target_fit_input='Unlabelled star U balanced prototype objective; clean and Q are test only',source_unknown_diagnostic_only=True,
        SOURCE_weights_source_only=True,TARGET_PCD_weights_source_only=False,TARGET_CCD_weights_source_only=False,target_DA=True,target_REG=False,
        new_projection_optimizer_updates_total=444,new_body_optimizer_updates_total=444,new_candidate_optimizer_updates=444,prediction_optimizer_updates=0,Sinc_optimizer_steps=0,
        new_source_density_models=1,new_source_threshold_sets=2,training_only_states_deployed=False,original_F0_updates=0,G_updates=0,threshold_values=48,threshold_bytes=384,
        costs={k:m[k] for k in ('source_IQ_reads','target_Q_IQ_reads','target_U_reads','U_ID_metadata_reads','training_target_U_count','real_F_forward_rows','G_forward_rows','real_signal_forward_rows','old_control_real_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows','synthetic_parent_signal_forward_rows','synthetic_CCD_signal_forward_rows')},
        training_costs=m['training_costs'],combined_new_real_signal_forward_rows=m['combined_new_real_signal_forward_rows'],combined_new_G_forward_rows=m['combined_new_G_forward_rows'],combined_new_real_F_forward_rows=m['combined_new_real_F_forward_rows'],
        inherited_control_prediction_identity=m['reference_prediction_identity'],inherited_control_training_updates=888,
        source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,registration_success=None,A_N=None,H_OW=None,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        original_F0_parameters=232993,open_signal_parameters_per_arm=334721,existing_restorer_parameters=266951,primary_deployed_neural_parameters=834665,
        scope='One independent source-r03 CCD student; six completed controls are copied, not retrained or re-forwarded. TARGET_CCD and its source-fitted density have target-U ancestry. Historical development/source exposure; no full UniOT reproduction, formalblind, discovery or registration proof')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_E0_predicted_ratio','paired_primary_vs_SOURCE_predicted_ratio','paired_primary_vs_PCD_predicted_ratio','inherited_control_prediction_identity')});print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
