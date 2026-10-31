"""Independent frozen body-prototype comparison; all predictions precede truths."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_body_prototype_run as evaluation
from . import star_density_core as density
from . import star_dura_best_score as previous
from . import star_consensus_score as common
from .star_crossset_metric_score import paired_counts
from .scoring import _truth as _target_truth

RUN_ID='20261031-phase3-body-prototype-s2026092701-r01'
SCHEMA='pamf_phase3_body_prototype_scores_v1'
STATUS='BODY_PROTOTYPE_SCORES_COMPLETE'
METHODS=('E0_pooled_ratio','E0_predicted_ratio','SOURCE_pooled_ratio','SOURCE_predicted_ratio','PCD_pooled_ratio','PCD_predicted_ratio')
PRIMARY=METHODS[5]
MODEL_NAMES=('E0','SOURCE','TARGET_PCD')
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
U_COUNT=384
STEPS=444
_read,_write,_source_truth=common._read,common._write,common._truth


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _validate_rows(a,n,fields,states,tau):
    if set(a)!=fields or not common._ids(a['ids'],n):raise ValueError('Complete public records with exact fields and unique IDs required')
    if (a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))):
        raise ValueError('All methods must preserve the actual same-DURA original F0 identity')
    if 'baseline_closed' in fields and (a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):raise ValueError('Original same-input F0 baseline cannot be replaced')
    z=a['signal_features']
    if (z.shape!=(n,3,160) or z.dtype!=np.float32 or not np.isfinite(z).all()
            or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1.,rtol=0,atol=2e-6)
            or any(a[k].shape!=(n,6) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):
        raise ValueError('Three ordered unit160 representations and six finite readout columns required')
    raw=[]
    for i,name in enumerate(MODEL_NAMES):
        raw.extend(evaluation._raw(states[name],z[:,i],a['closed']).T)
    if not np.allclose(a['raw_scores'],np.column_stack(raw),rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('Every model/feature/readout and original-class threshold route must reconstruct')
    return a


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],all_three_DURA_conditions_complete=True,classification_unchanged=True,
        source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_conditions_through_DURA=True,source_E0_controls_reused_exactly=True,
        source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,source_density_models_fitted=2,new_threshold_sets_fitted=4,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,
        private_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,U_truth_reads=0,original_U_s_reads=0,
        source_IQ_reads=sum(COUNTS.values()),target_Q_IQ_reads=2*Q_COUNT,target_U_reads=0,U_ID_metadata_reads=U_COUNT,training_target_U_count=U_COUNT,
        new_neural_updates=0,inherited_new_SOURCE_updates=STEPS,inherited_new_TARGET_PCD_updates=STEPS,real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,
        real_signal_forward_rows=2*sum(COUNTS.values())+6*Q_COUNT,real_signal_forward_rows_by_arm=dict(E0=2*Q_COUNT,SOURCE=sum(COUNTS.values())+2*Q_COUNT,TARGET_PCD=sum(COUNTS.values())+2*Q_COUNT),
        synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,synthetic_signal_forward_rows=6,
        formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Both trained arms and every six-column three-condition prediction must complete before truth')
    c=evaluation.validate_config(m['config']);pm,old_state,old_tau,old_cal=evaluation._reference(c)
    if (c['run_id']!=m['run_id'] or c['code_commit']!=m['code_commit'] or m['reference_prediction_identity']!=pm
            or m['parent_F0_identity']!=pm['parent_F0_identity'] or m['dura_identity']!=pm['dura_identity'] or m['loaded_dura_identity']!=pm['loaded_dura_identity'] or m['parent_calibration_identity']!=old_cal):raise ValueError('Actual run and frozen r03/F0/DURA reference identity required')
    fit=_read(root/'models/complete.json');cal=_read(root/'calibration/complete.json')
    if (any(fit.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm_order=list(MODEL_NAMES),neural_states_frozen=True,source_fit_inputs_only=True,
                source_V_fit=False,novel_fit=False,target_Q_fit=False,source_density_models_fitted=2,normalizer_fit_role='L',background_fit_role='OE',TARGET_PCD_has_target_U_ancestor=True,E0_state_reused_exactly=True).items())
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,group_count=42,
                known_only=True,unknown_inputs_used=False,target_reads=0,all_shared_source_thresholds=True,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,TARGET_PCD_neural_ancestor_U_count=U_COUNT,
                hard_names=list(evaluation.risk.HARD_NAMES),diagnostic_names=list(evaluation.risk.DIAGNOSTIC_NAMES)).items())):
        raise ValueError('Neural models/source densities must freeze before V; known labels only calibrate shared thresholds')
    identities=_identities(c,m,fit,pm)
    states={name:_arrays(root/'models'/(name+'.npz')) for name in MODEL_NAMES}
    for state in states.values():
        density.validate_state(state,'density_ratio')
        if state['schema'].item()!=density.SCHEMA:raise ValueError('Only fixed D160 ninety-known/source-OE density states allowed')
    if not evaluation._same(states['E0'],old_state):raise ValueError('E0 density must be the original exact best state')
    saved=_arrays(root/'thresholds.npz');tau=saved.get('conditional')
    if set(saved)!={'conditional'} or tau is None or tau.dtype!=np.float64 or tau.shape!=(6,6) or not np.isfinite(tau).all() or not np.array_equal(tau[:2],old_tau) or not np.array_equal(tau,np.asarray(cal['conditional_thresholds'],np.float64)):
        raise ValueError('All six class-threshold vectors must freeze; the original two remain exact')
    if set(cal.get('methods',{}))!=set(METHODS):raise ValueError('Complete six-method known risk reports required')
    for j,name in enumerate(METHODS):
        if not np.array_equal(tau[j],np.asarray(cal['methods'][name]['conditional_thresholds'],np.float64)):raise ValueError('Calibrated six-class vectors differ from deployed thresholds')
        if j<2 and cal['methods'][name]!=old_cal['methods'][evaluation.parent.METHODS[(1,3)[j]]]:raise ValueError('Source DA0 risk reports must remain exact, not newly calibrated')
    declared,U_ids=evaluation._training_metadata(c,pm)
    if declared!=identities:raise ValueError('Actual training metadata identity differs from prediction feature axis')
    rows={'source_DURA':{}};source_ids=set()
    for role in ('known','novel'):
        a=evaluation._source_rows(root,'source_DURA',role);_validate_rows(a,COUNTS['V' if role=='known' else 'novel'],evaluation.KNOWN_FIELDS if role=='known' else evaluation.NOVEL_FIELDS,states,tau)
        old=evaluation.parent._rows(c['reference_config']['output_root'],role)
        for k,v in old.items():
            actual=a[k][:,:2] if k in ('raw_scores','scores') else a[k][:,0] if k=='signal_features' else a[k]
            expected=v[:,[1,3]] if k in ('raw_scores','scores') else v
            if not np.array_equal(actual,expected):raise ValueError('Source E0 representation, ratio controls and original identity must remain exact')
        if source_ids&set(a['ids'].tolist()):raise ValueError('Source known and novel roles overlap')
        source_ids.update(a['ids'].tolist());rows['source_DURA'][role]=a
    if source_ids&set(U_ids.tolist()):raise ValueError('Source evaluation and target fit-U physical IDs overlap')
    for condition in CONDITIONS[1:]:
        a=evaluation._target_rows(root,condition);_validate_rows(a,Q_COUNT,evaluation.TARGET_FIELDS,states,tau)
        if (source_ids|set(U_ids.tolist()))&set(a['ids'].tolist()):raise ValueError('Future query cannot overlap source or fit-U physical records')
        old=evaluation.parent._target_rows(c['reference_config']['output_root'],condition)
        for k,v in old.items():
            actual=a[k][:,:2] if k in ('raw_scores','scores') else a[k][:,0] if k=='signal_features' else a[k]
            expected=v[:,[1,3]] if k in ('raw_scores','scores') else v
            if not np.array_equal(actual,expected):raise ValueError('Actual shared-G original F0 and E0 controls must be exact')
        rows[condition]=a
    return m,cal,rows


def _identities(c,m,fit,pm):
    # Completion metadata only: independent scoring never loads neural weights.
    identities=fit['identities']
    if set(identities)!=set(MODEL_NAMES) or identities['E0']!=pm['model_identity'] or m['model_identities']!=identities:
        raise ValueError('Three exact model identities must map to the stored feature axis')
    for arm in MODEL_NAMES[1:]:evaluation._identity(identities[arm],arm,c,pm)
    return identities


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
    for reference in (1,3):
        by_condition={}
        for condition in CONDITIONS:
            if condition=='source_DURA':
                k,n=rows[condition]['known'],rows[condition]['novel'];km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],nt['scenario']));groups={g:_pair(k,n,5,reference,mask,nm[g]) for g,mask in km.items()};uy=nt['tx'];unknown_roster=common.UNKNOWN_TX
            else:
                t=truths[condition.split('_')[0]];y=t['y'];mask=y<6;a=rows[condition];k=dict(y=y[mask],closed=a['closed'][mask],scores=a['scores'][mask]);n=dict(scores=a['scores'][~mask]);scenes=('clean',) if condition=='clean_DURA' else common.SCENES
                groups={g:_pair(k,n,5,reference,t['scenario'][mask]==g,t['scenario'][~mask]==g) for g in scenes};uy=y[~mask];unknown_roster=(6,7,8)
            by_condition[condition]=dict(counts=_pair(k,n,5,reference),groups=groups,delta=previous._delta(main[condition]['overall'],methods[METHODS[reference]]['paths'][condition]['overall']),
                by_known_TX={str(c):_pair(k,n,5,reference,k['y']==c) for c in range(6)},by_unknown_TX={str(c):_pair(k,n,5,reference,nm=uy==c) for c in unknown_roster})
        paired[METHODS[reference]]=by_condition
    gains={reference:paired[reference]['star_DURA']['delta']['A_K'] for reference in ('E0_predicted_ratio','SOURCE_predicted_ratio')}
    positive=all(value is not None and value>1e-12 for value in gains.values())
    rejection_kept=all(paired[name]['star_DURA']['delta']['R_U'] is not None and paired[name]['star_DURA']['delta']['R_U']>=-1e-12 for name in ('E0_predicted_ratio','SOURCE_predicted_ratio'))
    source_known=main['source_DURA']['known_12_groups_passed'] and bool(cal['methods'][PRIMARY]['constraints_passed'])
    protection=all(main[c]['goals']['passed'] for c in CONDITIONS[1:]);success=bool(source_known and positive and rejection_kept and protection)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,primary_summary={c:main[c]['overall'] for c in CONDITIONS},
        paired_primary_vs_E0_predicted_ratio=paired['E0_predicted_ratio'],paired_primary_vs_SOURCE_predicted_ratio=paired['SOURCE_predicted_ratio'],star_A_K_gain_vs_E0=gains['E0_predicted_ratio'],star_A_K_gain_vs_SOURCE=gains['SOURCE_predicted_ratio'],star_A_K_gain_over_both_controls=bool(positive),star_R_U_not_lower_than_both_controls=bool(rejection_kept),source_known_12_passed=bool(source_known),target_protection_passed=bool(protection),adaptation_success=success,development_gate_passed=success,eligible_primary=success,
        next_step='DEVELOPMENT_ADAPTATION_PASSED_FOR_SEPARATE_REGISTRATION_PREREGISTRATION' if success else 'NO_PROMOTION',truth_last=True,all_three_predictions_before_truth=True,U_truth_reads=0,source_truth_reads=1,target_truth_reads=2,
        classification_unchanged=True,source_E0_controls_reused_exactly=True,target_fit_input='Unlabelled star U balanced prototype objective; clean and Q are test only',source_unknown_diagnostic_only=True,
        SOURCE_weights_source_only=True,TARGET_PCD_weights_source_only=False,TARGET_PCD_weights_use_target_U=True,
        new_projection_optimizer_updates_total=2*STEPS,projection_updates_per_arm=STEPS,new_body_optimizer_updates_total=2*STEPS,Sinc_optimizer_steps=0,new_source_density_models=2,new_source_threshold_sets=4,prototypes_deployed=False,original_F0_updates=0,G_updates=0,threshold_values=36,threshold_bytes=288,
        costs={k:m[k] for k in ('source_IQ_reads','target_Q_IQ_reads','target_U_reads','U_ID_metadata_reads','training_target_U_count','real_F_forward_rows','G_forward_rows','real_signal_forward_rows','real_signal_forward_rows_by_arm','synthetic_F_forward_rows','G_synthetic_forward_rows','synthetic_signal_forward_rows')},per_record_source_rows_deployed=0,
        training_costs=m['training_costs'],combined_real_signal_forward_rows=m['combined_real_signal_forward_rows'],combined_G_forward_rows=m['combined_G_forward_rows'],combined_real_F_forward_rows=m['combined_real_F_forward_rows'],
        source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,target_REG=False,registration_success=None,A_N=None,H_OW=None,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        inherited_cost_scope='Original F0 E50/11100, r03 signal 2220 and DURA training remain inherited; SOURCE/TARGET_PCD independently add444 body/projection updates each with Sinc frozen, two new source densities and four source-known threshold sets',
        original_F0_parameters=232993,open_signal_parameters_per_arm=334721,existing_restorer_parameters=266951,primary_deployed_neural_parameters=834665,scope='TARGET_PCD body/projection and derived density have target-U ancestry; SOURCE is source-only. This is a limited prototype candidate, not a full SwAV/UniOT reproduction. Historical source/Q exposure, one development RX/seed, no U true TX/K, no registration or full Phase3 completion')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_E0_predicted_ratio','paired_primary_vs_SOURCE_predicted_ratio')});print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
