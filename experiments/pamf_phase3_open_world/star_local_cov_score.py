"""Independent truth-last scoring of source-fitted local covariance evidence."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_class_density_score as previous

RUN_ID='20261031-phase3-local-cov-s2026092701-r01'
SCHEMA='pamf_phase3_local_cov_scores_v1'
STATUS='LOCAL_COV_SCORES_COMPLETE'
METHODS=('pooled_ratio','predicted_ratio','local_pooled_ratio','local_predicted_ratio')
PRIMARY=METHODS[3]
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
CONDITIONS=previous.CONDITIONS
common=previous.common
_read,_write=previous._read,previous._write
_source_truth,_target_truth=previous._source_truth,previous._target_truth
_target,_paired=previous._target,previous._paired


def _prediction(root):
    from . import star_local_cov_run as evaluation
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,
        classification_path='P1',identity_baseline_path='same_DURA_output_P1',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        source_only=True,private_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,target_U_reads=0,original_U_s_reads=0,formal_target_accessed=False,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_conditions_complete=True,all_conditions_through_DURA=True,
        model_parameters_buffers_unchanged=True,source_states_unchanged=True,controls_reused_exactly=True,identity_predictions_unchanged=True,source_bank_deployed=False,
        per_record_source_rows_deployed=0,neural_updates=0,density_fit_updates=0,local_covariance_fit_modes=90,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('All frozen three-condition local covariance predictions required before truth')
    c=evaluation.validate_config(m['config'])
    if c['code_commit']!=m['code_commit']:raise ValueError('Prediction/config code identity differs')
    pm,state,old_tau,parent_cal=evaluation._reference(c)
    if (m['reference_prediction_identity']!=pm or m['model_identity']!=pm['model_identity']
            or m['parent_F0_identity']!=pm['parent_F0_identity'] or m['dura_identity']!=pm['dura_identity']):
        raise ValueError('Same frozen r03 signal, original F0 and DURA identities required')
    stored=_arrays(root/'models/density.npz')
    if not evaluation._same(stored,state):raise ValueError('Original density must be reused exactly without refit')
    local=_arrays(root/'models/local.npz');evaluation.core.validate_local(local,state)
    fit=_read(root/'models/complete.json')
    fit_expected=dict(status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],models_frozen=True,original_density_refitted=False,
        local_covariance_modes=90,source_L_count=COUNTS['L'],source_V_fit=False,novel_fit=False,target_fit=False)
    if any(fit.get(k)!=v for k,v in fit_expected.items()):raise ValueError('L-only local covariance must freeze before V and all novel/query predictions')
    thresholds=_arrays(root/'thresholds.npz')
    if set(thresholds)!={'conditional'}:raise ValueError('Four conditional threshold sets required')
    tau=thresholds['conditional'];cal=_read(root/'calibration/complete.json')
    cal_expected=dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),
        primary_method=PRIMARY,known_only=True,unknown_inputs_used=False,target_reads=0,group_count=42,conditional_thresholds=tau.tolist())
    if (tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or not np.array_equal(tau[:2],old_tau)
            or any(cal.get(k)!=v for k,v in cal_expected.items()) or set(cal.get('methods',{}))!=set(METHODS)):
        raise ValueError('Two exact parent thresholds and two V-only known12 calibrations required')
    for j,name in enumerate(METHODS):
        report=cal['methods'][name]
        if (set(report.get('groups',{}))!=set(common.GROUP_NAMES) or report.get('hard_names')!=list(evaluation.risk.HARD_NAMES)
                or report.get('diagnostic_names')!=list(evaluation.risk.DIAGNOSTIC_NAMES) or report.get('conditional_thresholds')!=tau[j].tolist()):
            raise ValueError('Each method requires known12 and thirty cross diagnostics')
    for j,parent_index in enumerate((1,3)):
        if cal['methods'][METHODS[j]]!=parent_cal['methods'][pm['method_names'][parent_index]]:raise ValueError('Original calibration controls must be exact')
    parent_root=c['reference_config']['output_root'];rows={}
    for role in ('known','novel'):
        a=evaluation._rows(root,role);old=evaluation._rows(parent_root,role)
        _exact_controls(a,old);_reconstruct(a,state,local,tau,evaluation.core);rows[role]=a
    for condition in CONDITIONS:
        a=evaluation._target_rows(root,condition);old=evaluation._target_rows(parent_root,condition)
        _exact_controls(a,old)
        if (not common._ids(a['ids'],Q_COUNT) or a['default_logits'].shape!=(Q_COUNT,6) or a['default_logits'].dtype!=np.float32
                or not np.isfinite(a['default_logits']).all() or a['closed'].shape!=(Q_COUNT,) or a['closed'].dtype!=np.int64
                or not np.array_equal(a['closed'],a['default_logits'].argmax(1)) or not np.array_equal(a['baseline_closed'],a['closed'])):
            raise ValueError('Original F0 identity/baseline must share the same DURA output')
        _reconstruct(a,state,local,tau,evaluation.core);rows[condition]=a
    source=set(rows['known']['ids'].tolist())|set(rows['novel']['ids'].tolist())
    if set(rows['known']['ids'].tolist())&set(rows['novel']['ids'].tolist()) or any(source&set(rows[c]['ids'].tolist()) for c in CONDITIONS):
        raise ValueError('Source known/novel and target physical roles must be disjoint')
    return m,cal,rows


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _exact_controls(a,parent):
    if set(a)!=set(parent):raise ValueError('Public prediction fields must remain unchanged')
    for key,value in parent.items():
        actual=a[key][:,:2] if key in ('raw_scores','scores') else a[key]
        expected=value[:,(1,3)] if key in ('raw_scores','scores') else value
        if not np.array_equal(actual,expected):raise ValueError('Exact parent ratio controls, features, metadata and default identity required')


def _reconstruct(a,state,local,tau,core):
    n=len(a['ids']);z=a['signal_features']
    if (z.shape!=(n,160) or z.dtype!=np.float32 or not np.isfinite(z).all()
            or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)
            or any(a[k].shape!=(n,4) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):
        raise ValueError('Four finite FP64 score columns and the fixed P1 unit160 representation required')
    raw=core.raw_scores(state,local,z,a['closed'])
    if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):
        raise ValueError('All local covariance raw scores and predicted-class threshold margins must reconstruct')


def _source(k,n,t,cal,index):
    # The prior helper indexes calibration by its own fixed method names.
    mapped=dict(methods={old:cal['methods'][new] for old,new in zip(previous.METHODS,METHODS)})
    return previous._source(k,n,t,mapped,index)


def _nonregression(main,control):
    return all(main[k] is not None and control[k] is not None and main[k]>=control[k]-1e-12 for k in ('A_K','R_U'))


def _comparison(rows,truths,nt,methods,reference):
    primary=3;sk,sn=rows['known'],rows['novel'];km=dict(common._masks(sk['rx_i'],sk['scenario']));nm=dict(common._masks(sn['rx_i'],nt['scenario']))
    result=dict(primary=PRIMARY,reference=METHODS[reference],counts={'source':_paired(sk,sn,primary,reference)},
        groups=dict(source={g:_paired(sk,sn,primary,reference,mask,nm[g]) for g,mask in km.items()}))
    for c in CONDITIONS:
        a=rows[c];y=truths[c]['y'];known=y<6;k=dict(y=y[known],closed=a['closed'][known],scores=a['scores'][known]);n=dict(scores=a['scores'][~known])
        result['counts'][c]=_paired(k,n,primary,reference)
        scenes=('clean',) if c=='clean_DURA' else common.SCENES
        result['groups'][c]={s:_paired(k,n,primary,reference,truths[c]['scenario'][known]==s,truths[c]['scenario'][~known]==s) for s in scenes}
    result['delta']={c:{key:None if methods[PRIMARY][c]['overall'][key] is None or methods[METHODS[reference]][c]['overall'][key] is None
        else methods[PRIMARY][c]['overall'][key]-methods[METHODS[reference]][c]['overall'][key]
        for key in ('A_K','C_K','R_U','H_score','auroc_known_macro','oscr_auc_macro','original_closed_minus_new_open')} for c in ('source',)+CONDITIONS}
    return result


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent score output required')
    m,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['novel'])
    truths={name:_target_truth(path,rows[name]['ids']) for name,path in zip(CONDITIONS,(clean_truth,star_truth))}
    for c,t in truths.items():
        scenes=('clean',) if c=='clean_DURA' else common.SCENES
        if not np.isin(t['y'],range(9)).all() or not np.all(t['receiver']==26) or not np.isin(t['scenario'],scenes).all():
            raise ValueError('Only the fixed development RX and condition-specific scorer roster are allowed')
    methods={name:dict(source=_source(rows['known'],rows['novel'],nt,cal,j),**{c:_target(rows[c],truths[c],c,j) for c in CONDITIONS},diagnostic_only=name!=PRIMARY) for j,name in enumerate(METHODS)}
    main=methods[PRIMARY];source_known=main['source']['known_12_groups_passed'] and main['source']['source_calibration_passed']
    no_regression=_nonregression(main['star_DURA']['overall'],methods['predicted_ratio']['star_DURA']['overall'])
    passed=bool(source_known and all(main[c]['goals']['passed'] for c in CONDITIONS) and no_regression)
    pairs={label:_comparison(rows,truths,nt,methods,index) for label,index in (('primary_vs_predicted_ratio',1),('primary_vs_pooled_ratio',0),('primary_vs_local_pooled_ratio',2))}
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,methods=methods,
        primary_summary={c:main[c]['overall'] for c in ('source',)+CONDITIONS},paired=pairs,source_known_12_passed=bool(source_known),
        source_unknown_90_diagnostic_passed=main['source']['source_unknown_90_diagnostic_passed'],star_not_worse_than_predicted_control=bool(no_regression),
        development_gate_passed=passed,eligible_primary=passed,next_step='DEVELOPMENT_PASSED_FOR_SEPARATE_UDA_PREREGISTRATION' if passed else 'NO_PROMOTION',
        truth_last=True,all_three_predictions_before_truth=True,source_truth_reads=1,target_truth_reads=2,classification_path='P1',identity_baseline_path='same_DURA_output_P1',
        clean_also_uses_DURA=True,formal_target_accessed=False,target_U_reads=0,target_DA=False,target_REG=False,neural_updates=0,density_fit_updates=0,
        local_covariance_fit_modes=m['local_covariance_fit_modes'],source_V_threshold_sets_fitted=2,source_bank_deployed=False,per_record_source_rows_deployed=0,
        A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False,full_goal_passed=False,
        costs={k:m[k] for k in ('real_signal_forward_rows','real_F_forward_rows','G_forward_rows','synthetic_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows')},
        inherited_training_cost_scope='Frozen original F0/r03 signal/DURA training and original density fit inherited; this run fits 90 local covariance modes on source L and two V-only threshold sets',
        scope='Source seen RX and historical novel; one historical development target RX; no anonymous registration, formal or full open-world success')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser()
    for name in ('prediction-root','source-novel-truth','clean-truth','star-truth','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
