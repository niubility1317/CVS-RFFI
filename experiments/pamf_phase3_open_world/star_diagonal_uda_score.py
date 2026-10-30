"""Independent future truth scoring after both U-only affine states freeze."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_class_density_score as previous

RUN_ID='20261031-phase3-diagonal-uda-s2026092701-r01'
SCHEMA='pamf_phase3_diagonal_uda_scores_v1'
STATUS='DIAGONAL_UDA_SCORES_COMPLETE'
METHODS=('pooled_ratio','predicted_ratio','shift_pooled_ratio','shift_predicted_ratio','diagonal_pooled_ratio','diagonal_predicted_ratio')
PRIMARY=METHODS[-1]
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
CONDITIONS=previous.CONDITIONS
common=previous.common
_read,_write=previous._read,previous._write
_source_truth,_target_truth=previous._source_truth,previous._target_truth
_target,_paired=previous._target,previous._paired


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _exact_controls(a,parent):
    if set(a)!=set(parent):raise ValueError('Inherited public prediction fields must be unchanged')
    for key,value in parent.items():
        current=a[key][:,:2] if key in ('raw_scores','scores') else a[key]
        expected=value[:,[1,3]] if key in ('raw_scores','scores') else value
        if not np.array_equal(current,expected):raise ValueError('Original ratio controls, same-DURA identity and features must be exact')


def _prediction(root):
    from . import star_diagonal_uda_run as evaluation
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=384,source_models_frozen_before_U=True,
        adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,all_thresholds_reused_exactly=True,
        source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,
        classification_unchanged=True,private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,
        new_neural_updates=0,density_fit_updates=0,threshold_fit_updates=0,target_U_reads=384,U_features_reused=True,
        source_L_reads=0,source_OE_reads=0,real_signal_forward_rows=0,real_F_forward_rows=0,G_forward_rows=0,
        source_only=False,original_neural_models_source_only=True,target_DA=True,target_REG=False,formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Complete frozen U-only affine three-DURA predictions required before truth')
    c=evaluation.validate_config(m['config'])
    if c['code_commit']!=m['code_commit']:raise ValueError('Prediction and actual config code differ')
    pm,state,old_tau,parent_cal=evaluation._reference(c)
    if (m['reference_prediction_identity']!=pm or m['model_identity']!=pm['model_identity']
            or m['parent_F0_identity']!=pm['parent_F0_identity'] or m['dura_identity']!=pm['dura_identity']):
        raise ValueError('Original source F0/r03/G identities must remain unchanged')
    if not evaluation._same(_arrays(root/'models/density.npz'),state):raise ValueError('Source Gaussian changed during U adaptation')
    fit=_read(root/'models/complete.json')
    if any(fit.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.SOURCE_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],
            models_frozen=True,models_refitted=False,thresholds_refitted=False,source_models_frozen_before_U=True).items()):
        raise ValueError('Original source state must freeze before U access')
    stored=_arrays(root/'thresholds.npz');tau=stored.get('conditional')
    if (set(stored)!={'conditional'} or tau.dtype!=np.float64 or tau.shape!=(6,6)
            or not np.array_equal(tau,np.tile(old_tau,(3,1))) or _read(root/'calibration/complete.json')!=parent_cal):
        raise ValueError('Only repeated original pooled/predicted thresholds and source calibration are allowed')
    cache=evaluation._cached_U_header(c,pm)
    U=_arrays(root/'adaptation/U_features.npz');oldU=evaluation._U_features(c,cache)
    if not evaluation._same(U,oldU):raise ValueError('Only unchanged public pre-adaptation star-U features may fit')
    done=_read(root/'adaptation/complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.ADAPTER_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],
        U_count=384,U_manifest=cache['U_manifest'],adapter_names=['SHIFT','DIAG'],fit_input='cached_star_U_signal_features_only',
        iterations_per_adapter=20,ridge=160.,group_prior=[.5,.5],query_fit=False,true_U_labels_used=False,true_unknown_count_used=False,
        source_density_refitted=False,thresholds_refitted=False,adapters_frozen=True,Q_reads=0,source_prediction_reads=0,
        U_bank_deployed=False,private_U_features_ref='U_features.npz')
    if any(done.get(k)!=v for k,v in required.items()):raise ValueError('Both adapters must freeze using only the fixed public U pool')
    diag=_read(root/'adaptation/fit_diagnostics.json');adapters={}
    if (diag.get('iterations')!=20 or diag.get('ridge')!=160. or diag.get('U_count')!=384
            or set(diag.get('arms',{}))!={'SHIFT','DIAG'}):raise ValueError('Both complete fixed twenty-step U fits required')
    for arm in ('SHIFT','DIAG'):
        adapter=_arrays(root/'adaptation'/(arm+'.npz'));evaluation.core.validate_adapter(adapter)
        if adapter['kind'].item()!=arm:raise ValueError('Adapter file and affine kind differ')
        report=diag['arms'][arm];trace=report.get('trace',[])
        if (report.get('group_prior')!=[.5,.5] or len(trace)!=21
                or [v.get('iteration') for v in trace]!=list(range(21))
                or any(set(v)!={'iteration','objective','scale_min','scale_max','offset_norm','known_posterior_mass'} for v in trace)):
            raise ValueError('Complete finite identity-start U-only EM trace required')
        values=np.asarray([[v[k] for k in ('objective','scale_min','scale_max','offset_norm','known_posterior_mass')] for v in trace],np.float64)
        if (not np.isfinite(values).all() or np.any(np.diff(values[:,0]) < -1e-7)
                or np.any(values[:,1]<=0) or np.any(values[:,2]<values[:,1]) or np.any(values[:,3]<0)
                or np.any(values[:,4]<-1e-12) or np.any(values[:,4]>1+1e-12)
                or not np.allclose(values[-1,1:4],[adapter['inverse_scale'].min(),adapter['inverse_scale'].max(),np.linalg.norm(adapter['offset'])],rtol=1e-12,atol=1e-12)):
            raise ValueError('Affine objective, positive scales or frozen terminal state differs')
        adapters[arm]=adapter
    rows={};seen=set(U['ids'].tolist())
    for role in ('known','novel'):
        a=evaluation._rows(root,role);old=evaluation.parent._rows(c['reference_config']['output_root'],role)
        _exact_controls(a,old)
        if (any(not np.array_equal(a[k],np.tile(old[k][:,[1,3]],(1,3))) for k in ('raw_scores','scores'))
                or seen & set(a['ids'].tolist())):raise ValueError('Source receivers require identity transform and disjoint physical roles')
        seen.update(a['ids'].tolist());rows[role]=a
    for condition in CONDITIONS:
        a=evaluation._target_rows(root,condition);old=evaluation.parent._target_rows(c['reference_config']['output_root'],condition)
        _exact_controls(a,old)
        raw=np.concatenate([old['raw_scores'][:,[1,3]]]+[evaluation.core.raw_scores(state,a['signal_features'],a['closed'],adapters[arm]) for arm in ('SHIFT','DIAG')],axis=1)
        if (not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9)
                or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T)
                or seen & set(a['ids'].tolist())):raise ValueError('Future Q must use only the same frozen U adapter, with no fit or physical overlap')
        rows[condition]=a
    cal=dict(methods={name:parent_cal['methods'][pm['method_names'][(1,3)[j%2]]] for j,name in enumerate(METHODS)})
    return m,cal,rows,diag


def _source(k,n,t,cal,index):
    km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],t['scenario']))
    allk=np.ones(len(k['ids']),bool);alln=np.ones(len(n['ids']),bool)
    def metric(ki,ni,kr=tuple(range(6)),ur=common.UNKNOWN_TX):
        return previous._metrics(k['y'][ki],k['closed'][ki],k['scores'][ki,index],t['tx'][ni],n['scores'][ni,index],k['baseline_closed'][ki],kr,ur)
    groups={name:metric(mask,nm[name]) for name,mask in km.items()};overall=groups['overall'];hard=previous.evaluation.risk.HARD_NAMES
    return dict(overall=overall,groups=groups,by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},by_scene={s:groups['scene/'+s] for s in common.SCENES},
        by_day={str(d):metric(k['day_i']==d,n['day_i']==d) for d in (1,2,3)},by_TX={str(c):metric(k['y']==c,alln,(c,)) for c in range(6)}|{c:metric(allk,t['tx']==c,ur=(c,)) for c in common.UNKNOWN_TX},
        confidence_intervals=previous._confidence(overall),cross_guard_violations_or_missing=previous._cross_report(groups),
        known_12_groups_passed=all(previous._known_pass(groups[g]) for g in hard),known_42_groups_passed=all(previous._known_pass(r) for r in groups.values()),
        source_calibration_passed=bool(cal['methods'][METHODS[index]]['constraints_passed']),source_RX_seen=True,source_novel_historically_exposed=True,
        source_unknown_90_diagnostic_passed=bool(overall['coverage']['unknown_complete'] and overall['R_U'] is not None and overall['R_U']>=.90-1e-12))


def _comparison(rows,truths,nt,methods,reference):
    primary=5;k,n=rows['known'],rows['novel'];km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],nt['scenario']))
    result=dict(primary=PRIMARY,reference=METHODS[reference],counts={'source':_paired(k,n,primary,reference)},
        groups=dict(source={g:_paired(k,n,primary,reference,mask,nm[g]) for g,mask in km.items()}))
    for c in CONDITIONS:
        a=rows[c];y=truths[c]['y'];known=y<6
        k=dict(y=y[known],closed=a['closed'][known],scores=a['scores'][known]);n=dict(scores=a['scores'][~known])
        result['counts'][c]=_paired(k,n,primary,reference)
        scenes=('clean',) if c=='clean_DURA' else common.SCENES
        result['groups'][c]={s:_paired(k,n,primary,reference,truths[c]['scenario'][known]==s,truths[c]['scenario'][~known]==s) for s in scenes}
    result['delta']={c:{key:None if methods[PRIMARY][c]['overall'][key] is None or methods[METHODS[reference]][c]['overall'][key] is None
        else methods[PRIMARY][c]['overall'][key]-methods[METHODS[reference]][c]['overall'][key]
        for key in ('A_K','C_K','R_U','H_score','auroc_known_macro','oscr_auc_macro','original_closed_minus_new_open')} for c in ('source',)+CONDITIONS}
    return result


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent scoring output required')
    m,cal,rows,diag=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['novel'])
    truths={name:_target_truth(path,rows[name]['ids']) for name,path in zip(CONDITIONS,(clean_truth,star_truth))}
    for c,t in truths.items():
        scenes=('clean',) if c=='clean_DURA' else common.SCENES
        if not np.isin(t['y'],range(9)).all() or not np.all(t['receiver']==26) or not np.isin(t['scenario'],scenes).all():
            raise ValueError('Only the fixed development RX/condition scoring roster is allowed')
    methods={name:dict(source=_source(rows['known'],rows['novel'],nt,cal,j),**{c:_target(rows[c],truths[c],c,j) for c in CONDITIONS},diagnostic_only=name!=PRIMARY) for j,name in enumerate(METHODS)}
    main=methods[PRIMARY];old=methods['predicted_ratio']['star_DURA']['overall'];now=main['star_DURA']['overall']
    gain=now['A_K'] is not None and old['A_K'] is not None and now['A_K']>old['A_K']+1e-12
    reject_kept=now['R_U'] is not None and old['R_U'] is not None and now['R_U']>=old['R_U']-1e-12
    source_known=main['source']['known_12_groups_passed'] and main['source']['source_calibration_passed']
    passed=bool(source_known and gain and reject_kept and all(main[c]['goals']['passed'] for c in CONDITIONS))
    pairs={label:_comparison(rows,truths,nt,methods,index) for label,index in (('primary_vs_predicted_ratio',1),('primary_vs_shift_predicted_ratio',3),('primary_vs_diagonal_pooled_ratio',4))}
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,methods=methods,
        primary_summary={c:main[c]['overall'] for c in ('source',)+CONDITIONS},paired=pairs,source_known_12_passed=bool(source_known),
        star_known_accuracy_gain=bool(gain),star_rejection_not_worse_than_predicted_control=bool(reject_kept),development_gate_passed=passed,eligible_primary=passed,
        next_step='DEVELOPMENT_PASSED_FOR_SEPARATE_REGISTRATION_PREREGISTRATION' if passed else 'NO_PROMOTION',
        truth_last=True,all_three_predictions_before_truth=True,U_truth_reads=0,source_truth_reads=1,target_truth_reads=2,
        formal_target_accessed=False,target_U_reads=384,target_DA=True,target_REG=False,neural_updates=0,density_fit_updates=0,threshold_fit_updates=0,
        adapter_iterations_per_arm=20,adapter_ridge=160.,adapter_parameter_counts=dict(SHIFT=160,DIAG=320),fit_diagnostics=diag,
        all_thresholds_reused_exactly=True,original_neural_models_source_only=True,source_only=False,U_bank_deployed=False,source_bank_deployed=False,
        costs={k:m[k] for k in ('real_signal_forward_rows','real_F_forward_rows','G_forward_rows','synthetic_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows')},
        A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False,full_goal_passed=False,
        scope='One historical development RX/seed, source seen receivers, star-U adapter shared by paired future clean/star; not formalblind or registration proof')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser()
    for name in ('prediction-root','source-novel-truth','clean-truth','star-truth','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
