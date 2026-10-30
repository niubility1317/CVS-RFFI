"""Independent class-aware U-only density adaptation; private truths are last."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_class_shift_run as evaluation
from . import star_class_shift_core as core
from . import star_mixture_uda_core as shared_core
from . import star_mixture_uda_score as previous_uda
from . import star_dura_best_score as previous
from . import star_consensus_score as common
from .star_crossset_metric_score import paired_counts
from .scoring import _truth as _target_truth

RUN_ID='20261031-phase3-class-shift-s2026092701-r01'
SCHEMA='pamf_phase3_class_shift_scores_v1'
STATUS='CLASS_SHIFT_SCORES_COMPLETE'
METHODS=('DA0_known','DA0_ratio','SHARED_known','SHARED_ratio','CLASS_known','CLASS_ratio')
PRIMARY=METHODS[5]
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
U_COUNT=384
_read,_write,_source_truth=common._read,common._write,common._truth
_arrays,_units=previous_uda._arrays,previous_uda._units


def _trace(d,prior,adapter,class_specific):
    expected=dict(iterations=20,U_count=U_COUNT,group_prior=[.5,.5],source_rank=len(prior['basis']),
        Q_inputs_used=False,true_U_labels_used=False,true_unknown_count_used=False,default_logits_used=False,source_state_unchanged_by_fit=True,fixed_iteration_checkpoint=True)
    if class_specific:expected.update(source_prior_frozen_before_U=True,U_only=True)
    if any(d.get(k)!=v for k,v in expected.items()) or not np.isclose(d.get('source_radius',np.nan),float(prior['radius']),rtol=1e-12,atol=1e-12):raise ValueError('Source-bounded U-only fixed twenty-step adapter metadata required')
    norm_key='shift_rms' if class_specific else 'shift_norm'
    keys={'iteration','objective','coefficient_norm',norm_key,'known_posterior_mass','constraint_multiplier'}
    if class_specific:keys.add('class_shift_max')
    trace=d.get('trace',[])
    if len(trace)!=21 or [r.get('iteration') for r in trace]!=list(range(21)) or any(set(r)!=keys or not all(np.isfinite(r[k]) for k in keys) for r in trace):raise ValueError('Complete finite iteration0..20 adapter trace required')
    radius=float(prior['radius'])
    if (any(b['objective']<a['objective']-1e-7 for a,b in zip(trace,trace[1:]))
            or any(r['coefficient_norm']<0 or r['coefficient_norm']>radius+1e-10 or r[norm_key]<0 or r[norm_key]>radius+1e-10
                or not np.isclose(r[norm_key],r['coefficient_norm'],rtol=1e-12,atol=1e-12)
                or not -1e-12<=r['known_posterior_mass']<=1+1e-12 or r['constraint_multiplier']<0 for r in trace)):
        raise ValueError('Source RMS trust region and numerical objective monotonicity must hold')
    delta=adapter['delta'];norm=float(np.sqrt(np.sum(delta*delta)/6)) if class_specific else float(np.linalg.norm(delta))
    if (not np.isclose(trace[-1][norm_key],norm,rtol=1e-12,atol=1e-12)
            or not np.isclose(trace[-1]['coefficient_norm'],np.linalg.norm(adapter['coefficients']),rtol=1e-12,atol=1e-12)
            or any(abs(trace[0][k])>1e-12 for k in ('coefficient_norm',norm_key))):raise ValueError('Saved final adapter and zero initialization must agree with its trace')
    if class_specific and (any(r['class_shift_max']<r['shift_rms']-1e-12 or r['class_shift_max']>np.sqrt(6)*r['shift_rms']+1e-10 for r in trace)
            or not np.isclose(trace[-1]['class_shift_max'],np.linalg.norm(delta,axis=1).max(),rtol=1e-12,atol=1e-12)):
        raise ValueError('Per-class maximum must match saved offsets; the trust bound is RMS, not maximum')


def _adapters(root,m,c,pm,state,tau,cal):
    folder=root/'adaptation';prior=_arrays(folder/'source_prior.npz');core.validate_prior(prior)
    expected=core.build_prior(state)
    if set(prior)!=set(expected) or any(not np.allclose(prior[k],expected[k],rtol=1e-12,atol=1e-12) if prior[k].dtype.kind=='f' else not np.array_equal(prior[k],expected[k]) for k in prior):raise ValueError('Class-domain prior must derive only from the fixed source aggregates')
    adapter=_arrays(folder/'CLASS.npz');core.validate_adapter(adapter,prior);diag=_read(folder/'fit_diagnostics.json');_trace(diag,prior,adapter,True)
    old,shared_prior,shared_adapter,shared_diag=evaluation._shared(c['reference_adaptation_root'],pm,state,tau,cal)
    if m['reference_adaptation_identity']!=old or m['U_manifest']!=old['U_manifest']:raise ValueError('Exact completed mixture-UDA shared control and cached-U handle required')
    for path,expected in ((folder/'shared_prior.npz',shared_prior),(folder/'SHARED.npz',shared_adapter)):
        if not evaluation._same(_arrays(path),expected):raise ValueError('Inherited shared adapter/prior cannot refit or change')
    if _read(folder/'shared_diagnostics.json')!=shared_diag:raise ValueError('Complete inherited shared fit diagnostics must remain exact')
    shared_core.validate_prior(shared_prior);shared_core.validate_adapter(shared_adapter,shared_prior);_trace(shared_diag['arms']['DA_M'],shared_prior,shared_adapter,False)
    u=_arrays(folder/'U_features.npz');old_u=_arrays(Path(c['reference_adaptation_root'])/'adaptation/U_features.npz')
    if not evaluation._same(u,old_u) or set(u)!={'ids','default_logits','closed','signal_features'} or not common._ids(u['ids'],U_COUNT):raise ValueError('Only the same complete cached label-free U export may fit CLASS')
    _units(u['signal_features'],U_COUNT)
    if u['default_logits'].shape!=(U_COUNT,6) or u['default_logits'].dtype!=np.float32 or not np.isfinite(u['default_logits']).all() or u['closed'].dtype!=np.int64 or not np.array_equal(u['closed'],u['default_logits'].argmax(1)):raise ValueError('Original F0 outputs must remain public diagnostics only')
    complete=_read(folder/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.ADAPTER_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],U_count=U_COUNT,adapters_frozen=True,
        fit_input='cached_star_U_signal_features_only',source_prior_frozen_before_U=True,query_fit=False,true_U_labels_used=False,true_unknown_count_used=False,source_density_refitted=False,thresholds_refitted=False,
        Q_reads=0,source_prediction_reads=0,U_bank_deployed=False,fit_diagnostics_ref='fit_diagnostics.json',shared_diagnostics_ref='shared_diagnostics.json')
    if any(complete.get(k)!=v for k,v in required.items()) or complete['U_manifest']!=m['U_manifest']:raise ValueError('Both class/shared adapters must freeze before source/Q prediction')
    return dict(prior=prior,adapter=adapter,diagnostics=diag,shared_prior=shared_prior,shared_adapter=shared_adapter,shared_diagnostics=shared_diag),u


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,
        classification_path='parent_original_F0_same_DURA_input',identity_baseline_path='parent_original_F0_same_DURA_input',classification_unchanged=True,score_threshold=0.,
        source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,source_controls_reused_exactly=True,DA0_controls_reused_exactly=True,shared_controls_reused_exactly=True,
        all_thresholds_reused_exactly=True,source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,
        source_IQ_reads=0,source_L_reads=0,source_OE_reads=0,target_Q_IQ_reads=0,target_U_reads=U_COUNT,U_feature_rows_read=U_COUNT,original_U_s_reads=0,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,new_neural_updates=0,native_updates=0,density_fit_updates=0,normalizer_fit_updates=0,threshold_fit_updates=0,
        adapter_iterations=20,new_target_statistical_iterations=20,inherited_shared_adapter_iterations=20,source_density_fit_updates=0,real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,
        source_bank_deployed=False,U_bank_deployed=False,per_record_source_rows_deployed=0,formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Complete frozen six-column three-condition predictions required before truth')
    c=evaluation.validate_config(m['config'])
    if any(c[k]!=m[k] for k in ('run_id','code_commit')):raise ValueError('Actual configuration/run identity differs')
    pm,state,tau,cal=evaluation._reference(c['reference_prediction_root'])
    if m['reference_prediction_identity']!=pm or m['model_identity']!=pm['model_identity'] or m['parent_F0_identity']!=pm['parent_F0_identity'] or m['dura_identity']!=pm['dura_identity'] or m['parent_calibration_identity']!=cal:raise ValueError('Original frozen P1 signal/F0/DURA/source state identity required')
    if not evaluation._same(_arrays(root/'models/density.npz'),state):raise ValueError('Original source density cannot change')
    saved=_arrays(root/'thresholds.npz')
    if set(saved)!={'conditional'} or saved['conditional'].dtype!=np.float64 or not np.array_equal(saved['conditional'],tau) or _read(root/'calibration/complete.json')!=cal:raise ValueError('Both original six-class thresholds must be reused without retau')
    fit=_read(root/'models/complete.json')
    if any(fit.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.SOURCE_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],models_frozen=True,models_refitted=False,thresholds_refitted=False,source_prior_frozen_before_U=True,U_reads=0,Q_reads=0,parent_calibration_identity=cal).items()):raise ValueError('Source models must freeze before cached U use')
    adaptation,u=_adapters(root,m,c,pm,state,tau,cal)
    if any(m['U_manifest'].get(k)!=v for k,v in dict(capsule_id=c['U_capsule_id'],split_id=c['U_split_id'],count=U_COUNT).items()):raise ValueError('Fixed legal unlabelled discovery capsule required')
    rows={'source_DURA':{}};source_ids=set();six_tau=np.tile(tau,(3,1))
    for role in ('known','novel'):
        a=evaluation._source_rows(root,'source_DURA',role);old=evaluation.mix.best._source_rows(c['reference_prediction_root'],'source_DURA',role)
        if any(not np.array_equal(a[k],np.tile(v,(1,3)) if k in ('raw_scores','scores') else v) for k,v in old.items()):raise ValueError('Source receivers must repeat the original two zero-shift columns')
        raw=core.scores(state,a['signal_features'],np.zeros((6,160),np.float64))
        if not np.allclose(a['raw_scores'],np.tile(raw,(1,3)),rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-six_tau[:,a['closed']].T):raise ValueError('All source raw scores and margins must reconstruct with no target adapter')
        if source_ids&set(a['ids'].tolist()):raise ValueError('Source known and novel roles overlap')
        source_ids.update(a['ids'].tolist());rows['source_DURA'][role]=a
    for condition in CONDITIONS[1:]:
        a=evaluation._target_rows(root,condition);old=evaluation.mix.best._target_rows(c['reference_prediction_root'],condition);shared=evaluation.mix._target_rows(c['reference_adaptation_root'],condition)
        if any(not np.array_equal(a[k][:,:2] if k in ('raw_scores','scores') else a[k],v) for k,v in old.items()):raise ValueError('DA0 and original identity/features must equal the completed best parent')
        if any(not np.array_equal(a[k][:,2:4],shared[k][:,4:6]) for k in ('raw_scores','scores')):raise ValueError('SHARED control must exactly copy the original DA_M predictions')
        raw=np.concatenate((shared_core.scores(state,a['signal_features'],np.zeros(160,np.float64)),shared_core.scores(state,a['signal_features'],adaptation['shared_adapter']['delta']),core.scores(state,a['signal_features'],adaptation['adapter']['delta'])),axis=1)
        if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-six_tau[:,a['closed']].T):raise ValueError('All class/shared likelihoods and original-prediction-routed margins must reconstruct')
        if (source_ids|set(u['ids'].tolist()))&set(a['ids'].tolist()):raise ValueError('Future Q cannot overlap fit-U or source roles')
        rows[condition]=a
    if source_ids&set(u['ids'].tolist()):raise ValueError('Cached U and source roles must remain disjoint')
    return m,cal,rows,adaptation


def _pair(k,n,main,reference,km=None,nm=None):
    return paired_counts(k,n,np.ones(len(k['y']),bool) if km is None else km,np.ones(len(n['scores']),bool) if nm is None else nm,primary_index=main,reference_index=reference)


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent UDA score output required')
    m,cal,rows,adaptation=_prediction(prediction_root)
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
    gain=paired['DA0_ratio']['star_DURA']['delta']['A_K'];positive=gain is not None and gain>1e-12;protection=all(main[c]['goals']['passed'] for c in CONDITIONS[1:]);success=bool(positive and protection)
    prior=adaptation['prior'];adapter=adaptation['adapter'];diag=adaptation['diagnostics'];shared=adaptation['shared_diagnostics']['arms']['DA_M']
    summary=dict(CLASS=dict(iterations=20,shift_rms=float(np.sqrt(np.sum(adapter['delta']**2)/6)),class_shift_norms=np.linalg.norm(adapter['delta'],axis=1).tolist(),background_shift_norm=float(np.linalg.norm(adapter['background_delta'])),coefficient_norm=float(np.linalg.norm(adapter['coefficients'])),initial_objective=diag['trace'][0]['objective'],final_objective=diag['trace'][-1]['objective'],group_prior=diag['group_prior']),
        SHARED=dict(iterations=20,inherited=True,delta_norm=float(np.linalg.norm(adaptation['shared_adapter']['delta'])),initial_objective=shared['trace'][0]['objective'],final_objective=shared['trace'][-1]['objective'],group_prior=shared['group_prior']))
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,primary_summary={c:main[c]['overall'] for c in CONDITIONS},
        paired_primary_vs_DA0_ratio=paired['DA0_ratio'],paired_primary_vs_SHARED_ratio=paired['SHARED_ratio'],star_A_K_gain=gain,star_A_K_gain_positive=bool(positive),target_protection_passed=bool(protection),adaptation_success=success,development_gate_passed=success,eligible_primary=success,
        next_step='DEVELOPMENT_ADAPTATION_PASSED_FOR_SEPARATE_REGISTRATION_PREREGISTRATION' if success else 'NO_PROMOTION',truth_last=True,all_three_predictions_before_truth=True,U_truth_reads=0,source_truth_reads=1,target_truth_reads=2,
        source_unchanged=True,DA0_controls_reused_exactly=True,shared_controls_reused_exactly=True,classification_unchanged=True,thresholds_refitted=False,target_fit_input='Cached unlabelled star U signal features only; clean is test only',source_unknown_diagnostic_only=True,
        source_prior_rank=len(prior['basis']),source_radius=float(prior['radius']),adapter_summary=summary,new_target_statistical_iterations=20,inherited_shared_adapter_iterations=20,target_class_mode_adaptation=True,new_neural_updates=0,source_density_fit_updates=0,threshold_fit_updates=0,
        costs={k:m[k] for k in ('real_F_forward_rows','real_signal_forward_rows','G_forward_rows','synthetic_F_forward_rows','synthetic_signal_forward_rows','G_synthetic_forward_rows','target_U_reads','U_feature_rows_read','target_Q_IQ_reads','source_IQ_reads')},
        source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,target_REG=False,registration_success=None,A_N=None,H_OW=None,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        inherited_cost_scope='Original F0 E50/11100, P1 signal r03 2220 updates, DURA and SHARED twenty-step fit remain inherited; CLASS twenty-step target mode adaptation is new, neural updates and real forwards are zero',
        scope='Source and Q historical exposure; one development RX/seed, no U true TX/K, no registration or full Phase3 completion')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_DA0_ratio','paired_primary_vs_SHARED_ratio')});print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
