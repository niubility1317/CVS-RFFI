"""Independent scoring of U-only frozen shifts; no U truth or adapter fitting."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_mixture_uda_run as evaluation
from . import star_mixture_uda_core as core
from . import star_dura_best_score as previous
from . import star_consensus_score as common
from .star_crossset_metric_score import paired_counts
from .scoring import _truth as _target_truth

RUN_ID='20261031-phase3-mixture-uda-s2026092701-r01'
SCHEMA='pamf_phase3_mixture_uda_scores_v1'
STATUS='MIXTURE_UDA_SCORES_COMPLETE'
METHODS=('DA0_known','DA0_ratio','DA_K_known','DA_K_ratio','DA_M_known','DA_M_ratio')
PRIMARY=METHODS[5]
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
U_COUNT=384
_read,_write,_source_truth=common._read,common._write,common._truth


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _adapters(root,m):
    folder=root/'adaptation';prior=_arrays(folder/'source_prior.npz');core.validate_prior(prior)
    complete=_read(folder/'complete.json');diag=_read(folder/'fit_diagnostics.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.ADAPTER_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],U_count=U_COUNT,adapter_names=list(core.ARMS),
        fit_input='U_signal_features_only',fit_diagnostics_ref='fit_diagnostics.json',iterations_per_adapter=20,source_prior_frozen_before_U=True,query_fit=False,
        true_U_labels_used=False,true_unknown_count_used=False,source_density_refitted=False,thresholds_refitted=False,adapters_frozen=True,Q_reads=0,source_prediction_reads=0,U_bank_deployed=False,private_U_features_ref='U_features.npz')
    if any(complete.get(k)!=v for k,v in expected.items()) or complete['U_manifest']!=m['U_manifest']:raise ValueError('Both U-only adapters must freeze before any source/Q prediction')
    u=_arrays(folder/'U_features.npz');n=U_COUNT
    if (set(u)!={'ids','default_logits','closed','signal_features'} or not common._ids(u['ids'],n)
            or u['default_logits'].shape!=(n,6) or u['default_logits'].dtype!=np.float32 or not np.isfinite(u['default_logits']).all()
            or u['closed'].shape!=(n,) or u['closed'].dtype!=np.int64 or not np.array_equal(u['closed'],u['default_logits'].argmax(1))):raise ValueError('Only complete label-free U public outputs may be stored; no true TX/K/scenario')
    _units(u['signal_features'],n)
    if any(diag.get(k)!=v for k,v in dict(iterations=20,U_count=U_COUNT,source_prior_frozen_before_U=True,U_only=True,Q_inputs_used=False,true_U_labels_used=False,true_unknown_count_used=False).items()) or set(diag.get('arms',{}))!=set(core.ARMS):raise ValueError('Exact two fixed twenty-step U-only fit diagnostics required')
    adapters={}
    for arm in core.ARMS:
        adapter=_arrays(folder/(arm+'.npz'));core.validate_adapter(adapter,prior);d=diag['arms'][arm]
        expected=dict(arm=arm,iterations=20,U_count=U_COUNT,group_prior=[1.,0.] if arm=='DA_K' else [.5,.5],source_rank=len(prior['basis']),
            true_U_labels_used=False,true_unknown_count_used=False,Q_inputs_used=False,default_logits_used=False,fixed_iteration_checkpoint=True,source_state_unchanged_by_fit=True)
        if any(d.get(k)!=v for k,v in expected.items()) or not np.isclose(d.get('source_radius',np.nan),float(prior['radius']),rtol=1e-12,atol=1e-12):raise ValueError('Recorded adapter source constraint or input permissions differ')
        trace=d.get('trace',[]);keys={'iteration','objective','coefficient_norm','shift_norm','known_posterior_mass','constraint_multiplier'}
        if len(trace)!=21 or [r.get('iteration') for r in trace]!=list(range(21)) or any(set(r)!=keys or not all(np.isfinite(r[k]) for k in keys) for r in trace):raise ValueError('Every adapter needs the complete finite iteration0..20 trace')
        if (any(b['objective']<a['objective']-1e-7 for a,b in zip(trace,trace[1:]))
                or any(r['coefficient_norm']<0 or r['coefficient_norm']>float(prior['radius'])+1e-10 or r['shift_norm']<0 or r['shift_norm']>float(prior['radius'])+1e-10
                    or not -1e-12<=r['known_posterior_mass']<=1+1e-12 or r['constraint_multiplier']<0 for r in trace)
                or not np.isclose(trace[-1]['coefficient_norm'],np.linalg.norm(adapter['coefficients']),rtol=1e-12,atol=1e-12)
                or not np.isclose(trace[-1]['shift_norm'],np.linalg.norm(adapter['delta']),rtol=1e-12,atol=1e-12)):
            raise ValueError('Stored fit must obey source bound, finite nondecreasing objective and final adapter norms')
        adapters[arm]=adapter
    return prior,adapters,diag,u


def _units(z,n):
    if z.shape!=(n,160) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6):raise ValueError('Finite same-row P1 signal unit160 required')


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,
        classification_path='parent_original_F0_same_DURA_input',identity_baseline_path='parent_original_F0_same_DURA_input',classification_unchanged=True,score_threshold=0.,
        source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,source_controls_reused_exactly=True,DA0_controls_reused_exactly=True,all_thresholds_reused_exactly=True,
        source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,source_IQ_reads=0,source_L_reads=0,source_OE_reads=0,target_Q_IQ_reads=0,target_U_reads=U_COUNT,
        original_U_s_reads=0,private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,new_neural_updates=0,native_updates=0,density_fit_updates=0,normalizer_fit_updates=0,threshold_fit_updates=0,adapter_iterations_per_arm=20,
        real_F_forward_rows=U_COUNT,real_signal_forward_rows=U_COUNT,G_forward_rows=U_COUNT,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,
        source_bank_deployed=False,U_bank_deployed=False,per_record_source_rows_deployed=0,formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('All three frozen six-column predictions must complete before any private evaluation truth')
    c=evaluation.validate_config(m['config'])
    if any(c[k]!=m[k] for k in ('run_id','code_commit')):raise ValueError('Actual run/config identity differs')
    pm,state,tau,cal=evaluation._reference(c['reference_prediction_root'])
    if (m['reference_prediction_identity']!=pm or m['model_identity']!=pm['model_identity'] or m['parent_F0_identity']!=pm['parent_F0_identity'] or m['dura_identity']!=pm['dura_identity'] or m['parent_calibration_identity']!=cal):raise ValueError('Frozen best P1 model and source identities must remain exact')
    if not evaluation._same(_arrays(root/'models/density.npz'),state):raise ValueError('Source density cannot refit on U')
    fit=_read(root/'models/complete.json')
    if any(fit.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.SOURCE_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],models_frozen=True,models_refitted=False,thresholds_refitted=False,parent_calibration_identity=cal,source_prior_frozen_before_U=True,U_reads=0,Q_reads=0).items()):raise ValueError('Original source model and thresholds must freeze before U')
    stored=_arrays(root/'thresholds.npz')
    if set(stored)!={'conditional'} or stored['conditional'].dtype!=np.float64 or stored['conditional'].shape!=(2,6) or not np.array_equal(stored['conditional'],tau) or _read(root/'calibration/complete.json')!=cal:raise ValueError('Both source-star thresholds/calibration must be copied without retau')
    prior,adapters,diag,U=_adapters(root,m)
    if (m['U_manifest'].get('capsule_id')!=c['U_capsule_id'] or m['U_manifest'].get('split_id')!=c['U_split_id'] or m['U_manifest'].get('count')!=U_COUNT):raise ValueError('Only the fixed discovery U public capsule may fit adapters')
    rows={'source_DURA':{}};source_ids=set();six_tau=np.tile(tau,(3,1));deltas=[np.zeros(160,np.float64)]+[adapters[arm]['delta'] for arm in core.ARMS]
    for role in ('known','novel'):
        a=evaluation._source_rows(root,'source_DURA',role);old=evaluation.best._source_rows(c['reference_prediction_root'],'source_DURA',role)
        for k,v in old.items():
            if not np.array_equal(a[k],np.tile(v,(1,3)) if k in ('raw_scores','scores') else v):raise ValueError('The target adapter must never alter a different source receiver; all six source columns repeat the original two')
        raw=core.scores(state,a['signal_features'],deltas[0])
        if not np.allclose(a['raw_scores'],np.tile(raw,(1,3)),rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-six_tau[:,a['closed']].T):raise ValueError('Source raw scores and margins must reconstruct with zero target shift')
        if source_ids&set(a['ids'].tolist()):raise ValueError('Source known and novel roles overlap')
        source_ids.update(a['ids'].tolist());rows['source_DURA'][role]=a
    for condition in CONDITIONS[1:]:
        a=evaluation._target_rows(root,condition);old=evaluation.best._target_rows(c['reference_prediction_root'],condition)
        for k,v in old.items():
            if not np.array_equal(a[k][:,:2] if k in ('raw_scores','scores') else a[k],v):raise ValueError('DA0, original F0 identity/baseline, public IDs and P1 features must remain exact')
        raw=np.concatenate([core.scores(state,a['signal_features'],delta) for delta in deltas],axis=1)
        if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-six_tau[:,a['closed']].T):raise ValueError('All six mixture scores and original-class-routed margins must reconstruct')
        if (source_ids|set(U['ids'].tolist()))&set(a['ids'].tolist()):raise ValueError('Future Q cannot overlap source or fit-U roles')
        rows[condition]=a
    if source_ids&set(U['ids'].tolist()):raise ValueError('U must be independent of source physical roles')
    return m,cal,rows,dict(prior=prior,adapters=adapters,diagnostics=diag)


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
    prior=adaptation['prior'];diag=adaptation['diagnostics'];summary={arm:dict(iterations=20,delta_norm=float(np.linalg.norm(adaptation['adapters'][arm]['delta'])),coefficient_norm=float(np.linalg.norm(adaptation['adapters'][arm]['coefficients'])),initial_objective=diag['arms'][arm]['trace'][0]['objective'],final_objective=diag['arms'][arm]['trace'][-1]['objective'],group_prior=diag['arms'][arm]['group_prior']) for arm in core.ARMS}
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,primary_summary={c:main[c]['overall'] for c in CONDITIONS},
        paired_primary_vs_DA0_ratio=paired['DA0_ratio'],paired_primary_vs_DA_K_ratio=paired['DA_K_ratio'],star_A_K_gain=gain,star_A_K_gain_positive=bool(positive),target_protection_passed=bool(protection),adaptation_success=success,development_gate_passed=success,eligible_primary=success,
        next_step='DEVELOPMENT_ADAPTATION_PASSED_FOR_SEPARATE_REGISTRATION_PREREGISTRATION' if success else 'NO_PROMOTION',truth_last=True,all_three_predictions_before_truth=True,U_truth_reads=0,source_truth_reads=1,target_truth_reads=2,
        source_unchanged=True,DA0_controls_reused_exactly=True,classification_unchanged=True,thresholds_refitted=False,target_fit_input='Unlabelled star U only; clean is test only',source_unknown_diagnostic_only=True,
        source_prior_rank=len(prior['basis']),source_radius=float(prior['radius']),adapter_summary=summary,target_statistical_iterations_total=40,new_neural_updates=0,source_density_fit_updates=0,threshold_fit_updates=0,
        costs={k:m[k] for k in ('real_F_forward_rows','real_signal_forward_rows','G_forward_rows','synthetic_F_forward_rows','synthetic_signal_forward_rows','G_synthetic_forward_rows','target_U_reads','target_Q_IQ_reads','source_IQ_reads')},
        source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,target_REG=False,registration_success=None,A_N=None,H_OW=None,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        inherited_cost_scope='Original F0 E50/11100, P1 signal r03 2220 updates and DURA training remain inherited; two20-step target statistical fits are new, neural updates and Q forwards are zero',
        scope='Source and Q historical exposure; one development RX/seed, no U true TX/K, no registration or full Phase3 completion')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_DA0_ratio','paired_primary_vs_DA_K_ratio')});print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
