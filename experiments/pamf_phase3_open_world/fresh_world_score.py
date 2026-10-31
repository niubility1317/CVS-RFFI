"""Fresh RX34 two-date, four-unknown truth-last scoring with full rosters."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import fresh_world_run as evaluation
from . import star_density_core as density
from . import star_dura_best_score as previous
from . import star_consensus_score as common
from .star_crossset_metric_score import paired_counts
from .star_physical_open_score import _metrics,_known_pass
from .star_ova_score import _ci

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_fresh_world_scores_v1'
STATUS='FRESH_WORLD_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=METHODS[5]
CONDITIONS=evaluation.CONDITIONS
COUNTS=dict(evaluation.COUNTS)
Q_COUNT=10238
TARGET_UNKNOWN=(6,7,8,9)
DAYS=('2021_03_15','2021_03_23')
TX_BY_LABEL=('14-10','14-7','20-15','20-19','6-15','8-20','5-5','10-7','2-19','7-11')
_read,_write,_source_truth=common._read,common._write,common._truth


def _arrays(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def _future_truth(path,ids):
    a=_arrays(path);n=len(ids)
    if set(a)!={'ids','y','tx','receiver','day','scenario'} or not common._ids(a['ids'],n):raise ValueError('Exact private future truth fields and opaque unique IDs required')
    if (any(a[k].shape!=(n,) or a[k].dtype!=np.int64 for k in ('y','receiver'))
            or any(a[k].shape!=(n,) or a[k].dtype.kind!='U' or np.any(np.char.strip(a[k])=='') for k in ('tx','day','scenario'))
            or not np.isin(a['y'],range(10)).all() or not np.all(a['receiver']==34) or not np.isin(a['day'],DAYS).all()
            or not np.array_equal(a['tx'],np.asarray(TX_BY_LABEL)[a['y']]) or set(a['ids'].tolist())!=set(ids.tolist())):
        raise ValueError('Only private RX34/two-date ten-class future mapping is allowed')
    lookup={v:i for i,v in enumerate(a['ids'].tolist())};order=np.asarray([lookup[v] for v in ids.tolist()],np.int64)
    return {k:v[order] for k,v in a.items()}


def _target(a,t,condition,index):
    y=t['y'];known=y<6;unknown=~known;scenes=('clean',) if condition=='clean_DURA' else common.SCENES;minimum=.95 if condition=='clean_DURA' else .90
    def metric(mask,kr=tuple(range(6)),ur=TARGET_UNKNOWN):
        k=mask&known;u=mask&unknown
        return _metrics(y[k],a['closed'][k],a['scores'][k,index],y[u],a['scores'][u,index],a['baseline_closed'][k],kr,ur)
    allrows=np.ones(len(y),bool);overall=metric(allrows);by_scene={s:metric(t['scenario']==s) for s in scenes}
    passed=lambda row:bool(_known_pass(row) and row['coverage']['unknown_complete'] and row['R_U'] is not None and row['R_U']>=minimum-1e-12)
    return dict(overall=overall,by_RX={str(r):metric(t['receiver']==r) for r in np.unique(t['receiver'])},by_day={d:metric(t['day']==d) for d in DAYS},by_scene=by_scene,
        by_day_scene={d:{s:metric((t['day']==d)&(t['scenario']==s)) for s in scenes} for d in DAYS},
        by_TX={TX_BY_LABEL[c]:metric((y==c)|unknown,(c,)) for c in range(6)}|{TX_BY_LABEL[c]:metric(known|(y==c),ur=(c,)) for c in TARGET_UNKNOWN},
        discoverable_unknown=metric(known|np.isin(y,(6,7,8)),ur=(6,7,8)),late_only_unknown=metric(known|(y==9),ur=(9,)),
        confidence_intervals=dict(level=.95,method='Per true-class Wilson; one development RX, global physical history UNKNOWN',
            A_K_per_class=_ci(overall['known_per_class'],'correct_accepted'),C_K_per_class=_ci(overall['known_per_class'],'accepted'),R_U_per_class=_ci(overall['unknown_per_class'],'rejected')),
        goals=dict(unknown_minimum=minimum,overall_passed=passed(overall),all_main_scenes_passed=all(passed(v) for v in by_scene.values()),
            required_scene_coverage_complete=all(v['coverage']['complete'] for v in by_scene.values()),passed=bool(passed(overall) and all(passed(v) for v in by_scene.values()))),
        date_groups_diagnostic_only=True,known_roster=list(range(6)),unknown_roster=list(TARGET_UNKNOWN),global_physical_history='UNKNOWN',new_blind_confirmation=False)


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,
        all_conditions_through_DURA=True,classification_unchanged=True,source_control_readouts_reused_exactly=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,
        source_density_models_fitted=1,new_threshold_sets_fitted=2,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,target_U_reads=0,original_U_s_reads=0,new_neural_updates=0,new_candidate_optimizer_updates=444,
        training_target_U_count=576,U_ID_metadata_reads=576,source_IQ_reads=sum(COUNTS.values()),target_Q_IQ_reads=2*Q_COUNT,
        real_signal_forward_rows=sum(COUNTS.values())+6*Q_COUNT,real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,
        formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('All fresh three-condition six-readout predictions must complete before truth')
    c=evaluation.validate_config(m['config']);pm,old_states,old_tau,old_cal=evaluation._reference(c)
    if c['code_commit']!=m['code_commit'] or m['source_control_prediction_identity']!=pm or any(m[k]!=pm[k] for k in ('parent_F0_identity','dura_identity','loaded_dura_identity')):raise ValueError('Actual fresh execution and source-only model ancestors differ')
    identity,Uids,_=evaluation._training_metadata(c,pm)
    identities=dict(E0=pm['model_identities']['E0'],SOURCE=pm['model_identities']['SOURCE'],TARGET_PCD=identity)
    fit=_read(root/'models/complete.json');cal=_read(root/'calibration/complete.json')
    if (fit.get('schema')!=evaluation.SCHEMA or fit.get('status')!=evaluation.MODELS_STATUS or fit.get('run_id')!=RUN_ID or fit.get('code_commit')!=m['code_commit']
            or fit.get('identities')!=identities or m['model_identities']!=identities or fit.get('neural_states_frozen') is not True or fit.get('source_V_fit') is not False or fit.get('target_Q_fit') is not False):
        raise ValueError('Only one new PCD density fit after all three neural models freeze is allowed')
    states={name:_arrays(root/'models'/(name+'.npz')) for name in evaluation.MODEL_NAMES}
    for state in states.values():density.validate_state(state,'density_ratio')
    if any(not evaluation._same(states[name],old_states[name]) for name in ('E0','SOURCE')):raise ValueError('Two inherited source densities changed')
    saved=_arrays(root/'thresholds.npz');tau=saved.get('conditional')
    if (set(saved)!={'conditional'} or tau.dtype!=np.float64 or tau.shape!=(6,6) or not np.isfinite(tau).all() or not np.array_equal(tau[:4],old_tau)
            or cal.get('schema')!=evaluation.SCHEMA or cal.get('status')!=evaluation.CALIBRATION_STATUS or cal.get('run_id')!=RUN_ID or cal.get('code_commit')!=m['code_commit']
            or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False or cal.get('target_reads')!=0
            or set(cal.get('methods',{}))!=set(METHODS) or cal.get('conditional_thresholds')!=tau.tolist()):raise ValueError('Exactly twenty-four inherited and twelve new source-known thresholds must freeze')
    for j,name in enumerate(METHODS):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist() or (j<4 and cal['methods'][name]!=old_cal['methods'][name]):raise ValueError('Class-routed source risk thresholds or inherited reports differ')
    used=set(Uids.tolist());rows={'source_DURA':{}}
    def reconstruct(a):
        raw=np.concatenate([evaluation._raw(states[name],a['signal_features'][:,i],a['closed']) for i,name in enumerate(evaluation.MODEL_NAMES)],axis=1)
        if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('All three actual unit160 geometries and predicted-class margins must reconstruct')
    for role in ('known','novel'):
        a=evaluation._source_rows(root,'source_DURA',role);old=evaluation.previous._source_rows(c['source_control_config']['output_root'],'source_DURA',role)
        evaluation._exact_source_controls(a,old);reconstruct(a);ids=set(a['ids'].tolist())
        if ids&used:raise ValueError('Source and discovery roles overlap')
        used.update(ids);rows['source_DURA'][role]=a
    paired=None
    for name in CONDITIONS[1:]:
        a=evaluation._target_rows(root,name);reconstruct(a)
        if used&set(a['ids'].tolist()):raise ValueError('Future Q overlaps source or discovery roles')
        if paired is None:paired=a['ids'].copy()
        elif not np.array_equal(paired,a['ids']):raise ValueError('Clean/star future predictions must have identical physical IDs and order')
        rows[name]=a
    return m,cal,rows


def _pair(k,n,main,reference,km=None,nm=None):
    return paired_counts(k,n,np.ones(len(k['y']),bool) if km is None else km,np.ones(len(n['scores']),bool) if nm is None else nm,primary_index=main,reference_index=reference)


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent UDA score output required')
    m,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['source_DURA']['novel']);truths={base:_future_truth(path,rows[base+'_DURA']['ids']) for base,path in (('clean',clean_truth),('star',star_truth))}
    for base,t in truths.items():
        if not np.isin(t['scenario'],('clean',) if base=='clean' else common.SCENES).all():raise ValueError('Fresh clean/star private scenario roster differs')
    if any(not np.array_equal(truths['clean'][k],truths['star'][k]) for k in ('ids','y','tx','receiver','day')):raise ValueError('Paired clean/star private physical labels and dates differ')
    methods={}
    for j,name in enumerate(METHODS):
        paths={'source_DURA':previous._source(rows['source_DURA'],nt,j)};paths.update({base+'_DURA':_target(rows[base+'_DURA'],truths[base],base+'_DURA',j) for base in ('clean','star')});methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY)
    main=methods[PRIMARY]['paths'];paired={}
    for reference in (1,3):
        by_condition={}
        for condition in CONDITIONS:
            if condition=='source_DURA':
                k,n=rows[condition]['known'],rows[condition]['novel'];km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],nt['scenario']));groups={g:_pair(k,n,5,reference,mask,nm[g]) for g,mask in km.items()};uy=nt['tx'];unknown_roster=common.UNKNOWN_TX
            else:
                t=truths[condition.split('_')[0]];y=t['y'];mask=y<6;a=rows[condition];k=dict(y=y[mask],closed=a['closed'][mask],scores=a['scores'][mask]);n=dict(scores=a['scores'][~mask]);scenes=('clean',) if condition=='clean_DURA' else common.SCENES
                groups={g:_pair(k,n,5,reference,t['scenario'][mask]==g,t['scenario'][~mask]==g) for g in scenes};uy=y[~mask];unknown_roster=TARGET_UNKNOWN
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
        classification_unchanged=True,source_control_readouts_reused_exactly=True,target_fit_input='Unlabelled star U balanced prototype objective; clean and Q are test only',source_unknown_diagnostic_only=True,
        SOURCE_weights_source_only=True,TARGET_PCD_weights_source_only=False,TARGET_PCD_weights_use_target_U=True,
        new_projection_optimizer_updates_total=444,new_body_optimizer_updates_total=444,new_candidate_optimizer_updates=444,prediction_optimizer_updates=0,Sinc_optimizer_steps=0,
        inherited_SOURCE_optimizer_updates=444,new_source_density_models=1,new_source_threshold_sets=2,prototypes_deployed=False,original_F0_updates=0,G_updates=0,threshold_values=36,threshold_bytes=288,
        costs={k:m[k] for k in ('source_IQ_reads','target_Q_IQ_reads','target_U_reads','U_ID_metadata_reads','training_target_U_count','real_F_forward_rows','G_forward_rows','real_signal_forward_rows','real_signal_forward_rows_by_arm','synthetic_F_forward_rows','G_synthetic_forward_rows','synthetic_signal_forward_rows')},
        training_costs=m['training_costs'],combined_new_real_signal_forward_rows=m['combined_new_real_signal_forward_rows'],combined_new_G_forward_rows=m['combined_new_G_forward_rows'],combined_new_real_F_forward_rows=m['combined_new_real_F_forward_rows'],
        source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,target_REG=False,registration_success=None,A_N=None,H_OW=None,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        original_F0_parameters=232993,open_signal_parameters_per_arm=334721,existing_restorer_parameters=266951,primary_deployed_neural_parameters=834665,
        source_RX_seen=True,source_novel_historically_exposed=True,fresh_development_RX=34,global_physical_history='UNKNOWN',new_blind_confirmation=False,
        scope='RX3-2 temporal development pool, source-only E0/SOURCE controls and independent fresh-U576 PCD. Missing date/scene classes retain full-roster N/A. Old RX24-1 failures remain; no formal/registration/full Phase3 claim')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_E0_predicted_ratio','paired_primary_vs_SOURCE_predicted_ratio')});print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
