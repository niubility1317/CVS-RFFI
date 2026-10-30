"""Independent three-test scoring of frozen P1 predicted-class densities."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_class_density_run as evaluation
from . import star_class_density_core as core
from . import star_consensus_score as common
from .star_physical_open_score import _metrics,_known_pass
from .star_crossset_metric_score import paired_counts
from .star_macro12_score import _cross_report
from .star_ova_score import _ci
from .scoring import _truth as _target_truth

RUN_ID='20261031-phase3-class-density-s2026092701-r01'
SCHEMA='pamf_phase3_class_density_scores_v1'
STATUS='CLASS_DENSITY_SCORES_COMPLETE'
METHODS=core.METHODS
PRIMARY=core.PRIMARY_METHOD
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
CONDITIONS=('clean_DURA','star_DURA')
TARGET_UNKNOWN=(6,7,8)  # Private scorer roster, never passed to the predictor.
_read,_write,_source_truth=common._read,common._write,common._truth


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,
        classification_path='P1',identity_baseline_path='same_DURA_output_P1',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        private_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,target_U_reads=0,formal_target_accessed=False,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_conditions_complete=True,all_conditions_through_DURA=True,model_parameters_buffers_unchanged=True,source_states_unchanged=True,
        controls_reused_exactly=True,source_bank_deployed=False,per_record_source_rows_deployed=0,neural_updates=0,density_fit_updates=0,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('All three frozen DURA prediction tests must complete before truth')
    c=evaluation.validate_config(m['config'])
    if c['run_id']!=m['run_id'] or c['code_commit']!=m['code_commit']:raise ValueError('Actual execution/config identity differs')
    pm,state,old_tau,parent_cal=evaluation._reference(c['reference_prediction_root'],c['train_config'])
    identity=pm['signal_reference_identity']['model_identity']
    if (m['reference_prediction_identity']!=pm or m['model_identity']!=identity or m['parent_F0_identity']!=identity['parent_F0_identity']
            or m['dura_identity']!=identity['dura_identity']):raise ValueError('Actual r03 P1 branch, original F0 and fixed DURA must match the parent, not its P0 candidate')
    with np.load(root/'models/density.npz',allow_pickle=False) as z:stored={k:z[k].copy() for k in z.files}
    core.density.validate_state(stored,'density_ratio')
    if not evaluation._same(stored,state):raise ValueError('No density refit or control-model change allowed')
    fit=_read(root/'models/complete.json')
    if any(fit.get(k)!=v for k,v in dict(status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],models_refitted=False,models_frozen=True).items()):raise ValueError('Frozen original P1 density state required')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Exact four conditional threshold sets required')
        tau=z['conditional'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.dtype!=np.float64 or tau.shape!=(4,6) or not np.isfinite(tau).all() or not np.array_equal(tau[:2],old_tau)
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],
                method_names=list(METHODS),primary_method=PRIMARY,known_only=True,unknown_inputs_used=False,target_reads=0,group_count=42,
                conditional_thresholds=tau.tolist()).items()) or set(cal.get('methods',{}))!=set(METHODS)):
        raise ValueError('Exact two frozen controls and two V-only calibrations required')
    for j,name in enumerate(METHODS):
        report=cal['methods'][name]
        if (set(report.get('groups',{}))!=set(common.GROUP_NAMES) or report.get('hard_names')!=list(evaluation.risk.HARD_NAMES)
                or report.get('diagnostic_names')!=list(evaluation.risk.DIAGNOSTIC_NAMES) or report.get('conditional_thresholds')!=tau[j].tolist()):raise ValueError('All methods need hard12 and thirty cross diagnostics')
    for j,name in enumerate(METHODS[:2]):
        if cal['methods'][name]!=parent_cal['methods'][pm['method_names'][j]]:raise ValueError('Original pooled calibration must be copied exactly')
    rows={}
    for role in ('known','novel'):
        a=evaluation._rows(root,role);old=evaluation._rows(c['reference_prediction_root'],role)
        for k,v in old.items():
            actual=a[k][:,:2] if k in ('scores','raw_scores') else a[k];expected=v[:,:2] if k in ('scores','raw_scores') else v
            if not np.array_equal(actual,expected):raise ValueError('Exact source controls, P1 identity, metadata and representations must remain unchanged')
        _reconstruct(a,state,tau);rows[role]=a
    for name in CONDITIONS:
        a=evaluation._target_rows(root,name);n=Q_COUNT
        if (not common._ids(a['ids'],n) or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
                or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
                or a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):
            raise ValueError('Original F0 identity and baseline must share the actual DURA output, including clean')
        _reconstruct(a,state,tau);rows[name]=a
    source=set(rows['known']['ids'].tolist())|set(rows['novel']['ids'].tolist())
    if set(rows['known']['ids'].tolist())&set(rows['novel']['ids'].tolist()) or any(source&set(rows[n]['ids'].tolist()) for n in CONDITIONS):raise ValueError('Source and query physical roles must remain disjoint')
    return m,cal,rows


def _reconstruct(a,state,tau):
    n=len(a['ids']);z=a['signal_features']
    if (z.shape!=(n,160) or z.dtype!=np.float32 or not np.isfinite(z).all()
            or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)
            or any(a[k].shape!=(n,4) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):raise ValueError('Complete four-column FP64 evidence and same P1 unit160 required')
    raw=core.raw_scores(state,z,a['closed'])
    if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('Predicted-class fifteen-mode likelihood and class-routed margins must reconstruct')


def _confidence(row):
    return dict(level=.95,method='Per true-class Wilson; one historical development RX, no independent-RX confidence claim',
        A_K_per_class=_ci(row['known_per_class'],'correct_accepted'),C_K_per_class=_ci(row['known_per_class'],'accepted'),R_U_per_class=_ci(row['unknown_per_class'],'rejected'))


def _full_pass(row,minimum):
    return bool(_known_pass(row) and row['coverage']['unknown_complete'] and row['R_U'] is not None and row['R_U']>=minimum-1e-12)


def _target(a,truth,condition,index):
    y=truth['y'];known=y<6;unknown=~known;minimum=.95 if condition=='clean_DURA' else .90
    scenes=('clean',) if condition=='clean_DURA' else common.SCENES
    def metric(mask,kr=tuple(range(6)),ur=TARGET_UNKNOWN):
        k=mask&known;u=mask&unknown
        return _metrics(y[k],a['closed'][k],a['scores'][k,index],y[u],a['scores'][u,index],a['baseline_closed'][k],kr,ur)
    overall=metric(np.ones(len(y),bool));by_scene={s:metric(truth['scenario']==s) for s in scenes}
    coverage=all(r['coverage']['complete'] for r in by_scene.values());passed=_full_pass(overall,minimum) and all(_full_pass(r,minimum) for r in by_scene.values())
    return dict(overall=overall,by_scene=by_scene,by_RX={str(r):metric(truth['receiver']==r) for r in np.unique(truth['receiver'])},by_day={'2021_03_15':overall},
        day_scope='Fixed future-Q public physical role; no query-side fit',by_TX={str(c):metric((y==c)|unknown,(c,)) for c in range(6)}|{str(c):metric(known|(y==c),ur=(c,)) for c in TARGET_UNKNOWN},
        confidence_intervals=_confidence(overall),goals=dict(unknown_minimum=minimum,overall_passed=_full_pass(overall,minimum),all_main_scenes_passed=all(_full_pass(r,minimum) for r in by_scene.values()),
            required_scene_coverage_complete=coverage,passed=bool(passed)),historically_exposed_development_RX=True,new_blind_confirmation=False)


def _source(k,n,t,cal,index):
    km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],t['scenario']));allk=np.ones(len(k['ids']),bool);alln=np.ones(len(n['ids']),bool)
    def metric(ki,ni,kr=tuple(range(6)),ur=common.UNKNOWN_TX):return _metrics(k['y'][ki],k['closed'][ki],k['scores'][ki,index],t['tx'][ni],n['scores'][ni,index],k['baseline_closed'][ki],kr,ur)
    groups={name:metric(mask,nm[name]) for name,mask in km.items()};overall=groups['overall'];hard=evaluation.risk.HARD_NAMES
    return dict(overall=overall,groups=groups,by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},by_scene={s:groups['scene/'+s] for s in common.SCENES},
        by_day={str(d):metric(k['day_i']==d,n['day_i']==d) for d in (1,2,3)},by_TX={str(c):metric(k['y']==c,alln,(c,)) for c in range(6)}|{c:metric(allk,t['tx']==c,ur=(c,)) for c in common.UNKNOWN_TX},
        confidence_intervals=_confidence(overall),cross_guard_violations_or_missing=_cross_report(groups),known_12_groups_passed=all(_known_pass(groups[g]) for g in hard),known_42_groups_passed=all(_known_pass(r) for r in groups.values()),
        source_calibration_passed=bool(cal['methods'][METHODS[index]]['constraints_passed']),source_unknown_90_diagnostic_passed=bool(overall['coverage']['unknown_complete'] and overall['R_U'] is not None and overall['R_U']>=.90-1e-12),
        unknown_is_development_diagnostic_not_target_entry_gate=True,source_RX_seen=True,source_novel_historically_exposed=True)


def _paired(k,n,primary,reference,kmask=None,nmask=None):
    if kmask is None:kmask=np.ones(len(k['y']),bool)
    if nmask is None:nmask=np.ones(len(n['scores']),bool)
    return paired_counts(k,n,kmask,nmask,primary_index=primary,reference_index=reference)


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent three-condition score root required')
    m,cal,rows=_prediction(prediction_root)
    # Exactly these three private joins, after every public prediction passes.
    nt=_source_truth(source_novel_truth,rows['novel']);truths={name:_target_truth(path,rows[name]['ids']) for name,path in zip(CONDITIONS,(clean_truth,star_truth))}
    for name,t in truths.items():
        scenes=('clean',) if name=='clean_DURA' else common.SCENES
        if not np.isin(t['y'],range(9)).all() or not np.all(t['receiver']==26) or not np.isin(t['scenario'],scenes).all():raise ValueError('Fixed development RX24-1 condition and scorer-only old/unknown roster required')
    methods={}
    for j,name in enumerate(METHODS):
        methods[name]=dict(source=_source(rows['known'],rows['novel'],nt,cal,j),**{c:_target(rows[c],truths[c],c,j) for c in CONDITIONS},diagnostic_only=name!=PRIMARY)
    main=methods[PRIMARY];source_known=main['source']['known_12_groups_passed'] and main['source']['source_calibration_passed'];target_passed=all(main[c]['goals']['passed'] for c in CONDITIONS)
    paired={}
    for label,a,b in (('conditional_known_vs_pooled',2,0),('conditional_ratio_vs_pooled',3,1),('pooled_ratio_vs_known',1,0),('conditional_ratio_vs_known',3,2)):
        comparison=dict(primary=METHODS[a],reference=METHODS[b],counts={'source':_paired(rows['known'],rows['novel'],a,b)},groups={})
        sk=rows['known'];sn=rows['novel'];km=dict(common._masks(sk['rx_i'],sk['scenario']));nm=dict(common._masks(sn['rx_i'],nt['scenario']))
        comparison['groups']['source']={g:_paired(sk,sn,a,b,mask,nm[g]) for g,mask in km.items()}
        for c in CONDITIONS:
            data=rows[c];y=truths[c]['y'];mask=y<6;k=dict(y=y[mask],closed=data['closed'][mask],scores=data['scores'][mask]);n=dict(scores=data['scores'][~mask]);comparison['counts'][c]=_paired(k,n,a,b)
            scenes=('clean',) if c=='clean_DURA' else common.SCENES
            comparison['groups'][c]={s:_paired(k,n,a,b,truths[c]['scenario'][mask]==s,truths[c]['scenario'][~mask]==s) for s in scenes}
        comparison['delta']={c:{key:None if methods[METHODS[a]][c]['overall'][key] is None or methods[METHODS[b]][c]['overall'][key] is None else methods[METHODS[a]][c]['overall'][key]-methods[METHODS[b]][c]['overall'][key]
            for key in ('A_K','C_K','R_U','H_score','auroc_known_macro','oscr_auc_macro','original_closed_minus_new_open')} for c in ('source',)+CONDITIONS};paired[label]=comparison
    passed=bool(target_passed)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,methods=methods,primary_summary={c:main[c]['overall'] for c in ('source',)+CONDITIONS},paired=paired,
        source_known_12_passed=bool(source_known),source_unknown_90_diagnostic_passed=main['source']['source_unknown_90_diagnostic_passed'],development_gate_passed=passed,eligible_primary=passed,
        next_step='DEVELOPMENT_PASSED_FOR_SEPARATE_UDA_PREREGISTRATION' if passed else 'NO_PROMOTION',truth_last=True,all_three_predictions_before_truth=True,source_truth_reads=1,target_truth_reads=2,
        classification_path='P1',identity_baseline_path='same_DURA_output_P1',clean_also_uses_DURA=True,formal_target_accessed=False,target_U_reads=0,target_DA=False,target_REG=False,
        neural_updates=0,density_fit_updates=0,source_bank_deployed=False,per_record_source_rows_deployed=0,A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False,full_goal_passed=False,
        original_F0_parameters=232993,signal_parameters=334721,combined_active_parameters=567714,existing_restorer_parameters=266951,combined_with_DURA_parameters=834665,
        inherited_training_cost_scope='Original F0 E50/11100, r03 copied signal 11100+2220 and MLP2220, and fixed DURA training are inherited; no new neural/statistical fitting',
        costs={k:m[k] for k in ('real_signal_forward_rows','real_F_forward_rows','G_forward_rows','synthetic_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows')},
        scope='Shared source-star thresholds; source historical novel and one exposed development RX, no formal or complete open-world claim')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired')});print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
