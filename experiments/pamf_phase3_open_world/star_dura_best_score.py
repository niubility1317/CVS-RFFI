"""Frozen P1 readouts: six prediction paths first, three private truth joins last."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_dura_best_run as evaluation
from . import star_density_core as density
from . import star_consensus_score as common
from .star_physical_open_score import _metrics,_known_pass
from .star_macro12_risk import HARD_NAMES
from .star_macro12_score import _cross_report
from .star_ova_score import _ci
from .scoring import _truth as _target_truth

RUN_ID='20261031-phase3-dura-best-s2026092701-r01'
SCHEMA='pamf_phase3_dura_best_scores_v1'
STATUS='DURA_BEST_SCORES_COMPLETE'
METHODS=('known_support','density_ratio')
PRIMARY=METHODS[1]
CONDITIONS=('source_raw','source_DURA','clean_raw','clean_DURA','star_raw','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
KNOWN_FIELDS={'ids','y','rx_i','day_i','scenario','default_logits','closed','baseline_closed','signal_features','raw_scores','scores'}
NOVEL_FIELDS=KNOWN_FIELDS-{'y','scenario','baseline_closed'}
TARGET_FIELDS=(NOVEL_FIELDS-{'rx_i','day_i'})|{'baseline_closed'}
TARGET_UNKNOWN=(6,7,8)
_read,_write,_source_truth=common._read,common._write,common._truth


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,
        classification_path='per_input_original_F0',identity_baseline_path='same_path_original_F0',all_six_paths_complete=True,all_DURA_main_conditions_complete=True,
        models_frozen_before_any_new_input=True,all_thresholds_frozen_before_source_raw_and_Q=True,model_parameters_buffers_unchanged=True,source_states_unchanged=True,
        source_DURA_controls_reused_exactly=True,new_neural_updates=0,density_fit_updates=0,threshold_fit_updates=0,private_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,target_U_reads=0,
        real_F_forward_rows=COUNTS['V']+COUNTS['novel']+4*Q_COUNT,real_signal_forward_rows=COUNTS['V']+COUNTS['novel']+4*Q_COUNT,G_forward_rows=2*Q_COUNT,
        synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('All six fixed-model raw/DURA prediction paths required before any truth')
    c=evaluation.validate_config(m['config'])
    if c['run_id']!=m['run_id'] or c['code_commit']!=m['code_commit']:raise ValueError('Actual run/config identity differs')
    pm,state,tau,parent_cal=evaluation._reference(c['reference_prediction_root'],c['train_config'])
    identity=pm['signal_reference_identity']['model_identity']
    if (m['reference_prediction_identity']!=pm or m['model_identity']!=identity or m['parent_F0_identity']!=identity['parent_F0_identity'] or m['dura_identity']!=identity['dura_identity']):raise ValueError('Only the frozen original P1 signal r03 and matching F0/DURA are allowed, never the P0-trained model')
    with np.load(root/'models/density.npz',allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
    density.validate_state(saved,'density_ratio')
    if not evaluation._same(saved,state):raise ValueError('Original ninety/thirty-two-mode density must remain unchanged')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'} or z['conditional'].dtype!=np.float64 or z['conditional'].shape!=(2,6) or not np.array_equal(z['conditional'],tau):raise ValueError('Exactly the two original FP64 six-class thresholds must be reused')
    fit=_read(root/'models/complete.json');cal=_read(root/'calibration/complete.json')
    if (any(fit.get(k)!=v for k,v in dict(status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],models_frozen=True,models_refitted=False,thresholds_refitted=False).items())
            or fit.get('parent_calibration_identity')!=parent_cal or cal!=parent_cal):raise ValueError('The original source-star calibration object must be copied intact, without retau')
    rows={}
    for condition in ('source_raw','source_DURA'):
        rows[condition]={}
        for role in ('known','novel'):
            a=evaluation._source_rows(root,condition,role);_validate(a,COUNTS['V' if role=='known' else 'novel'],KNOWN_FIELDS if role=='known' else NOVEL_FIELDS,state,tau)
            for k,roster in (('rx_i',common.RECEIVERS),('day_i',(1,2,3))):
                if a[k].shape!=(len(a['ids']),) or a[k].dtype!=np.int64 or not np.isin(a[k],roster).all():raise ValueError('Aligned source public RX/day required')
            if role=='known' and (a['y'].shape!=(len(a['ids']),) or a['y'].dtype!=np.int64 or not np.isin(a['y'],range(6)).all() or a['scenario'].shape!=(len(a['ids']),) or a['scenario'].dtype.kind!='U' or not np.isin(a['scenario'],common.SCENES).all()):raise ValueError('Only public source-known labels and fixed scenes allowed')
            if condition=='source_DURA':
                with np.load(Path(c['reference_prediction_root'])/(role+'_predictions.npz'),allow_pickle=False) as z:
                    for k in a:
                        expected=z[k][:,:2] if k in ('scores','raw_scores') else z[k]
                        if not np.array_equal(a[k],expected):raise ValueError('Source-DURA fields and both pooled operating points must exactly equal the completed parent')
            rows[condition][role]=a
    for role in ('known','novel'):
        _align(rows['source_raw'][role],rows['source_DURA'][role],('ids','rx_i','day_i')+(('y','scenario') if role=='known' else ()))
    for name in CONDITIONS[2:]:
        rows[name]=evaluation._target_rows(root,name);_validate(rows[name],Q_COUNT,TARGET_FIELDS,state,tau)
    for group in ('clean','star'):_align(rows[group+'_raw'],rows[group+'_DURA'],('ids',))
    known_ids=set(rows['source_DURA']['known']['ids'].tolist());novel_ids=set(rows['source_DURA']['novel']['ids'].tolist())
    if known_ids&novel_ids or any((known_ids|novel_ids)&set(rows[n]['ids'].tolist()) for n in CONDITIONS[2:]):raise ValueError('Source fitting/calibration/heldout and future query roles must remain disjoint')
    return m,cal,rows


def _align(a,b,keys):
    if any(not np.array_equal(a[k],b[k]) for k in keys):raise ValueError('Raw and DURA must refer to the same ordered physical records and metadata')


def _validate(a,n,fields,state,tau):
    if set(a)!=fields or not common._ids(a['ids'],n):raise ValueError('Every complete public path needs exact fields and unique IDs; no private query labels')
    if (a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))):raise ValueError('Each path needs its actual original-F0 default prediction')
    if 'baseline_closed' in fields and (a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):raise ValueError('Known/query baselines must be the same-path original-F0 prediction')
    z=a['signal_features']
    if (z.shape!=(n,160) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)
            or any(a[k].shape!=(n,2) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):raise ValueError('Two finite pooled raw scores/margins and fixed signal unit160 required')
    known={k:state[k] for k in density.KNOWN_FIELDS};raw=np.column_stack((density.score(known,z,'known_only'),density.score(state,z,'density_ratio')))
    if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):raise ValueError('Both frozen raw density scores and path-specific class-routed margins must reconstruct')


def _confidence(row):
    return dict(level=.95,method='Per true-class Wilson on the historical development pool; not independent-RX confidence',
        A_K_per_class=_ci(row['known_per_class'],'correct_accepted'),C_K_per_class=_ci(row['known_per_class'],'accepted'),R_U_per_class=_ci(row['unknown_per_class'],'rejected'))


def _full_pass(row,minimum):return bool(_known_pass(row) and row['coverage']['unknown_complete'] and row['R_U'] is not None and row['R_U']>=minimum-1e-12)


def _source(data,truth,index):
    k,n=data['known'],data['novel'];km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],truth['scenario']));allk=np.ones(len(k['ids']),bool);alln=np.ones(len(n['ids']),bool)
    def metric(ki,ni,kr=tuple(range(6)),ur=common.UNKNOWN_TX):return _metrics(k['y'][ki],k['closed'][ki],k['scores'][ki,index],truth['tx'][ni],n['scores'][ni,index],k['baseline_closed'][ki],kr,ur)
    groups={name:metric(mask,nm[name]) for name,mask in km.items()};overall=groups['overall']
    return dict(overall=overall,groups=groups,by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},by_scene={s:groups['scene/'+s] for s in common.SCENES},
        by_day={str(d):metric(k['day_i']==d,n['day_i']==d) for d in (1,2,3)},by_TX={str(c):metric(k['y']==c,alln,(c,)) for c in range(6)}|{c:metric(allk,truth['tx']==c,ur=(c,)) for c in common.UNKNOWN_TX},
        known_12_groups_passed=all(_known_pass(groups[g]) for g in HARD_NAMES),known_42_groups_passed=all(_known_pass(r) for r in groups.values()),cross_guard_violations_or_missing=_cross_report(groups),
        source_unknown_90_diagnostic_passed=bool(overall['coverage']['unknown_complete'] and overall['R_U'] is not None and overall['R_U']>=.90-1e-12),confidence_intervals=_confidence(overall),
        source_RX_seen=True,source_novel_historically_exposed=True,unknown_is_diagnostic_not_target_entry_gate=True)


def _target(data,truth,condition,index):
    y=truth['y'];k=y<6;u=~k;clean=condition.startswith('clean');scenes=('clean',) if clean else common.SCENES;minimum=.95 if clean else .90
    def metric(mask,kr=tuple(range(6)),ur=TARGET_UNKNOWN):return _metrics(y[mask&k],data['closed'][mask&k],data['scores'][mask&k,index],y[mask&u],data['scores'][mask&u,index],data['baseline_closed'][mask&k],kr,ur)
    overall=metric(np.ones(len(y),bool));by_scene={s:metric(truth['scenario']==s) for s in scenes}
    return dict(overall=overall,by_scene=by_scene,by_RX={str(r):metric(truth['receiver']==r) for r in np.unique(truth['receiver'])},by_day={'2021_03_15':overall},
        by_TX={str(c):metric((y==c)|u,(c,)) for c in range(6)}|{str(c):metric(k|(y==c),ur=(c,)) for c in TARGET_UNKNOWN},confidence_intervals=_confidence(overall),
        goals=dict(unknown_minimum=minimum,overall_passed=_full_pass(overall,minimum),all_main_scenes_passed=all(_full_pass(r,minimum) for r in by_scene.values()),
            required_scene_coverage_complete=all(r['coverage']['complete'] for r in by_scene.values()),passed=bool(_full_pass(overall,minimum) and all(_full_pass(r,minimum) for r in by_scene.values()))),
        historically_exposed_development_RX=True,new_blind_confirmation=False)


def paired_counts(new,old,new_novel,old_novel,index,known_mask=None,unknown_mask=None):
    """DURA minus raw; identity correctness is recomputed on each actual path."""
    _align(new,old,('ids','y'));_align(new_novel,old_novel,('ids',))
    k=np.ones(len(new['y']),bool) if known_mask is None else known_mask;u=np.ones(len(new_novel['ids']),bool) if unknown_mask is None else unknown_mask
    accepted_new=new['scores'][:,index]>=0;accepted_old=old['scores'][:,index]>=0;correct_new=new['closed']==new['y'];correct_old=old['closed']==old['y']
    result=dict(known_count=int(k.sum()),unknown_count=int(u.sum()),known_closed_prediction_changed=int((k&(new['closed']!=old['closed'])).sum()))
    for name,a,b,mask in (('known_correct_accepted',correct_new&accepted_new,correct_old&accepted_old,k),('known_accepted',accepted_new,accepted_old,k),
            ('known_correct_rejected',correct_new&~accepted_new,correct_old&~accepted_old,k),('known_wrong_accepted',~correct_new&accepted_new,~correct_old&accepted_old,k),
            ('known_closed_correct',correct_new,correct_old,k),('unknown_rejected',new_novel['scores'][:,index]<0,old_novel['scores'][:,index]<0,u)):
        gain=int((mask&a&~b).sum());lost=int((mask&b&~a).sum());result.update({name+'_gained':gain,name+'_lost':lost,name+'_net':gain-lost})
    return result


def _delta(new,old):
    keys=('A_K','C_K','R_U','H_score','original_F0_macro_closed_accuracy','candidate_macro_closed_accuracy','original_closed_minus_new_open','candidate_closed_minus_open','auroc_known_macro','oscr_auc_macro')
    return {k:None if new[k] is None or old[k] is None else new[k]-old[k] for k in keys}


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable six-path independent score output required')
    m,cal,rows=_prediction(prediction_root)
    # All six paths verified before these three private reads. Each truth is
    # then shared by its exact same-ID raw/DURA pair, without further IO.
    nt=_source_truth(source_novel_truth,rows['source_DURA']['novel']);truths={base:_target_truth(path,rows[base+'_DURA']['ids']) for base,path in (('clean',clean_truth),('star',star_truth))}
    for base,t in truths.items():
        scenes=('clean',) if base=='clean' else common.SCENES
        if not np.isin(t['y'],range(9)).all() or not np.all(t['receiver']==26) or not np.isin(t['scenario'],scenes).all():raise ValueError('Fixed development RX24-1, condition and private scorer roster required')
    methods={};paired={}
    for j,name in enumerate(METHODS):
        paths={c:_source(rows[c],nt,j) for c in CONDITIONS[:2]};paths.update({c:_target(rows[c],truths[c.split('_')[0]],c,j) for c in CONDITIONS[2:]});methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY,
            parent_source_calibration_passed=bool(cal['methods'][m['reference_prediction_identity']['method_names'][j]]['constraints_passed']))
        pairs={}
        for base in ('source','clean','star'):
            if base=='source':
                new,old=rows['source_DURA']['known'],rows['source_raw']['known'];nn,on=rows['source_DURA']['novel'],rows['source_raw']['novel'];km=dict(common._masks(new['rx_i'],new['scenario']));nm=dict(common._masks(nn['rx_i'],nt['scenario']))
                groups={g:paired_counts(new,old,nn,on,j,mask,nm[g]) for g,mask in km.items()}
            else:
                y=truths[base]['y'];k=y<6;u=~k
                def known(a):return dict(ids=a['ids'][k],scores=a['scores'][k],y=y[k],closed=a['closed'][k])
                def unknown(a):return dict(ids=a['ids'][u],scores=a['scores'][u])
                new,old=known(rows[base+'_DURA']),known(rows[base+'_raw']);nn,on=unknown(rows[base+'_DURA']),unknown(rows[base+'_raw'])
                scenes=('clean',) if base=='clean' else common.SCENES;groups={s:paired_counts(new,old,nn,on,j,truths[base]['scenario'][k]==s,truths[base]['scenario'][u]==s) for s in scenes}
            pairs[base]=dict(counts=paired_counts(new,old,nn,on,j),groups=groups,delta=_delta(paths[base+'_DURA']['overall'],paths[base+'_raw']['overall']),
                by_known_TX={str(c):paired_counts(new,old,nn,on,j,new['y']==c) for c in range(6)},
                by_unknown_TX={str(c):paired_counts(new,old,nn,on,j,unknown_mask=(nt['tx']==c if base=='source' else truths[base]['y'][u]==c)) for c in (common.UNKNOWN_TX if base=='source' else TARGET_UNKNOWN)},
                identity_correctness='Each path uses its own original F0 argmax and same-path baseline; six frozen source-star thresholds are unchanged')
        paired[name]=pairs
    main=methods[PRIMARY]['paths'];passed=all(main[c]['goals']['passed'] for c in ('clean_DURA','star_DURA'))
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,
        primary_summary={c:main[c]['overall'] for c in CONDITIONS},paired_DURA_minus_raw=paired,development_gate_passed=bool(passed),eligible_primary=bool(passed),
        next_step='DEVELOPMENT_PASSED_FOR_SEPARATE_UDA_PREREGISTRATION' if passed else 'NO_PROMOTION',truth_last=True,all_six_predictions_before_truth=True,source_truth_reads=1,target_truth_reads=2,
        primary_paths=['source_DURA','clean_DURA','star_DURA'],raw_paths_diagnostic_only=True,source_unknown_diagnostic_only=True,shared_frozen_model_and_thresholds=True,
        classification_path='per_input_original_F0',identity_baseline_path='same_path_original_F0',neural_updates=0,density_fit_updates=0,threshold_fit_updates=0,
        source_bank_deployed=False,per_record_source_rows_deployed=0,formal_target_accessed=False,target_U_reads=0,target_DA=False,target_REG=False,A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False,full_goal_passed=False,
        original_F0_parameters=232993,signal_parameters=334721,existing_restorer_parameters=266951,combined_active_parameters=834665,
        costs={k:m[k] for k in ('real_F_forward_rows','real_signal_forward_rows','G_forward_rows','synthetic_F_forward_rows','synthetic_signal_forward_rows','G_synthetic_forward_rows')},
        inherited_training_cost_scope='Original F0 E50/11100, P1 signal r03 feature11100+2220 and MLP2220, and DURA training remain inherited; new training/statistics/threshold fitting are zero',
        scope='Source seen-RX historical-novel and one exposed development RX; raw/DURA same-ID comparison is not new physical data or full open-world confirmation')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_DURA_minus_raw')});print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
