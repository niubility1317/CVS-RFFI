"""Independent U-only anonymous matching and frozen future registration scores."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from . import star_signal_register_run as evaluation
from . import star_signal_register_core as core
from . import star_class_density_score as parent
from . import unlabeled_registration_score as prior
from .metrics import open_set_metrics,open_world_h,_harmonic
from .scoring import _truth,wilson95

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_signal_register_scores_v1'
STATUS='SIGNAL_REGISTER_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=METHODS[3]
CONDITIONS=('source','clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
U_COUNT=384
OLD=tuple(range(6))
UNKNOWN=(6,7,8)  # Scorer-only true roster, never used by the predictor.
common=parent.common
_read,_write,_source_truth=common._read,common._write,parent._source_truth
EXTRA={'predictions','decision_scores','anonymous_raw_scores'}
U_FIELDS={'ids','default_logits','closed','signal_features','base_scores','assignments','dbscan_labels','rejected'}


def _load(path,fields=None):
    with np.load(path,allow_pickle=False) as z:
        if fields is not None and set(z.files)!=set(fields):raise ValueError('Exact public prediction/state schema required; no private labels')
        return {k:z[k].copy() for k in z.files}


def _require(value,expected,message):
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError(message)


def _head(a,n):
    if (not common._ids(a['ids'],n) or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32
            or not np.isfinite(a['default_logits']).all() or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64
            or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
            or a['signal_features'].shape!=(n,160) or a['signal_features'].dtype!=np.float32 or not np.isfinite(a['signal_features']).all()
            or not np.allclose(np.linalg.norm(a['signal_features'].astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)):
        raise ValueError('Original frozen F0 default identity and same-P1 unit160 must remain intact')
    if 'baseline_closed' in a and (a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):
        raise ValueError('Baseline must be the same DURA output original F0, never the registered winner')


def _reconstruct(a,state,anonymous,tau,registration_tau):
    n=len(a['ids']);K=len(anonymous['means']);_head(a,n)
    for key,shape,dtype in (('raw_scores',(n,2),np.float64),('scores',(n,2),np.float64),
            ('predictions',(n,4),np.int64),('decision_scores',(n,4),np.float64),('anonymous_raw_scores',(n,K),np.float64)):
        if a[key].shape!=shape or a[key].dtype!=dtype or not np.isfinite(a[key]).all():raise ValueError('All four complete frozen registration decisions required')
    raw=evaluation.parent.core.raw_scores(state,a['signal_features'],a['closed'])[:,[1,3]]
    if (not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9)
            or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T)):
        raise ValueError('Original selected raw scores and six predicted-old-class thresholds must reconstruct')
    new=core.anonymous_ratios(state,anonymous,a['signal_features'])
    if not np.allclose(new,a['anonymous_raw_scores'],rtol=1e-12,atol=1e-9):raise ValueError('Frozen anonymous Gaussian evidence differs')
    predictions,decision=core.predict(a['closed'],a['scores'],a['anonymous_raw_scores'],registration_tau)
    if not np.array_equal(a['predictions'],predictions) or not np.array_equal(a['decision_scores'],decision):
        raise ValueError('Old and anonymous classes must compete through the actual frozen joint decision')


def _U_state(U,state,source,anonymous,tau):
    _head(U,U_COUNT);K=len(anonymous['means']);labels=U['dbscan_labels'];assigned=U['assignments']
    if (labels.shape!=(U_COUNT,) or labels.dtype!=np.int64 or np.any(labels< -1)
            or assigned.shape!=(U_COUNT,) or assigned.dtype!=np.int64 or np.any((assigned< -1)|(assigned>=K))
            or U['base_scores'].shape!=(U_COUNT,2) or U['base_scores'].dtype!=np.float64 or not np.isfinite(U['base_scores']).all()
            or U['rejected'].shape!=(U_COUNT,) or U['rejected'].dtype!=np.bool_
            or not np.array_equal(U['rejected'],U['base_scores'][:,1]<0)):
        raise ValueError('Full unlabeled U partition, predicted-rejection majority and anonymous assignments required')
    _,margin=evaluation._base(state,U['signal_features'],U['closed'],tau)
    if not np.allclose(margin,U['base_scores'],rtol=1e-12,atol=1e-9):raise ValueError('U rejection rule must use the original predicted margin')
    white=(evaluation.parent.core.density.unit(U['signal_features'])-state['origin'])@state['whitener']
    selected=[]
    for label in np.unique(labels[labels>=0]):
        mask=labels==label;n=int(mask.sum());eligible=n>=16 and int(U['rejected'][mask].sum())*2>n
        slots=np.unique(assigned[mask])
        if len(slots)!=1 or (eligible and slots[0]<0) or (not eligible and slots[0]!=-1):
            raise ValueError('Register only complete DBSCAN clusters of size sixteen and strict rejected majority')
        if eligible:
            j=int(slots[0]);selected.append(j);x=white[mask];mean=x.mean(0)
            variance=max(n/(n+160)*float(np.mean(np.sum((x-mean)**2,axis=1)))/160+160/(n+160)*float(state['pooled_L_variance']),1e-8)
            if not np.allclose(anonymous['means'][j],mean,rtol=1e-12,atol=1e-9) or not np.isclose(anonymous['variances'][j],variance,rtol=1e-12,atol=1e-9):
                raise ValueError('Anonymous Gaussian must use all assigned U members and the frozen source shrinkage')
    if sorted(selected)!=list(range(K)) or np.any(assigned[labels<0]!=-1):raise ValueError('Anonymous states cannot omit, split or invent U clusters')


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    _require(m,dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,source_radius_frozen_before_U=True,anonymous_frozen_before_V_and_Q=True,
        all_thresholds_frozen_before_novel_and_Q=True,all_three_conditions_complete=True,all_conditions_through_DURA=True,
        original_F0_default_identity_unchanged=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,REG0_controls_reused_exactly=True,
        original_density_refitted=False,source_radius_fitted=True,registration_threshold_sets_fitted=2,neural_updates=0,source_OE_reads=0,
        original_U_s_reads=0,private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,target_Q_IQ_reads=0,
        formal_target_accessed=False,source_bank_deployed=False,U_bank_deployed=False,per_record_source_rows_deployed=0,
        target_DA=False,target_REG=True,final_phase3_complete=False), 'All frozen four-method/three-condition predictions must precede any truth')
    c=evaluation.validate_config(m['config'])
    if any(c[k]!=m[k] for k in ('run_id','code_commit')):raise ValueError('Execution/config identity differs')
    pm,state,tau,parent_cal=evaluation._reference(c);identity=pm['model_identity']
    if (m['reference_prediction_identity']!=pm or m['model_identity']!=identity or m['parent_F0_identity']!=identity['parent_F0_identity']
            or m['dura_identity']!=identity['dura_identity']):raise ValueError('Original source r03/F0/DURA ancestry must remain fixed')
    stored=_load(root/'models/density.npz');evaluation.parent.core.density.validate_state(stored,'density_ratio')
    if not evaluation._same(stored,state):raise ValueError('The original source density cannot be refitted or changed')
    source=_load(root/'models/source_radius.npz');anonymous=_load(root/'models/anonymous.npz');core.validate_source(source);core.validate_anonymous(anonymous)
    K=len(anonymous['means'])
    fit=_read(root/'models/complete.json');reg=_read(root/'registration/complete.json')
    _require(fit,dict(schema=evaluation.SCHEMA,status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],model_identity=identity,
        source_L_count=COUNTS['L'],source_radius_fitted=True,source_radius_frozen_before_U=True,original_density_refitted=False,source_V_fit=False,target_U_reads=0),'Source-only radius must freeze before U')
    _require(reg,dict(schema=evaluation.SCHEMA,status=evaluation.REGISTRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],U_count=U_COUNT,
        U_manifest=m['U_manifest'],anonymous_count=K,anonymous_ids=anonymous['anonymous_ids'].tolist(),anonymous_frozen_before_V_and_Q=True,
        true_U_labels_used=False,true_unknown_count_used=False,query_fit=False,source_prediction_reads=0,source_radius_frozen_before_U=True,U_bank_deployed=False),'Anonymous U state must freeze before V or Q')
    if m['anonymous_count']!=K:raise ValueError('Actual anonymous count differs from completed predictions')
    thresholds=_load(root/'thresholds.npz',{'conditional','registration'});registration_tau=thresholds['registration']
    if (not np.array_equal(thresholds['conditional'],tau) or thresholds['conditional'].dtype!=np.float64
            or registration_tau.shape!=(2,) or registration_tau.dtype!=np.float64 or not np.isfinite(registration_tau).all()):raise ValueError('Original two six-threshold controls and finite registration thresholds required')
    cal=_read(root/'calibration/complete.json')
    _require(cal,dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],
        method_names=list(METHODS),primary_method=PRIMARY,conditional_thresholds=tau.tolist(),registration_thresholds=registration_tau.tolist(),
        known_count=COUNTS['V'],known_only=True,unknown_inputs_used=False,target_label_reads=0,U_state_frozen=True,
        source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,shared_thresholds_for_all_conditions=True),'Only frozen-U, source-V known calibration is legal')
    report=cal['report']
    if set(report['methods'])!=set(METHODS[2:]) or report.get('hard_names')!=list(evaluation.parent.risk.HARD_NAMES):raise ValueError('Registration calibration must report the fixed known twelve groups')
    for j,name in enumerate(METHODS[2:]):
        if set(report['methods'][name]['groups'])!=set(common.GROUP_NAMES) or report['methods'][name]['selected_threshold']!=float(registration_tau[j]):
            raise ValueError('All forty-two known risk diagnostics and the selected registration threshold required')
    U=_load(root/'registration/U.npz',U_FIELDS);_U_state(U,state,source,anonymous,tau)
    rows={};p=c['reference_config'];seen=set(U['ids'].tolist())
    for name,role in (('known','known'),('novel','novel'),('clean_DURA',None),('star_DURA',None)):
        old=evaluation._rows(p['output_root'],role) if role else evaluation._target_rows(p['output_root'],name)
        path=root/(role+'_predictions.npz') if role else root/'targets'/name/'predictions.npz'
        a=_load(path,set(old)|EXTRA);n=COUNTS['V'] if role=='known' else COUNTS['novel'] if role=='novel' else Q_COUNT
        _head(a,n)
        for key,value in old.items():
            expected=value[:,[1,3]] if key in ('raw_scores','scores') else value
            if not np.array_equal(a[key],expected):raise ValueError('REG0, same-DURA identity, representations and physical roles must exactly reuse the parent')
        _reconstruct(a,state,anonymous,tau,registration_tau)
        ids=set(a['ids'].tolist())
        # Clean/star are two realizations of the same Q physical role; both
        # must be disjoint from U and all source roles.
        if ids&seen:raise ValueError('U/source/future-Q physical roles cannot overlap')
        if role:seen.update(ids)
        rows[name]=a
    return m,cal,rows,U,anonymous


def _mean(rows,key):
    return None if not rows or any(r['count']==0 for r in rows) else float(np.mean([r[key] for r in rows]))


def _rejection(y,raw,labels):
    rows=[dict(label=int(c),count=int(np.sum(y==c)),rejected=int(np.sum((y==c)&(raw==-1)))) for c in labels]
    for r in rows:r['rejection']=r['rejected']/r['count'] if r['count'] else None
    return _mean(rows,'rejection'),rows


def _metrics(y,raw,mapped,baseline,evidence,closed,registered,arrivals,later,actual):
    old=prior._class_rows(y,raw,mapped,baseline,OLD,missing_zero=False)
    new=prior._class_rows(y,raw,mapped,baseline,arrivals,missing_zero=False) if registered else []
    intended=old+new if registered else []
    A,C,original=(_mean(old,k) for k in ('open_accuracy','acceptance','original_F0_closed_accuracy'))
    AN=_mean(new,'open_accuracy') if registered else None
    RU,unknown=_rejection(y,raw,later if registered else UNKNOWN)
    lr,lrows=_rejection(y,raw,later);ar,arows=_rejection(y,raw,actual);allru,allrows=_rejection(y,raw,UNKNOWN)
    complete=lambda labels:bool(labels) and all(np.any(y==c) for c in labels)
    coverage=dict(known_complete=complete(OLD),unknown_complete=complete(UNKNOWN),arrival_complete=complete(arrivals),
        later_complete=complete(later),registered_complete=complete(OLD+tuple(arrivals)))
    coverage['complete']=coverage['known_complete'] and coverage['unknown_complete']
    auc=oscr=None
    if not registered:
        k=y<6;u=~k;r=open_set_metrics(y[k],closed[k],evidence[k],y[u],evidence[u],0.,OLD,UNKNOWN,baseline_prediction=baseline[k])
        auc,oscr=r['auroc_known_macro'],r['oscr_auc_macro']
    confidence=lambda rows,key:[dict(label=r['label'],count=r['count'],successes=r[key],interval95=wilson95(r[key],r['count'])) for r in rows]
    return dict(count=len(y),coverage=coverage,A_K=A,C_K=C,R_U=RU,H_score=_harmonic((A,RU)),A_N=AN,
        H_KN=_harmonic((A,AN)) if registered else None,H_OW=open_world_h(A,AN,lr) if registered else None,
        original_F0_macro_closed_accuracy=original,original_closed_minus_new_open=None if original is None or A is None else original-A,
        known_macro_open_accuracy=A,known_macro_acceptance=C,known_macro_frr=None if C is None else 1-C,
        known_per_class=old,unknown_per_class=unknown,new_per_class=new if registered else None,
        old_count=int(np.sum(y<6)),old_correct_accepted=int(np.sum((y<6)&(mapped==y))),old_accepted=int(np.sum((y<6)&(raw!=-1))),
        old_rejected=int(np.sum((y<6)&(raw==-1))),original_F0_closed_correct=int(np.sum((y<6)&(baseline==y))),
        arrival_count=int(np.isin(y,arrivals).sum()),arrival_correct_accepted=int(np.sum(np.isin(y,arrivals)&(mapped==y))) if registered else None,
        registered_macro_accuracy=_mean(intended,'open_accuracy') if registered else None,registered_macro_acceptance=_mean(intended,'acceptance') if registered else None,
        registered_macro_FRR=_mean(intended,'frr') if registered else None,registered_per_class=intended if registered else None,
        new_macro_denominator=len(arrivals) if registered else None,R_U_labels=list(later if registered else UNKNOWN),
        R_U_scope='planned later outside all true U arrivals' if registered else 'all initially unregistered old-external labels',
        planned_later_macro_rejection=lr,planned_later_per_class=lrows,actual_unregistered_macro_rejection=ar,actual_unregistered_per_class=arows,
        all_nonold_macro_rejection=allru,all_nonold_per_class=allrows,unmatched_anonymous_accepted=int(np.sum(mapped==-2)),
        auroc_known_macro=auc,oscr_auc_macro=oscr,REG1_curves_definition=None if registered else 'Original old head and actual REG0 margin',
        confidence_intervals=dict(level=.95,method='Per true-class Wilson; one exposed development RX, no independent-RX macro interval',
            A_K_per_class=confidence(old,'correct_accepted'),C_K_per_class=confidence(old,'accepted'),R_U_per_class=confidence(unknown,'rejected'),
            A_N_per_class=confidence(new,'correct_accepted') if registered else None))


def _old_ok(r):
    return bool(r['coverage']['known_complete'] and r['C_K'] is not None and r['C_K']>=.90-1e-12
        and r['original_closed_minus_new_open'] is not None and r['original_closed_minus_new_open']<=.02+1e-12)


def _checks(row,minimum,*,overall):
    checks=dict(complete_required_rosters=row['coverage']['complete'],old_CK90_and_same_F0_protection=_old_ok(row),
        old_loss_from_REG0_at_most_2pp=row['old_loss_from_REG0'] is not None and row['old_loss_from_REG0']<=.02+1e-12,
        registered_CK90=row['registered_macro_acceptance'] is not None and row['registered_macro_acceptance']>=.90-1e-12,
        later_RU=row['R_U'] is not None and row['R_U']>=minimum-1e-12,
        arrival_coverage=row['coverage']['arrival_complete'],later_coverage=row['coverage']['later_complete'])
    if overall:checks['arrival_accuracy80']=row['A_N'] is not None and row['A_N']>=.80-1e-12
    return checks


def _condition(a,y,rx,scenes,days,index,mapping,arrivals,later,condition):
    registered=index>=2;raw=a['predictions'][:,index];mapped=prior._mapped(raw,mapping) if registered else raw.copy()
    actual=sorted(set(UNKNOWN)-(set(mapping.values()) if registered else set()))
    baseline=a.get('baseline_closed',a['closed'])
    def metric(mask):return _metrics(y[mask],raw[mask],mapped[mask],baseline[mask],a['decision_scores'][mask,index],a['closed'][mask],registered,arrivals,later,actual)
    masks=dict(common._masks(rx,scenes)) if condition=='source' else dict(overall=np.ones(len(y),bool),**{'scene/'+s:scenes==s for s in (('clean',) if condition=='clean_DURA' else common.SCENES)})
    groups={g:metric(mask) for g,mask in masks.items()};overall=groups['overall']
    bytx={str(r['label']):r for r in overall['known_per_class']+overall['all_nonold_per_class']}
    if registered:
        for r in overall['new_per_class']:bytx[str(r['label'])]=dict(bytx[str(r['label'])],**r)
    return dict(overall=overall,groups=groups,by_scene={g[6:]:r for g,r in groups.items() if g.startswith('scene/')},
        by_RX={str(r):metric(rx==r) for r in np.unique(rx)},by_day={str(d):metric(days==d) for d in np.unique(days)},by_TX=bytx,
        registered=registered,matched_arrival_labels=sorted(mapping.values()) if registered else [],actual_unregistered_labels=actual,
        new_denominator='All true U arrival classes, including undiscovered/unmatched; missing query coverage is N/A',
        unmatched_anonymous_rule='Accepted wrong (-2 after mapping); only raw -1 rejects')


def _counts(y,left,right,lmapped,rmapped,arrivals,later):
    result={}
    for name,cohort,le,re in (
        ('old_correct_accept',y<6,lmapped==y,rmapped==y),('old_accept',y<6,left!=-1,right!=-1),
        ('arrival_correct_accept',np.isin(y,arrivals),lmapped==y,rmapped==y),
        ('later_reject',np.isin(y,later),left==-1,right==-1),('all_nonold_reject',y>=6,left==-1,right==-1)):
        result[name+'_gain']=int(np.sum(cohort&le&~re));result[name+'_loss']=int(np.sum(cohort&re&~le))
        result[name+'_net']=int(np.sum(cohort&le)-np.sum(cohort&re))
    return result


def _difference(a,b):
    fields=('A_K','C_K','A_N','H_KN','H_OW','registered_macro_accuracy','registered_macro_acceptance','registered_macro_FRR','planned_later_macro_rejection')
    result={k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in fields}
    same=a['R_U_labels']==b['R_U_labels']
    result.update(R_U=None if not same or a['R_U'] is None or b['R_U'] is None else a['R_U']-b['R_U'],
        H_score=None if not same or a['H_score'] is None or b['H_score'] is None else a['H_score']-b['H_score'],same_primary_unknown_denominator=same,
        R_U_rosters=dict(left=a['R_U_labels'],right=b['R_U_labels']))
    return result


def score(prediction_root,U_truth,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent registration score root required')
    m,cal,rows,U,anonymous=_prediction(prediction_root)
    # Every public prediction/state/threshold above passes before the first
    # private join. No Q labels can enter anonymous matching.
    ut=_truth(U_truth,U['ids'])
    if not np.isin(ut['y'],OLD+UNKNOWN).all() or not np.all(ut['receiver']==26) or not np.isin(ut['scenario'],common.SCENES).all():raise ValueError('Only development RX24-1 U truth is allowed for scorer-side matching')
    arrivals=sorted(int(c) for c in np.unique(ut['y']) if c>=6);later=sorted(set(UNKNOWN)-set(arrivals))
    mapping,clusters=prior._matching(ut['y'],U['assignments'],U['assignments']>=0,len(anonymous['means']),arrivals)
    nt=_source_truth(source_novel_truth,rows['novel'])
    truths={name:_truth(path,rows[name]['ids']) for name,path in zip(CONDITIONS[1:],(clean_truth,star_truth))}
    for name,t in truths.items():
        scenes=('clean',) if name=='clean_DURA' else common.SCENES
        if not np.isin(t['y'],OLD+UNKNOWN).all() or not np.all(t['receiver']==26) or not np.isin(t['scenario'],scenes).all():raise ValueError('Only the fixed development future-Q roles and true roster may score')
    k,n=rows['known'],rows['novel'];source={key:np.concatenate((k[key],n[key])) for key in ('ids','closed','predictions','decision_scores')}
    source['baseline_closed']=np.concatenate((k['baseline_closed'],n['closed']))
    sy=np.concatenate((k['y'],np.asarray([6+common.UNKNOWN_TX.index(str(tx)) for tx in nt['tx']],np.int64)))
    data=dict(source=source,**{c:rows[c] for c in CONDITIONS[1:]})
    labels=dict(source=(sy,np.r_[k['rx_i'],n['rx_i']],np.r_[k['scenario'],nt['scenario']],np.r_[k['day_i'],n['day_i']]))
    for c,t in truths.items():labels[c]=(t['y'],t['receiver'],t['scenario'],np.full(len(t['y']),'2021-03-15'))
    methods={name:{c:_condition(data[c],*labels[c],j,mapping,arrivals,later,c) for c in CONDITIONS} for j,name in enumerate(METHODS)}
    for j in (2,3):
        for condition in CONDITIONS:
            current=methods[METHODS[j]][condition];previous=methods[METHODS[j-2]][condition]
            for name,row in current['groups'].items():
                a,b=row['A_K'],previous['groups'][name]['A_K'];row['old_loss_from_REG0']=None if a is None or b is None else b-a
            if condition=='source':
                current['known_12_groups_passed']=all(_old_ok(current['groups'][g]) for g in evaluation.parent.risk.HARD_NAMES)
                current['known_42_groups_passed']=all(_old_ok(r) for r in current['groups'].values())
                current['source_calibration_passed']=bool(cal['report']['methods'][METHODS[j]]['constraints_passed'])
            else:
                minimum=.95 if condition=='clean_DURA' else .90;overall=_checks(current['overall'],minimum,overall=True)
                scenes={s:_checks(row,minimum,overall=False) for s,row in current['by_scene'].items()}
                current['goals']=dict(overall=overall,by_scene=scenes,unknown_minimum=minimum,scene_AN80_required=False,
                    passed=all(overall.values()) and all(all(checks.values()) for checks in scenes.values()))
    paired={}
    for name,left,right in (('pooled_REG1_minus_REG0',2,0),('predicted_REG1_minus_REG0',3,1),('predicted_REG1_minus_pooled_REG1',3,2)):
        result={}
        for c in CONDITIONS:
            a=data[c];y,rx,scenes,days=labels[c];lp=a['predictions'][:,left];rp=a['predictions'][:,right]
            lm=prior._mapped(lp,mapping) if left>=2 else lp;rm=prior._mapped(rp,mapping) if right>=2 else rp
            masks=dict(common._masks(rx,scenes)) if c=='source' else dict(overall=np.ones(len(y),bool),**{'scene/'+s:scenes==s for s in methods[PRIMARY][c]['by_scene']})
            result[c]=dict(counts=_counts(y,lp,rp,lm,rm,arrivals,later),delta=_difference(methods[METHODS[left]][c]['overall'],methods[METHODS[right]][c]['overall']),
                groups={g:_counts(y[mask],lp[mask],rp[mask],lm[mask],rm[mask],arrivals,later) for g,mask in masks.items()},
                fixed_cohort_rule='later rejection compared on the same U-derived roster; REG0 and REG1 main RU/H have different denominators')
        paired[name]=dict(left=METHODS[left],right=METHODS[right],conditions=result)
    main=methods[PRIMARY];source_pass=main['source']['known_12_groups_passed'] and main['source']['source_calibration_passed']
    passed=bool(source_pass and all(main[c]['goals']['passed'] for c in CONDITIONS[1:]))
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,
        methods=methods,primary_summary={c:main[c]['overall'] for c in CONDITIONS},paired=paired,cluster_diagnostics=clusters,
        U_true_arrival_labels=arrivals,planned_later_labels=later,U_mapping_frozen_before_source_and_Q_truth=True,Q_truth_used_for_cluster_mapping=False,
        source_known_12_passed=bool(source_pass),registration_success=passed,development_gate_passed=passed,eligible_primary=passed,
        next_step='DEVELOPMENT_REGISTRATION_PASSED_FOR_SEPARATE_CONFIRMATION' if passed else 'NO_PROMOTION',
        truth_last=True,all_three_predictions_before_truth=True,U_truth_reads=1,source_novel_truth_reads=1,target_truth_reads=2,
        target_DA=False,target_REG=True,target_U_reads=m['target_U_reads'],neural_updates=0,registration_threshold_sets_fitted=2,source_radius_fitted=True,
        formal_target_accessed=False,full_goal_status='ACTIVE',final_phase3_complete=False,full_goal_passed=False,
        costs={key:m[key] for key in ('real_signal_forward_rows','real_F_forward_rows','G_forward_rows','synthetic_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows')},
        metrics_scope='H_score=harmonic(old AK,later RU) after REG; H_KN and H_OW separately. REG0 AN/H_KN/H_OW are N/A.',
        scope='Source V is reused calibration; source RX/novel and one future development RX historically exposed; no formal or complete Phase3 claim',
        source_bank_deployed=False,U_bank_deployed=False,per_record_source_rows_deployed=0)
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired','cluster_diagnostics')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--U-truth',required=True)
    p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.U_truth,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
