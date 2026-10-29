"""Independent U-only matching and four-state future-Q neighborhood scores.

All state/array checks precede private truth. U truth fixes the anonymous
mapping before Q truth is opened. Curves use the current DA/head or joint
winner, never the unchanged original-F0 baseline as identity correctness.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import unlabeled_registration_score as prior
from .metrics import open_set_metrics, open_world_h
from .scoring import _truth

RUN='20261029-phase3-star-neighborhood-s2026092701-r01'
SCHEMA='pamf_phase3_star_neighborhood_scores_v1'
STATUS='STAR_NEIGHBORHOOD_SCORES_COMPLETE'
STATES=('DA0_REG0','DA1_REG0','DA0_REG1','DA1_REG1')
PRIMARY='DA1_REG1'
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
UNKNOWN_LABELS=(6,7,8)  # Independent scorer-only roster; never passed to fit/predict.
OLD_LABELS=tuple(range(6))


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def _prediction_gate(root):
    from .star_neighborhood_run import _predictions
    return _predictions(root)


def _targets(path):
    with Path(path).open(encoding='utf-8-sig') as stream:spec=json.load(stream)
    if (spec.get('schema')!='pamf_phase3_scientific_targets_v3'
            or spec.get('known_macro_acceptance',{}).get('minimum')!=.90
            or spec.get('unknown_macro_rejection_min',{}).get('practical_cross_receiver')!=.90
            or spec.get('known_open_accuracy_drop_from_same_input_original_closed_max')!=.02
            or spec.get('anonymous_registration',{}).get('future_independent_new_class_macro_accuracy_min')!=.8
            or spec['anonymous_registration'].get('old_class_drop_from_pre_registration_max')!=.02
            or spec.get('unlabeled_adaptation',{}).get('known_open_accuracy_gain_required') is not True
            or spec.get('each_preregistered_main_scene_required') is not True):
        raise ValueError('Current v3 CK90/starRU90/F0loss2pp/new80/retention2pp targets required')
    return spec


def _gate(done,p,diag,state):
    if (done.get('status')!='STAR_NEIGHBORHOOD_PREDICTIONS_COMPLETE'
            or done.get('schema')!='pamf_phase3_star_neighborhood_target_v1' or done.get('run_id')!=RUN
            or done.get('condition')!='P1' or done.get('view_family')!='practical_cross_receiver'
            or done.get('classification_path')!='P1' or done.get('rejection_input_paths')!=['P1','P0']
            or done.get('all_new_IQ_satellite') is not True or done.get('truth_read') is not False
            or done.get('query_fit') is not False or done.get('registry_frozen_before_Q') is not True
            or done.get('states')!=list(STATES) or done.get('U_day')!='2021-03-08' or done.get('Q_day')!='2021-03-15'):
        raise ValueError('All four frozen, truth-blind star P1 states must complete before truth')
    ids,U_ids=p['ids'],p['U_ids'];n=len(ids);nu=len(U_ids)
    if (any(a.ndim!=1 or a.dtype.kind!='U' or len(set(a.tolist()))!=len(a) or not len(a) for a in (ids,U_ids))
            or set(ids.tolist())&set(U_ids.tolist()) or not np.array_equal(diag['ids'],U_ids)
            or done.get('count')!=n or done.get('U_count')!=nu or p['state_names'].tolist()!=list(STATES)):
        raise ValueError('Disjoint unique Q/U IDs, aligned U diagnostics and four-state counts required')
    counts=done.get('prototype_counts',{})
    if set(counts)!={'DA0','DA1'} or type(counts['DA0']) is not int or counts['DA0']<0 or counts['DA0']!=counts['DA1']:
        raise ValueError('The same frozen U anonymous registry, including K=0, must serve both DA states')
    K=counts['DA0'];closed=p['closed_prediction'];baseline=p['baseline_closed_prediction']
    if (closed.shape!=(2,n) or closed.dtype.kind not in 'iu' or np.any((closed<0)|(closed>=6))
            or baseline.shape!=(n,) or baseline.dtype.kind not in 'iu' or np.any((baseline<0)|(baseline>=6))
            or p['knownness'].shape!=(4,n) or p['knownness'].dtype.kind!='f' or not np.isfinite(p['knownness']).all()
            or p['prediction'].shape!=(4,n) or p['prediction'].dtype.kind not in 'iu'):
        raise ValueError('Current DA six-head choices, original P1 baseline and four finite evidence/decision rows required')
    for i,name in enumerate(STATES):
        joint=p['joint_margins_'+name];width=6+(K if i>=2 else 0)
        if joint.shape!=(n,width) or joint.dtype.kind!='f' or not np.isfinite(joint).all():
            raise ValueError('Complete old-six plus frozen-anonymous joint margins required')
        evidence=joint.max(1);winner=joint.argmax(1)
        if not np.array_equal(p['knownness'][i],evidence) or not np.array_equal(p['prediction'][i],np.where(evidence>=0,winner,-1)):
            raise ValueError('Every state must use its own joint maximum, intact zero ties and actual winner')
        if i<2 and not np.array_equal(closed[i],winner):
            raise ValueError('REG0 current DA closed prediction must match its own head/joint scores')
    assignment=diag['assignments'];candidate=diag['candidate_mask'];labels=diag['cluster_labels']
    C=len(diag['cluster_sizes'])
    labels_valid=(np.all(labels==-1) and K==0) if C==0 else (
        not np.any((labels<0)|(labels>=C)) and np.array_equal(np.bincount(labels,minlength=C),diag['cluster_sizes']))
    if (assignment.shape!=(nu,) or assignment.dtype.kind not in 'iu' or np.any((assignment<-1)|(assignment>=K))
            or candidate.shape!=(nu,) or candidate.dtype.kind!='b' or labels.shape!=(nu,) or labels.dtype.kind not in 'iu'
            or not labels_valid
            or diag['anonymous_ids'].shape!=(K,) or diag['anonymous_ids'].dtype.kind!='U'
            or len(set(diag['anonymous_ids'].tolist()))!=K or state['cluster_embedding'].shape!=(C,688)):
        raise ValueError('Complete aligned unlabeled partition and arbitrary-K registry diagnostics required')
    return K


def _mean(rows,key,complete):
    if not rows or (complete and any(row['count']==0 for row in rows)):return None
    values=[row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def _rejection(y,raw,labels,complete):
    rows=[]
    for label in labels:
        m=y==label;n=int(m.sum());rejected=int(np.sum(m&(raw==-1)))
        rows.append(dict(label=int(label),count=n,rejected=rejected,rejection=rejected/n if n else None))
    return _mean(rows,'rejection',complete),rows


def _coverage(y,arrivals,later,registered):
    present=set(int(label) for label in np.unique(y))
    covered=lambda labels:bool(labels) and set(labels).issubset(present)
    old,all_unknown,new,future=(covered(v) for v in (OLD_LABELS,UNKNOWN_LABELS,arrivals,later))
    return dict(old_classes=old,unknown_classes=all_unknown,new_arrivals=new,planned_later=future,
        R_U_classes=covered(later if registered else UNKNOWN_LABELS),intended_registered=old and new,
        complete=old and all_unknown and (new and future if registered else True),
        missing_old_labels=sorted(set(OLD_LABELS)-present),missing_unknown_labels=sorted(set(UNKNOWN_LABELS)-present),
        missing_arrival_labels=sorted(set(arrivals)-present),missing_later_labels=sorted(set(later)-present))


def _wilson(rows,success_key):
    result=[];z=1.959963984540054
    for row in rows:
        n=row['count'];k=row[success_key]
        if n:
            p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
            lo,hi=float(max(0.,c-h)),float(min(1.,c+h))
        else:lo=hi=None
        result.append(dict(label=row['label'],count=n,successes=k,lower=lo,upper=hi))
    return result


def _metrics(y,raw,mapped,baseline,evidence,current_choice,registered,arrivals,later,actual_unregistered,*,complete=True):
    present=set(int(label) for label in np.unique(y))
    roster=lambda labels:list(labels) if complete else sorted(present&set(labels))
    old=prior._class_rows(y,raw,mapped,baseline,roster(OLD_LABELS),missing_zero=False)
    new=prior._class_rows(y,raw,mapped,baseline,roster(arrivals),missing_zero=False) if registered else []
    intended=prior._class_rows(y,raw,mapped,baseline,roster(list(OLD_LABELS)+arrivals),missing_zero=False) if registered else []
    A,C,frr=(_mean(old,key,complete) for key in ('open_accuracy','acceptance','frr'))
    original=_mean(old,'original_F0_closed_accuracy',complete);AN=_mean(new,'open_accuracy',complete) if registered else None
    ru_labels=roster(later if registered else UNKNOWN_LABELS);RU,ru_rows=_rejection(y,raw,ru_labels,complete)
    planned,later_rows=_rejection(y,raw,roster(later),complete)
    actual,actual_rows=_rejection(y,raw,roster(actual_unregistered),complete)
    all_nonold,all_rows=_rejection(y,raw,roster(UNKNOWN_LABELS),complete)
    coverage=_coverage(y,arrivals,later,registered)
    if not complete:coverage['complete']=False
    harmonic=lambda a,b:None if a is None or b is None else 0. if a+b==0 else 2*a*b/(a+b)
    H=harmonic(A,RU);H_old_new=harmonic(A,AN) if registered else None
    if complete and not coverage['complete']:H=H_old_new=None
    H_OW=open_world_h(A,AN,planned) if registered else None
    if complete and not coverage['complete']:H_OW=None
    old_mask=y<6;arrival_mask=np.isin(y,arrivals)
    auc=oscr=None
    if roster(OLD_LABELS) and ru_labels and (not complete or coverage['complete']):
        # The baseline is supplied only as a protection reference. Correctness
        # comes from this state's current pre-rejection DA/head or joint winner.
        rank=open_set_metrics(y[old_mask],current_choice[old_mask],evidence[old_mask],
            y[np.isin(y,ru_labels)],evidence[np.isin(y,ru_labels)],0.,roster(OLD_LABELS),ru_labels,
            baseline_prediction=baseline[old_mask])
        auc,oscr=rank['auroc_known_macro'],rank['oscr_auc_macro']
    return dict(count=len(y),coverage=coverage,old_count=int(old_mask.sum()),
        old_accepted=int(np.sum(old_mask&(raw!=-1))),old_rejected=int(np.sum(old_mask&(raw==-1))),
        old_correct_accepted=int(np.sum(old_mask&(mapped==y))),original_F0_closed_correct=int(np.sum(old_mask&(baseline==y))),
        old_per_class=old,old_class_coverage=coverage['old_classes'],old_macro_open_accuracy=A,old_macro_acceptance=C,old_macro_frr=frr,
        original_F0_macro_closed_accuracy=original,original_F0_minus_old_open=None if original is None or A is None else original-A,
        known_macro_open_accuracy=A,known_macro_acceptance=C,known_macro_frr=frr,
        original_closed_minus_new_open=None if original is None or A is None else original-A,
        A_K=A,C_K=C,R_U=RU,H_score=H,H_OS=H,unknown_macro_rejection=RU,unknown_per_class=ru_rows,R_U_labels=ru_labels,
        R_U_scope='planned later not observed in U' if registered else 'all initially unregistered future labels',
        A_N=AN,new_macro_open_accuracy=AN,new_macro_denominator=len(roster(arrivals)) if registered else None,
        global_true_arrival_denominator=len(arrivals),new_arrival_count=int(arrival_mask.sum()),
        new_correct_accepted=int(np.sum(arrival_mask&(mapped==y))) if registered else None,new_per_class=new if registered else None,
        H_old_new=H_old_new,H_OW=H_OW,intended_registered_macro_acceptance=_mean(intended,'acceptance',complete) if registered else None,
        intended_registered_macro_frr=_mean(intended,'frr',complete) if registered else None,
        intended_registered_per_class=intended if registered else None,
        planned_later_macro_rejection=planned,planned_later_per_class=later_rows,
        actual_unregistered_macro_rejection=actual,actual_unregistered_per_class=actual_rows,
        all_future_nonold_macro_rejection=all_nonold,all_future_nonold_per_class=all_rows,
        unmatched_anonymous_accepted=int(np.sum(mapped==-2)),auroc_known_macro=auc,oscr_auc_macro=oscr,
        auroc_known=auc,oscr=oscr,curve_correctness='current DA old-head winner for REG0; U-mapped current joint winner for REG1; never original baseline',
        confidence_intervals=dict(level=.95,method='per true-class Wilson; no pooled or independent-RX macro interval',
            A_K_per_class=_wilson(old,'correct_accepted'),R_U_per_class=_wilson(ru_rows,'rejected'),
            limitation='one development RX/seed and reused paired physical records; class intervals are descriptive'))


def _old_ok(row):
    return (row['coverage']['complete'] and row['old_macro_acceptance'] is not None and row['old_macro_acceptance']>=.90-1e-12
        and row['original_F0_minus_old_open'] is not None and row['original_F0_minus_old_open']<=.02+1e-12)


def _reg0_checks(row):
    return dict(complete_required_class_coverage=row['coverage']['complete'],old_known_and_F0_protection=_old_ok(row),
        unknown_protection=row['coverage']['complete'] and row['R_U'] is not None and row['R_U']>=.90-1e-12)


def _registration_checks(row):
    covered=row['coverage']['complete'];loss=row['old_loss_from_same_DA_REG0']
    return dict(complete_required_class_coverage=covered,old_known_and_absolute_F0_protection=_old_ok(row),
        old_retention_from_same_DA_REG0=covered and loss is not None and loss<=.02+1e-12,
        intended_registered_macro_CK90=covered and row['intended_registered_macro_acceptance'] is not None and row['intended_registered_macro_acceptance']>=.90-1e-12,
        planned_later_unknown_protection=covered and row['R_U'] is not None and row['R_U']>=.90-1e-12)


def _goals(row,registered):
    check=_registration_checks if registered else _reg0_checks
    overall=check(row['overall']);scenes={s:check(row['by_scenario'][s]) for s in SCENES}
    if registered:overall['new_arrival_macro_accuracy80']=row['overall']['coverage']['complete'] and row['overall']['A_N'] is not None and row['overall']['A_N']>=.80-1e-12
    return dict(overall=overall,by_scenario=scenes,main_scene_coverage=all(row['by_scenario'][s]['count']>0 for s in SCENES),
        new_accuracy_each_scene_reported_not_required=registered,
        passed=all(overall.values()) and all(all(c.values()) for c in scenes.values()))


_DELTA=('old_macro_open_accuracy','old_macro_acceptance','old_macro_frr','original_F0_minus_old_open','R_U','H_score',
    'A_N','H_old_new','H_OW','intended_registered_macro_acceptance','planned_later_macro_rejection',
    'all_future_nonold_macro_rejection','auroc_known_macro','oscr_auc_macro')


def _difference(a,b):return {key:None if a[key] is None or b[key] is None else a[key]-b[key] for key in _DELTA}


def _counts(y,a,b,ma,mb,cohort):
    old=y<6;unknown=np.isin(y,cohort);aa=a!=-1;ba=b!=-1;ac=old&(ma==y);bc=old&(mb==y)
    return dict(old_accept_only_left=int(np.sum(old&aa&~ba)),old_accept_only_right=int(np.sum(old&ba&~aa)),
        old_accept_net=int(np.sum(old&aa)-np.sum(old&ba)),old_correct_accept_only_left=int(np.sum(ac&~bc)),
        old_correct_accept_only_right=int(np.sum(bc&~ac)),old_correct_accept_net=int(ac.sum()-bc.sum()),
        unknown_reject_only_left=int(np.sum(unknown&(a==-1)&(b!=-1))),unknown_reject_only_right=int(np.sum(unknown&(b==-1)&(a!=-1))),
        unknown_rejected_net=int(np.sum(unknown&(a==-1))-np.sum(unknown&(b==-1))))


def _partition_diagnostics(uy,diag,state):
    rows=[];registered={int(c):i for i,c in enumerate(state['anonymous_indices'])}
    for c,n in enumerate(diag['cluster_sizes']):
        selected=diag['cluster_labels']==c;labels,counts=np.unique(uy[selected],return_counts=True)
        old=int(np.sum(selected&(uy<6)))
        rows.append(dict(cluster_index=c,count=int(n),composition={str(int(k)):int(v) for k,v in zip(labels,counts)},
            old_count=old,old_contamination=old/int(n) if n else None,purity=int(counts.max())/int(n) if n else None,
            registered_anonymous_index=registered.get(c),mean_raw_margin=float(state['cluster_margin'][c]),radius=float(state['cluster_radius'][c])))
    return dict(partition_cluster_count=len(rows),clusters=rows,truth_side_only=True,not_true_K_input=True)


def score(predictions_root,U_truth,Q_truth,targets,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable neighborhood score output root is required')
    done,p,diag,state=_prediction_gate(predictions_root);K=_gate(done,p,diag,state);spec=_targets(targets)
    u=_truth(U_truth,p['U_ids'])
    if not np.all(u['receiver']==26) or np.any((u['y']<0)|(u['y']>7)) or not np.isin(u['scenario'],SCENES).all():
        raise ValueError('Only fixed RX24-1 U role truth may fit scorer-side anonymous matching')
    arrivals=sorted(int(label) for label in np.unique(u['y']) if label>=6)
    # Freeze one mapping on U before Q truth is even opened.
    mapping,clusters=prior._matching(u['y'],diag['assignments'],diag['candidate_mask'],K,arrivals)
    partition=_partition_diagnostics(u['y'],diag,state)
    future=_truth(Q_truth,p['ids']);y=future['y'];scenes=future['scenario']
    if not np.all(future['receiver']==26) or not np.isin(y,OLD_LABELS+UNKNOWN_LABELS).all() or not np.isin(scenes,SCENES).all():
        raise ValueError('Only fixed RX24-1 future-Q truth is allowed; formal RX is excluded')
    later=sorted(set(UNKNOWN_LABELS)-set(arrivals));rows={};mapped_states={};choices={}
    masks=dict(overall=np.ones(len(y),bool),**{'scene/'+s:scenes==s for s in SCENES})
    for i,name in enumerate(STATES):
        registered=i>=2;raw=p['prediction'][i];mapped=prior._mapped(raw,mapping) if registered else raw.copy()
        choice=p['joint_margins_'+name].argmax(1).astype(np.int64)
        choice=prior._mapped(choice,mapping) if registered else p['closed_prediction'][i%2]
        mapped_states[name]=mapped;choices[name]=choice
        actual=sorted(set(UNKNOWN_LABELS)-(set(mapping.values()) if registered else set()))
        def calculate(mask,complete=True):
            return _metrics(y[mask],raw[mask],mapped[mask],p['baseline_closed_prediction'][mask],p['knownness'][i,mask],
                choice[mask],registered,arrivals,later,actual,complete=complete)
        rows[name]=dict(overall=calculate(masks['overall']),by_scenario={s:calculate(masks['scene/'+s]) for s in SCENES},
            by_transmitter={str(label):calculate(y==label,False) for label in OLD_LABELS+UNKNOWN_LABELS},
            by_receiver={'26':calculate(masks['overall'])},by_day={done['Q_day']:calculate(masks['overall'])},
            by_receiver_day={'26/'+done['Q_day']:calculate(masks['overall'])},registered=registered,
            matched_arrival_labels=sorted(mapping.values()) if registered else [],actual_unregistered_labels=actual,
            unknown_rejection_rule='Only raw -1 rejects; unmatched anonymous -2 is accepted wrong',
            new_macro_denominator_rule='All true U arrival labels, including undiscovered/unmatched; missing Q coverage is N/A; REG0 N/A')
    for da in ('DA0','DA1'):
        current,previous=rows[da+'_REG1'],rows[da+'_REG0']
        for group in ('overall','by_scenario','by_transmitter','by_receiver','by_day','by_receiver_day'):
            pairs=[(current[group],previous[group])] if group=='overall' else [(v,previous[group][key]) for key,v in current[group].items()]
            for a,b in pairs:a['old_loss_from_same_DA_REG0']=None if a['old_macro_open_accuracy'] is None or b['old_macro_open_accuracy'] is None else b['old_macro_open_accuracy']-a['old_macro_open_accuracy']
        previous['goals']=_goals(previous,False);current['goals']=_goals(current,True)
    first,second=rows['DA0_REG0']['overall'],rows['DA1_REG0']['overall']
    gain=None if first['A_K'] is None or second['A_K'] is None else second['A_K']-first['A_K']
    adaptation_checks=dict(neighborhood_fusion_applied=done.get('DA1_fusion_applied') is True,
        old_macro_open_accuracy_gain=gain is not None and gain>0.,six_scene_v3_constraints=rows['DA1_REG0']['goals']['passed'])
    adaptation=dict(checks=adaptation_checks,old_macro_open_accuracy_gain=gain,passed=all(adaptation_checks.values()))
    registration={da:rows[da+'_REG1']['goals'] for da in ('DA0','DA1')}
    comparisons=(('DA1_REG0_minus_DA0_REG0','DA1_REG0','DA0_REG0'),('DA1_REG1_minus_DA0_REG1','DA1_REG1','DA0_REG1'),
        ('DA0_REG1_minus_DA0_REG0','DA0_REG1','DA0_REG0'),('DA1_REG1_minus_DA1_REG0','DA1_REG1','DA1_REG0'))
    deltas={}
    for key,left,right in comparisons:
        a,b=rows[left],rows[right];li,ri=STATES.index(left),STATES.index(right)
        counts=_counts(y,p['prediction'][li],p['prediction'][ri],mapped_states[left],mapped_states[right],UNKNOWN_LABELS)
        same=a['overall']['R_U_labels']==b['overall']['R_U_labels']
        actual_counts=_counts(y,p['prediction'][li],p['prediction'][ri],mapped_states[left],mapped_states[right],a['overall']['R_U_labels'])
        deltas[key]=dict(_difference(a['overall'],b['overall']),left_state=left,right_state=right,counts=counts,
            unknown_rejected_net=actual_counts['unknown_rejected_net'] if same else None,
            unknown_rejected_net_scope='same main R_U roster' if same else 'N/A: REG changes R_U roster; use fixed-cohort counts',
            R_U_rosters=dict(left=a['overall']['R_U_labels'],right=b['overall']['R_U_labels']),
            primary_R_U_and_H_comparable=same,
            fixed_cohort_counts=dict(all_future_nonold=counts,planned_later=_counts(y,p['prediction'][li],p['prediction'][ri],mapped_states[left],mapped_states[right],later)),
            by_scenario={s:dict(delta=_difference(a['by_scenario'][s],b['by_scenario'][s]),
                counts=_counts(y[masks['scene/'+s]],p['prediction'][li,masks['scene/'+s]],p['prediction'][ri,masks['scene/'+s]],
                    mapped_states[left][masks['scene/'+s]],mapped_states[right][masks['scene/'+s]],UNKNOWN_LABELS)) for s in SCENES})
    passed=adaptation['passed'] and registration['DA1']['passed']
    result=dict(status=STATUS,schema=SCHEMA,run_id=RUN,condition='P1',primary_state=PRIMARY,primary_method='paired1000_predicted_class_neighborhood',
        states=rows,primary_summary=rows[PRIMARY]['overall'],primary_pass=passed,
        next_step='STAR_NEIGHBORHOOD_DEVELOPMENT_PIPELINE_PASSED' if passed else 'NO_PROMOTION',
        cluster_diagnostics=clusters,all_U_partition_diagnostics=partition,deltas=deltas,
        adaptation_goals=adaptation,registration_goals=registration,required_main_scenes=list(SCENES),observed_scenes=sorted(set(scenes.tolist())),
        future_unknown_labels=list(UNKNOWN_LABELS),U_true_arrival_labels=arrivals,planned_later_labels=later,
        primary_metrics='True-TX macro A_K, state-specific R_U, H-score=harmonic(A_K,R_U); A_N separately; H_old_new is not H-score',
        unknown_denominator_rule='REG0 all initially unregistered [6,7,8]; REG1 planned later outside U; actual unregistered also diagnostic',
        curve_correctness='Current DA/head or mapped joint winner; original F0 prediction only baseline for protection',
        targets_schema=spec['schema'],condition_unknown_rejection_min=.90,intended_registered_CK_min=.90,
        development_only=True,truth_last=True,U_mapping_frozen_before_Q_truth=True,Q_truth_used_for_cluster_mapping=False,
        future_Q_fitting=0,target_truth_used_for_prediction=False,algorithm_target_label_inputs=0,
        source_parameters_updated=0,identity_head_updates=0,optimizer_updates=0,formal_target_reads=0,
        anonymous_K_may_be_zero=True,true_unknown_K_inputs=0,all_new_IQ_satellite=True,
        day_grouping='Q single date from immutable role metadata; truth has no per-record day field',
        final_phase3_complete=False,full_goal_status='ACTIVE',automatic_formal_or_multiseed_promotion=False,
        missing_group_policy='N/A and coverage false; no empty-group certification or zero filling')
    output.mkdir(parents=True,exist_ok=False);_write(output/'scores.json',result)
    _write(output/'complete.json',{key:result[key] for key in ('status','schema','run_id','condition','truth_last','primary_state','primary_method','primary_pass','next_step','final_phase3_complete')})
    return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_NEIGHBORHOOD_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions-root',required=True);p.add_argument('--U-truth',required=True)
    p.add_argument('--Q-truth',required=True);p.add_argument('--targets',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);done=score(a.predictions_root,a.U_truth,a.Q_truth,a.targets,a.output)
    print(json.dumps(dict(status=done['status'],primary_pass=done['primary_pass'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
