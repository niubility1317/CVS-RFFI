"""Truth-last U-only alignment and future all-cluster registration evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import fresh_world_score as fresh
from . import star_consensus_score as common
from . import unlabeled_registration_score as prior
from . import star_signal_register_score as previous
from .metrics import open_set_metrics,_harmonic,open_world_h
from .scoring import wilson95
from .star_macro12_risk import HARD_NAMES

RUN_ID='20261031-phase3-pcd-cluster-registry-s2026092701-r01'
SCHEMA='pamf_phase3_pcd_cluster_registry_scores_v1'
STATUS='PCD_CLUSTER_REGISTRY_SCORES_COMPLETE'
METHODS=('pooled_REG0','predicted_REG0','pooled_REG1','predicted_REG1')
PRIMARY=METHODS[-1]
OLD=tuple(range(6))
ARRIVALS=(6,7,8)
UNKNOWN=(6,7,8,9)
LATER=(9,)
SOURCE_UNKNOWN=(100,101,102)
CONDITIONS=('source','clean_DURA','star_DURA')
_read,_write=common._read,common._write


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _prediction_gate(predict_config):
    from . import pcd_cluster_registry_run as producer
    c=producer.validate_config(_read(predict_config));done,state,registry,U,rows=producer._prediction_gate(c['output_root'])
    if done.get('config')!=c or done.get('run_id')!=RUN_ID or done.get('method_names')!=list(METHODS) or done.get('primary_method')!=PRIMARY:
        raise ValueError('The exact completed four-readout/three-condition prediction config is required before truth')
    return done,state,registry,U,rows


def _matching(y,assignment,cluster_labels):
    """Strict true-new majority cells, then one U-only one-to-one match."""
    y,assignment,cluster_labels=map(np.asarray,(y,assignment,cluster_labels));n=len(y)
    if (y.shape!=(n,) or y.dtype.kind not in 'iu' or not np.isin(y,OLD+ARRIVALS).all()
            or assignment.shape!=(n,) or assignment.dtype.kind not in 'iu' or np.any(assignment< -1) or np.any(assignment>=len(cluster_labels))
            or cluster_labels.ndim!=1 or cluster_labels.dtype.kind not in 'iu' or np.any(cluster_labels<0)):
        raise ValueError('Aligned U truth and frozen whole-cluster assignment required')
    anonymous=np.flatnonzero(cluster_labels>=6);slots=cluster_labels[anonymous]-6;count=len(anonymous)
    if sorted(slots.tolist())!=list(range(count)):raise ValueError('Canonical one-to-one anonymous slot labels required')
    table=np.zeros((count,len(ARRIVALS)),np.int64);sizes=np.zeros(count,np.int64);clusters=[]
    for cluster in anonymous:
        slot=int(cluster_labels[cluster]-6);mask=assignment==cluster;size=int(mask.sum());sizes[slot]=size
        labels,counts=np.unique(y[mask],return_counts=True);composition={str(int(c)):int(n) for c,n in zip(labels,counts)}
        for j,c in enumerate(ARRIVALS):table[slot,j]=int((mask&(y==c)).sum())
        clusters.append(dict(cluster_index=int(cluster),anonymous_index=slot,size=size,old_count=int((mask&(y<6)).sum()),composition=composition,
            strict_majority_true_arrivals=[c for j,c in enumerate(ARRIVALS) if 2*table[slot,j]>size]))
    eligible=2*table>sizes[:,None];objective=np.where(eligible,table,0);mapping={}
    if count:
        from scipy.optimize import linear_sum_assignment
        r,c=linear_sum_assignment(-objective)
        mapping={int(i):int(ARRIVALS[j]) for i,j in zip(r,c) if objective[i,j]>0}
    return mapping,dict(mapping={str(k):v for k,v in mapping.items()},mapping_source='U truth only; strict new-class majority before maximum-count one-to-one Hungarian',
        strict_majority_required=True,majority_denominator='all U members of the frozen cluster, including known contamination',Q_truth_used=False,old_classes_renamed=False,
        known_associated_cluster_count=int((cluster_labels<6).sum()),known_associated_cluster_indices=np.flatnonzero(cluster_labels<6).tolist(),anonymous_count=count,
        contingency=dict(anonymous_indices=list(range(count)),true_arrival_labels=list(ARRIVALS),counts=table.tolist(),eligible=eligible.tolist()),clusters=clusters,
        matched_count=len(mapping),unmatched_anonymous_indices=[i for i in range(count) if i not in mapping],unmatched_true_arrivals=[c for c in ARRIVALS if c not in mapping.values()],
        U_noise_count=int((assignment<0).sum()),U_labels_used_for_deployment=False)


_mapped=prior._mapped


def _mean(rows,key):return None if not rows or any(r['count']==0 for r in rows) else float(np.mean([r[key] for r in rows]))


def _rejection(y,raw,roster):
    rows=[]
    for c in roster:
        mask=y==c;n=int(mask.sum());r=int((mask&(raw==-1)).sum())
        rows.append(dict(label=int(c),count=n,rejected=r,rejection=r/n if n else None))
    return _mean(rows,'rejection'),rows


def _metrics(y,raw,mapped,baseline,evidence,registered,mapping,*,source=False):
    unknowns=SOURCE_UNKNOWN if source else UNKNOWN;arrivals=() if source else ARRIVALS;later=() if source else LATER
    old=prior._class_rows(y,raw,mapped,baseline,OLD,missing_zero=False)
    new=prior._class_rows(y,raw,mapped,baseline,arrivals,missing_zero=False) if registered and not source else []
    intended=old+new if registered else []
    A,C,F0=(_mean(old,k) for k in ('open_accuracy','acceptance','original_F0_closed_accuracy'))
    AN=_mean(new,'open_accuracy') if new else None;ru_roster=unknowns if source or not registered else later
    RU,unknown=_rejection(y,raw,ru_roster);lr,lrows=_rejection(y,raw,later)
    actual=unknowns if source or not registered else tuple(c for c in UNKNOWN if c not in mapping.values())
    ar,arows=_rejection(y,raw,actual);allru,allrows=_rejection(y,raw,unknowns)
    complete=lambda r:bool(r) and all(np.any(y==c) for c in r)
    coverage=dict(known_complete=complete(OLD),unknown_complete=complete(unknowns),arrival_complete=None if source else complete(ARRIVALS),later_complete=None if source else complete(LATER),
        registered_complete=complete(OLD+tuple(arrivals)) if registered else None)
    coverage['complete']=coverage['known_complete'] and coverage['unknown_complete']
    # A rejected point has no supported U prototype and its pre-reject fallback
    # is its original old-head class. Accepted prototype choices use mapped IDs.
    choice=np.where(raw>=0,mapped,baseline);k=y<6;u=np.isin(y,ru_roster)
    curve=open_set_metrics(y[k],choice[k],evidence[k],y[u],evidence[u],0.,OLD,ru_roster,baseline_prediction=baseline[k]) if ru_roster else None
    ci=lambda rows,key:[dict(label=r['label'],count=r['count'],successes=r[key],interval95=wilson95(r[key],r['count'])) for r in rows]
    return dict(count=len(y),coverage=coverage,A_K=A,C_K=C,R_U=RU,H_score=_harmonic((A,RU)),A_N=AN,H_KN=_harmonic((A,AN)) if new else None,
        H_OW=open_world_h(A,AN,lr) if new else None,original_F0_macro_closed_accuracy=F0,original_closed_minus_new_open=None if F0 is None or A is None else F0-A,
        known_macro_open_accuracy=A,known_macro_acceptance=C,known_macro_frr=None if C is None else 1-C,known_per_class=old,unknown_per_class=unknown,new_per_class=new if new else None,
        old_count=int(k.sum()),old_correct_accepted=int((k&(mapped==y)).sum()),old_accepted=int((k&(raw!=-1)).sum()),old_rejected=int((k&(raw==-1)).sum()),original_F0_closed_correct=int((k&(baseline==y)).sum()),
        new_macro_denominator=len(arrivals) if new else None,arrival_count=int(np.isin(y,arrivals).sum()),arrival_correct_accepted=int((np.isin(y,arrivals)&(mapped==y)).sum()) if new else None,
        registered_macro_accuracy=_mean(intended,'open_accuracy') if registered else None,registered_macro_acceptance=_mean(intended,'acceptance') if registered else None,
        registered_macro_FRR=_mean(intended,'frr') if registered else None,registered_per_class=intended if registered else None,
        R_U_labels=list(ru_roster),R_U_scope='all distinct source-NV devices, never target arrivals' if source else 'fixed future-only label9' if registered else 'all four initially unknown target devices',
        planned_later_macro_rejection=lr,planned_later_per_class=lrows,actual_unregistered_macro_rejection=ar,actual_unregistered_per_class=arows,actual_unregistered_labels=list(actual),
        all_nonold_macro_rejection=allru,all_nonold_per_class=allrows,unmatched_anonymous_accepted=int((mapped==-2).sum()),
        auroc_known_macro=curve['auroc_known_macro'] if curve else None,oscr_auc_macro=curve['oscr_auc_macro'] if curve else None,
        curves_scope='old-six versus the explicitly reported R_U cohort, fixed decision evidence; rejected fallback is original F0 old choice',
        confidence_intervals=dict(level=.95,A_K_per_class=ci(old,'correct_accepted'),C_K_per_class=ci(old,'accepted'),R_U_per_class=ci(unknown,'rejected'),A_N_per_class=ci(new,'correct_accepted') if new else None))


def _align_truth(a,ids):
    n=len(ids)
    if not common._ids(a.get('ids',np.asarray([])),n) or set(a['ids'].tolist())!=set(ids.tolist()):raise ValueError('Exact unique private/public physical IDs must align')
    lookup={v:i for i,v in enumerate(a['ids'].tolist())};order=np.asarray([lookup[v] for v in ids.tolist()],np.int64)
    return {k:v[order] for k,v in a.items()}


def _target_truth(path,ids,*,U=False):
    a=_arrays(path);n=len(ids)
    if (set(a)!={'ids','y','tx','receiver','day','scenario'} or any(a[k].shape!=(n,) or a[k].dtype!=np.int64 for k in ('y','receiver'))
            or any(a[k].shape!=(n,) or a[k].dtype.kind!='U' for k in ('tx','day','scenario')) or not np.isin(a['y'],OLD+ARRIVALS if U else OLD+UNKNOWN).all()
            or not np.all(a['receiver']==34) or not np.isin(a['day'],('2021_03_01',) if U else fresh.DAYS).all()
            or not np.array_equal(a['tx'],np.asarray(fresh.TX_BY_LABEL)[a['y']])):raise ValueError('Only the exact RX34 private U/future source-independent TX/date namespace is allowed')
    if U and (not np.isin(a['scenario'],common.SCENES).all() or set(a['y'].tolist())!=set(OLD+ARRIVALS)):raise ValueError('U truth must contain all intended old-six/three arrivals, never future-only9')
    return _align_truth(a,ids)


def _source_truth(path,novel):
    a=_arrays(path);n=len(novel['ids'])
    if (set(a)!={'ids','tx','rx_i','day_i','scenario'} or any(a[k].shape!=(n,) or a[k].dtype.kind!='U' for k in ('tx','scenario'))
            or any(a[k].shape!=(n,) or a[k].dtype!=np.int64 for k in ('rx_i','day_i')) or not np.isin(a['tx'],common.UNKNOWN_TX).all()
            or not np.isin(a['scenario'],common.SCENES).all()):raise ValueError('Source-NV truth contains its own three TX identities, not target arrivals')
    a=_align_truth(a,novel['ids'])
    if any(not np.array_equal(a[k],novel[k]) for k in ('rx_i','day_i')):raise ValueError('Source-NV private/public domains differ')
    return a


def _old_ok(row):
    return bool(row['coverage']['known_complete'] and row['C_K'] is not None and row['C_K']>=.9-1e-12
        and row['original_closed_minus_new_open'] is not None and row['original_closed_minus_new_open']<=.02+1e-12)


def _condition(a,y,rx,scene,day,index,mapping,*,source=False,condition='source'):
    registered=index>=2;raw=a['raw_prediction'][:,index];mapped=_mapped(raw,{} if source else mapping) if registered else raw.copy()
    def metric(mask):return _metrics(y[mask],raw[mask],mapped[mask],a['baseline_closed'][mask],a['decision_score'][mask,index],registered,mapping,source=source)
    masks=dict(common._masks(rx,scene)) if source else dict(overall=np.ones(len(y),bool),**{'scene/'+s:scene==s for s in (('clean',) if condition=='clean_DURA' else common.SCENES)})
    groups={g:metric(mask) for g,mask in masks.items()};overall=groups['overall']
    label_name=lambda c:common.UNKNOWN_TX[c-100] if source and c>=100 else fresh.TX_BY_LABEL[c]
    tx={label_name(r['label']):dict(r) for r in overall['known_per_class']+overall['all_nonold_per_class']}
    for r in overall['new_per_class'] or []:tx[label_name(r['label'])].update(r)
    return dict(overall=overall,groups=groups,by_scene={g[6:]:v for g,v in groups.items() if g.startswith('scene/')},
        by_RX={str(r):metric(rx==r) for r in np.unique(rx)},by_day={str(d):metric(day==d) for d in np.unique(day)},by_TX=tx,registered=registered,
        source_unknown_namespace=list(SOURCE_UNKNOWN) if source else None,A_N_denominator='all three true U arrival devices, including missed or unmatched' if not source else 'N/A: source NV are not target arrivals',
        unmatched_anonymous_rule='accepted wrong (-2 after U-only matching), never a reject')


def _checks(row,*,registered,minimum):
    result=dict(required_unknown_coverage=row['coverage']['unknown_complete'],old_CK90_and_original_F0_gap2pp=_old_ok(row))
    if registered:result.update(expected_registered_coverage=row['coverage']['registered_complete'],registered_CK90=row['registered_macro_acceptance'] is not None and row['registered_macro_acceptance']>=.9-1e-12,
        old_loss_from_REG0_at_most2pp=row['old_loss_from_REG0'] is not None and row['old_loss_from_REG0']<=.02+1e-12,later_RU95=row['R_U'] is not None and row['R_U']>=.95-1e-12)
    else:result['initial_unknown_RU']=row['R_U'] is not None and row['R_U']>=minimum-1e-12
    return result


def score(predict_config,U_truth,source_novel_truth,clean_truth,star_truth,output_root):
    out=Path(output_root)
    if out.exists():raise FileExistsError('New immutable independent registry score output root required')
    done,state,registry,U,rows=_prediction_gate(predict_config)
    # No private IO precedes the complete all-condition artifact gate.
    ut=_target_truth(U_truth,U['ids'],U=True);mapping,matching=_matching(ut['y'],U['assignment'],registry['labels'])
    nv=rows['source_NV'];nt=_source_truth(source_novel_truth,nv)
    truths={name:_target_truth(path,rows['Q_'+name]['ids']) for name,path in (('clean_DURA',clean_truth),('star_DURA',star_truth))}
    for name,t in truths.items():
        if not np.isin(t['scenario'],('clean',) if name=='clean_DURA' else common.SCENES).all():raise ValueError('Future truth must match its clean/star condition')
    v=rows['source_V'];source={k:np.concatenate((v[k],nv[k])) for k in ('ids','closed','baseline_closed','raw_prediction','decision_score')}
    sy=np.r_[v['y'],np.asarray([100+common.UNKNOWN_TX.index(str(x)) for x in nt['tx']],np.int64)]
    labels=dict(source=(sy,np.r_[v['rx_i'],nv['rx_i']],np.r_[v['scenario'],nt['scenario']],np.r_[v['day_i'],nv['day_i']]))
    data=dict(source=source,**{name:rows['Q_'+name] for name in CONDITIONS[1:]})
    for name,t in truths.items():labels[name]=(t['y'],t['receiver'],t['scenario'],t['day'])
    methods={name:{condition:_condition(data[condition],*labels[condition],index,mapping,source=condition=='source',condition=condition) for condition in CONDITIONS} for index,name in enumerate(METHODS)}
    for index,name in enumerate(METHODS):
        for condition in CONDITIONS:
            row=methods[name][condition];control=methods[METHODS[index-2]][condition] if index>=2 else None
            for group,current in row['groups'].items():
                old=control['groups'][group]['A_K'] if control else current['A_K'];current['old_loss_from_REG0']=None if old is None or current['A_K'] is None else old-current['A_K']
            for grouping in ('by_RX','by_day'):
                for group_name,current in row[grouping].items():
                    old=control[grouping][group_name]['A_K'] if control else current['A_K']
                    current['old_loss_from_REG0']=None if old is None or current['A_K'] is None else old-current['A_K']
            if condition=='source':
                row['known_12_groups_passed']=all(_old_ok(row['groups'][g]) for g in HARD_NAMES)
                row['known_42_groups_passed']=all(_old_ok(r) for r in row['groups'].values());row['source_NV_diagnostic_only']=True
            else:
                minimum=.95 if condition=='clean_DURA' else .90;checks={g:_checks(r,registered=index>=2,minimum=minimum) for g,r in row['groups'].items()}
                coverage=all(r['coverage']['complete'] and (index<2 or r['coverage']['arrival_complete'] and r['coverage']['later_complete']) for r in row['groups'].values())
                AN=row['overall']['A_N'];AN_ok=index<2 or AN is not None and AN>=.8-1e-12
                row['goals']=dict(checks=checks,required_scene_coverage_complete=bool(coverage),arrival_accuracy80_overall_only=bool(AN_ok),arrival_accuracy_each_scene_reported_not_required=True,
                    passed=bool(coverage and AN_ok and all(all(c.values()) for c in checks.values())))
    paired={}
    for left,right in ((2,0),(3,1)):
        comparisons={}
        for condition in CONDITIONS:
            y,rx,scene,day=labels[condition];a=data[condition];lp=a['raw_prediction'][:,left];rp=a['raw_prediction'][:,right]
            lm=_mapped(lp,{} if condition=='source' else mapping);rm=rp.copy();arrivals=() if condition=='source' else ARRIVALS;later=() if condition=='source' else LATER
            masks=dict(common._masks(rx,scene)) if condition=='source' else dict(overall=np.ones(len(y),bool),**{'scene/'+s:scene==s for s in methods[METHODS[left]][condition]['by_scene']})
            metric_left=methods[METHODS[left]][condition]['overall'];metric_right=methods[METHODS[right]][condition]['overall']
            comparisons[condition]=dict(counts=previous._counts(y,lp,rp,lm,rm,arrivals,later),groups={g:previous._counts(y[m],lp[m],rp[m],lm[m],rm[m],arrivals,later) for g,m in masks.items()},
                by_RX={str(r):previous._counts(y[rx==r],lp[rx==r],rp[rx==r],lm[rx==r],rm[rx==r],arrivals,later) for r in np.unique(rx)},
                by_day={str(d):previous._counts(y[day==d],lp[day==d],rp[day==d],lm[day==d],rm[day==d],arrivals,later) for d in np.unique(day)},
                by_TX={str(c):previous._counts(y[y==c],lp[y==c],rp[y==c],lm[y==c],rm[y==c],arrivals,later) for c in np.unique(y)},
                delta=previous._difference(metric_left,metric_right),same_later_cohort=dict(labels=list(later),REG0_rejection=metric_right['planned_later_macro_rejection'],REG1_rejection=metric_left['planned_later_macro_rejection'],
                    gain=None if metric_left['planned_later_macro_rejection'] is None or metric_right['planned_later_macro_rejection'] is None else metric_left['planned_later_macro_rejection']-metric_right['planned_later_macro_rejection']))
        paired[METHODS[left]+'_vs_'+METHODS[right]]=comparisons
    main=methods[PRIMARY];source_ok=main['source']['known_12_groups_passed'];registration_ok=all(main[c]['goals']['passed'] for c in CONDITIONS[1:])
    control_ok=bool(methods['predicted_REG0']['source']['known_12_groups_passed'] and all(methods['predicted_REG0'][c]['goals']['passed'] for c in CONDITIONS[1:]))
    success=bool(source_ok and registration_ok and control_ok)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=done['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,methods=methods,conditions=list(CONDITIONS),
        primary_summary={c:main[c]['overall'] for c in CONDITIONS},paired=paired,U_matching=matching,U_true_arrival_labels=list(ARRIVALS),planned_later_labels=list(LATER),U_mapping_frozen_before_source_and_Q_truth=True,Q_truth_used_for_mapping=False,
        source_known_12_passed=bool(source_ok),main_registration_gates_passed=bool(registration_ok),predicted_REG0_gates_passed=bool(control_ok),development_gate_passed=success,registration_success=success,
        next_step='DEVELOPMENT_CLUSTER_REGISTRATION_PASSED_REQUIRES_SEPARATE_CONFIRMATION' if success else 'NO_PROMOTION',truth_last=True,all_three_predictions_before_truth=True,U_truth_reads=1,source_novel_truth_reads=1,target_truth_reads=2,
        old_classes_renamed=False,source_NV_target_identity_namespaces_disjoint=True,source_unknown_namespace=list(SOURCE_UNKNOWN),source_A_N_H_KN_H_OW=None,source_V_is_calibration_not_independent_test=True,
        target_DA=True,target_REG=True,neural_updates=0,original_F0_updates=0,G_updates=0,query_fit=False,formal_target_accessed=False,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        source_RX_seen=True,source_novel_historically_exposed=True,fresh_development_RX=34,global_physical_history='UNKNOWN',new_blind_confirmation=False,
        prediction_identity={k:done[k] for k in ('run_id','code_commit','method_names','primary_method')},scope='Frozen PCD U-only all-cluster registry on exposed RX34 development future; A_N includes all true arrivals and unmatched anonymous is accepted wrong')
    result['prediction_costs']={k:done[k] for k in ('source_L_reads','target_U_reads','real_signal_forward_rows','real_F_forward_rows','G_forward_rows',
        'synthetic_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows','optimizer_updates','new_density_fits','new_threshold_fits') if k in done}
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired','U_matching')});print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--predict-config',required=True);p.add_argument('--U-truth',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output-root',required=True)
    a=p.parse_args();score(a.predict_config,a.U_truth,a.source_novel_truth,a.clean_truth,a.star_truth,a.output_root)


if __name__=='__main__':main()
