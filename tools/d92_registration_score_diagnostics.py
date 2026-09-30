"""Pure post-prediction decomposition of old ordering and new competition.

Inputs are independently fixed scores and evaluator labels, never fitting data.
Restricting C to old columns is an offline counterfactual, not a deployable rule:
the deployed C prediction always competes over its complete registered classes.
No cross-head score subtraction, calibration, concatenation or scale comparison.
"""
from collections.abc import Mapping
import math
from numbers import Real


def _check(condition,message):
    if not condition:raise ValueError(message)


def _strings(values,name):
    _check(not isinstance(values,(str,bytes,Mapping)),name+' must be a sequence of identities')
    try:values=list(values)
    except TypeError as exc:raise ValueError(name+' must be a sequence of identities') from exc
    _check(bool(values),name+' must not be empty')
    _check(all(isinstance(v,str) and bool(v) for v in values),name+' must contain nonempty strings')
    _check(len(values)==len(set(values)),name+' contains duplicate identities')
    return values


def _scores(values,ids,classes,name):
    try:rows=list(values)
    except TypeError as exc:raise ValueError(name+' must be a two-dimensional score array') from exc
    _check(len(rows)==len(ids),name+' row count does not match physical IDs')
    result={}
    for pid,row in zip(ids,rows):
        _check(not isinstance(row,(str,bytes,Mapping)),name+' must be a two-dimensional score array')
        try:row=list(row)
        except TypeError as exc:raise ValueError(name+' must be a two-dimensional score array') from exc
        _check(len(row)==len(classes),name+' column count does not match classes')
        _check(all(isinstance(v,Real) and not isinstance(v,bool) for v in row),name+' must contain real numeric scores')
        try:row=[float(v) for v in row]
        except (ValueError,OverflowError) as exc:raise ValueError(name+' cannot be represented as finite scores') from exc
        _check(all(math.isfinite(v) for v in row),name+' contains nonfinite scores')
        result[pid]=dict(zip(classes,row))
    return result


def _prediction(scores,classes):
    maximum=max(scores[c] for c in classes)
    ties=sorted(c for c in classes if scores[c]==maximum)
    # Identical to LocalRidge's lexical column order followed by np.argmax.
    return ties[0],ties,maximum


def _margin(a,b):
    if b is None:return None,'NO_COMPARISON_CLASS'
    value=a-b
    if not math.isfinite(value):return None,'DIFFERENCE_NOT_REPRESENTABLE_AS_FINITE_FLOAT'
    return value,None


def _transitions(before,after):
    result=dict(correct_to_correct=0,correct_to_incorrect=0,incorrect_to_correct=0,incorrect_to_incorrect=0)
    for left,right in zip(before,after):
        key=('correct' if left else 'incorrect')+'_to_'+('correct' if right else 'incorrect')
        result[key]+=1
    result['net_correct_loss']=result['correct_to_incorrect']-result['incorrect_to_correct']
    return result


def _quantity(numerator,denominator):
    fraction=numerator/denominator
    return dict(correct_count_difference=numerator,physical_count=denominator,
                accuracy_fraction=fraction,percentage_points=100*fraction)


def diagnose_registration(*,b_scores,b_classes,b_ids,c_scores,c_classes,c_ids,held_labels,old_classes):
    """Diagnose one fixed B/C pair, aligned by physical IDs and class identities.

    ``held_labels`` is the independent evaluator's {physical_id: class_id} map
    covering exactly C's IDs. B must contain exactly the old-labelled subset of
    C, with no silently discarded IDs. C may additionally contain new held IDs;
    their presence/count and full-registry accuracy are reported explicitly.
    Scores may have unrelated scales: only each head's own argmax is compared.
    The old-column counterfactual must never be used to mask deployed queries.
    """
    old=_strings(old_classes,'old_classes');bc=_strings(b_classes,'b_classes');cc=_strings(c_classes,'c_classes')
    bi=_strings(b_ids,'b_ids');ci=_strings(c_ids,'c_ids');oldset=set(old);cset=set(cc)
    _check(set(bc)==oldset,'B classes must equal the complete old registry')
    _check(oldset<=cset,'C registry is missing old classes')
    _check(isinstance(held_labels,Mapping),'held_labels must be an independent physical-ID to class-ID mapping')
    _check(set(held_labels)==set(ci),'held_labels must cover exactly C physical IDs; missing or extra IDs are not allowed')
    _check(all(isinstance(v,str) and v in cset for v in held_labels.values()),'Held label belongs to an unregistered class')
    _check(set(bi)<=set(ci),'B physical ID is missing from C')
    _check(all(held_labels[pid] in oldset for pid in bi),'B held labels must all belong to old classes')
    expected_old={pid for pid in ci if held_labels[pid] in oldset}
    _check(set(bi)==expected_old,'B must contain exactly the same complete old held physical-ID set as C')
    B=_scores(b_scores,bi,bc,'b_scores');C=_scores(c_scores,ci,cc,'c_scores')
    new=sorted(cset-oldset);old=sorted(oldset);old_records=[];new_records=[]
    ties=dict(B_old_top_tie_records=0,C_old_columns_top_tie_records=0,C_full_old_top_tie_records=0,
              C_full_new_top_tie_records=0,C_old_new_max_tie_records=0,C_true_old_max_new_tie_records=0)
    for pid in sorted(expected_old):
        truth=held_labels[pid]
        bp,bt,_=_prediction(B[pid],old);op,ot,omax=_prediction(C[pid],old);cp,ct,_=_prediction(C[pid],cc)
        nmax=max(C[pid][cls] for cls in new) if new else None
        values={};reasons={}
        for name,left,right in (
            ('C_true_old_minus_max_new',C[pid][truth],nmax),('C_max_old_minus_max_new',omax,nmax),
            ('B_true_minus_max_other_old',B[pid][truth],max((B[pid][cls] for cls in old if cls!=truth),default=None)),
            ('C_true_minus_max_other_old',C[pid][truth],max((C[pid][cls] for cls in old if cls!=truth),default=None))):
            values[name],reasons[name]=_margin(left,right)
        bo,co,cf=bp==truth,op==truth,cp==truth
        # If the best old-class decision was wrong, adding only new-class
        # competitors cannot make an old truth win with the same tie policy.
        _check(not (cf and not co),'Internal full-registry/old-column decision inconsistency')
        if not new:_check(op==cp and ot==ct,'No-new registry must have identical C old/full decisions')
        ties['B_old_top_tie_records']+=int(len(bt)>1)
        ties['C_old_columns_top_tie_records']+=int(len(ot)>1)
        ties['C_full_old_top_tie_records']+=int(len(ct)>1)
        ties['C_old_new_max_tie_records']+=int(nmax is not None and omax==nmax)
        ties['C_true_old_max_new_tie_records']+=int(nmax is not None and C[pid][truth]==nmax)
        old_records.append(dict(physical_id=pid,true_class=truth,B_predicted_class=bp,
            C_old_columns_predicted_class=op,C_full_predicted_class=cp,
            B_correct=bo,C_old_columns_correct=co,C_full_correct=cf,
            old_order_correct_loss=int(bo)-int(co),new_competition_correct_loss=int(co)-int(cf),
            total_correct_loss=int(bo)-int(cf),old_winner_changed=bp!=op,
            C_old_winner_displaced_by_new=cp in new,B_top_tie_classes=bt,
            C_old_columns_top_tie_classes=ot,C_full_top_tie_classes=ct,
            margins=values,margin_unavailable_reasons=reasons))
    for pid in sorted(set(ci)-expected_old):
        pred,tied,_=_prediction(C[pid],cc)
        ties['C_full_new_top_tie_records']+=int(len(tied)>1)
        new_records.append(dict(physical_id=pid,true_class=held_labels[pid],C_full_predicted_class=pred,
                                C_full_correct=pred==held_labels[pid],C_full_top_tie_classes=tied))
    n=len(old_records);bn=sum(r['B_correct'] for r in old_records)
    on=sum(r['C_old_columns_correct'] for r in old_records);cn=sum(r['C_full_correct'] for r in old_records)
    ordering=_transitions([r['B_correct'] for r in old_records],[r['C_old_columns_correct'] for r in old_records])
    competition=_transitions([r['C_old_columns_correct'] for r in old_records],[r['C_full_correct'] for r in old_records])
    overall=_transitions([r['B_correct'] for r in old_records],[r['C_full_correct'] for r in old_records])
    _check(competition['incorrect_to_correct']==0 and on-cn>=0,'New competition accuracy component must be nonnegative')
    _check((bn-on)+(on-cn)==bn-cn,'Internal exact decomposition failure')
    new_correct=sum(r['C_full_correct'] for r in new_records)
    new_accuracy=new_correct/len(new_records) if new_records else None
    old_accuracy=cn/n
    harmonic=(2*old_accuracy*new_accuracy/(old_accuracy+new_accuracy)
              if old_accuracy+new_accuracy else 0.) if new_accuracy is not None else None
    return dict(schema='d92_registration_score_diagnostics_v1',status='ANALYZED_FIXED_SCORES',analysis_only=True,
        fit_performed=False,query_role_mask_for_deployment_allowed=False,
        decision_rule='physical_class_id_lexicographic_exact_argmax_tie',old_classes=old,new_classes=new,
        old_held_count=n,additional_new_held_count=len(new_records),additional_new_held_explicitly_present=bool(new_records),
        B_acc=bn/n,C_old_columns_acc=on/n,C_full_acc=cn/n,
        Cnew_acc=new_accuracy,C_harmonic_mean=harmonic,
        Cnew_acc_unavailable_reason=None if new_records else 'NO_NEW_HELD_RECORDS',
        accuracy_unit='fraction',B_correct_count=bn,C_old_columns_correct_count=on,C_full_correct_count=cn,
        Cnew_correct_count=new_correct,
        decomposition=dict(old_order_change=_quantity(bn-on,n),new_competition_loss=_quantity(on-cn,n),
            total_old_accuracy_drop=_quantity(bn-cn,n),exact_count_identity=True,
            equation='(B_correct-C_old_columns_correct)+(C_old_columns_correct-C_full_correct)=B_correct-C_full_correct',
            accuracy_equation='(B_acc-C_old_columns_acc)+(C_old_columns_acc-C_full_acc)=B_acc-C_full_acc',
            interpretation='Exact integer-count identity over one common denominator; floating fractions/pp are representations of those counts'),
        transitions=dict(old_order_change=ordering,new_competition=competition,total=overall),
        old_winner_changed_count=sum(r['old_winner_changed'] for r in old_records),
        old_winner_displaced_by_new_count=sum(r['C_old_winner_displaced_by_new'] for r in old_records),
        tie_counts=ties,old_records=old_records,new_records=new_records,
        margin_scope='Each margin compares columns within its own fixed head only. B/C raw scores are never subtracted or concatenated; margin magnitudes across heads are not comparable.',
        limitation='Offline post-prediction counterfactual using independently supplied held labels. '
            'C_old_columns is not a deployed predictor and must not route or mask queries by their truth/role. '
            'The decomposition describes fixed decisions, not a causal training attribution or a new performance gate.')
