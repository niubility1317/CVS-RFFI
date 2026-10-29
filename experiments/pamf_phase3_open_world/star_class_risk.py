"""Known-only empirical risk calibration routed by the frozen predicted class.

No IO, neural fitting, geometry/normalizer fitting, unknown inputs or target
inputs occur here. Source score curves are temporary calibration workspace;
deployment stores only the global control scalar and six conditional scalars.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import star_consensus_readout as global_risk


RUN_ID='20261029-phase3-star-class-risk-s2026092701-r01'
SCHEMA='pamf_phase3_star_class_risk_prediction_v1'
STATE_SCHEMA='pamf_phase3_star_class_risk_state_v1'
CALIBRATION_STATUS='STAR_CLASS_RISK_THRESHOLDS_FROZEN'
METHODS=('global_control','predicted_class')
PRIMARY='predicted_class'
PRIMARY_INDEX=1
IDENTITY_PATH='P1'
RECEIVERS=global_risk.RECEIVERS
SCENES=global_risk.SCENES
FRR=Fraction(1,10)
TOTAL_LOSS=Fraction(1,50)


def _rate(q):
    return dict(numerator=q.numerator,denominator=q.denominator)


def _curves(score, closed):
    curves=[];candidates={Fraction(0),Fraction(1)}
    for c in range(6):
        values=score[closed==c]
        if not len(values):curves.append(None);continue
        levels,sizes=np.unique(values,return_counts=True)
        prefix=np.r_[np.int64(0),np.cumsum(sizes,dtype=np.int64)]
        curves.append((levels,prefix,len(values)))
        candidates.update(Fraction(int(count),len(values)) for count in prefix)
    return curves,sorted(candidates)


def _thresholds(q, curves, score, closed):
    thresholds=np.full(6,float(score.min()),np.float64)
    accepted=np.ones(len(score),bool);rejected_counts=[]
    for c,curve in enumerate(curves):
        if curve is None:rejected_counts.append(0);continue
        levels,prefix,n=curve
        budget=(q.numerator*n)//q.denominator
        block=int(np.searchsorted(prefix,budget,side='right')-1)
        rejected_counts.append(int(prefix[block]))
        if block==len(levels):
            with np.errstate(over='ignore'):thresholds[c]=np.nextafter(float(levels[-1]),np.inf)
            accepted[closed==c]=False
        else:
            thresholds[c]=levels[block]
            accepted[closed==c]=score[closed==c]>=thresholds[c]
    return thresholds,accepted,rejected_counts


def _groups(y, closed, baseline, accepted, masks, coverage):
    # Reuse the existing exact Fraction risk accounting on this decision mask.
    # Passing 0/1 evidence at threshold .5 does not refit or transform knownness.
    return {key:global_risk._risk_group(y[mask],closed[mask],baseline[mask],accepted[mask].astype(np.float64),.5)
        if coverage[key] else dict(certified=False,constraints_passed=False,reason='missing_known_class',
            known_count=int(mask.sum()),class_counts=np.bincount(y[mask],minlength=6).tolist()) for key,mask in masks}


def calibrate_known(ids, y, closed, baseline, score, rx, scenario):
    """Return (small deployment state, JSON calibration diagnostics).

    Candidate q is the union of complete-tie prefix fractions in each *predicted*
    class. Each prefix obeys floor(q*n_c). All 42 groups use true-known-class
    macro risks for auditing; true labels never route the deployed threshold.
    """
    score=np.asarray(score)
    if score.ndim!=1 or not len(score) or score.dtype.kind!='f' or not np.isfinite(score).all():
        raise ValueError('One aligned finite floating raw knownness vector is required')
    # This also validates all known-only labels, IDs, RX and scene metadata.
    global_threshold,control_report=global_risk.calibrate_known(ids,y,closed,baseline,score[:,None],rx,scenario,
        method_names=('global_control',))
    ids,y,closed,baseline,rx,scenario=map(np.asarray,(ids,y,closed,baseline,rx,scenario))
    masks=list(global_risk._masks(rx,scenario))
    coverage={key:set(y[mask].tolist())==set(range(6)) for key,mask in masks}
    missing_known={key:sorted(set(range(6))-set(y[mask].tolist())) for key,mask in masks if not coverage[key]}
    missing_predicted=sorted(set(range(6))-set(closed.tolist()))
    cover=dict(known_groups_complete=all(coverage.values()),predicted_classes_complete=not missing_predicted,
        missing_known_classes_by_group=missing_known,missing_predicted_classes=missing_predicted)
    curves,candidates=_curves(score,closed)
    def trial(q):
        thresholds,accepted,rejected=_thresholds(q,curves,score,closed)
        groups=_groups(y,closed,baseline,accepted,masks,coverage)
        feasible=not missing_predicted and all(coverage.values()) and all(row['constraints_passed'] for row in groups.values())
        return thresholds,groups,bool(feasible),rejected
    zero=trial(Fraction(0));zero_feasible=zero[2]
    index=0
    if zero_feasible:
        left,right=0,len(candidates)-1
        while left<right:
            middle=(left+right+1)//2
            if trial(candidates[middle])[2]:left=middle
            else:right=middle-1
        index=left
    q=candidates[index] if zero_feasible else Fraction(0)
    thresholds,groups,passed,rejected=trial(q) if q else zero
    # q=1 necessarily fails FRR<=.1, so a selected finite state cannot contain
    # an unrepresentable nextafter(float64.max,+inf). It may appear only in the
    # failed-candidate diagnostic, represented as None rather than JSON inf.
    if not np.isfinite(thresholds).all():raise ValueError('Finite six-scalar deployment thresholds required')
    following=candidates[index+1] if zero_feasible and index+1<len(candidates) else None
    next_thresholds,next_groups,next_pass=(None,None,None)
    if following is not None:
        nt,ng,np_,_=trial(following)
        next_thresholds=nt.tolist() if np.isfinite(nt).all() else None
        next_groups={key:value for key,value in ng.items() if not value['constraints_passed']}
        next_pass=bool(np_)
    predicted=dict(constraints_passed=passed,no_rejection_feasible=zero_feasible,zero_rejection_fallback=not zero_feasible,
        coverage=cover,groups=groups,conditional_thresholds=thresholds.tolist(),common_tail_rate=_rate(q),
        predicted_class_counts=np.bincount(closed,minlength=6).tolist(),selected_prefix_counts=rejected,
        candidate_count=len(candidates),selected_candidate_index=index if zero_feasible else None,
        next_common_tail_rate=_rate(following) if following is not None else None,
        next_constraints_passed=next_pass,next_failure_groups=next_groups or {})
    control=control_report['methods']['global_control']
    control.update(global_threshold=float(global_threshold[0]),coverage=dict(
        known_groups_complete=all(coverage.values()),missing_known_classes_by_group=missing_known))
    state=dict(schema=STATE_SCHEMA,global_threshold=float(global_threshold[0]),conditional_thresholds=thresholds)
    report=dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,
        identity_path=IDENTITY_PATH,primary_path=IDENTITY_PATH,known_count=len(score),group_count=42,known_only=True,source_only=True,
        global_threshold=state['global_threshold'],conditional_thresholds=thresholds.tolist(),common_tail_rate=_rate(q),
        methods=dict(global_control=control,predicted_class=predicted),coverage=cover,
        constraints_passed=bool(control['constraints_passed'] and passed),
        next_common_tail_rate=_rate(following) if following is not None else None,
        next_constraints_passed=next_pass,next_conditional_thresholds=next_thresholds,next_failure_groups=next_groups or {},
        source_V_labels_used=True,source_V_neural_fit=False,source_V_normalizer_fit=False,source_V_threshold_fit=True,
        unknown_inputs_used=False,novel_inputs_used=False,target_reads=0,
        limits=dict(known_macro_acceptance_min=.9,known_macro_frr=.1,original_closed_minus_new_open=.02),
        exact_limits=dict(known_macro_frr='1/10',original_closed_minus_new_open='1/50'),
        routing='own frozen six-class predicted label; never true TX',
        acceptance_rule='raw knownness >= threshold indexed by own frozen predicted class; complete ties',
        risk_scope='Empirical known-V calibration only; no conformal coverage guarantee',
        deployment_state='one global control scalar and six conditional scalars; no CDF or source row bank')
    return state,report


def apply(score, closed, calibration):
    """Return [global, predicted-class] FP64 margins; acceptance is margin>=0."""
    score,closed=np.asarray(score),np.asarray(closed)
    if (score.ndim!=1 or not len(score) or score.dtype.kind!='f' or not np.isfinite(score).all()
            or closed.shape!=score.shape or closed.dtype.kind not in 'iu' or np.any((closed<0)|(closed>=6))):
        raise ValueError('Aligned finite score and frozen predicted-class vectors are required')
    if (not isinstance(calibration,dict) or set(calibration)!={'schema','global_threshold','conditional_thresholds'}
            or calibration['schema']!=STATE_SCHEMA or not isinstance(calibration['global_threshold'],(float,np.floating))
            or not np.isfinite(calibration['global_threshold'])):
        raise ValueError('Small frozen class-risk scalar state required')
    thresholds=np.asarray(calibration['conditional_thresholds'])
    if thresholds.shape!=(6,) or thresholds.dtype!=np.float64 or not np.isfinite(thresholds).all():
        raise ValueError('Exactly six finite FP64 predicted-class thresholds are required')
    with np.errstate(over='ignore',invalid='ignore'):
        result=np.column_stack((score.astype(np.float64)-calibration['global_threshold'],
            score.astype(np.float64)-thresholds[closed]))
    if not np.isfinite(result).all():raise ValueError('Finite class-risk margins required')
    return result
