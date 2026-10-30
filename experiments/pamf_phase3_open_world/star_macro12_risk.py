"""Known-V common-tail risk: twelve hard groups, thirty cross diagnostics."""
from __future__ import annotations
from fractions import Fraction
import numpy as np
from . import star_class_risk as old

RUN_ID='20261030-phase3-star-macro12-risk-s2026092701-r01'
SCHEMA='pamf_phase3_star_macro12_prediction_v1'
STATE_SCHEMA='pamf_phase3_star_macro12_risk_state_v1'
CALIBRATION_STATUS='STAR_MACRO12_THRESHOLDS_FROZEN'
RECEIVERS,SCENES=old.RECEIVERS,old.SCENES
HARD_NAMES=('overall',)+tuple('rx/'+str(r) for r in RECEIVERS)+tuple('scene/'+s for s in SCENES)
DIAGNOSTIC_NAMES=tuple('rxscene/'+str(r)+'/'+s for r in RECEIVERS for s in SCENES)
FRR=Fraction(1,10)
TOTAL_LOSS=Fraction(1,50)


def calibrate_known(ids,y,closed,baseline,score,rx,scenario):
    """Return six deployment thresholds and all42 empirical risk diagnostics."""
    ids,y,closed,baseline,score,rx,scenario=map(np.asarray,(ids,y,closed,baseline,score,rx,scenario));n=len(ids) if ids.ndim==1 else 0
    if (not n or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)=='')
            or score.shape!=(n,) or score.dtype.kind!='f' or not np.isfinite(score).all()
            or any(a.shape!=(n,) or a.dtype.kind not in 'iu' for a in (y,closed,baseline,rx))
            or any(np.any((a<0)|(a>=6)) for a in (y,closed,baseline)) or not np.isin(rx,RECEIVERS).all()
            or scenario.shape!=(n,) or scenario.dtype.kind!='U' or not np.isin(scenario,SCENES).all()):
        raise ValueError('Aligned source-known V only: IDs, old-six labels/predictions, finite scores and source domains')
    masks=list(old.global_risk._masks(rx,scenario));coverage={k:set(y[m].tolist())==set(range(6)) for k,m in masks}
    missing={k:sorted(set(range(6))-set(y[m].tolist())) for k,m in masks if not coverage[k]}
    missing_predicted=sorted(set(range(6))-set(closed.tolist()));curves,candidates=old._curves(score,closed)
    def trial(q):
        tau,accepted,rejected=old._thresholds(q,curves,score,closed)
        groups=old._groups(y,closed,baseline,accepted,masks,coverage)
        passed=not missing_predicted and all(coverage[k] and groups[k]['constraints_passed'] for k in HARD_NAMES)
        return tau,groups,bool(passed),rejected
    zero=trial(Fraction(0));index=0
    if zero[2]:
        left,right=0,len(candidates)-1
        while left<right:
            middle=(left+right+1)//2
            if trial(candidates[middle])[2]:left=middle
            else:right=middle-1
        index=left
    q=candidates[index] if zero[2] else Fraction(0);tau,groups,passed,rejected=trial(q) if q else zero
    if not np.isfinite(tau).all():raise ValueError('Finite six-scalar deployment state required')
    following=candidates[index+1] if zero[2] and index+1<len(candidates) else None
    nt,ng,np_=None,None,None
    if following is not None:
        nt,ng,np_,_=trial(following)
    failures={k:ng[k] for k in HARD_NAMES if not ng[k]['constraints_passed']} if ng is not None else {}
    state=dict(schema=STATE_SCHEMA,conditional_thresholds=tau)
    report=dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,known_count=n,group_count=42,hard_group_count=12,diagnostic_group_count=30,
        hard_names=list(HARD_NAMES),diagnostic_names=list(DIAGNOSTIC_NAMES),groups=groups,constraints_passed=passed,
        no_rejection_feasible=zero[2],zero_rejection_fallback=not zero[2],conditional_thresholds=tau.tolist(),common_tail_rate=old._rate(q),
        coverage=dict(hard_groups_complete=all(coverage[k] for k in HARD_NAMES),all_groups_complete=all(coverage.values()),
            predicted_classes_complete=not missing_predicted,missing_known_classes_by_group=missing,missing_predicted_classes=missing_predicted),
        predicted_class_counts=np.bincount(closed,minlength=6).tolist(),selected_prefix_counts=rejected,candidate_count=len(candidates),
        selected_candidate_index=index if zero[2] else None,next_common_tail_rate=old._rate(following) if following is not None else None,
        next_constraints_passed=np_,next_conditional_thresholds=nt.tolist() if nt is not None and np.isfinite(nt).all() else None,
        next_groups=ng,next_failure_groups=failures,known_only=True,source_only=True,source_V_labels_used=True,
        source_V_neural_fit=False,source_V_normalizer_fit=False,source_V_threshold_fit=True,unknown_inputs_used=False,novel_inputs_used=False,target_reads=0,
        limits=dict(known_macro_acceptance_min=.90,original_closed_minus_new_open=.02),exact_limits=dict(known_macro_frr='1/10',original_closed_minus_new_open='1/50'),
        routing='own frozen old-six predicted class only',acceptance_rule='raw >= conditional threshold; complete ties',
        risk_scope='Empirical known-V twelve-group protection; RX-by-scene diagnostics are not hard constraints; no conformal guarantee')
    return state,report


def apply(score,closed,calibration):
    score,closed=np.asarray(score),np.asarray(closed)
    if (score.ndim!=1 or not len(score) or score.dtype.kind!='f' or not np.isfinite(score).all()
            or closed.shape!=score.shape or closed.dtype.kind not in 'iu' or np.any((closed<0)|(closed>=6))):
        raise ValueError('Finite raw evidence and aligned frozen predicted classes required')
    if not isinstance(calibration,dict) or set(calibration)!={'schema','conditional_thresholds'} or calibration['schema']!=STATE_SCHEMA:
        raise ValueError('Six-scalar frozen macro12 state required')
    tau=np.asarray(calibration['conditional_thresholds'])
    if tau.shape!=(6,) or tau.dtype!=np.float64 or not np.isfinite(tau).all():raise ValueError('Six finite FP64 thresholds required')
    with np.errstate(over='ignore',invalid='ignore'):result=score.astype(np.float64)-tau[closed]
    if not np.isfinite(result).all():raise ValueError('Finite macro12 margins required')
    return result
