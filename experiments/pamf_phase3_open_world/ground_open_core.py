"""Known-only calibration and explicit receiver/date groups for ground training."""
from fractions import Fraction
import numpy as np
from . import star_class_risk as previous

SCHEMA='pamf_phase3_ground_open_calibration_v1'
RECEIVERS=(1,3,4,6,8)
DAYS=(1,2,3)
GROUP_NAMES=('overall',)+tuple('rx/'+str(r) for r in RECEIVERS)+tuple('day/'+str(d) for d in DAYS)+tuple('rxday/'+str(r)+'/'+str(d) for r in RECEIVERS for d in DAYS)


def masks(rx,day):
    rx,day=np.asarray(rx),np.asarray(day)
    if (rx.ndim!=1 or not len(rx) or day.shape!=rx.shape or rx.dtype.kind not in 'iu' or day.dtype.kind not in 'iu'
            or not np.isin(rx,RECEIVERS).all() or not np.isin(day,DAYS).all()):
        raise ValueError('Aligned public source receiver/date codes required')
    return {'overall':np.ones(len(rx),bool)}|{'rx/'+str(r):rx==r for r in RECEIVERS}|{'day/'+str(d):day==d for d in DAYS}|{'rxday/'+str(r)+'/'+str(d):(rx==r)&(day==d) for r in RECEIVERS for d in DAYS}


def energy(logits):
    a=np.asarray(logits)
    if a.ndim!=2 or a.shape[1]!=6 or not len(a) or a.dtype!=np.float32 or not np.isfinite(a).all():
        raise ValueError('Finite complete-six native FP32 logits required')
    return np.logaddexp.reduce(a.astype(np.float64),axis=1)


def _known_groups(y,closed,baseline,accepted,groups):
    result={}
    for name,mask in groups.items():
        complete=set(y[mask].tolist())==set(range(6))
        if not complete:
            result[name]=dict(count=int(mask.sum()),coverage_complete=False,C_K=None,A_K=None,original_F0_closed=None,original_closed_minus_open=None,passed=False)
            continue
        macro=lambda a:float(np.mean([a[mask&(y==c)].mean() for c in range(6)]))
        ck,ak,base=macro(accepted),macro(accepted&(closed==y)),macro(baseline==y)
        result[name]=dict(count=int(mask.sum()),coverage_complete=True,C_K=ck,A_K=ak,original_F0_closed=base,original_closed_minus_open=base-ak,
            passed=bool(ck>=.95-1e-12 and base-ak<=.02+1e-12))
    return result


def calibrate_known(ids,y,closed,baseline,raw,rx,day):
    """Maximal common empirical lower tail preserving all 24 known groups.

    This consumes only the separate known calibration subset. Unknown/test
    observations, their class count, and class quotas have no interface here.
    """
    ids,y,closed,baseline,raw,rx,day=map(np.asarray,(ids,y,closed,baseline,raw,rx,day));n=len(ids) if ids.ndim==1 else 0
    if (not n or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)=='')
            or raw.shape!=(n,) or raw.dtype.kind!='f' or not np.isfinite(raw).all()
            or any(a.shape!=(n,) or a.dtype.kind not in 'iu' or not np.isin(a,range(6)).all() for a in (y,closed,baseline))):
        raise ValueError('Known calibration IDs/y/own prediction/original baseline and finite evidence required')
    groups=masks(rx,day)
    if len(rx)!=n:raise ValueError('Source calibration public metadata must align')
    curves,candidates=previous._curves(raw,closed);missing=sorted(set(range(6))-set(closed.tolist()))
    def trial(q):
        tau,accepted,counts=previous._thresholds(q,curves,raw,closed)
        report=_known_groups(y,closed,baseline,accepted,groups)
        passed=not missing and all(v['passed'] for v in report.values())
        return tau,report,bool(passed),counts
    at_zero=trial(Fraction(0));index=0
    if at_zero[2]:
        lo,hi=0,len(candidates)-1
        while lo<hi:
            mid=(lo+hi+1)//2
            if trial(candidates[mid])[2]:lo=mid
            else:hi=mid-1
        index=lo
    q=candidates[index] if at_zero[2] else Fraction(0)
    tau,report,passed,counts=trial(q)
    if tau.shape!=(6,) or tau.dtype!=np.float64 or not np.isfinite(tau).all():raise ValueError('Finite source-frozen six-class thresholds required')
    state=dict(schema=SCHEMA,conditional_thresholds=tau)
    info=dict(schema=SCHEMA,conditional_thresholds=tau.tolist(),calibration_count=n,group_names=list(GROUP_NAMES),groups=report,
        constraints_passed=passed,zero_tail_feasible=at_zero[2],common_tail_fraction=str(q),predicted_class_missing=missing,
        rejected_counts_by_prediction=counts,known_only=True,unknown_inputs_used=False,test_inputs_used=False,U_labels_used=False,
        limits=dict(C_K=.95,original_F0_closed_minus_A_K=.02),
        failure_policy='If the student already violates known protection, freeze zero-tail source thresholds and report failure; no test-driven repair',
        missing_prediction_policy='Existing source common-tail implementation uses the global minimum for absent predicted classes; missing classes explicitly fail calibration',
        empirical_guarantee_only=True,global_history_independent_blind_test=False)
    return state,info


def apply(raw,closed,state):
    a,c=np.asarray(raw),np.asarray(closed)
    if (a.ndim!=1 or not len(a) or a.dtype.kind!='f' or not np.isfinite(a).all() or c.shape!=a.shape
            or c.dtype.kind not in 'iu' or not np.isin(c,range(6)).all()
            or not isinstance(state,dict) or set(state)!={'schema','conditional_thresholds'} or state['schema']!=SCHEMA):
        raise ValueError('Finite evidence, own predicted identity and exact ground calibration state required')
    tau=np.asarray(state['conditional_thresholds'])
    if tau.shape!=(6,) or tau.dtype!=np.float64 or not np.isfinite(tau).all():raise ValueError('Six finite FP64 thresholds required')
    result=a.astype(np.float64)-tau[c]
    if not np.isfinite(result).all():raise ValueError('Finite source-frozen margins required')
    return result
