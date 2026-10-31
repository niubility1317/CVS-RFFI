"""R6 kernel plus one support-calibrated old/new score offset.

No query input is accepted by fit. Old/new are registered support identities,
never a query-role oracle. K1 requires a source-frozen prior coefficient.
"""
from dataclasses import replace
import numpy as np
from . import support_calibrated as base


def score_scale(scores):
    scores=np.asarray(scores,dtype=np.float64)
    scale=float(np.sqrt(np.mean((scores-scores.mean(1,keepdims=True))**2)))
    if not np.isfinite(scale) or scale<=0:raise FloatingPointError('No positive support score scale')
    return scale


def add_offset(scores,delta,old_count):
    out=np.array(scores,dtype=np.float64,copy=True);out[:,:old_count]+=delta
    return out


def choose_offset(scores,labels,old_count):
    scores=np.asarray(scores,dtype=np.float64);y=np.asarray(labels,dtype=np.int64)
    if scores.shape!=(len(y),int(y.max())+1) or not np.isfinite(scores).all() or not 0<old_count<scores.shape[1]:
        raise ValueError('Invalid support calibration scores')
    gaps=scores[:,old_count:].max(1)-scores[:,:old_count].max(1)
    gaps=np.unique(gaps)
    candidates=np.unique(np.r_[0.,np.nextafter(gaps[0],-np.inf),gaps[:-1]+np.diff(gaps)/2,np.nextafter(gaps[-1],np.inf)])
    best=None
    for delta in candidates:
        pred=add_offset(scores,float(delta),old_count).argmax(1)
        h=base.target_and_score(y,pred,old_count)
        rank=(h,-abs(float(delta)),-float(delta))
        if best is None or rank>best[0]:best=(rank,float(delta))
    return best[1],dict(conditional_LOO_H_before=base.target_and_score(y,scores.argmax(1),old_count),
        conditional_LOO_H_after=best[0][0],thresholds=len(candidates),tie='closest_zero_then_smaller_delta')


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    state=base.fit(features,labels,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo)
    y=np.asarray(labels,dtype=np.int64);count=int(y.max())+1;k=len(y)//count
    if count==old_count:
        delta=0.;audit=dict(rule='B_no_new_group')
    elif k==1:
        if fixed is None or 'old_new_beta' not in fixed:raise ValueError('K1 requires frozen source old/new coefficient')
        beta=float(fixed['old_new_beta'])
        if not np.isfinite(beta):raise ValueError('Nonfinite source coefficient')
        scale=score_scale(state.score(features));delta=beta*np.log((count-old_count)/old_count)*scale
        audit=dict(rule='K1_source_beta_log_group_count_ratio',beta=beta,support_score_scale=scale)
    else:
        g,_,mix,scales=state.config
        K=base.make_kernel(base.similarity(state.features,state.features,g),state.tau,mix,scales)
        _,_,loo=base.coefficients(base.decompose(K,y),state.ridge)
        delta,audit=choose_offset(loo,y,old_count);audit['rule']='support_conditional_LOO_group_H'
    intercept=state.intercept.copy();intercept[:old_count]+=delta;intercept.flags.writeable=False
    audit.update(delta=float(delta),registered_old_count=old_count,registered_new_count=count-old_count,
                 query_role_read=False,query_fit=False)
    return replace(state,intercept=intercept,selection=dict(state.selection,group_calibration=audit))
