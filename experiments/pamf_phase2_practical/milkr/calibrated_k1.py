"""Source-calibrated K1 kernel with empirical-mean-loss ridge scaling."""
from dataclasses import replace
import numpy as np
from .nested_fisher import fit as nested_fit


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('ridge_reference_classes')!=6:raise ValueError('Frozen six-class source ridge reference required')
    y=np.asarray(labels,np.int64)
    if y.ndim!=1 or not len(y) or (y<0).any():raise ValueError('Invalid current support labels')
    count=int(y.max())+1;k=len(y)//count;rule=dict(fixed)
    reference=float(rule['ridge'])
    if not np.isfinite(reference) or reference<=0:raise ValueError('Positive source ridge required')
    if k==1:rule['ridge']=reference*count/6
    state=nested_fit(features,y,ids,old_count=old_count,fixed=rule,conditional_loo=conditional_loo)
    return replace(state,selection=dict(state.selection,K1_source_calibration=dict(reference_classes=6,current_classes=count,
        reference_ridge=reference,effective_ridge=float(state.ridge),class_count_scaling=k==1,
        current_support_only=True,query_count_or_role_read=False)))
