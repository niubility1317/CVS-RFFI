"""Truth-side paired feature comparison with an explicit bounded reference pool."""
import numpy as np


def capped_pair_margin(unit,reference_unit,labels,cap):
    if not isinstance(cap,int) or cap<1:
        raise ValueError('Positive fixed reference count required')
    if unit.shape!=reference_unit.shape or len(unit)!=len(labels) or not len(unit):
        raise ValueError('Matching paired features and truth-side labels required')
    index=np.linspace(0,len(unit)-1,min(cap,len(unit)),dtype=np.int64)
    margin=np.empty(len(unit),dtype=np.float32)
    valid=np.zeros(len(unit),dtype=bool)
    for begin in range(0,len(unit),2048):
        end=min(len(unit),begin+2048)
        scores=unit[begin:end]@reference_unit[index].T
        scores[labels[begin:end,None]==labels[index][None,:]]=-np.inf
        competitor=scores.max(1)
        valid[begin:end]=np.isfinite(competitor)
        paired=(unit[begin:end]*reference_unit[begin:end]).sum(1)
        margin[begin:end]=paired-competitor
    return dict(mean=float(margin[valid].mean()) if valid.any() else None,
        own_pair_higher_percent=float((margin[valid]>0).mean()*100) if valid.any() else None,
        inference_or_selection_use=False,reference_pool_count=len(index),query_count=len(unit),
        valid_comparison_count=int(valid.sum()),
        reference_pool_rule='Fixed evenly spaced opaque-record positions; independent of TX truth',
        full_global_nearest_other_TX=False)
