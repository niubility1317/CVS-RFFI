"""Select descriptor-weight strength by physical support folds, never queries."""
from dataclasses import replace
import numpy as np
from . import fisher_signature as fisher,support_calibrated as base


def folds(labels,ids,maximum=3):
    y=np.asarray(labels,dtype=np.int64);ids=tuple(map(str,ids))
    if y.ndim!=1 or not len(y) or (y<0).any() or len(ids)!=len(y) or len(set(ids))!=len(ids):
        raise ValueError('Invalid physical support for outer folds')
    counts=np.bincount(y);k=int(counts[0]);c=len(counts)
    if (counts!=k).any() or k<2 or maximum!=3:raise ValueError('Balanced support K>=2 and frozen3 folds required')
    groups=[np.asarray(sorted(np.flatnonzero(y==i),key=lambda j:ids[j])) for i in range(c)]
    result=[]
    for fold in range(min(maximum,k)):
        held=np.concatenate([g[fold::min(maximum,k)] for g in groups]);keep=np.ones(len(y),bool);keep[held]=False;train=np.flatnonzero(keep)
        if set(ids[i] for i in train)&set(ids[i] for i in held):raise ValueError('Physical outer-fold overlap')
        result.append((train,held))
    return result


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('outer_folds')!=3:raise ValueError('Frozen outer support-fold rule required')
    if tuple(fixed.get('outer_strengths',()))!=fisher.STRENGTHS:raise ValueError('Wrong frozen strength grid')
    y=np.asarray(labels,dtype=np.int64);ids=tuple(map(str,ids))
    if y.ndim!=1 or not len(y) or (y<0).any() or len(set(ids))!=len(y) or len(ids)!=len(y):raise ValueError('Invalid physical support')
    counts=np.bincount(y);k=int(counts[0]);c=len(counts)
    if c<old_count or (counts!=k).any() or k<1:raise ValueError('Positive equal K per class required')
    control=fixed.get('force_uniform',False)
    if control not in (True,False):raise ValueError('Invalid control rule')
    audits=[];objectives={}
    if k==1 or control:selected=0.
    else:
        split=folds(y,ids);seen=np.concatenate([held for _,held in split])
        if sorted(seen.tolist())!=list(range(len(y))):raise ValueError('Outer folds do not cover support once')
        for strength in fisher.STRENGTHS:
            out=np.full(len(y),-1,np.int64)
            for train,held in split:
                state=fisher.fit({name:value[train] for name,value in features.items()},y[train],[ids[i] for i in train],
                    old_count=old_count,fixed=dict(fixed,fisher_strength=strength))
                scores=state.score({name:value[held] for name,value in features.items()});out[held]=scores.argmax(1)
            objectives[str(strength)]=base.target_and_score(y,out,old_count)
        # Smaller strength wins an exact tie; labels only from allocated support.
        selected=max(fisher.STRENGTHS,key=lambda a:(objectives[str(a)],-a))
        audits=[dict(train=len(train),held=len(held),train_K=int(np.bincount(y[train])[0]),held_K=int(np.bincount(y[held])[0])) for train,held in split]
    state=fisher.fit(features,y,ids,old_count=old_count,fixed=dict(fixed,fisher_strength=selected),conditional_loo=conditional_loo)
    return replace(state,selection=dict(state.selection,outer_support_selection=dict(selected_strength=selected,
        objectives=objectives,folds=audits,held_physical_support_excluded_from_weights_kernel_and_coefficients=True,
        support_coverage_once=not control and k>1,K1_uniform=k==1,control=control,query_fit=False)))
