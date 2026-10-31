"""Shrinkage diagonal between/within weighting, fitted on current support only."""
from dataclasses import dataclass,replace
import numpy as np
from . import signature_kernel as sig,support_calibrated as base

STRENGTHS=(0.,.25,.5,.75)


@dataclass(frozen=True)
class Normalizer:
    weights:np.ndarray

    def __call__(self,features):
        out=sig.normalize(features);x=out['sig_views'].reshape(-1,3,112)*np.sqrt(self.weights)
        norms=np.linalg.norm(x,axis=2,keepdims=True)
        if not np.isfinite(norms).all() or (norms<=1e-12).any():raise FloatingPointError('Invalid weighted signature view')
        value=(x/norms/np.sqrt(3)).reshape(-1,336);value.flags.writeable=False;out['sig_views']=value
        return out


def make_normalizer(features,labels,strength):
    x=sig.normalize(features)['sig_views'].reshape(-1,3,112);y=np.asarray(labels,dtype=np.int64)
    c=int(y.max())+1;n=len(y);means=np.stack([x[y==i].mean(0) for i in range(c)])
    between=np.mean((means-means.mean(0))**2,axis=0,keepdims=True)
    within=np.sum((x-means[y])**2,axis=0,keepdims=True)/(n-c)
    ratio=between/(within+within.mean(2,keepdims=True)+np.finfo(np.float64).eps)
    average=ratio.mean(2,keepdims=True)
    # With no between-class evidence this view remains uniformly weighted.
    relative=np.divide(ratio,average,out=np.ones_like(ratio),where=average>0)
    weights=(1-strength)+strength*relative
    if not np.isfinite(weights).all() or (weights<=0).any():raise FloatingPointError('Invalid support Fisher weights')
    weights.flags.writeable=False
    return Normalizer(weights),int(np.count_nonzero(average==0))


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('fisher_strength') not in STRENGTHS:raise ValueError('Frozen source weighting strength required')
    if tuple(fixed['config']) not in sig.CONFIGS:raise ValueError('Wrong frozen signature rule')
    y=np.asarray(labels,dtype=np.int64)
    if y.ndim!=1 or not len(y) or (y<0).any():raise ValueError('Invalid support labels')
    counts=np.bincount(y);c=len(counts)
    if c<old_count or (counts!=counts[0]).any() or counts[0]<1:raise ValueError('Positive equal K per class required')
    strength=float(fixed['fisher_strength'])
    if strength==0 or counts[0]==1:
        state=sig.fit(features,y,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo)
        return replace(state,selection=dict(state.selection,fisher_weighting=dict(strength=strength,applied=False,K1_uniform=bool(counts[0]==1))))
    normalizer,uniform=make_normalizer(features,y,strength)
    state=base.fit(features,y,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo,
                   normalizer=normalizer,similarity_fn=sig.similarity,configs=sig.CONFIGS)
    return replace(state,selection=dict(state.selection,fisher_weighting=dict(strength=strength,applied=True,
        weights_from='current_support_labels_only',uniform_no_between_views=uniform,extra_frozen_array_bytes=normalizer.weights.nbytes,
        query_updates=0,within_between_conditioned_on_full_support=True)))
