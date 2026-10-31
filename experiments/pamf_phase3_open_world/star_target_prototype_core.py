"""Fixed-capacity balanced target prototypes; no identities or true class count."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F

CAPACITY=16
SINKHORN_ITERATIONS=20
EPSILON=.05
TEMPERATURE=.1
INITIALIZATION_SEED=2026103113
LLOYD_ITERATIONS=25


def _matrix(x,name):
    if (not torch.is_tensor(x) or x.ndim!=2 or not len(x) or x.shape[1]!=160 or x.dtype!=torch.float32
            or not bool(torch.isfinite(x).all()) or bool(x.norm(dim=1).eq(0).any())):
        raise ValueError('Nonzero finite FP32 [N,160] '+name+' required')
    return x


def balanced_assignments(weak,prototypes):
    _matrix(weak,'weak features');_matrix(prototypes,'prototypes')
    if prototypes.shape!=(CAPACITY,160) or prototypes.device!=weak.device:raise ValueError('Sixteen same-device prototypes required')
    with torch.no_grad():
        logq=F.normalize(weak.detach(),dim=1).double()@F.normalize(prototypes.detach(),dim=1).double().T/EPSILON
        target=float(len(weak))/CAPACITY
        for _ in range(SINKHORN_ITERATIONS):
            logq=logq-torch.logsumexp(logq,dim=0,keepdim=True)+np.log(target)
            logq=logq-torch.logsumexp(logq,dim=1,keepdim=True)
        return logq.exp().float().detach()


def initialize_prototypes(features):
    _matrix(features,'frozen parent weak features')
    if len(features)<CAPACITY:raise ValueError('At least sixteen unlabeled records required for fixed capacity')
    x=np.asarray(features.detach().cpu().tolist(),np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True)
    rng=np.random.default_rng(INITIALIZATION_SEED);selected=[int(rng.integers(len(x)))];closest=np.maximum(2-2*x@x[selected[0]],0)
    for _ in range(1,CAPACITY):
        distance=closest.copy();distance[selected]=0;total=float(distance.sum())
        index=int(rng.choice(len(x),p=distance/total)) if total>0 else next(j for j in range(len(x)) if j not in selected)
        selected.append(index);closest=np.minimum(closest,np.maximum(2-2*x@x[index],0))
    centers=x[selected].copy();counts=None
    for _ in range(LLOYD_ITERATIONS):
        assignments=(x@centers.T).argmax(1);counts=np.bincount(assignments,minlength=CAPACITY)
        for j in range(CAPACITY):
            if counts[j]:
                mean=x[assignments==j].mean(0);norm=float(np.linalg.norm(mean))
                if norm>0:centers[j]=mean/norm
    result=torch.tensor(centers.astype(np.float32).tolist(),dtype=torch.float32,device=features.device)
    return F.normalize(result,dim=1),dict(seed=INITIALIZATION_SEED,iterations=LLOYD_ITERATIONS,capacity=CAPACITY,
        last_assignment_counts=counts.tolist(),empty_centers_retained=True,capacity_is_true_class_count=False)


def nearest_neighbors(features,ids=None):
    _matrix(features,'parent weak neighbors');n=len(features)
    if n<2:raise ValueError('Another physical record is required for every local neighbor')
    if ids is None:order=np.arange(n)
    else:
        ids=np.asarray(ids)
        if ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)==''):
            raise ValueError('Unique nonempty aligned opaque IDs required')
        order=np.argsort(ids,kind='stable')
    x=np.asarray(features.detach().cpu().tolist(),np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True)
    similarity=x@x[order].T;similarity[np.arange(n),np.argsort(order)]=-np.inf
    return torch.tensor(order[similarity.argmax(1)].tolist(),dtype=torch.long,device=features.device)


def consistency(weak,strong0,strong1,prototypes,indices,neighbors):
    _matrix(weak,'current weak features');_matrix(strong0,'strong view zero');_matrix(strong1,'strong view one')
    n=len(weak);b=len(strong0)
    if (strong1.shape!=strong0.shape or any(x.device!=weak.device for x in (strong0,strong1,prototypes))
            or not torch.is_tensor(indices) or indices.dtype!=torch.long or indices.shape!=(b,) or indices.device!=weak.device
            or bool(((indices<0)|(indices>=n)).any()) or not torch.is_tensor(neighbors) or neighbors.dtype!=torch.long
            or neighbors.shape!=(n,) or neighbors.device!=weak.device or bool(((neighbors<0)|(neighbors>=n)).any())
            or bool(neighbors.eq(torch.arange(n,device=weak.device)).any())):
        raise ValueError('Aligned strong batches, U indices and fixed nonself parent neighbors required')
    q=balanced_assignments(weak,prototypes);unit=F.normalize(prototypes,dim=1)
    logp=[(F.normalize(view,dim=1)@unit.T/TEMPERATURE).log_softmax(1) for view in (strong0,strong1)]
    global_loss=-.5*sum((q[indices]*value).sum(1).mean() for value in logp)
    local_loss=-.5*sum((q[neighbors[indices]]*value).sum(1).mean() for value in logp)
    mass=q.sum(0)
    return dict(loss=global_loss+local_loss,**{'global':global_loss,'local':local_loss},
        prototype_mass_min=mass.min().detach(),prototype_mass_max=mass.max().detach(),assignments_detached=True)
