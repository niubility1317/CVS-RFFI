"""Label-blind direct cross-RX/day neighbors and source-known edge precision.

Prototype-relation ideas motivate this source mining experiment; this is not
an OpenNCD reproduction. Components describe connectivity only. Future training
may consume direct edges, never propagated component pseudo-labels. No IO,
models, fitting on OE identity labels, true K, or target data enters here.
"""
from __future__ import annotations

import numpy as np


RECEIVERS=(1,3,4,6,8)
DAYS=(1,2,3)
FEATURE_DIM=160
MIN_EDGES=25
P1_RULE='P1_per_other_RX_cross_day_mutual_nearest'
PAIR_RULE='PAIR_per_other_RX_cross_day_both_views_same_mutual_nearest'


def _metadata(ids,rx,day):
    ids,rx,day=map(np.asarray,(ids,rx,day));n=len(ids) if ids.ndim==1 else 0
    if (not n or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)=='')
            or rx.shape!=(n,) or rx.dtype.kind not in 'iu' or not np.isin(rx,RECEIVERS).all()
            or day.shape!=(n,) or day.dtype.kind not in 'iu' or not np.isin(day,DAYS).all()):
        raise ValueError('Unique nonempty Unicode IDs and aligned legal source RX/day required')
    return ids,rx.astype(np.int64,copy=False),day.astype(np.int64,copy=False)


def _features(z,n):
    a=np.asarray(z)
    if a.shape!=(n,FEATURE_DIM) or a.dtype!=np.float32 or not np.isfinite(a).all():
        raise ValueError('Finite FP32[N,160] original F0 unit features required')
    x=a.astype(np.float64);norm=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.all(np.isclose(norm,1.,rtol=0.,atol=2e-6)):raise ValueError('Every source feature must have unit norm')
    return x/norm


def _choices(x,ids,rx,day,block_size):
    n=len(ids);order=np.argsort(ids,kind='stable');result=np.full((n,len(RECEIVERS)),-1,np.int64)
    for column,receiver in enumerate(RECEIVERS):
        reference=order[rx[order]==receiver]
        if not len(reference):continue
        for start in range(0,n,block_size):
            end=min(start+block_size,n);similarity=x[start:end]@x[reference].T
            valid=(rx[start:end,None]!=receiver)&(day[start:end,None]!=day[reference][None,:])
            selected=np.where(valid,similarity,-np.inf).argmax(axis=1)
            result[start:end,column]=np.where(valid.any(axis=1),reference[selected],-1)
    return result


def candidates(ids,z1,rx,day,z0=None,block_size=128):
    """Return undirected i<j edges sorted by distance and opaque endpoint IDs.

    P1 requires per-other-RX mutual nearest neighbors, excluding the node's
    day. PAIR additionally requires both directed choices to agree in P0/P1.
    Class labels never participate in any choice, filter, or tie break.
    """
    ids,rx,day=_metadata(ids,rx,day);n=len(ids)
    if type(block_size) is not int or block_size<=0:raise ValueError('Positive integer block_size required')
    x1=_features(z1,n);x0=None if z0 is None else _features(z0,n)
    choice1=_choices(x1,ids,rx,day,block_size);choice0=None if x0 is None else _choices(x0,ids,rx,day,block_size)
    source_column=np.searchsorted(RECEIVERS,rx);index=np.arange(n);edges=[]
    for column in range(len(RECEIVERS)):
        neighbor=choice1[:,column];usable=(neighbor>=0)&(index<neighbor);i=index[usable];j=neighbor[usable]
        mutual=choice1[j,source_column[i]]==i
        if choice0 is not None:mutual&=(choice0[i,column]==j)&(choice0[j,source_column[i]]==i)
        edges.append(np.column_stack((i[mutual],j[mutual])))
    pairs=np.concatenate(edges,axis=0).astype(np.int64,copy=False)
    distances=np.clip(1.-np.einsum('ed,ed->e',x1[pairs[:,0]],x1[pairs[:,1]],optimize=False),0.,2.)
    if x0 is not None:
        distances=np.maximum(distances,np.clip(1.-np.einsum('ed,ed->e',x0[pairs[:,0]],x0[pairs[:,1]],optimize=False),0.,2.))
    left,right=ids[pairs[:,0]],ids[pairs[:,1]];low=np.where(left<right,left,right);high=np.where(left<right,right,left)
    order=np.lexsort((high,low,distances))
    return dict(pair_indices=pairs[order].copy(),distances=distances[order].astype(np.float64,copy=True),
                rule=P1_RULE if x0 is None else PAIR_RULE)


def _pairs(value,n=None):
    pairs=np.asarray(value)
    if (pairs.ndim!=2 or pairs.shape[1]!=2 or pairs.dtype!=np.int64 or np.any(pairs<0)
            or np.any(pairs[:,0]>=pairs[:,1]) or (n is not None and np.any(pairs>=n))
            or len(np.unique(pairs,axis=0))!=len(pairs)):
        raise ValueError('Unique undirected int64[E,2] node-index pairs with i<j required')
    return pairs


def _candidate(candidate,n=None):
    if (not isinstance(candidate,dict) or set(candidate)!={'pair_indices','distances','rule'}
            or not isinstance(candidate['rule'],str) or not candidate['rule'].strip()):
        raise ValueError('Only label-free pair indices, distances and rule are accepted')
    pairs=_pairs(candidate['pair_indices'],n);distance=np.asarray(candidate['distances'])
    if distance.shape!=(len(pairs),) or distance.dtype!=np.float64 or not np.isfinite(distance).all() or np.any((distance<0)|(distance>2)):
        raise ValueError('Aligned finite FP64 cosine distances in [0,2] required')
    return pairs,distance


def _coverage(n,pairs,rx,y=None):
    covered=np.zeros(n,bool);covered[pairs.reshape(-1)]=True
    def row(mask):
        count=int(mask.sum());hit=int((mask&covered).sum())
        return dict(node_count=count,covered_node_count=hit,covered_node_fraction=hit/count if count else None)
    result=dict(edge_count=len(pairs),**row(np.ones(n,bool)),by_RX={str(r):row(rx==r) for r in RECEIVERS})
    if y is not None:result['by_TX']={str(t):row(y==t) for t in range(6)}
    return result


def calibrate(candidate,y,rx):
    """Largest whole-tie distance with >=98% precision and >=25 edges/group.

    Only source L's true old labels are legal. Global counts each edge once;
    each RX/TX group contains edges incident on at least one node in that
    group. Wrong edges count as errors in both incident TX groups. Feasibility
    need not be monotone, so every distinct complete distance level is checked.
    Node coverage is reported after selection and never optimizes the cutoff.
    """
    y,rx=np.asarray(y),np.asarray(rx);n=len(y) if y.ndim==1 else 0
    if (not n or y.dtype.kind not in 'iu' or not np.isin(y,range(6)).all()
            or rx.shape!=(n,) or rx.dtype.kind not in 'iu' or not np.isin(rx,RECEIVERS).all()):
        raise ValueError('Only aligned visible source L six-class labels and source RX are accepted')
    pairs,distance=_candidate(candidate,n);correct=y[pairs[:,0]]==y[pairs[:,1]]
    masks={'global':np.ones(len(pairs),bool)}|{
        'RX/'+str(r):(rx[pairs[:,0]]==r)|(rx[pairs[:,1]]==r) for r in RECEIVERS}|{
        'TX/'+str(t):(y[pairs[:,0]]==t)|(y[pairs[:,1]]==t) for t in range(6)}
    def groups(accepted):
        result={}
        for name,mask in masks.items():
            count=int((mask&accepted).sum());hits=int((mask&accepted&correct).sum())
            result[name]=dict(edge_count=count,correct_edge_count=hits,precision=hits/count if count else None,
                constraints_passed=bool(count>=MIN_EDGES and 50*hits>=49*count))
        return result
    order=np.argsort(distance,kind='stable');sorted_distance=distance[order]
    ends=np.r_[np.flatnonzero(sorted_distance[:-1]!=sorted_distance[1:]),len(order)-1] if len(order) else np.empty(0,np.int64)
    feasible=np.ones(len(ends),bool)
    for mask in masks.values():
        counts=np.cumsum(mask[order],dtype=np.int64)[ends];hits=np.cumsum((mask&correct)[order],dtype=np.int64)[ends]
        feasible&=(counts>=MIN_EDGES)&(50*hits>=49*counts)
    eligible=np.flatnonzero(feasible);threshold=float(sorted_distance[ends[eligible[-1]]]) if len(eligible) else None
    accepted=np.zeros(len(pairs),bool) if threshold is None else distance<=threshold
    return dict(threshold=threshold,constraints_passed=threshold is not None,groups=groups(accepted),
        full_candidate_groups=groups(np.ones(len(pairs),bool)),summary=_coverage(n,pairs[accepted],rx,y),
        candidate_edge_count=len(pairs),candidate_threshold_count=len(ends),feasible_threshold_count=int(feasible.sum()),
        minimum_group_edges=MIN_EDGES,minimum_group_precision=.98,exact_precision_check='50*correct >= 49*incident',
        group_scope='global once; each of five RX and six true known TX includes an edge if either endpoint belongs',
        threshold_rule='largest feasible observed distance; all equal distances included together; all distinct levels checked',
        coverage_used_for_threshold_selection=False,source_L_labels_used=True,OE_identity_labels_used=False,
        no_feasible_threshold_means_zero_edges=True)


def apply(candidate,threshold):
    """Apply one frozen source threshold to any label-free candidate graph."""
    pairs,distance=_candidate(candidate)
    if threshold is None:return np.empty((0,2),np.int64)
    if (isinstance(threshold,(bool,np.bool_)) or not isinstance(threshold,(float,int,np.floating,np.integer))
            or not np.isfinite(threshold) or not 0<=threshold<=2):
        raise ValueError('A finite frozen cosine-distance threshold or None is required')
    return pairs[distance<=threshold].copy()


def summary(ids,rx,day,pairs):
    """Coverage, degrees and diagnostic component sizes; never identity labels."""
    ids,rx,day=_metadata(ids,rx,day);n=len(ids);pairs=_pairs(pairs,n);result=_coverage(n,pairs,rx)
    degree=np.bincount(pairs.reshape(-1),minlength=n);levels,counts=np.unique(degree,return_counts=True)
    result['degree']=dict(minimum=int(degree.min()),maximum=int(degree.max()),mean=float(degree.mean()),
        median=float(np.median(degree)),isolated_nodes=int((degree==0).sum()),histogram={str(k):int(v) for k,v in zip(levels,counts)})
    result['by_day']={str(d):dict(node_count=int((day==d).sum()),covered_node_fraction=float((degree[day==d]>0).mean())
        if np.any(day==d) else None) for d in DAYS}
    parent=np.arange(n)
    def find(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=int(parent[index])
        return index
    for left,right in pairs:
        a,b=find(int(left)),find(int(right))
        if a!=b:parent[b]=a
    component_sizes=np.unique(np.asarray([find(i) for i in range(n)]),return_counts=True)[1]
    result.update(all_edges_cross_RX=bool(np.all(rx[pairs[:,0]]!=rx[pairs[:,1]])),
        all_edges_cross_day=bool(np.all(day[pairs[:,0]]!=day[pairs[:,1]])),
        same_RX_edges=int((rx[pairs[:,0]]==rx[pairs[:,1]]).sum()),same_day_edges=int((day[pairs[:,0]]==day[pairs[:,1]]).sum()),
        component_sizes=sorted(component_sizes.tolist(),reverse=True),component_count_diagnostic=len(component_sizes),
        components_are_identity_labels=False,component_count_is_discovered_class_count=False,
        training_edges='direct pairs only; do not propagate component labels',
        component_risk='A single erroneous edge can chain-merge identities; connectivity is not precision or true K')
    return result
