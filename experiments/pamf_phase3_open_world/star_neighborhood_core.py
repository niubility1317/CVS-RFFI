"""All-U average-linkage neighborhoods, frozen local fusion and registration.

This is a project candidate, not a native FINCH implementation. No IO, true
labels, true class count, model fitting or Q statistics occur here. The state
contains only cluster aggregates; per-U diagnostics are returned separately.
"""
from __future__ import annotations

from math import isqrt

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score as _silhouette_score


RUN_ID='20261029-phase3-star-neighborhood-s2026092701-r01'
STATE_SCHEMA='pamf_phase3_star_neighborhood_state_v1'
DIAGNOSTIC_SCHEMA='pamf_phase3_star_neighborhood_diagnostics_v1'
DIMENSION=688
MIN_CLUSTER=8
MIN_REGISTER=16
FUSION_WEIGHT=.5
RADIUS_QUANTILE=.95
RADIUS_FLOOR=1e-6
LOGIT_TEMPERATURE=30.


def _matrix(value, columns, dtype, name, count=None):
    result=np.asarray(value)
    if (result.ndim!=2 or not len(result) or result.shape[1]!=columns or result.dtype!=np.dtype(dtype)
            or (count is not None and len(result)!=count) or not np.isfinite(result).all()):
        raise ValueError('Aligned nonempty finite matrix with declared dtype required: '+name)
    return result


def _vector(value, count, dtype, name):
    result=np.asarray(value)
    if result.shape!=(count,) or result.dtype!=np.dtype(dtype) or not np.isfinite(result).all():
        raise ValueError('Aligned finite vector with declared dtype required: '+name)
    return result


def _inputs(embedding, logits, margins):
    z=_matrix(embedding,DIMENSION,np.float32,'embedding')
    l=_matrix(logits,6,np.float32,'complete logits',len(z))
    m=_vector(margins,len(z),np.float64,'raw margin')
    return z.astype(np.float64),l.astype(np.float64),m


def _canonical_labels(labels):
    rows=[np.flatnonzero(labels==label) for label in np.unique(labels)]
    rows.sort(key=lambda indices:int(indices[0]))  # Input is opaque-ID sorted.
    result=np.empty(len(labels),np.int64)
    for index,indices in enumerate(rows):result[indices]=index
    return result,rows


def fit(embedding_U, logits_U, margins_U, opaque_ids, anchor_embedding, anchor_margin):
    z,logits,margin=_inputs(embedding_U,logits_U,margins_U);n=len(z)
    ids=np.asarray(opaque_ids)
    if (ids.shape!=(n,) or ids.dtype.kind!='U' or np.any(np.char.strip(ids)=='') or len(set(ids.tolist()))!=n):
        raise ValueError('Unique aligned physical opaque IDs are required')
    anchors=_matrix(anchor_embedding,DIMENSION,np.float32,'ninety source group anchors',90).astype(np.float64)
    anchor_q=_vector(anchor_margin,90,np.float64,'ninety source raw margins')
    order=np.argsort(ids,kind='stable');z=z[order];logits=logits[order];margin=margin[order]
    best_rows=[];best_labels=np.full(n,-1,np.int64);best_score=None;candidates=[]
    for k in range(2,isqrt(n)+1):
        raw=AgglomerativeClustering(n_clusters=k,metric='euclidean',linkage='average').fit_predict(z)
        labels,rows=_canonical_labels(raw);sizes=[len(indices) for indices in rows]
        eligible=min(sizes)>=MIN_CLUSTER
        value=float(_silhouette_score(z,labels,metric='euclidean')) if eligible else None
        if value is not None and not np.isfinite(value):raise ValueError('Finite full-U silhouette required')
        candidates.append(dict(K=k,cluster_sizes=sizes,eligible=eligible,silhouette=value,
            reason=None if eligible else 'cluster_smaller_than_8'))
        # Ascending K and strict improvement implement exact-tie smaller-K priority.
        if eligible and (best_score is None or value>best_score):
            best_rows,best_labels,best_score=rows,labels,value
    count=len(best_rows)
    means=np.empty((count,DIMENSION),np.float64);mean_logits=np.empty((count,6),np.float64)
    mean_margin=np.empty(count,np.float64);radius=np.empty(count,np.float64)
    anonymous=[];penalties=[];clusters=[]
    assignment=np.full(n,-1,np.int64);candidate=np.zeros(n,bool)
    for c,rows in enumerate(best_rows):
        current=z[rows];means[c]=current.mean(axis=0);mean_logits[c]=logits[rows].mean(axis=0)
        mean_margin[c]=margin[rows].mean()
        loo=(current.sum(axis=0)-current)/(len(rows)-1)
        radius[c]=max(float(np.quantile(np.linalg.norm(current-loo,axis=1),RADIUS_QUANTILE)),RADIUS_FLOOR)
        half=[float(margin[rows[::2]].mean()),float(margin[rows[1::2]].mean())]
        registered=bool(len(rows)>=MIN_REGISTER and mean_margin[c]<0. and all(value<0. for value in half))
        slot=None
        if registered:
            slot=len(anonymous);anonymous.append(c);assignment[rows]=slot;candidate[rows]=True
            distance=np.linalg.norm(anchors-means[c],axis=1)
            penalties.append(.05+max(0.,float(np.max(1.-distance/radius[c]-np.tanh(anchor_q)))))
        clusters.append(dict(cluster_index=c,size=len(rows),mean_margin=float(mean_margin[c]),
            half_mean_margins=half,radius=float(radius[c]),registered=registered,anonymous_index=slot))
    state=dict(schema=STATE_SCHEMA,cluster_embedding=means,cluster_logits=mean_logits,cluster_margin=mean_margin,
        cluster_radius=radius,anonymous_indices=np.asarray(anonymous,np.int64),anonymous_penalty=np.asarray(penalties,np.float64))
    # Restore all diagnostics to caller order. Sorted opaque IDs are not saved.
    labels_original=np.empty(n,np.int64);labels_original[order]=best_labels
    assignments_original=np.empty(n,np.int64);assignments_original[order]=assignment
    candidate_original=np.empty(n,bool);candidate_original[order]=candidate
    diag=dict(schema=DIAGNOSTIC_SCHEMA,input_count=n,all_U_rows_used=True,pre_rejection_applied=False,
        selected_cluster_count=count,selected_silhouette=best_score,candidates=candidates,
        cluster_labels=labels_original,assignments=assignments_original,candidate_mask=candidate_original,
        cluster_sizes=np.asarray([len(rows) for rows in best_rows],np.int64),clusters=clusters,
        registered_count=len(anonymous),min_cluster_size=MIN_CLUSTER,min_registration_size=MIN_REGISTER,
        half_rule='alternating opaque-ID-sorted members; both raw-margin means strictly negative',
        cluster_order='minimum opaque ID',true_labels_used=False,true_class_count_used=False,
        query_fit=False,algorithm='average-linkage Euclidean hierarchy with full-U silhouette; not FINCH')
    _state(state)
    return state,diag


def _state(state):
    keys={'schema','cluster_embedding','cluster_logits','cluster_margin','cluster_radius','anonymous_indices','anonymous_penalty'}
    if not isinstance(state,dict) or set(state)!=keys or state['schema']!=STATE_SCHEMA:
        raise ValueError('Frozen aggregate neighborhood schema differs')
    centers=np.asarray(state['cluster_embedding'])
    if centers.ndim!=2 or centers.shape[1]!=DIMENSION:raise ValueError('Frozen neighborhood centers require 688 coordinates')
    c=len(centers)
    shapes=dict(cluster_embedding=(c,DIMENSION),cluster_logits=(c,6),cluster_margin=(c,),cluster_radius=(c,))
    for name,shape in shapes.items():
        value=np.asarray(state[name])
        if value.shape!=shape or value.dtype!=np.float64 or not np.isfinite(value).all():
            raise ValueError('Finite FP64 aggregate shape required: '+name)
    indices=np.asarray(state['anonymous_indices']);penalty=np.asarray(state['anonymous_penalty'])
    if (indices.ndim!=1 or indices.dtype!=np.int64 or len(set(indices.tolist()))!=len(indices)
            or np.any((indices<0)|(indices>=c)) or penalty.shape!=indices.shape or penalty.dtype!=np.float64
            or not np.isfinite(penalty).all() or np.any(penalty<.05) or np.any(state['cluster_radius']<RADIUS_FLOOR)):
        raise ValueError('Valid anonymous cluster indices, positive radii and source penalties are required')
    return c,len(indices)


def _neighbors(embedding, state):
    c,_=_state(state)
    if not c:return np.zeros(len(embedding),np.int64),np.zeros(len(embedding),bool)
    distances=np.linalg.norm(embedding[:,None,:]-state['cluster_embedding'][None,:,:],axis=2)
    nearest=distances.argmin(axis=1)
    return nearest,distances[np.arange(len(embedding)),nearest]<=state['cluster_radius'][nearest]


def fusion_mask(embedding_Q, state):
    """Actual DA1 eligibility, without label/score changes as a proxy."""
    z=_matrix(embedding_Q,DIMENSION,np.float32,'query embedding').astype(np.float64)
    return _neighbors(z,state)[1]


def infer(embedding_Q, logits_Q, margins_Q, state, adapt: bool, registered: bool):
    if type(adapt) is not bool or type(registered) is not bool:raise ValueError('Explicit boolean adaptation/registration flags required')
    z,logits,margin=_inputs(embedding_Q,logits_Q,margins_Q);_,k=_state(state)
    current_logits=logits.copy();current_margin=margin.copy()
    if adapt:
        nearest,inside=_neighbors(z,state)
        if inside.any():
            current_logits[inside]=.5*logits[inside]+.5*state['cluster_logits'][nearest[inside]]
            current_margin[inside]=.5*margin[inside]+.5*state['cluster_margin'][nearest[inside]]
    knownness=np.tanh(current_margin)
    old=(current_logits-current_logits.max(axis=1,keepdims=True))/LOGIT_TEMPERATURE+knownness[:,None]
    joint=old
    if registered and k:
        selected=state['anonymous_indices']
        distance=np.linalg.norm(z[:,None,:]-state['cluster_embedding'][selected][None,:,:],axis=2)
        new=1.-distance/state['cluster_radius'][selected]-state['anonymous_penalty']
        joint=np.concatenate((old,new),axis=1)
    if not np.isfinite(joint).all():raise ValueError('Finite joint margins required')
    winner=joint.argmax(axis=1)
    prediction=np.where(joint[np.arange(len(joint)),winner]<0.,-1,winner).astype(np.int64)
    return dict(prediction=prediction,closed=current_logits.argmax(axis=1).astype(np.int64),
        knownness=knownness,joint_margins=joint)
