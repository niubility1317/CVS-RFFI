"""Fixed source-L class-conditioned support, routed by the frozen old head.

Only source L carries true old-class labels. Query identity is supplied as
the unchanged default-head prediction; no query truth or RX is accepted.
"""
from __future__ import annotations

import numpy as np

from . import star_local_support_core as previous

METHODS=('pooled_relative','predicted_identity_relative')
K=8
EPS=1e-6
BLOCK=128
_features,_distances,_mean_eight=previous._features,previous._distances,previous._mean_eight


def _labels(value,n):
    labels=np.asarray(value)
    if (labels.shape!=(n,) or labels.dtype.kind not in 'iu' or np.any((labels<0)|(labels>=6))
            or any(np.count_nonzero(labels==c)<K for c in range(6))):
        raise ValueError('Six old source L classes with at least eight records each are required')
    return labels


def fit(L_features,L_y,OE_features):
    """Copy FP32 units into FP64 working banks and preserve visible L labels."""
    L,ln=_features(L_features,np.float32);OE,on=_features(OE_features,np.float32)
    labels=_labels(L_y,len(L))
    if len(OE)<K:raise ValueError('At least eight source OE records required')
    return dict(L=L/ln[:,None],L_y=labels.astype(np.int64,copy=True),OE=OE/on[:,None])


def _state(state):
    if not isinstance(state,dict) or set(state)!={'L','L_y','OE'}:
        raise ValueError('Only fixed L/OE banks and source L labels may be stored')
    L,_=_features(state['L'],np.float64);OE,_=_features(state['OE'],np.float64)
    labels=_labels(state['L_y'],len(L))
    if labels.dtype!=np.int64 or len(OE)<K:raise ValueError('Int64 source labels and sufficient fixed OE bank required')
    return L,OE,[np.flatnonzero(labels==c) for c in range(6)]


def score(query_features,closed,state):
    """Return pooled control and predicted-old-class-relative FP64 scores."""
    query,norm=_features(query_features,np.float32);query=query/norm[:,None];closed=np.asarray(closed)
    if closed.shape!=(len(query),) or closed.dtype.kind not in 'iu' or np.any((closed<0)|(closed>=6)):
        raise ValueError('Aligned frozen default-head predictions in the old six classes required')
    L,OE,classes=_state(state);result=np.empty((len(query),2),np.float64)
    for start in range(0,len(query),BLOCK):
        stop=min(start+BLOCK,len(query));block=query[start:stop]
        known_distances=_distances(block,L);oe_distances=_distances(block,OE)
        pooled=_mean_eight(known_distances);outlier=_mean_eight(oe_distances)
        conditional=np.empty(len(block),np.float64)
        for c,columns in enumerate(classes):
            rows=np.flatnonzero(closed[start:stop]==c)
            if len(rows):conditional[rows]=_mean_eight(known_distances[np.ix_(rows,columns)])
        log_oe=np.log(outlier+EPS)
        result[start:stop,0]=log_oe-np.log(pooled+EPS)
        result[start:stop,1]=log_oe-np.log(conditional+EPS)
    return result
