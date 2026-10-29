"""Fixed local support from original-F0 source L/OE unit-feature banks.

The banks retain source RX grouping only. Queries carry no RX, label, class
count or fitting permission. All reductions are row-local; at most 128 query
rows share the two temporary distance matrices.
"""
from __future__ import annotations

import numpy as np

METHODS=('pooled_known','pooled_relative','rx_consensus_relative')
K=8
EPS=1e-6
RECEIVERS=(1,3,4,6,8)
BLOCK=128


def _features(value,dtype):
    array=np.asarray(value)
    if (array.ndim!=2 or not len(array) or array.shape[1]!=160 or array.dtype!=dtype
            or not np.isfinite(array).all()):
        raise ValueError('Nonempty finite unit-feature matrix with the declared dtype and 160 columns required')
    work=array.astype(np.float64,copy=False)
    norm=np.sqrt(np.einsum('nd,nd->n',work,work,optimize=False))
    tolerance=2e-6 if dtype==np.float32 else 2e-12
    if not np.allclose(norm,1.,rtol=0.,atol=tolerance):raise ValueError('Per-record unit features required')
    return work,norm


def _receivers(value,n):
    rx=np.asarray(value)
    if (rx.shape!=(n,) or rx.dtype.kind not in 'iu' or not np.isin(rx,RECEIVERS).all()
            or any(np.count_nonzero(rx==receiver)<K for receiver in RECEIVERS)):
        raise ValueError('Each of the five declared source RX must contain at least eight records')
    return rx


def fit(L_features,L_rx,OE_features,OE_rx):
    """Copy and normalize source FP32 banks into the exact FP64 working state."""
    L,ln=_features(L_features,np.float32);OE,on=_features(OE_features,np.float32)
    lr=_receivers(L_rx,len(L));ore=_receivers(OE_rx,len(OE))
    return dict(L=L/ln[:,None],OE=OE/on[:,None],L_rx=lr.astype(np.int64,copy=True),OE_rx=ore.astype(np.int64,copy=True))


def _state(state):
    if not isinstance(state,dict) or set(state)!={'L','OE','L_rx','OE_rx'}:
        raise ValueError('Only the fixed two banks and their source RX arrays are allowed')
    L,_=_features(state['L'],np.float64);OE,_=_features(state['OE'],np.float64)
    lr=_receivers(state['L_rx'],len(L));ore=_receivers(state['OE_rx'],len(OE))
    if lr.dtype!=np.int64 or ore.dtype!=np.int64:raise ValueError('FP64 working banks require int64 source RX')
    return L,OE,[np.flatnonzero(lr==rx) for rx in RECEIVERS],[np.flatnonzero(ore==rx) for rx in RECEIVERS]


def _distances(query,bank):
    # Fixed per-pair accumulation preserves row/block and bank-order semantics.
    result=np.einsum('nd,kd->nk',query,bank,optimize=False)
    np.multiply(result,-2.,out=result);np.add(result,2.,out=result);np.clip(result,0.,4.,out=result)
    return result


def _mean_eight(distances):
    # Sorting only the selected eight values makes their sum invariant to bank
    # order, including distance ties; no neighbor ID or class label is selected.
    smallest=np.partition(distances,K-1,axis=1)[:,:K]
    return np.ascontiguousarray(np.sort(smallest,axis=1)).mean(axis=1)


def score(query_features,state):
    """Return three FP64 source-support scores without changing any bank."""
    query,norm=_features(query_features,np.float32);query=query/norm[:,None]
    L,OE,L_groups,OE_groups=_state(state);result=np.empty((len(query),3),np.float64)
    for start in range(0,len(query),BLOCK):
        stop=min(start+BLOCK,len(query));block=query[start:stop]
        known_distances=_distances(block,L);oe_distances=_distances(block,OE)
        known=_mean_eight(known_distances);outlier=_mean_eight(oe_distances)
        known_rx=np.median(np.column_stack([_mean_eight(known_distances[:,ix]) for ix in L_groups]),axis=1)
        outlier_rx=np.median(np.column_stack([_mean_eight(oe_distances[:,ix]) for ix in OE_groups]),axis=1)
        result[start:stop,0]=-known
        result[start:stop,1]=np.log(outlier+EPS)-np.log(known+EPS)
        result[start:stop,2]=np.log(outlier_rx+EPS)-np.log(known_rx+EPS)
    return result
