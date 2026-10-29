"""Frozen binary U-graph smoothing and row-wise Q extrapolation, without IO."""
import numpy as np
from . import star_reciprocal_core as registry

STATE_SCHEMA='pamf_phase3_star_binary_graph_state_v1'
STATE_FIELDS={'schema','unit','ids','sigma','f','skipped'}


def distances(left,right):
    dot=np.einsum('id,jd->ij',left.astype(np.float64),right.astype(np.float64),optimize=False)
    return np.maximum(0.,2.-2.*np.clip(dot,-1.,1.))


def seeds(logits,margin):
    n=len(margin);logits=registry._array(logits,np.float32,(n,6),'logits')
    margin=registry._array(margin,np.float64,(n,),'margin')
    exp=np.exp(logits.astype(np.float64)-logits.max(axis=1,keepdims=True))
    msp=exp.max(axis=1)/exp.sum(axis=1)
    return (margin>=0)&(msp>=.95),(margin<0)&(msp<=.5)


def _state(state):
    if not isinstance(state,dict) or set(state)!=STATE_FIELDS or state['schema']!=STATE_SCHEMA or type(state['skipped']) is not bool:
        raise ValueError('Exact frozen binary-graph state required')
    unit=registry._unit(state['unit']);n=len(unit);registry._ids(state['ids'],n)
    registry._array(state['sigma'],np.float64,(n,),'sigma');registry._array(state['f'],np.float64,(n,),'f')
    if np.any(state['sigma']<1e-6) or np.any(abs(state['f'])>1+1e-10):
        raise ValueError('Positive local scales and bounded finite graph scores required')
    return state


def fit(unit_U,logits_U,original_margin,ids):
    unit=registry._unit(unit_U);n=len(unit);ids=registry._ids(ids,n)
    margin=registry._array(original_margin,np.float64,(n,),'original_margin')
    positive,negative=seeds(logits_U,margin)
    adjacency,neighbors=registry.reciprocal_graph(unit,ids)
    d2=distances(unit,unit)
    sigma=np.maximum(np.sqrt(d2[np.arange(n),neighbors[:,-1]]),1e-6) if neighbors.shape[1] else np.full(n,1e-6)
    W=np.exp(-d2/(sigma[:,None]*sigma[None,:]))*adjacency
    np.fill_diagonal(W,0.)
    L=np.diag(W.sum(axis=1))-W
    selected=positive|negative;sign=positive.astype(np.float64)-negative.astype(np.float64)
    base=np.tanh(margin);skipped=not positive.any() or not negative.any()
    A=np.eye(n)+L+10*np.diag(selected.astype(np.float64));rhs=base+10*sign
    if skipped:f=base.copy()
    else:
        # Canonical solve order makes U-row permutations numerically identical.
        order=np.argsort(ids,kind='stable');f=np.empty(n,np.float64)
        f[order]=np.linalg.solve(A[np.ix_(order,order)],rhs[order])
    if not np.isfinite(f).all() or np.any(abs(f)>1+1e-10):
        raise ValueError('Binary graph solve violated finite maximum principle')
    def energy(value):
        return float(.5*np.sum((value-base)**2)+.25*np.sum(W*(value[:,None]-value[None,:])**2)+5*np.sum(selected*(value-sign)**2))
    state=dict(schema=STATE_SCHEMA,unit=unit.copy(),ids=ids.copy(),sigma=sigma,f=f,skipped=skipped)
    diag=dict(status='SKIPPED_EMPTY_SEED_SIDE' if skipped else 'SOLVED',positive_count=int(positive.sum()),
        negative_count=int(negative.sum()),seed_count=int(selected.sum()),edge_count=int(adjacency.sum()//2),
        solve_residual_linf=None if skipped else float(np.max(abs(A@f-rhs))),
        energy_before=energy(base),energy_after=energy(f),f_min=float(f.min()) if n else None,
        f_max=float(f.max()) if n else None,maximum_principle_passed=True,optimizer_updates=0)
    return _state(state),diag


def margins(unit_Q,original_margin_Q,state):
    unit=registry._unit(unit_Q);n=len(unit)
    margin=registry._array(original_margin_Q,np.float64,(n,),'original_margin_Q');_state(state)
    base=np.tanh(margin)
    if state['skipped']:return base
    d2=distances(unit,state['unit']);k=min(10,len(state['unit']))
    if not k:raise ValueError('Non-skipped graph requires U memory')
    order=np.argsort(state['ids'],kind='stable')
    nearest=order[np.argsort(d2[:,order],axis=1,kind='stable')[:,:k]]
    local=d2[np.arange(n)[:,None],nearest]
    sigma_q=np.maximum(np.sqrt(local[:,-1]),1e-6)
    weight=np.exp(-local/(sigma_q[:,None]*state['sigma'][nearest]))
    return (base+np.sum(weight*state['f'][nearest],axis=1))/(1+weight.sum(axis=1))
