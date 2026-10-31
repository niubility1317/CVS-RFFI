"""Known-only positive affine MAP EM with a frozen original OE background."""
from __future__ import annotations
import numpy as np
from . import star_diagonal_uda_core as base

density,original=base.density,base.original
ADAPTER_SCHEMA='pamf_phase3_selective_uda_adapter_v1'
ITERATIONS=20
RIDGE=160.
U_COUNT=384
MONOTONIC_ATOL=1e-8


def validate_adapter(adapter):
    if not isinstance(adapter,dict) or set(adapter)!={'schema','kind','inverse_scale','offset'}:
        raise ValueError('Exact flat selective-affine adapter required')
    for name in ('schema','kind'):
        value=adapter[name]
        if not isinstance(value,np.ndarray) or value.shape!=() or value.dtype.kind!='U':raise ValueError('Scalar Unicode schema/kind required')
    if adapter['schema'].item()!=ADAPTER_SCHEMA or adapter['kind'].item()!='SELECTIVE':raise ValueError('Matching known-only selective adapter schema/kind required')
    for name in ('inverse_scale','offset'):
        value=adapter[name]
        if not isinstance(value,np.ndarray) or value.dtype!=np.float64 or value.shape!=(160,) or not np.isfinite(value).all():raise ValueError('Finite FP64 selective affine vectors required')
    if np.any(adapter['inverse_scale']<=0):raise ValueError('Every inverse scale must be strictly positive')
    return adapter


def _expectation(state,w,a,b):
    h=a*w+b;variance=state['known_variances']
    known=(np.log(.5*state['known_weights'])[None]-80*np.log(2*np.pi*variance)[None]
        -density._distances(h,state['known_means'])/(2*variance[None])+np.log(a).sum())
    # Only known Gaussians move. OE stays at the original w coordinates.
    background=density._log_density(state,w,'background')-float(state['log_abs_det_whitener'])+np.log(.5)
    terms=np.column_stack((known,background));peak=terms.max(axis=1);exp=np.exp(terms-peak[:,None]);mass=exp.sum(axis=1)
    responsibilities=exp[:,:90]/mass[:,None]
    objective=float(np.sum(peak+np.log(mass))+len(w)*float(state['log_abs_det_whitener'])
        -RIDGE/2*(np.square(a-1).sum()+np.square(b).sum()))
    if not np.isfinite(objective) or not np.isfinite(responsibilities).all():raise ValueError('Finite selective MAP objective and known responsibilities required')
    return objective,responsibilities


def _m_step(w,rknown,means,variance):
    known_mass=float(rknown.sum());p=rknown/variance[None];t=p.sum(axis=1);M=p@means
    s0=float(t.sum());sx=(t[:,None]*w).sum(0);sxx=(t[:,None]*np.square(w)).sum(0)
    m0=M.sum(0);mx=(w*M).sum(0);D=s0+RIDGE
    H=sxx+RIDGE-sx*sx/D;g=mx+RIDGE-sx*m0/D
    if not np.isfinite(H).all() or np.any(H<=0) or not np.isfinite(g).all():raise ValueError('Finite strictly positive exact selective M-step curvature required')
    root=np.hypot(g,2*np.sqrt(H*known_mass));a=np.empty(160,np.float64);negative=g<0
    a[negative]=2*known_mass/(root[negative]-g[negative]);a[~negative]=(g[~negative]+root[~negative])/(2*H[~negative])
    b=(m0-a*sx)/D
    # At zero known mass H=g=RIDGE, giving exact identity without pseudo-counts.
    if not np.isfinite(a).all() or np.any(a<=0) or not np.isfinite(b).all():raise ValueError('Positive finite selective affine M-step required')
    return a,b


def fit_adapter(state,U_features):
    w=base._white(state,U_features)
    if len(w)!=U_COUNT:raise ValueError('The complete fixed star-G10 unlabeled U role is required')
    a=np.ones(160,np.float64);b=np.zeros(160,np.float64);trace=[]
    objective,rknown=_expectation(state,w,a,b)
    def record(iteration):
        return dict(iteration=iteration,objective=objective,scale_min=float(a.min()),scale_max=float(a.max()),
            offset_norm=float(np.linalg.norm(b)),known_posterior_mass=float(rknown.sum(axis=1).mean()))
    trace.append(record(0))
    for iteration in range(1,ITERATIONS+1):
        a,b=_m_step(w,rknown,state['known_means'],state['known_variances'])
        value,rknown=_expectation(state,w,a,b)
        if value<objective-MONOTONIC_ATOL:raise ValueError('Exact selective MAP EM objective decreased beyond floating-point tolerance')
        objective=value;trace.append(record(iteration))
    adapter=dict(schema=np.asarray(ADAPTER_SCHEMA),kind=np.asarray('SELECTIVE'),inverse_scale=a.copy(),offset=b.copy());validate_adapter(adapter)
    diag=dict(iterations=ITERATIONS,ridge=RIDGE,U_count=len(w),trace=trace,group_prior=[.5,.5],parameters_fitted=320,
        identity_initialization=True,objective_non_decreasing=True,known_only_transform=True,background_frozen=True,U_only=True,
        Q_inputs_used=False,true_U_labels_used=False,true_unknown_count_used=False,default_logits_used=False,source_state_unchanged_by_fit=True,
        monotonic_absolute_tolerance=MONOTONIC_ATOL,Jacobian_in_objective='known component log terms include sum(log inverse_scale); common N log_abs_det_whitener',
        transform='known h = inverse_scale * original_white + offset; OE original_white; no post-transform normalization')
    return adapter,diag


def raw_scores(state,features,closed,adapter):
    validate_adapter(adapter);w=base._white(state,features);c=np.asarray(closed)
    if c.shape!=(len(w),) or c.dtype.kind not in 'iu' or not np.isin(c,range(6)).all():raise ValueError('Aligned frozen original F0 classes required; no true-label routing')
    if np.array_equal(adapter['inverse_scale'],np.ones(160)) and np.array_equal(adapter['offset'],np.zeros(160)):
        return original.raw_scores(state,features,c)[:,[1,3]]
    h=adapter['inverse_scale']*w+adapter['offset'];background=density._log_density(state,w,'background')
    pooled=density._log_density(state,h,'known');conditional=np.empty(len(w),np.float64);jacobian=float(np.log(adapter['inverse_scale']).sum())
    for label in range(6):
        rows=np.flatnonzero(c==label)
        if not len(rows):continue
        selection=slice(label*15,(label+1)*15);variance=state['known_variances'][selection]
        terms=-np.log(15.)-80*np.log(2*np.pi*variance)[None]-density._distances(h[rows],state['known_means'][selection])/(2*variance[None])
        peak=terms.max(axis=1);conditional[rows]=peak+np.log(np.exp(terms-peak[:,None]).sum(axis=1))+float(state['log_abs_det_whitener'])
    # The whitening Jacobian cancels; the known-only affine Jacobian does not.
    result=np.column_stack((pooled-background+jacobian,conditional-background+jacobian))
    if not np.isfinite(result).all():raise ValueError('Finite known-only selective FP64 likelihood ratios required')
    return result
