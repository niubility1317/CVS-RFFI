"""U-only MAP EM for a positive diagonal affine map in frozen whitened space."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density
from . import star_class_density_core as original

ADAPTER_SCHEMA='pamf_phase3_diagonal_uda_adapter_v1'
ARMS=('SHIFT','DIAG')
ITERATIONS=20
RIDGE=160.
U_COUNT=384
DIMENSION=160
MONOTONIC_ATOL=1e-8


def _state(state):
    density.validate_state(state,'density_ratio')
    if state['schema'].item()!=density.SCHEMA or state['known_means'].shape!=(90,160):
        raise ValueError('Original six-by-fifteen D160 frozen Gaussian state required')
    return state


def validate_adapter(adapter):
    if not isinstance(adapter,dict) or set(adapter)!={'schema','kind','inverse_scale','offset'}:
        raise ValueError('Exact flat diagonal-affine adapter fields required')
    for key in ('schema','kind'):
        if not isinstance(adapter[key],np.ndarray) or adapter[key].shape!=() or adapter[key].dtype.kind!='U':
            raise ValueError('Scalar Unicode adapter schema/kind required')
    if adapter['schema'].item()!=ADAPTER_SCHEMA or adapter['kind'].item() not in ARMS:
        raise ValueError('Matching preregistered affine adapter schema/kind required')
    for key in ('inverse_scale','offset'):
        value=adapter[key]
        if not isinstance(value,np.ndarray) or value.shape!=(160,) or value.dtype!=np.float64 or not np.isfinite(value).all():
            raise ValueError('Finite FP64 affine aggregate vectors required')
    if np.any(adapter['inverse_scale']<=0) or (adapter['kind'].item()=='SHIFT' and not np.array_equal(adapter['inverse_scale'],np.ones(160))):
        raise ValueError('Positive inverse scales; SHIFT must fix every scale to one')
    return adapter


def _white(state,features):
    _state(state)
    with np.errstate(over='ignore',invalid='ignore'):w=(density.unit(features)-state['origin'])@state['whitener']
    if not np.isfinite(w).all():raise ValueError('Finite original whitened feature coordinates required')
    return w


def _expectation(w,a,b,means,variance,weight,logdet):
    h=a*w+b
    terms=np.log(weight)[None]-80*np.log(2*np.pi*variance)[None]-density._distances(h,means)/(2*variance[None])
    peak=terms.max(axis=1);exp=np.exp(terms-peak[:,None]);mass=exp.sum(axis=1)
    responsibilities=exp/mass[:,None]
    objective=float(np.sum(peak+np.log(mass))+len(w)*(np.log(a).sum()+logdet)
                    -RIDGE/2*(np.square(a-1).sum()+np.square(b).sum()))
    if not np.isfinite(objective) or not np.isfinite(responsibilities).all():raise ValueError('Finite MAP objective and mixture responsibilities required')
    return objective,responsibilities


def _m_step(w,responsibilities,means,variance,kind):
    p=responsibilities/variance[None];t=p.sum(axis=1);M=p@means
    s0=float(t.sum());sx=(t[:,None]*w).sum(axis=0);sxx=(t[:,None]*np.square(w)).sum(axis=0)
    m0=M.sum(axis=0);mx=(w*M).sum(axis=0);denominator=s0+RIDGE
    if kind=='SHIFT':a=np.ones(160,np.float64)
    else:
        H=sxx+RIDGE-sx*sx/denominator;g=mx+RIDGE-sx*m0/denominator
        if not np.isfinite(H).all() or np.any(H<=0) or not np.isfinite(g).all():
            raise ValueError('Every exact scale M-step must have finite positive curvature')
        root=np.hypot(g,2*np.sqrt(H*len(w)));a=np.empty(160,np.float64);negative=g<0
        a[negative]=2*len(w)/(root[negative]-g[negative])
        a[~negative]=(g[~negative]+root[~negative])/(2*H[~negative])
    b=(m0-a*sx)/denominator
    if not np.isfinite(a).all() or np.any(a<=0) or not np.isfinite(b).all():raise ValueError('Exact affine M-step must remain finite with positive scales')
    return a,b


def fit_adapters(state,U_features):
    w=_white(state,U_features)
    if len(w)!=U_COUNT:raise ValueError('The complete fixed star-G10 unlabeled U role is required')
    means=np.concatenate((state['known_means'],state['background_means']))
    variance=np.concatenate((state['known_variances'],state['background_variances']))
    weight=np.concatenate((.5*state['known_weights'],.5*state['background_weights']))
    adapters={};reports={}
    for kind in ARMS:
        a=np.ones(160,np.float64);b=np.zeros(160,np.float64);trace=[]
        objective,responsibilities=_expectation(w,a,b,means,variance,weight,float(state['log_abs_det_whitener']))
        def record(iteration):
            return dict(iteration=iteration,objective=objective,scale_min=float(a.min()),scale_max=float(a.max()),
                offset_norm=float(np.linalg.norm(b)),known_posterior_mass=float(responsibilities[:,:90].sum(axis=1).mean()))
        trace.append(record(0))
        for iteration in range(1,ITERATIONS+1):
            a,b=_m_step(w,responsibilities,means,variance,kind)
            value,responsibilities=_expectation(w,a,b,means,variance,weight,float(state['log_abs_det_whitener']))
            if value<objective-MONOTONIC_ATOL:raise ValueError('Exact MAP EM objective decreased beyond floating-point tolerance')
            objective=value;trace.append(record(iteration))
        adapter=dict(schema=np.asarray(ADAPTER_SCHEMA),kind=np.asarray(kind),inverse_scale=a.copy(),offset=b.copy())
        validate_adapter(adapter);adapters[kind]=adapter
        reports[kind]=dict(group_prior=[.5,.5],iterations=ITERATIONS,trace=trace,parameters_fitted=160 if kind=='SHIFT' else 320,
            identity_initialization=True,objective_non_decreasing=True)
    return adapters,dict(iterations=ITERATIONS,ridge=RIDGE,U_count=len(w),arms=reports,U_only=True,Q_inputs_used=False,
        true_U_labels_used=False,true_unknown_count_used=False,default_logits_used=False,source_state_unchanged_by_fit=True,
        monotonic_absolute_tolerance=MONOTONIC_ATOL,Jacobian_in_objective='N sum(log inverse_scale) + N log_abs_det_whitener',
        transform='h = inverse_scale * original_white + offset; no post-transform unit normalization')


def raw_scores(state,features,closed,adapter):
    validate_adapter(adapter);w=_white(state,features);c=np.asarray(closed)
    if c.shape!=(len(w),) or c.dtype.kind not in 'iu' or not np.isin(c,range(6)).all():
        raise ValueError('Original frozen F0 predicted classes required; no true-label routing')
    if np.array_equal(adapter['inverse_scale'],np.ones(160)) and np.array_equal(adapter['offset'],np.zeros(160)):
        return original.raw_scores(state,features,c)[:,[1,3]]
    h=adapter['inverse_scale']*w+adapter['offset'];background=density._log_density(state,h,'background')
    pooled=density._log_density(state,h,'known');conditional=np.empty(len(w),np.float64)
    for label in range(6):
        rows=np.flatnonzero(c==label)
        if not len(rows):continue
        selection=slice(label*15,(label+1)*15);variance=state['known_variances'][selection]
        terms=-np.log(15.)-80*np.log(2*np.pi*variance)[None]-density._distances(h[rows],state['known_means'][selection])/(2*variance[None])
        peak=terms.max(axis=1)
        conditional[rows]=peak+np.log(np.exp(terms-peak[:,None]).sum(axis=1))+float(state['log_abs_det_whitener'])
    # Both the affine and original whitening Jacobians cancel in each ratio.
    result=np.column_stack((pooled-background,conditional-background))
    if not np.isfinite(result).all():raise ValueError('Finite transformed FP64 density ratios required')
    return result
