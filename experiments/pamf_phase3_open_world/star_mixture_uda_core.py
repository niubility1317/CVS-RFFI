"""Unlabelled target translation in a source-derived nuisance subspace."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density

PRIOR_SCHEMA='pamf_phase3_mixture_uda_prior_v1'
ADAPTER_SCHEMA='pamf_phase3_mixture_uda_adapter_v1'
ARMS=('DA_K','DA_M')
ITERATIONS=20
U_COUNT=384


def validate_prior(prior):
    if (not isinstance(prior,dict) or set(prior)!={'schema','basis','variances','radius'}
            or not isinstance(prior['schema'],np.ndarray) or prior['schema'].shape!=() or prior['schema'].item()!=PRIOR_SCHEMA):
        raise ValueError('Flat source nuisance prior required')
    b,v,r=prior['basis'],prior['variances'],prior['radius']
    if (any(not isinstance(a,np.ndarray) or a.dtype!=np.float64 or not np.isfinite(a).all() for a in (b,v,r))
            or b.ndim!=2 or b.shape[1]!=160 or not 1<=len(b)<=14 or v.shape!=(len(b),) or r.shape!=()
            or np.any(v<=0) or float(r)<=0 or not np.allclose(b@b.T,np.eye(len(b)),rtol=0,atol=1e-12)):
        raise ValueError('Finite orthonormal source basis, positive variances and source radius required')
    return prior


def build_prior(state):
    density.validate_state(state,'density_ratio')
    if state['known_means'].shape!=(90,160):raise ValueError('Original class-major ninety-mode P1 state required')
    means=state['known_means'].reshape(6,15,160)
    offsets=(means-means.mean(axis=1,keepdims=True)).mean(axis=0)
    offsets-=offsets.mean(axis=0,keepdims=True)
    _,s,vh=np.linalg.svd(offsets,full_matrices=False)
    rank=int(np.sum(s>np.finfo(np.float64).eps*max(offsets.shape)*s[0]))
    prior=dict(schema=np.asarray(PRIOR_SCHEMA),basis=vh[:rank].copy(),variances=s[:rank]**2/15.,
               radius=np.asarray(np.linalg.norm(offsets,axis=1).max(),np.float64))
    validate_prior(prior)
    for value in prior.values():value.setflags(write=False)
    return prior


def validate_adapter(adapter,prior):
    validate_prior(prior)
    if (not isinstance(adapter,dict) or set(adapter)!={'schema','delta','coefficients'}
            or not isinstance(adapter['schema'],np.ndarray) or adapter['schema'].shape!=() or adapter['schema'].item()!=ADAPTER_SCHEMA):
        raise ValueError('Flat frozen target adapter required')
    delta,a=adapter['delta'],adapter['coefficients']
    if (any(not isinstance(x,np.ndarray) or x.dtype!=np.float64 or not np.isfinite(x).all() for x in (delta,a))
            or delta.shape!=(160,) or a.shape!=(len(prior['basis']),)
            or not np.allclose(delta,a@prior['basis'],rtol=1e-12,atol=1e-12)
            or np.linalg.norm(a)>float(prior['radius'])+1e-10):
        raise ValueError('Target shift must lie in the declared source subspace and radius')
    return adapter


def _whiten(state,features):
    density.validate_state(state,'density_ratio')
    if state['known_means'].shape!=(90,160):raise ValueError('Original D160 density required')
    return (density.unit(features)-state['origin'])@state['whitener']


def scores(state,features,delta):
    z=_whiten(state,features);d=np.asarray(delta)
    if d.shape!=(160,) or d.dtype!=np.float64 or not np.isfinite(d).all():raise ValueError('One finite frozen FP64 target shift required')
    h=z-d
    known=density._log_density(state,h,'known');background=density._log_density(state,h,'background')
    result=np.column_stack((known,known-background))
    if not np.isfinite(result).all():raise ValueError('Finite translated density scores required')
    return result


def _components(state,arm):
    if arm not in ARMS:raise ValueError('Fixed known-only or background-aware arm required')
    means=state['known_means'];variance=state['known_variances'];weights=state['known_weights']
    if arm=='DA_M':
        means=np.concatenate((means,state['background_means']))
        variance=np.concatenate((variance,state['background_variances']))
        weights=np.concatenate((.5*weights,.5*state['background_weights']))
    return means,variance,np.log(weights)


def _expectation(w,means,variance,logweight,a,prior,jacobian):
    h=w-a@prior['basis']
    terms=logweight[None]-80.*np.log(2*np.pi*variance)[None]-density._distances(h,means)/(2*variance[None])
    peak=terms.max(axis=1);normalizer=peak+np.log(np.exp(terms-peak[:,None]).sum(axis=1))
    responsibility=np.exp(terms-normalizer[:,None])
    objective=float(normalizer.sum()+len(w)*jacobian-.5*np.sum(a*a/prior['variances']))
    if not np.isfinite(objective) or not np.isfinite(responsibility).all():raise ValueError('Finite U-only likelihood and responsibilities required')
    return responsibility,objective


def _maximization(w,means,variance,responsibility,prior):
    rp=responsibility/variance[None];row_precision=rp.sum(axis=1)
    total=float(row_precision.sum())
    residual=(row_precision[:,None]*w).sum(axis=0)-rp.sum(axis=0)@means
    numerator=prior['basis']@residual;diagonal=total+1./prior['variances']
    a=numerator/diagonal;radius=float(prior['radius']);multiplier=0.
    if np.linalg.norm(a)>radius:
        low=0.;high=max(1.,float(np.linalg.norm(numerator))/radius)
        for _ in range(60):
            middle=(low+high)/2
            if np.linalg.norm(numerator/(diagonal+middle))>radius:low=middle
            else:high=middle
        multiplier=high;a=numerator/(diagonal+multiplier)
    return a,multiplier


def fit_whitened(state,w,prior,arm):
    """Fixed twenty-step MAP-EM; internal input is already whitened once."""
    density.validate_state(state,'density_ratio');validate_prior(prior)
    w=np.asarray(w)
    if w.shape!=(U_COUNT,160) or w.dtype!=np.float64 or not np.isfinite(w).all():
        raise ValueError('Only the complete finite 384-row unlabelled U coordinates may fit')
    means,variance,logweight=_components(state,arm);a=np.zeros(len(prior['basis']),np.float64)
    r,obj=_expectation(w,means,variance,logweight,a,prior,float(state['log_abs_det_whitener']))
    def row(iteration,objective,responsibility,multiplier):
        return dict(iteration=iteration,objective=objective,coefficient_norm=float(np.linalg.norm(a)),
            shift_norm=float(np.linalg.norm(a@prior['basis'])),known_posterior_mass=float(responsibility[:,:90].sum()/len(w)),
            constraint_multiplier=multiplier)
    trace=[row(0,obj,r,0.)]
    for step in range(1,ITERATIONS+1):
        a,multiplier=_maximization(w,means,variance,r,prior)
        new_r,new_obj=_expectation(w,means,variance,logweight,a,prior,float(state['log_abs_det_whitener']))
        if new_obj<obj-1e-7:raise ValueError('U-only MAP objective decreased during exact constrained EM')
        r,obj=new_r,new_obj;trace.append(row(step,obj,r,multiplier))
    adapter=dict(schema=np.asarray(ADAPTER_SCHEMA),delta=a@prior['basis'],coefficients=a.copy())
    validate_adapter(adapter,prior)
    for value in adapter.values():value.setflags(write=False)
    return adapter,dict(arm=arm,iterations=ITERATIONS,trace=trace,U_count=U_COUNT,
        group_prior=[1.,0.] if arm=='DA_K' else [.5,.5],source_radius=float(prior['radius']),source_rank=len(prior['basis']),
        objective='sum U log mixture likelihood including fixed whitening Jacobian minus Gaussian source-domain prior',
        true_U_labels_used=False,true_unknown_count_used=False,Q_inputs_used=False,default_logits_used=False,
        fixed_iteration_checkpoint=True,source_state_unchanged_by_fit=True)


def fit_adapters(state,U_features,prior):
    w=_whiten(state,U_features)
    if len(w)!=U_COUNT:raise ValueError('Only the fixed complete U384 pool may adapt')
    adapters={};reports={}
    for arm in ARMS:adapters[arm],reports[arm]=fit_whitened(state,w,prior,arm)
    return adapters,dict(iterations=ITERATIONS,U_count=U_COUNT,arms=reports,source_prior_frozen_before_U=True,
        U_only=True,Q_inputs_used=False,true_U_labels_used=False,true_unknown_count_used=False)
