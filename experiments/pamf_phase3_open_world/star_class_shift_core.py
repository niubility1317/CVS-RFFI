"""Source-class nuisance tensor with one unlabelled target latent state."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density

PRIOR_SCHEMA='pamf_phase3_class_shift_prior_v1'
ADAPTER_SCHEMA='pamf_phase3_class_shift_adapter_v1'
ITERATIONS=20
U_COUNT=384


def validate_prior(prior):
    if (not isinstance(prior,dict) or set(prior)!={'schema','basis','variances','radius'}
            or not isinstance(prior['schema'],np.ndarray) or prior['schema'].shape!=() or prior['schema'].item()!=PRIOR_SCHEMA):
        raise ValueError('Flat source class-domain prior required')
    b,v,r=prior['basis'],prior['variances'],prior['radius']
    if (any(not isinstance(x,np.ndarray) or x.dtype!=np.float64 or not np.isfinite(x).all() for x in (b,v,r))
            or b.ndim!=3 or b.shape[1:]!=(6,160) or not 1<=len(b)<=14 or v.shape!=(len(b),) or r.shape!=()
            or np.any(v<=0) or float(r)<=0
            or not np.allclose(np.einsum('kcd,lcd->kl',b,b)/6,np.eye(len(b)),atol=1e-12,rtol=0)):
        raise ValueError('Source basis must be orthonormal in class-RMS coordinates with positive prior variances/radius')
    return prior


def build_prior(state):
    density.validate_state(state,'density_ratio')
    if state['known_means'].shape!=(90,160):raise ValueError('Original class-major six by fifteen source means required')
    mu=state['known_means'].reshape(6,15,160)
    offsets=mu-mu.mean(axis=1,keepdims=True)
    matrix=offsets.transpose(1,0,2).reshape(15,960)/np.sqrt(6.)
    matrix-=matrix.mean(axis=0,keepdims=True)
    _,s,vh=np.linalg.svd(matrix,full_matrices=False)
    rank=int(np.sum(s>np.finfo(np.float64).eps*max(matrix.shape)*s[0]))
    prior=dict(schema=np.asarray(PRIOR_SCHEMA),basis=(np.sqrt(6.)*vh[:rank]).reshape(rank,6,160).copy(),
        variances=s[:rank]**2/15.,radius=np.asarray(np.linalg.norm(matrix,axis=1).max(),np.float64))
    validate_prior(prior)
    for a in prior.values():a.setflags(write=False)
    return prior


def validate_adapter(adapter,prior):
    validate_prior(prior)
    if (not isinstance(adapter,dict) or set(adapter)!={'schema','delta','background_delta','coefficients'}
            or not isinstance(adapter['schema'],np.ndarray) or adapter['schema'].shape!=() or adapter['schema'].item()!=ADAPTER_SCHEMA):
        raise ValueError('Flat class-aware target adapter required')
    d,b,a=adapter['delta'],adapter['background_delta'],adapter['coefficients']
    if (any(not isinstance(x,np.ndarray) or x.dtype!=np.float64 or not np.isfinite(x).all() for x in (d,b,a))
            or d.shape!=(6,160) or b.shape!=(160,) or a.shape!=(len(prior['basis']),)
            or not np.allclose(d,np.einsum('k,kcd->cd',a,prior['basis']),rtol=1e-12,atol=1e-12)
            or not np.allclose(b,d.mean(axis=0),rtol=1e-12,atol=1e-12)
            or np.linalg.norm(a)>float(prior['radius'])+1e-10):
        raise ValueError('One shared latent state must define all class offsets and the mean background offset inside the source-RMS ball')
    return adapter


def _whiten(state,features):
    density.validate_state(state,'density_ratio')
    if state['known_means'].shape!=(90,160):raise ValueError('Original D160 ninety-mode state required')
    return (density.unit(features)-state['origin'])@state['whitener']


def _logsum(terms):
    peak=terms.max(axis=1)
    return peak+np.log(np.exp(terms-peak[:,None]).sum(axis=1))


def _terms(w,means,variances,weights):
    return np.log(weights)[None]-80.*np.log(2*np.pi*variances)[None]-density._distances(w,means)/(2*variances[None])


def scores(state,features,delta):
    """Move known class modes, and background by their mean; do not move Q."""
    w=_whiten(state,features);d=np.asarray(delta)
    if d.shape!=(6,160) or d.dtype!=np.float64 or not np.isfinite(d).all():raise ValueError('Six frozen FP64 class shifts required')
    means=state['known_means']+np.repeat(d,15,axis=0)
    known=_logsum(_terms(w,means,state['known_variances'],state['known_weights']))+float(state['log_abs_det_whitener'])
    background=density._log_density(state,w-d.mean(axis=0),'background')
    result=np.column_stack((known,known-background))
    if not np.isfinite(result).all():raise ValueError('Finite fixed class-shift likelihoods required')
    return result


def _expectation(state,w,a,prior):
    delta=np.einsum('k,kcd->cd',a,prior['basis'])
    known=state['known_means']+np.repeat(delta,15,axis=0)
    background=state['background_means']+delta.mean(axis=0)
    terms=np.concatenate((_terms(w,known,state['known_variances'],.5*state['known_weights']),
        _terms(w,background,state['background_variances'],.5*state['background_weights'])),axis=1)
    normalizer=_logsum(terms);responsibility=np.exp(terms-normalizer[:,None])
    objective=float(normalizer.sum()+len(w)*float(state['log_abs_det_whitener'])-.5*np.sum(a*a/prior['variances']))
    if not np.isfinite(objective) or not np.isfinite(responsibility).all():raise ValueError('Finite U-only likelihood and soft responsibilities required')
    return responsibility,objective


def _maximization(state,w,responsibility,prior):
    """Exact ridge MAP M step, with a coupled source-RMS trust region."""
    q=len(prior['basis']);hessian=np.diag(1./prior['variances']);numerator=np.zeros(q,np.float64)
    for c in range(7):
        if c<6:
            selection=slice(c*15,(c+1)*15);basis=prior['basis'][:,c,:]
            means=state['known_means'][selection];variance=state['known_variances'][selection]
        else:
            selection=slice(90,None);basis=prior['basis'].mean(axis=1)
            means=state['background_means'];variance=state['background_variances']
        precision=responsibility[:,selection]/variance[None]
        row_precision=precision.sum(axis=1);total=float(row_precision.sum())
        residual=(row_precision[:,None]*w).sum(axis=0)-precision.sum(axis=0)@means
        hessian+=total*(basis@basis.T);numerator+=basis@residual
    hessian=(hessian+hessian.T)/2
    a=np.linalg.solve(hessian,numerator);multiplier=0.;radius=float(prior['radius'])
    if np.linalg.norm(a)>radius:
        eigenvalues,eigenvectors=np.linalg.eigh(hessian)
        if np.any(eigenvalues<=0):raise ValueError('Positive definite source-ridge M step required')
        projected=eigenvectors.T@numerator;low=0.;high=max(1.,float(np.linalg.norm(numerator))/radius)
        for _ in range(60):
            middle=(low+high)/2
            if np.linalg.norm(projected/(eigenvalues+middle))>radius:low=middle
            else:high=middle
        multiplier=high;a=eigenvectors@(projected/(eigenvalues+multiplier))
    return a,multiplier


def fit_adapter(state,U_features,prior):
    validate_prior(prior);w=_whiten(state,U_features)
    if w.shape!=(U_COUNT,160) or w.dtype!=np.float64 or not np.isfinite(w).all():raise ValueError('Only the complete unlabelled star-U feature pool may fit')
    a=np.zeros(len(prior['basis']),np.float64);r,obj=_expectation(state,w,a,prior)
    def row(step,multiplier):
        delta=np.einsum('k,kcd->cd',a,prior['basis'])
        return dict(iteration=step,objective=obj,coefficient_norm=float(np.linalg.norm(a)),shift_rms=float(np.sqrt(np.sum(delta*delta)/6)),
            class_shift_max=float(np.linalg.norm(delta,axis=1).max()),known_posterior_mass=float(r[:,:90].sum()/len(w)),constraint_multiplier=multiplier)
    trace=[row(0,0.)]
    for step in range(1,ITERATIONS+1):
        a,multiplier=_maximization(state,w,r,prior);new_r,new_obj=_expectation(state,w,a,prior)
        if new_obj<obj-1e-7:raise ValueError('Exact source-constrained U-only MAP objective decreased')
        r,obj=new_r,new_obj;trace.append(row(step,multiplier))
    delta=np.einsum('k,kcd->cd',a,prior['basis'])
    adapter=dict(schema=np.asarray(ADAPTER_SCHEMA),delta=delta,background_delta=delta.mean(axis=0),coefficients=a.copy())
    validate_adapter(adapter,prior)
    for value in adapter.values():value.setflags(write=False)
    return adapter,dict(iterations=ITERATIONS,U_count=U_COUNT,trace=trace,group_prior=[.5,.5],source_rank=len(prior['basis']),source_radius=float(prior['radius']),
        source_prior_frozen_before_U=True,U_only=True,Q_inputs_used=False,true_U_labels_used=False,true_unknown_count_used=False,
        default_logits_used=False,source_state_unchanged_by_fit=True,fixed_iteration_checkpoint=True,
        objective='sum U log fixed known90/OE32 mixture likelihood minus Gaussian source class-domain prior; class-RMS ball')
