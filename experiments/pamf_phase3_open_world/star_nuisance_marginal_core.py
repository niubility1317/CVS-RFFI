"""Analytic source nuisance convolution of fixed known and OE Gaussians."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density
from . import star_class_density_core as conditional

SCHEMA='pamf_phase3_star_nuisance_marginal_prior_v1'
METHODS=('pooled_ratio','predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
DIMENSION=160


def _source(state):
    density.validate_state(state,'density_ratio')
    if state['schema'].item()!=density.SCHEMA or state['known_means'].shape!=(90,DIMENSION):raise ValueError('Only the original D160 class-major source density is allowed')


def validate_prior(prior):
    if (not isinstance(prior,dict) or set(prior)!={'schema','basis','variances'} or not isinstance(prior['schema'],np.ndarray)
            or prior['schema'].shape!=() or prior['schema'].dtype.kind!='U' or prior['schema'].item()!=SCHEMA):raise ValueError('Flat source nuisance prior schema required')
    basis,variance=prior['basis'],prior['variances']
    if (not isinstance(basis,np.ndarray) or basis.dtype!=np.float64 or basis.ndim!=2 or basis.shape[1]!=DIMENSION or not 0<=len(basis)<=14
            or not isinstance(variance,np.ndarray) or variance.dtype!=np.float64 or variance.shape!=(len(basis),)
            or not np.isfinite(basis).all() or not np.isfinite(variance).all() or np.any(variance<=0)
            or not np.allclose(basis@basis.T,np.eye(len(basis)),rtol=0,atol=1e-10)):
        raise ValueError('Zero-to-fourteen orthonormal source directions with positive finite variances required')
    return prior


def build_prior(state):
    _source(state);means=state['known_means'].reshape(6,15,DIMENSION)
    offsets=(means-means.mean(axis=1,keepdims=True)).mean(axis=0)
    # Subtracting a reference row first makes constant offsets exactly zero;
    # the following mean centering has the same population covariance.
    offsets=offsets-offsets[:1];offsets-=offsets.mean(axis=0,keepdims=True)
    _,singular,vh=np.linalg.svd(offsets,full_matrices=False)
    tolerance=np.finfo(np.float64).eps*max(offsets.shape)*singular[0]
    rank=int(np.count_nonzero(singular>tolerance))
    prior=dict(schema=np.asarray(SCHEMA),basis=vh[:rank].copy(),variances=singular[:rank]**2/15.)
    validate_prior(prior)
    for value in prior.values():value.setflags(write=False)
    return prior,dict(rank=rank,source_mode_count=90,source_domains=15,population_divisor=15,
        basis_orthogonality_error=float(np.linalg.norm(prior['basis']@prior['basis'].T-np.eye(rank))),
        source_only=True,source_state_refitted=False,query_fit=False,hard_direction_removal=False,
        same_additive_covariance_for_known_and_background=True)


def _quadratic(residual,basis,variance,spherical):
    coordinate=residual@basis.T
    orthogonal=residual-coordinate@basis
    return (np.einsum('nd,nd->n',orthogonal,orthogonal)/spherical
        +np.sum(coordinate*coordinate/(spherical+variance)[None],axis=1))


def _terms(state,prior,white,prefix):
    means=state[prefix+'_means'];spherical=state[prefix+'_variances'];basis=prior['basis'];variance=prior['variances'];rank=len(basis)
    result=np.empty((len(white),len(means)),np.float64)
    for j,(mean,v) in enumerate(zip(means,spherical)):
        logdet=(DIMENSION-rank)*np.log(v)+np.log(v+variance).sum()
        quadratic=_quadratic(white-mean,basis,variance,v)
        result[:,j]=-.5*(DIMENSION*np.log(2*np.pi)+logdet+quadratic)
    return result


def _logsum(values):
    high=values.max(axis=1)
    return high+np.log(np.exp(values-high[:,None]).sum(axis=1))


def raw_scores(state,prior,x,closed):
    _source(state);validate_prior(prior)
    if len(prior['basis'])==0:return conditional.raw_scores(state,x,closed)[:,[1,3]]
    features=density.unit(x);labels=np.asarray(closed)
    if labels.shape!=(len(features),) or labels.dtype.kind not in 'iu' or not np.isin(labels,density.CLASSES).all():raise ValueError('Aligned frozen F0 predicted old-six classes required')
    white=(features-state['origin'])@state['whitener']
    known=_terms(state,prior,white,'known');background=_logsum(_terms(state,prior,white,'background')+np.log(state['background_weights'])[None])
    pooled=_logsum(known+np.log(state['known_weights'])[None]);predicted=np.empty(len(white),np.float64)
    for c in density.CLASSES:
        rows=np.flatnonzero(labels==c)
        if len(rows):predicted[rows]=_logsum(known[rows,c*15:(c+1)*15]-np.log(15.))
    # The same fixed whitening Jacobian cancels from both ratios.
    result=np.column_stack((pooled-background,predicted-background))
    if result.shape!=(len(features),2) or result.dtype!=np.float64 or not np.isfinite(result).all():raise ValueError('Two finite analytic nuisance-marginal ratios required')
    return result
