"""Frozen-head source geometry: standard or domain-orthogonal ViM residuals.

Only actual FP32 cosine-classifier coordinates and public source domain codes
enter fitting. Neither coordinates nor their uncentered second moment are
renormalized, centered, or shrunk here. The classifier logits stay unchanged.
"""
from __future__ import annotations
import numpy as np
from . import vim_readout_core as vim

SCHEMA='pamf_phase3_ground_geometry_v1'
MODES=('standard','domain_orthogonal')
METHODS=vim.METHODS
DIMENSION=160
RESIDUAL_DIM=80
DOMAIN_COUNT=15
STATE_FIELDS={'schema','mode','origin','residual_basis','nuisance_basis','alpha'}
classifier_coordinates=vim.classifier_coordinates


def _vim_state(state):
    return dict(schema=np.asarray(vim.SCHEMA),origin=state['origin'],
                residual_basis=state['residual_basis'],alpha=state['alpha'])


def validate_state(state):
    if not isinstance(state,dict) or set(state)!=STATE_FIELDS:
        raise ValueError('Exact six-array source geometry state required; no source rows or labels')
    for name,allowed in (('schema',(SCHEMA,)),('mode',MODES)):
        a=state[name]
        if not isinstance(a,np.ndarray) or a.shape!=() or a.dtype.kind!='U' or a.item() not in allowed:
            raise ValueError('Matching scalar Unicode source geometry '+name+' required')
    vim.validate_state(_vim_state(state))
    nuisance=state['nuisance_basis']
    if (not isinstance(nuisance,np.ndarray) or nuisance.ndim!=2 or nuisance.shape[0]!=160
            or nuisance.shape[1]>14 or nuisance.dtype!=np.float64 or not np.isfinite(nuisance).all()):
        raise ValueError('Finite FP64 nuisance basis must have shape160 by rank0..14')
    r=nuisance.shape[1]
    if state['mode'].item()=='standard' and r:
        raise ValueError('The standard mode has no nuisance basis')
    if (not np.allclose(nuisance.T@nuisance,np.eye(r),atol=1e-10,rtol=0)
            or not np.allclose(nuisance.T@state['residual_basis'],0.,atol=1e-10,rtol=0)):
        raise ValueError('Nuisance and residual bases must be internally and mutually orthogonal')
    return state


def _domains(domains,n):
    a=np.asarray(domains)
    if (a.shape!=(n,) or a.dtype!=np.int64 or not np.isin(a,range(DOMAIN_COUNT)).all()
            or set(a.tolist())!=set(range(DOMAIN_COUNT))):
        raise ValueError('All fifteen nonempty public source RX/day domains0..14 are required')
    return a


def _orthogonal_residual(a,domains):
    means=np.stack([a[domains==d].mean(axis=0) for d in range(DOMAIN_COUNT)])
    # These fourteen differences span precisely the centered fifteen-mean
    # space. They preserve identical centroids as exact zero without creating
    # a common rounding direction when computing their floating-point mean.
    differences=means[1:]-means[0]
    _,singular,vh=np.linalg.svd(differences,full_matrices=True)
    if not np.isfinite(singular).all() or not np.isfinite(vh).all():
        raise ValueError('Finite full source-domain SVD required')
    tolerance=np.finfo(np.float64).eps*max(differences.shape)*float(singular[0])
    r=int(np.count_nonzero(singular>tolerance))
    if r>14:raise ValueError('Centered fifteen-domain nuisance rank must not exceed fourteen')
    nuisance=vh[:r].T.copy()
    complement=vh[r:].T if r else np.eye(160,dtype=np.float64)
    projected=a@complement
    eigenvalues,vectors=np.linalg.eigh(projected.T@projected/len(a))
    if not np.isfinite(eigenvalues).all() or not np.isfinite(vectors).all():
        raise ValueError('Finite uncentered complement eigendecomposition required')
    residual=complement@vectors[:,:RESIDUAL_DIM]
    return residual,nuisance,eigenvalues,singular,tolerance


def fit(x,logits,domains,mode):
    if type(mode) is not str or mode not in MODES:
        raise ValueError('Only the two preregistered source geometry modes are permitted')
    a,l=vim._inputs(x,logits);codes=_domains(domains,len(a))
    if len(a)<=RESIDUAL_DIM:raise ValueError('Source row count must exceed fixed residual dimension80')
    counts=np.bincount(codes,minlength=DOMAIN_COUNT).tolist()
    singular=[];tolerance=None
    if mode=='standard':
        original,numerical=vim.fit(x,logits)
        residual=original['residual_basis'];nuisance=np.empty((160,0),np.float64)
        eigenvalues=np.asarray(numerical['eigenvalues_ascending'],np.float64)
        alpha=original['alpha'];origin=original['origin']
    else:
        residual,nuisance,eigenvalues,singular,tolerance=_orthogonal_residual(a,codes)
        if nuisance.shape[1]==0:
            # No domain-mean direction: precisely the same standard operation,
            # not a fallback from an invalid or degenerate fitted geometry.
            original,numerical=vim.fit(x,logits)
            residual=original['residual_basis'];alpha=original['alpha'];origin=original['origin']
        else:
            origin=np.zeros(160,np.float64)
            values=np.linalg.norm(a@residual,axis=1)
            denominator=float(values.mean());numerator=float(l.max(axis=1).mean())
            roundoff=64*np.finfo(np.float64).eps*float(np.linalg.norm(a,axis=1).mean())
            if not np.isfinite(denominator) or denominator<=roundoff:
                raise ValueError('Degenerate source residual; nonzero source ViM calibration required')
            if not np.isfinite(numerator) or numerator<=0:
                raise ValueError('Mean source maximum logit must produce positive alpha')
            alpha=np.asarray(numerator/denominator,np.float64)
    state=dict(schema=np.asarray(SCHEMA),mode=np.asarray(mode),origin=origin,
               residual_basis=residual,nuisance_basis=nuisance,alpha=alpha)
    validate_state(state)
    for value in state.values():value.setflags(write=False)
    r=nuisance.shape[1];values=np.linalg.norm(a@residual,axis=1)
    numerator=float(l.max(axis=1).mean());denominator=float(values.mean())
    roundoff=64*np.finfo(np.float64).eps*float(np.linalg.norm(a,axis=1).mean())
    state_values=sum(v.size for v in state.values() if v.dtype==np.float64)
    centroids=np.stack([a[codes==d].mean(axis=0) for d in range(DOMAIN_COUNT)])
    projected_centroids=centroids@residual
    diagnostic=dict(schema=SCHEMA,mode=mode,source_role='U_s',source_count=len(a),source_Us_count=len(a),
        domain_count=DOMAIN_COUNT,domain_counts=counts,feature_dimension=160,nuisance_rank=r,
        effective_dimension=160-r,principal_dimension=160-r-RESIDUAL_DIM,residual_dimension=RESIDUAL_DIM,
        assume_centered=True,empirical_mean_subtracted=False,covariance_shrinkage_applied=False,
        coordinates_renormalized=False,origin='zero_for_bias_free_native_cosine_head',
        second_moment='X.T@X/N in FP64' if mode=='standard' else '(X@C).T@(X@C)/N in FP64',
        eigenvalues_ascending=eigenvalues.tolist(),nuisance_singular_values=np.asarray(singular).tolist(),
        basis_construction='uncentered source second moment' if mode=='standard' else 'reference-domain mean differences, equivalent centered span',
        nuisance_rank_tolerance=tolerance,domain_centroid_weighting='equal_nonempty_source_domains',
        alpha=float(alpha),alpha_numerator_mean_max_logit=numerator,alpha_denominator_mean_residual=denominator,
        residual_minimum=float(values.min()),residual_maximum=float(values.max()),residual_roundoff_bound=roundoff,
        coordinate_norm_minimum=float(np.linalg.norm(a,axis=1).min()),coordinate_norm_maximum=float(np.linalg.norm(a,axis=1).max()),
        maximum_nuisance_residual_overlap=float(np.max(np.abs(nuisance.T@residual))) if r else 0.,
        domain_residual_centroid_maximum_distance=float(np.linalg.norm(projected_centroids-projected_centroids[0],axis=1).max()),
        source_only=True,source_TX_labels_used=False,OE_inputs_used=False,CAL_inputs_used=False,V_inputs_used=False,
        target_U_reads=0,target_Q_reads=0,neural_updates=0,source_rows_deployed=False,nuisance_basis_audit_only=True,
        logit_path_domain_invariance_guaranteed=False,deployed_numeric_values=12961,deployed_numeric_bytes=103688,
        audit_numeric_values=160*r,audit_numeric_bytes=160*r*8,state_numeric_values=state_values,state_numeric_bytes=state_values*8)
    return state,diagnostic


def raw_scores(state,x,logits):
    validate_state(state)
    return vim.raw_scores(_vim_state(state),x,logits)
