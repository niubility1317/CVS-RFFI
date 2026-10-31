"""Source-L population MD/RMD over frozen unit PCD features.

RMD subtracts the class-independent background distance (Ren et al., 2021,
arXiv:2106.09022). Every exported score is oriented higher-is-known.
"""
from __future__ import annotations
import numpy as np
from .star_density_core import unit

SCHEMA='pamf_phase3_pcd_relative_metric_v1'
METHODS=('MD_pooled','MD_predicted','RMD_pooled','RMD_predicted')
METRIC_FIELDS={'schema','class_means','background_mean','within_cholesky','background_cholesky'}
DIMENSION=160


def _features(value):
    a=np.asarray(value)
    if a.ndim!=2 or a.shape[1]!=DIMENSION or not len(a) or a.dtype!=np.float32 or not np.isfinite(a).all():
        raise ValueError('Nonempty finite FP32 unit160 features required')
    norm=np.linalg.norm(a.astype(np.float64),axis=1)
    if not np.allclose(norm,1.,rtol=0.,atol=2e-6):raise ValueError('Unit160 features required within tolerance 2e-6')
    return unit(a)


def _labels(value,n,complete=False):
    y=np.asarray(value)
    if y.shape!=(n,) or y.dtype.kind not in 'iu' or not np.isin(y,range(6)).all() or (complete and set(y.tolist())!=set(range(6))):
        raise ValueError('Aligned old-six integer classes required; source fit needs every class')
    return y


def _cholesky(residual):
    covariance=residual.T@residual/len(residual)
    shrunk=.9*covariance+.1*np.trace(covariance)/DIMENSION*np.eye(DIMENSION)
    if not np.isfinite(shrunk).all():raise ValueError('Finite population covariance required')
    try:return np.linalg.cholesky(shrunk)
    except np.linalg.LinAlgError as exc:raise ValueError('Degenerate source covariance; no fallback or jitter') from exc


def validate_metric(state):
    if not isinstance(state,dict) or set(state)!=METRIC_FIELDS:raise ValueError('Exact five-array aggregate metric state required; no row bank or truth')
    schema=state['schema']
    if not isinstance(schema,np.ndarray) or schema.shape!=() or schema.dtype.kind!='U' or schema.item()!=SCHEMA:
        raise ValueError('Matching scalar Unicode metric schema required')
    for key,shape in (('class_means',(6,DIMENSION)),('background_mean',(DIMENSION,)),('within_cholesky',(DIMENSION,DIMENSION)),('background_cholesky',(DIMENSION,DIMENSION))):
        a=state[key]
        if not isinstance(a,np.ndarray) or a.shape!=shape or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Finite FP64 metric aggregate required: '+key)
        if key.endswith('cholesky'):
            if not np.array_equal(a,np.tril(a)) or not np.all(np.diag(a)>0):raise ValueError('Lower triangular positive-diagonal Cholesky factors required')
        elif np.any(np.linalg.norm(a,axis=-1)>1.+2e-6):raise ValueError('Arithmetic means of unit inputs cannot exceed unit norm')
    return state


def fit_metric(L_features,L_y):
    z=_features(L_features);y=_labels(L_y,len(z),complete=True)
    means=np.stack([z[y==c].mean(0) for c in range(6)]);background=z.mean(0)
    state=dict(schema=np.asarray(SCHEMA),class_means=means,background_mean=background,
        within_cholesky=_cholesky(z-means[y]),background_cholesky=_cholesky(z-background))
    validate_metric(state)
    for value in state.values():value.setflags(write=False)
    return state


def metric_scores(features,closed,state):
    z=_features(features);classes=_labels(closed,len(z));validate_metric(state)
    # Whiten each query once; no source records or query-derived statistics.
    try:
        white=np.linalg.solve(state['within_cholesky'],np.concatenate((z,state['class_means']),axis=0).T).T
        queries,centers=white[:-6],white[-6:]
        distances=np.column_stack([np.einsum('nd,nd->n',queries-center,queries-center,optimize=False) for center in centers])
        bg=np.linalg.solve(state['background_cholesky'],(z-state['background_mean']).T).T
    except np.linalg.LinAlgError as exc:raise ValueError('Metric factors could not be solved') from exc
    background=np.einsum('nd,nd->n',bg,bg,optimize=False);minimum=distances.min(1);predicted=distances[np.arange(len(z)),classes]
    result=np.column_stack((-minimum,-predicted,background-minimum,background-predicted))
    if not np.isfinite(result).all():raise ValueError('Finite MD/RMD scores required')
    return result
