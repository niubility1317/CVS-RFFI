"""Source-L local Gaussian shapes with frozen mode means and OE background."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density
from . import star_class_density_core as conditional

SCHEMA='pamf_phase3_star_local_cov_state_v1'
METHODS=('pooled_ratio','predicted_ratio','local_pooled_ratio','local_predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=3
DIMENSION=160
MODES=90
CELL_SIZE=70
ALPHA=70/230
FIELDS={'schema','transforms','log_determinants','trace_variances','alpha'}
RTOL,ATOL=1e-5,1e-7


def _source(state):
    density.validate_state(state,'density_ratio')
    if state['known_means'].shape!=(MODES,DIMENSION):raise ValueError('Original D160 class-major ninety-mode density required')


def validate_local(local,state):
    _source(state)
    if (not isinstance(local,dict) or set(local)!=FIELDS or not isinstance(local['schema'],np.ndarray)
            or local['schema'].shape!=() or local['schema'].dtype.kind!='U' or local['schema'].item()!=SCHEMA):raise ValueError('Exact aggregate local-covariance schema required')
    shapes=dict(transforms=(MODES,DIMENSION,DIMENSION),log_determinants=(MODES,),trace_variances=(MODES,),alpha=())
    if any(not isinstance(local[k],np.ndarray) or local[k].dtype!=np.float64 or local[k].shape!=shape or not np.isfinite(local[k]).all() for k,shape in shapes.items()):
        raise ValueError('Finite FP64 local inverse-Cholesky, logdet, trace and alpha arrays required')
    transform=local['transforms'];diagonal=np.diagonal(transform,axis1=1,axis2=2)
    if (float(local['alpha'])!=ALPHA or np.any(diagonal<=0) or not np.allclose(np.tril(transform,-1),0.,rtol=0,atol=1e-10)
            or np.any(local['trace_variances']<density.VARIANCE_FLOOR*(1-1e-12))
            or not np.allclose(local['trace_variances'],state['known_variances'],rtol=RTOL,atol=ATOL)
            or not np.allclose(local['log_determinants'],-2*np.log(diagonal).sum(axis=1),rtol=1e-10,atol=1e-9)):
        raise ValueError('Positive local Gaussian scales, matching logdet and unchanged scalar traces required')
    return local


def fit(state,L_features,y,rx,day):
    _source(state);x=density.unit(L_features);n=len(x)
    y=density._vector(y,n,'source old y');rx,day,_,_=density._domains(rx,day,n)
    if n!=MODES*CELL_SIZE or not np.isin(y,density.CLASSES).all():raise ValueError('Only the complete original old-six source L6300 may fit local covariance')
    groups=[]
    for c in density.CLASSES:
        for r in density.RECEIVERS:
            for d in density.DAYS:
                rows=np.flatnonzero((y==c)&(rx==r)&(day==d))
                if len(rows)!=CELL_SIZE:raise ValueError('Every original class/RX/day cell must contain exactly seventy rows')
                groups.append(rows)
    white=(x-state['origin'])@state['whitener'];identity=np.eye(DIMENSION)
    transforms=[];logdet=[];traces=[];floored=[];mean_errors=[]
    for j,rows in enumerate(groups):
        values=white[rows];mean=values.mean(axis=0)
        if not np.allclose(mean,state['known_means'][j],rtol=RTOL,atol=ATOL):raise ValueError('Recomputed source mode mean differs from frozen r03 density')
        mean_errors.append(float(np.max(np.abs(mean-state['known_means'][j]))))
        residual=values-mean;population=residual.T@residual/CELL_SIZE
        covariance=ALPHA*population+(1-ALPHA)*float(state['pooled_L_variance'])*identity
        covariance=(covariance+covariance.T)*.5
        expected=float(state['known_variances'][j]);raw_trace=float(np.trace(covariance)/DIMENSION)
        if not np.isclose(max(raw_trace,density.VARIANCE_FLOOR),expected,rtol=RTOL,atol=ATOL):raise ValueError('Local population trace does not reproduce the frozen scalar variance')
        if (1-ALPHA)*float(state['pooled_L_variance'])<density.VARIANCE_FLOOR:
            eigenvalues,eigenvectors=np.linalg.eigh(covariance)
            if eigenvalues.min()<density.VARIANCE_FLOOR:
                covariance=(eigenvectors*np.maximum(eigenvalues,density.VARIANCE_FLOOR))@eigenvectors.T
                # Flooring is legal only if it retains the original scalar trace.
                if not np.isclose(float(np.trace(covariance)/DIMENSION),expected,rtol=1e-10,atol=1e-14):
                    raise ValueError('Eigenvalue floor would change frozen variance trace; no silent rescaling')
                floored.append(j)
        try:
            chol=np.linalg.cholesky(covariance);inverse=np.linalg.solve(chol,identity).T
        except np.linalg.LinAlgError as exc:raise ValueError('Source-shrunk local covariance is not positive definite') from exc
        transforms.append(inverse);logdet.append(float(2*np.log(np.diag(chol)).sum()));traces.append(float(np.trace(covariance)/DIMENSION))
    local=dict(schema=np.asarray(SCHEMA),transforms=np.asarray(transforms,np.float64),log_determinants=np.asarray(logdet,np.float64),
        trace_variances=np.asarray(traces,np.float64),alpha=np.asarray(ALPHA,np.float64))
    validate_local(local,state)
    for value in local.values():value.setflags(write=False)
    diagnostics=dict(source_count=n,known_modes=MODES,cell_size=CELL_SIZE,dimension=DIMENSION,alpha=ALPHA,population_covariance=True,
        mean_max_abs_error=max(mean_errors),trace_max_abs_error=float(np.max(np.abs(local['trace_variances']-state['known_variances']))),
        floor_components=floored,mode_order=conditional.MODE_ORDER,source_state_refitted=False,query_inputs_used=False,
        original_means_used_for_scoring=True,background_unchanged=True,source_role='L_only')
    return local,diagnostics


def _logsum(values):
    high=values.max(axis=1)
    return high+np.log(np.exp(values-high[:,None]).sum(axis=1))


def raw_scores(state,local,x,closed):
    validate_local(local,state)
    original=conditional.raw_scores(state,x,closed);white=(density.unit(x)-state['origin'])@state['whitener']
    terms=np.empty((len(white),MODES),np.float64)
    for j in range(MODES):
        transformed=(white-state['known_means'][j])@local['transforms'][j]
        squared=np.einsum('nd,nd->n',transformed,transformed)
        terms[:,j]=-.5*(DIMENSION*np.log(2*np.pi)+local['log_determinants'][j]+squared)
    jacobian=float(state['log_abs_det_whitener']);background=density._log_density(state,white,'background')
    pooled=_logsum(terms)-np.log(MODES)+jacobian;predicted=np.empty(len(white),np.float64);labels=np.asarray(closed)
    for c in density.CLASSES:
        rows=np.flatnonzero(labels==c)
        if len(rows):predicted[rows]=_logsum(terms[rows,c*15:(c+1)*15])-np.log(15.)+jacobian
    result=np.column_stack((original[:,1],original[:,3],pooled-background,predicted-background))
    if result.shape!=(len(white),4) or result.dtype!=np.float64 or not np.isfinite(result).all():raise ValueError('Four complete finite old/local density-ratio scores required')
    return result
