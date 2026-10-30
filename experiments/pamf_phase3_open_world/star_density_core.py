"""Source-only spherical Gaussian mixtures in a shared full-whitened space.

The known model has one equal-mass component per old-class/RX/day cell.
Background components use domain-balanced weighted k-means, not TX labels.
All persistent state consists only of flat aggregate NumPy arrays.
"""
from __future__ import annotations
import numpy as np

SCHEMA='pamf_phase3_star_density_state_v1'
CONTRIBUTION_SCHEMA='pamf_phase3_star_contribution_density_state_v1'
JOINT_SCHEMA='pamf_phase3_star_joint_feature_density_state_v1'
METHODS=('known_only','density_ratio')
DIMENSION=160
CLASSES=(0,1,2,3,4,5)
RECEIVERS=(1,3,4,6,8)
DAYS=(1,2,3)
KNOWN_CELL_SIZE=70
OE_COUNT=6134
CAPACITY=32
LLOYD_ITERATIONS=25
KMEANS_SEED=2026103019
VARIANCE_FLOOR=1e-8
KNOWN_FIELDS={'schema','origin','whitener','log_abs_det_whitener','pooled_L_variance',
              'known_means','known_variances','known_weights'}
BACKGROUND_FIELDS={'background_means','background_variances','background_weights'}


def _dimension(dimension):
    if dimension is None:return DIMENSION
    if type(dimension) is not int or dimension not in (DIMENSION,2*DIMENSION,4*DIMENSION):
        raise ValueError('Only original fused160, joint320 or four-contribution640 geometry is allowed')
    return dimension


def _state_dimension(state):
    schema=state.get('schema')
    if not isinstance(schema,np.ndarray) or schema.shape!=() or schema.dtype.kind!='U':
        raise ValueError('Matching flat-array density state schema required')
    if schema.item()==SCHEMA:return DIMENSION
    if schema.item()==JOINT_SCHEMA:return 2*DIMENSION
    if schema.item()==CONTRIBUTION_SCHEMA:return 4*DIMENSION
    raise ValueError('Only original fused160, joint320 and four-contribution density schemas are allowed')


def unit(x,dimension=None):
    """Canonical FP64 row normalization, shared by every fit and score path."""
    width=_dimension(dimension);a=np.asarray(x)
    if a.ndim!=2 or a.shape[1]!=width or not len(a) or a.dtype.kind!='f':
        raise ValueError('Nonempty floating features with the declared geometry required')
    a=a.astype(np.float64,copy=True)
    if not np.isfinite(a).all():raise ValueError('Features must be finite')
    with np.errstate(over='ignore',invalid='ignore'):norm=np.linalg.norm(a,axis=1,keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm==0):raise ValueError('Every source/query feature must be nonzero with finite norm')
    return a/norm


def contribution_unit(c):
    """Unit each nonzero native branch, then unit their concatenation in FP64."""
    a=np.asarray(c)
    if a.ndim!=3 or a.shape[1:]!=(4,DIMENSION) or not len(a) or a.dtype.kind!='f':
        raise ValueError('Finite floating four-contribution features [N,4,160] required')
    a=a.astype(np.float64,copy=True)
    if not np.isfinite(a).all():raise ValueError('Contributions must be finite')
    with np.errstate(over='ignore',invalid='ignore'):norm=np.linalg.norm(a,axis=2,keepdims=True)
    if not np.isfinite(norm).all():raise ValueError('Every branch norm must be finite')
    branches=np.divide(a,norm,out=np.zeros_like(a),where=norm>0)
    return unit(branches.reshape(len(a),4*DIMENSION),dimension=4*DIMENSION).astype(np.float32)


def _vector(a,n,name):
    a=np.asarray(a)
    if a.shape!=(n,) or a.dtype.kind not in 'iu':raise ValueError('Aligned integer public metadata required: '+name)
    return a


def _domains(rx,day,n):
    rx=_vector(rx,n,'rx');day=_vector(day,n,'day')
    if not np.isin(rx,RECEIVERS).all() or not np.isin(day,DAYS).all():raise ValueError('Only the original source RX/day roster is allowed')
    groups=[];counts=[]
    for r in RECEIVERS:
        for d in DAYS:
            rows=np.flatnonzero((rx==r)&(day==d))
            if not len(rows):raise ValueError('Every public source RX/day cell must be nonempty')
            groups.append(rows);counts.append(len(rows))
    return rx,day,groups,np.asarray(counts,np.int64)


def _frozen(state):
    for a in state.values():a.setflags(write=False)
    return state


def validate_state(state,method):
    if method not in METHODS:raise ValueError('Explicit known_only or density_ratio method required')
    fields=KNOWN_FIELDS|(BACKGROUND_FIELDS if method=='density_ratio' else set())
    if not isinstance(state,dict) or set(state)!=fields:raise ValueError('Exact aggregate-only density state fields required')
    width=_state_dimension(state)
    for name in fields-{'schema'}:
        a=state[name]
        if not isinstance(a,np.ndarray) or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Finite FP64 aggregate arrays required: '+name)
    modes=len(CLASSES)*len(RECEIVERS)*len(DAYS)
    shapes=dict(origin=(width,),whitener=(width,width),log_abs_det_whitener=(),pooled_L_variance=(),
        known_means=(modes,width),known_variances=(modes,),known_weights=(modes,))
    if any(state[k].shape!=shape for k,shape in shapes.items()):raise ValueError('Known Gaussian aggregate shapes differ')
    if (float(state['pooled_L_variance'])<0 or np.any(state['known_variances']<VARIANCE_FLOOR)
            or not np.allclose(state['known_weights'],np.full(modes,1/modes),rtol=0.,atol=1e-15)):
        raise ValueError('Known variances and equal cell weights must be valid')
    sign,logdet=np.linalg.slogdet(state['whitener'])
    if sign==0 or not np.isfinite(logdet) or not np.isclose(float(state['log_abs_det_whitener']),logdet,rtol=1e-12,atol=1e-12):
        raise ValueError('Nonsingular shared whitening and its actual log-Jacobian required')
    if method=='density_ratio':
        means=state['background_means'];variance=state['background_variances'];weights=state['background_weights']
        if (means.ndim!=2 or means.shape[1]!=width or not 1<=len(means)<=CAPACITY
                or variance.shape!=(len(means),) or weights.shape!=(len(means),)
                or np.any(variance<VARIANCE_FLOOR) or np.any(weights<=0) or not np.isclose(weights.sum(),1.,rtol=0.,atol=1e-12)):
            raise ValueError('Only positive-mass finite background Gaussian components may deploy')
    return state


def fit_known(x,y,rx,day,*,dimension=None):
    width=_dimension(dimension);a=unit(x,dimension=width);n=len(a);y=_vector(y,n,'visible old y');rx,day,_,domain_counts=_domains(rx,day,n)
    if not np.isin(y,CLASSES).all():raise ValueError('Only original old-class source-L labels are permitted')
    groups=[];means=[];residual=np.empty_like(a)
    for c in CLASSES:
        for r in RECEIVERS:
            for d in DAYS:
                rows=np.flatnonzero((y==c)&(rx==r)&(day==d))
                if len(rows)!=KNOWN_CELL_SIZE:raise ValueError('Every original class/RX/day mode must contain exactly 70 records')
                mean=a[rows].mean(axis=0);residual[rows]=a[rows]-mean;means.append(mean);groups.append(rows)
    origin=a.mean(axis=0);covariance=residual.T@residual/n
    isotropic=max(float(np.trace(covariance))/width,VARIANCE_FLOOR)
    covariance=.9*covariance+.1*isotropic*np.eye(width)
    whitener=np.linalg.inv(np.linalg.cholesky(covariance)).T
    zresidual=residual@whitener
    pooled=float(np.square(zresidual).sum()/(n*width));alpha=KNOWN_CELL_SIZE/(KNOWN_CELL_SIZE+width)
    radial=np.asarray([np.square(zresidual[rows]).sum()/(len(rows)*width) for rows in groups])
    variance=np.maximum(alpha*radial+(1-alpha)*pooled,VARIANCE_FLOOR)
    schema=SCHEMA if width==DIMENSION else JOINT_SCHEMA if width==2*DIMENSION else CONTRIBUTION_SCHEMA
    state=dict(schema=np.asarray(schema),origin=origin,whitener=whitener,
        log_abs_det_whitener=np.asarray(np.linalg.slogdet(whitener)[1],np.float64),pooled_L_variance=np.asarray(pooled,np.float64),
        known_means=(np.asarray(means)-origin)@whitener,known_variances=variance,known_weights=np.full(len(groups),1/len(groups),np.float64))
    validate_state(state,'known_only')
    diag=dict(known_count=n,known_mode_count=len(groups),known_cell_count=KNOWN_CELL_SIZE,dimension=width,
        domain_counts=domain_counts.tolist(),covariance='pooled population residuals within class/RX/day mode',
        covariance_data_weight=.9,covariance_isotropic_weight=.1,variance_floor=VARIANCE_FLOOR,
        pooled_L_variance=pooled,variance_shrinkage_alpha=alpha,source_only=True,background_values_read=0)
    return _frozen(state),diag


def _distances(x,means):
    # Subtract directly to retain precision near tight, distant components.
    # Only [N,D] temporary memory is used, never an [N,K,D] feature bank.
    out=np.empty((len(x),len(means)),np.float64)
    for j,mean in enumerate(means):
        difference=x-mean;out[:,j]=np.einsum('nd,nd->n',difference,difference)
    return out


def fit_background(known_state,x,rx,day):
    validate_state(known_state,'known_only');width=_state_dimension(known_state);a=unit(x,dimension=width);n=len(a)
    if n!=OE_COUNT or n<CAPACITY:raise ValueError('The complete original 6134-row source-OE role is required')
    _,_,domains,domain_counts=_domains(rx,day,n);w=np.empty(n,np.float64)
    for rows in domains:w[rows]=1/(len(domains)*len(rows))
    z=(a-known_state['origin'])@known_state['whitener'];rng=np.random.default_rng(KMEANS_SEED)
    selected=[int(rng.choice(n,p=w/w.sum()))];closest=_distances(z,z[selected])[:,0]
    for _ in range(1,CAPACITY):
        potential=w*closest;potential[selected]=0;total=float(potential.sum())
        if total>0:choice=int(rng.choice(n,p=potential/total))
        else:choice=next(i for i in range(n) if i not in selected)
        selected.append(choice);closest=np.minimum(closest,_distances(z,z[[choice]])[:,0])
    means=z[selected].copy();trace=[]
    for iteration in range(1,LLOYD_ITERATIONS+1):
        labels=_distances(z,means).argmin(axis=1);counts=np.bincount(labels,minlength=CAPACITY)
        for j in range(CAPACITY):
            mask=labels==j
            if counts[j]:means[j]=np.sum(w[mask,None]*z[mask],axis=0)/w[mask].sum()
        error=z-means[labels];objective=float(np.sum(w*np.einsum('nd,nd->n',error,error)))
        trace.append(dict(iteration=iteration,weighted_sse=objective,empty_modes=int((counts==0).sum()),active_modes=int((counts>0).sum())))
    active=np.flatnonzero(counts);mass=[];variance=[];effective=[]
    for j in active:
        mask=labels==j;weights=w[mask];weight=float(weights.sum());neff=weight*weight/float(np.square(weights).sum())
        residual=z[mask]-means[j];radial=float(np.sum(weights*np.einsum('nd,nd->n',residual,residual))/(weight*width))
        alpha=neff/(neff+width)
        mass.append(weight);effective.append(neff);variance.append(max(alpha*radial+(1-alpha)*float(known_state['pooled_L_variance']),VARIANCE_FLOOR))
    state={k:v.copy() for k,v in known_state.items()}
    state.update(background_means=means[active].copy(),background_variances=np.asarray(variance,np.float64),background_weights=np.asarray(mass,np.float64))
    validate_state(state,'density_ratio')
    diag=dict(background_count=n,dimension=width,capacity=CAPACITY,lloyd_iterations=LLOYD_ITERATIONS,
        kmeans_seed=KMEANS_SEED,lloyd_trace=trace,domain_counts=domain_counts.tolist(),domain_total_weights=[float(w[rows].sum()) for rows in domains],
        effective_mode_count=len(active),empty_modes_removed=CAPACITY-len(active),mode_row_counts=counts[active].tolist(),
        effective_counts=effective,mode_weights=mass,background_modes_are_TX_classes=False,source_only=True,
        final_assignment='last fixed Lloyd assignment; means, variances and weights share that assignment',
        zero_residual_initialization='first unselected source row in stable input order')
    return _frozen(state),diag


def _log_density(state,z,prefix):
    means=state[prefix+'_means'];variance=state[prefix+'_variances'];weights=state[prefix+'_weights']
    components=(np.log(weights)[None]-_state_dimension(state)/2*np.log(2*np.pi*variance)[None]
        -_distances(z,means)/(2*variance[None]))
    maximum=components.max(axis=1)
    return maximum+np.log(np.exp(components-maximum[:,None]).sum(axis=1))+float(state['log_abs_det_whitener'])


def score(state,x,method):
    validate_state(state,method);a=unit(x,dimension=_state_dimension(state));z=(a-state['origin'])@state['whitener']
    result=_log_density(state,z,'known')
    if method=='density_ratio':result=result-_log_density(state,z,'background')
    if result.shape!=(len(a),) or result.dtype!=np.float64 or not np.isfinite(result).all():
        raise ValueError('Density arithmetic must produce finite FP64 scores')
    return result
