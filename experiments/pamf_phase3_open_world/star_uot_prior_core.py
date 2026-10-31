"""U-only unbalanced transport reweights frozen source mixture priors only."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density
from . import star_class_density_core as original

ADAPTER_SCHEMA='pamf_phase3_uot_prior_adapter_v1'
METHODS=('pooled_ratio','predicted_ratio','UOT_pooled_ratio','UOT_predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=3
ITERATIONS=200
EPSILON=.05
RHO_SOURCE=1.
RHO_TARGET=1.
BLEND=.5
U_COUNT=384


def _logsumexp(x,axis):
    peak=x.max(axis=axis,keepdims=True)
    return np.squeeze(peak+np.log(np.exp(x-peak).sum(axis=axis,keepdims=True)),axis=axis)


def _state(state):
    density.validate_state(state,'density_ratio')
    if state['schema'].item()!=density.SCHEMA or state['known_means'].shape!=(90,160):raise ValueError('Only original class-major six-by-fifteen D160 source statistics are allowed')
    return state


def _white(state,features):
    _state(state)
    with np.errstate(over='ignore',invalid='ignore'):w=(density.unit(features)-state['origin'])@state['whitener']
    if not np.isfinite(w).all():raise ValueError('Finite original whitened coordinates required')
    return w


def _sinkhorn(cost,a,b):
    """Fixed generalized Sinkhorn on positive measures; return log row masses only."""
    cost,a,b=(np.asarray(v,np.float64) for v in (cost,a,b))
    if (cost.ndim!=2 or not cost.size or a.shape!=(cost.shape[0],) or b.shape!=(cost.shape[1],)
            or any(not np.isfinite(v).all() for v in (cost,a,b)) or np.any(cost<0) or np.any(a<=0) or np.any(b<=0)):
        raise ValueError('Finite nonnegative rectangular transport costs and positive aligned measures required')
    with np.errstate(over='ignore',invalid='ignore'):logK=-cost/EPSILON
    if not np.isfinite(logK).all():raise ValueError('Finite log-domain kernel required')
    loga,logb=np.log(a),np.log(b);u=np.zeros_like(a);v=np.zeros_like(b);trace=[]
    theta_s=RHO_SOURCE/(RHO_SOURCE+EPSILON);theta_t=RHO_TARGET/(RHO_TARGET+EPSILON)
    for iteration in range(1,ITERATIONS+1):
        new_u=theta_s*(loga-_logsumexp(logK+v[None],axis=1))
        new_v=theta_t*(logb-_logsumexp(logK+new_u[:,None],axis=0))
        change=max(float(np.abs(new_u-u).max()),float(np.abs(new_v-v).max()));u,v=new_u,new_v
        if not np.isfinite(change) or not np.isfinite(u).all() or not np.isfinite(v).all():raise ValueError('Finite generalized Sinkhorn scaling updates required')
        trace.append(dict(iteration=iteration,max_log_scaling_change=change))
    row=u+_logsumexp(logK+v[None],axis=1);column=v+_logsumexp(logK+u[:,None],axis=0)
    # C + eps log(P) + rho_s log(row/a) + rho_t log(column/b).
    source_residual=EPSILON*u+RHO_SOURCE*(row-loga);target_residual=EPSILON*v+RHO_TARGET*(column-logb)
    stationarity=float(np.abs(source_residual[:,None]+target_residual[None]).max());total=float(_logsumexp(row,axis=0))
    if not np.isfinite(row).all() or not np.isfinite(total) or not np.isfinite(stationarity):raise ValueError('Finite log transport masses and KKT residual required')
    return row,dict(iterations=ITERATIONS,epsilon=EPSILON,rho_source=RHO_SOURCE,rho_target=RHO_TARGET,
        theta_source=theta_s,theta_target=theta_t,log_total_mass=total,kkt_stationarity_residual=stationarity,
        max_log_scaling_change=change,trace=trace)


def validate_adapter(adapter,state):
    _state(state)
    if (not isinstance(adapter,dict) or set(adapter)!={'schema','domain_weights','background_weights'}
            or not isinstance(adapter['schema'],np.ndarray) or adapter['schema'].shape!=() or adapter['schema'].dtype.kind!='U'
            or adapter['schema'].item()!=ADAPTER_SCHEMA):raise ValueError('Exact aggregate-only UOT prior adapter required')
    for key,shape,floor in (('domain_weights',(15,),np.full(15,BLEND/15)),
            ('background_weights',(len(state['background_means']),),BLEND*state['background_weights'])):
        value=adapter[key]
        if (not isinstance(value,np.ndarray) or value.dtype!=np.float64 or value.shape!=shape or not np.isfinite(value).all()
                or np.any(value<=0) or np.any(value<floor) or not np.isclose(value.sum(),1.,rtol=0,atol=1e-12)):
            raise ValueError('Positive normalized FP64 prior weights retaining half the original source support required')
    return adapter


def fit_adapter(state,U_features):
    w=_white(state,U_features)
    if len(w)!=U_COUNT:raise ValueError('Only the complete fixed unlabeled star-G10 U feature role may fit')
    means=np.concatenate((state['known_means'],state['background_means']))
    variance=np.concatenate((state['known_variances'],state['background_variances']))
    with np.errstate(over='ignore',invalid='ignore'):cost=density._distances(w,means).T/(160*variance[:,None])
    prior=np.concatenate((.5*state['known_weights'],.5*state['background_weights']))
    row,diag=_sinkhorn(cost,prior,np.full(len(w),1/len(w),np.float64))
    domain_log_mass=_logsumexp(row[:90].reshape(6,15),axis=0)
    learned_domain=np.exp(domain_log_mass-_logsumexp(domain_log_mass,axis=0))
    learned_background=np.exp(row[90:]-_logsumexp(row[90:],axis=0))
    adapter=dict(schema=np.asarray(ADAPTER_SCHEMA),domain_weights=BLEND/15+(1-BLEND)*learned_domain,
        background_weights=BLEND*state['background_weights']+(1-BLEND)*learned_background)
    validate_adapter(adapter,state)
    diag.update(blend=BLEND,U_count=len(w),U_only=True,Q_inputs_used=False,true_U_labels_used=False,true_unknown_count_used=False,
        default_logits_used=False,source_state_unchanged_by_fit=True,U_bank_deployed=False,coupling_deployed=False,
        known_transport_fraction=float(np.exp(_logsumexp(row[:90],axis=0)-diag['log_total_mass'])),
        known_transport_fraction_is_true_class_proportion=False,learned_domain_weights=learned_domain.tolist(),learned_background_weights=learned_background.tolist(),
        fitted_prior_scalars=15+len(learned_background),simplex_degrees_of_freedom=14+len(learned_background)-1,
        known_class_prior='equal one-sixth, shared fifteen-domain conditional weights',source_gaussians_refitted=False)
    return adapter,diag


def raw_scores(state,features,closed,adapter):
    validate_adapter(adapter,state);w=_white(state,features);c=np.asarray(closed)
    if c.shape!=(len(w),) or c.dtype.kind not in 'iu' or not np.isin(c,range(6)).all():raise ValueError('Aligned frozen original F0 classes required; no true-label routing')
    if np.array_equal(adapter['domain_weights'],np.full(15,1/15)) and np.array_equal(adapter['background_weights'],state['background_weights']):
        return original.raw_scores(state,features,c)[:,[1,3]]
    variance=state['known_variances'];bg_variance=state['background_variances'];domain=adapter['domain_weights']
    known=-80*np.log(2*np.pi*variance)[None]-density._distances(w,state['known_means'])/(2*variance[None])
    background=-80*np.log(2*np.pi*bg_variance)[None]-density._distances(w,state['background_means'])/(2*bg_variance[None])+np.log(adapter['background_weights'])[None]
    bg=_logsumexp(background,axis=1);pooled=_logsumexp(known+np.log(np.tile(domain/6,6))[None],axis=1)
    conditional=_logsumexp(known.reshape(len(w),6,15)[np.arange(len(w)),c]+np.log(domain)[None],axis=1)
    result=np.column_stack((pooled-bg,conditional-bg))
    if not np.isfinite(result).all():raise ValueError('Finite prior-reweighted FP64 Gaussian ratios required')
    return result
