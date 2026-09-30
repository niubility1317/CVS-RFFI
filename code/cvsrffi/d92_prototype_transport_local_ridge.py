"""Train-only prototype transport, differentiated through the LocalRidge head.

The original interaction remains one half of the joint squared distance.
Only ten shared scalar parameters (nine constrained degrees of freedom) train.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import solve_triangular
from . import d92_joint_channel_local_ridge as ch
from . import d92_branch_local_ridge as local
from . import d92_branch_interaction as interaction
from .d92_branch_ridge import _freeze, _plain, _readonly

_EPS=np.finfo(np.float64).eps
_NAMES=ch._NAMES
_SLICES=ch._SLICES
NumericalFailure=ch.NumericalFailure
_finite=ch._finite
_norm=ch._norm
_row_norm=ch._row_norm
FROZEN_CONFIG=dict(
    schema='d92_prototype_transport_local_ridge_v1',method='D92-PrototypeTransportLocalRidge-v1',
    base_algorithm=deepcopy(local.FROZEN_CONFIG),
    channel=dict(route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25000000),
    allowed_scenarios=['practical_high','practical_mid','practical_low_urban'],
    arms=['local_ridge','transport_seq','transport_reset'],
    adapter='prototype_conditioned_norm_preserving_tangent_rotation',
    block_dimensions=[160,96,160,160,160],trainable_parameter_count=10,effective_parameter_count=9,
    beta_bound=float(np.pi/4),eta_bound=float(np.log(2)/2),eta_sum=0.,
    eta_constraint='fixed_geometric_mean_branch_weight_not_scale_redundancy',
    prototype='train_class_mean_original_block_and_mean_unit_direction',
    neighbors='original_sum_block_squared_distance_nearest_two_include_all_second_ties',
    prototype_scale='train_median_nearest_wrong_prototype_including_zero',
    joint_original_distance_weight=.5,trace_target='original_inner_train_interaction_trace',
    objective='RMS_across_classes_of_inner_held_mean_half_squared_unit_margin_hinge_plus_theta_anchor_norm2_over_2N',
    optimizer='normalized_projected_gradient_Armijo',max_iterations=4,max_trials=3,
    initial_step_size=.125,backtrack_factor=.5,armijo_coefficient=1e-4,
    armijo_tolerance_multiplier=128,armijo_tolerance='128*eps64*max(1,abs(loss_before),abs(loss_trial),abs(rhs))',
    projection_max_bisections=80,projection_tolerance_multiplier=128,
    projection_tolerance='128*eps64*5*eta_bound',
    rotation_small_angle=1e-4,numerical_tolerance_multiplier=128,
    inner_folds='per_class_physical_id_sort_position_mod_min_K_3',
    tie_gradient='uniform_exact_wrong_max;uniform_exact_nearest_and_tied_median_groups',
    initialization_B='zero',initialization_C_seq='inherit_B_theta',initialization_C_reset='zero',
    final_old_prototypes='inherit_B_exact_columns_after_ID_label_raw_binding',
    zero_beta='bitwise_R0_forward_live_beta_derivative',
    no_information='physical_K1_or_single_class_preserve_anchor',
    prototype_degenerate='single_class_or_zero_nu_identity_adapter',
    n0='reuse_corresponding_B',query_decision_policy='per_sample_all_registered_classes',
    source_inputs=False,query_fit=False,phase1_frozen=True,encoder_backward=False,
    parameter_search=False,dtype='float64')
_CONFIG=deepcopy(FROZEN_CONFIG)


def _parameter(theta):
    x=np.asarray(theta,dtype=np.float64)
    h=_CONFIG['eta_bound'];tol=128*_EPS*5*h
    if x.shape!=(10,) or not np.isfinite(x).all():raise ValueError('theta must be finite shape (10,)')
    if np.max(np.abs(x[:5]))>_CONFIG['beta_bound']+tol or np.max(np.abs(x[5:]))>h+tol or abs(float(x[5:].sum()))>tol:
        raise ValueError('theta outside beta box / zero-sum eta box')
    return x


def project_transport(value):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(10,):raise ValueError('theta shape')
    _finite(value);out=value.copy();out[:5]=np.clip(out[:5],-_CONFIG['beta_bound'],_CONFIG['beta_bound'])
    v=value[5:];h=_CONFIG['eta_bound'];tol=128*_EPS*5*h
    if np.max(np.abs(v))<=h and abs(float(v.sum()))<=tol:return out
    lo=float(np.min(v-h));hi=float(np.max(v+h))
    for _ in range(_CONFIG['projection_max_bisections']):
        mid=lo+(hi-lo)/2;x=np.clip(v-mid,-h,h);total=float(x.sum())
        if abs(total)<=tol:break
        if total>0:lo=mid
        else:hi=mid
    if abs(float(x.sum()))>tol:raise FloatingPointError('PROTOTYPE_ETA_PROJECTION_UNRESOLVED')
    out[5:]=x;return _parameter(out)


@dataclass(frozen=True)
class PrototypeTable:
    p: np.ndarray
    m: np.ndarray
    nu: float | None
    classes: tuple
    audit: Mapping
    def __post_init__(self):
        for k in ('p','m'):object.__setattr__(self,k,_readonly(getattr(self,k)))
        object.__setattr__(self,'audit',_freeze(self.audit))


def _unit_blocks(x):
    rho=np.column_stack([_row_norm(x[:,sl]) for sl in _SLICES]);w=np.zeros_like(x)
    for l,sl in enumerate(_SLICES):
        active=rho[:,l]>0
        w[active,sl]=x[active,sl]/rho[active,l,None]
    _finite(w,rho);return w,rho


def _prototype_distances(x,p):
    # Direct subtraction is essential for nearby records and prototypes.
    d=np.empty((len(x),len(p),5))
    for l,sl in enumerate(_SLICES):
        for i in range(len(x)):
            delta=x[i,sl]-p[:,sl];norms=_row_norm(delta);d[i,:,l]=norms*norms
            if np.any(np.any(delta!=0,axis=1)&(d[i,:,l]==0)):
                raise FloatingPointError('POSITIVE_PROTOTYPE_DISTANCE_UNDERFLOW')
    _finite(d);return d


def _make_prototypes(b,a,y,classes,ids,*,inherited=None):
    start=time.perf_counter();x=np.concatenate((b,a),axis=1);w,_=_unit_blocks(x)
    p=np.stack([x[y==c].mean(axis=0) for c in range(len(classes))])
    m=np.stack([w[y==c].mean(axis=0) for c in range(len(classes))])
    copied=[]
    if inherited is not None:
        for j,cls in enumerate(inherited.classes):
            dest=classes.index(cls)
            if not np.array_equal(p[dest],inherited.prototypes.p[j]) or not np.array_equal(m[dest],inherited.prototypes.m[j]):
                raise ValueError('Inherited old prototype recomputation mismatch')
            p[dest]=inherited.prototypes.p[j];m[dest]=inherited.prototypes.m[j];copied.append(cls)
    d=_prototype_distances(x,p).sum(axis=2)
    nu=None if len(classes)==1 else float(np.median(np.min(np.where(np.arange(len(classes))[None,:]!=y[:,None],d,np.inf),axis=1)))
    if nu is not None and (not np.isfinite(nu) or nu<0):raise FloatingPointError('PROTOTYPE_SCALE_UNRESOLVED')
    return PrototypeTable(p,m,nu,classes,dict(training_physical_ids=list(ids),classes=list(classes),
        prototype_scale=nu,prototype_identity=nu is None or nu==0,
        prototype_construction_count=1,prototype_distance_evaluation_count=1,
        inherited_old_prototype_classes=copied,old_prototypes_bitwise_inherited=bool(copied),
        prototype_numeric_state_bytes=p.nbytes+m.nbytes+8,
        prototype_seconds=time.perf_counter()-start))


def _map_context(b,a,table):
    x=np.concatenate((b,a),axis=1);w,rho=_unit_blocks(x)
    d=_prototype_distances(x,table.p);total=d.sum(axis=2)
    if len(table.classes)>1:
        threshold=np.partition(total,1,axis=1)[:,1];mask=total<=threshold[:,None]
    else:mask=np.ones_like(total,dtype=bool)
    return dict(x=x,w=w,rho=rho,d=d,neighbors=mask)


def _adapt(context,table,theta,*,derivative_cache=False):
    theta=_parameter(theta);x,w,rho=context['x'],context['w'],context['rho']
    if table.nu is None or table.nu==0:
        return x[:,:256],x[:,256:],None
    gain=np.exp(theta[5:]);weighted=np.sum(context['d']*gain,axis=2)
    nearest=np.min(np.where(context['neighbors'],weighted,np.inf),axis=1)
    with np.errstate(over='ignore'):
        logits=-(weighted-nearest[:,None])/table.nu
    logits=np.where(context['neighbors'],logits,-np.inf)
    att=np.exp(logits);att/=att.sum(axis=1)[:,None]
    # GEMM for a batch and GEMV for a single sample can use different BLAS
    # reduction trees. Keep one identical per-sample reduction in fit and score.
    mu=np.empty_like(x)
    for i in range(len(x)):mu[i]=np.sum(att[i,:,None]*table.m,axis=0)
    mapped=np.empty_like(x);tall=np.empty_like(x);qs=np.empty((len(x),5))
    for l,sl in enumerate(_SLICES):
        wl=w[:,sl];t=mu[:,sl]-np.sum(wl*mu[:,sl],axis=1)[:,None]*wl
        q=_row_norm(t);z=theta[l]*q
        mapped[:,sl]=rho[:,l,None]*(np.cos(z)[:,None]*wl+(theta[l]*np.sinc(z/np.pi))[:,None]*t)
        tall[:,sl]=t;qs[:,l]=q
        err=np.abs(_row_norm(mapped[:,sl])-rho[:,l])
        if np.any(err>128*_EPS*len(range(*sl.indices(736)))*rho[:,l]):raise FloatingPointError('PROTOTYPE_ROTATION_NORM_FAILED')
    _finite(mapped,att)
    # Preserve the actual original bits without dropping the live VJP cache.
    if np.all(theta[:5]==0):mapped=x
    cache=dict(att=att,mu=mu,t=tall,q=qs) if derivative_cache else None
    return mapped[:,:256],mapped[:,256:],cache


def _rotation_vjp(context,table,theta,gb,ga,cache=None):
    if table.nu is None or table.nu==0:return np.zeros(10)
    if cache is None:_,_,cache=_adapt(context,table,theta,derivative_cache=True)
    g=np.concatenate((gb,ga),axis=1);w,rho=context['w'],context['rho'];grad=np.zeros(10);gmu=np.zeros_like(w)
    for l,sl in enumerate(_SLICES):
        t=cache['t'][:,sl];q=cache['q'][:,l];beta=theta[l];z=beta*q
        wl=w[:,sl];gl=g[:,sl];dot=np.sum(gl*t,axis=1);dotw=np.sum(gl*wl,axis=1)
        grad[l]=np.sum(rho[:,l]*(-q*np.sin(z)*dotw+np.cos(z)*dot))
        A=beta*np.sinc(z/np.pi);B=np.empty(len(q));small=np.abs(z)<_CONFIG['rotation_small_angle']
        zz=z[small]**2
        B[small]=beta**3*(-1/3+zz/30-zz**2/840+zz**3/45360)
        B[~small]=(z[~small]*np.cos(z[~small])-np.sin(z[~small]))/(q[~small]**3)
        gt=rho[:,l,None]*(A[:,None]*gl+(B*dot-beta**2*np.sinc(z/np.pi)*dotw)[:,None]*t)
        gmu[:,sl]=gt-np.sum(gt*wl,axis=1)[:,None]*wl
    # Backprop through the same per-sample neighbor softmax, never a label route.
    gatt=gmu@table.m.T;att=cache['att'];center=gatt-np.sum(att*gatt,axis=1)[:,None]
    for l in range(5):
        # Use centered distances: all excluded entries have exactly zero weight.
        d=context['d'][:,:,l];d=d-np.sum(att*d,axis=1)[:,None]
        grad[5+l]=-np.sum(att*center*d)*np.exp(theta[5+l])/table.nu
    _finite(grad);return grad


def _joint_distances(b,a,tb,ta,b0,a0,tb0,ta0,*,symmetric=False,d0=None):
    if d0 is None:d0=local._distances(b0,a0,None if symmetric else tb0,None if symmetric else ta0)
    da=local._distances(b,a,None if symmetric else tb,None if symmetric else ta)
    for i in range(len(b)):
        before=np.all(b0[i]==tb0,axis=1)&np.all(a0[i]==ta0,axis=1)
        after=np.all(b[i]==tb,axis=1)&np.all(a[i]==ta,axis=1)
        if np.any(before&~after):raise FloatingPointError('ORIGINAL_EQUAL_ADAPTED_UNEQUAL')
    out=.5*d0+.5*da
    if np.any(out<.5*d0) or np.any((d0>0)&(out<=0)):raise FloatingPointError('JOINT_DISTANCE_BOUND_FAILED')
    return out


@dataclass(frozen=True)
class TransportProblem:
    original: ch.ChannelProblem
    prototypes: PrototypeTable
    train_context: Mapping
    held_context: Mapping
    def __post_init__(self):
        for name in ('train_context','held_context'):
            object.__setattr__(self,name,_freeze({k:_readonly(v) for k,v in getattr(self,name).items()}))
    @property
    def audit(self):
        result=_plain(self.original.audit);result['prototypes']=_plain(self.prototypes.audit)
        return result


def _problem(original,*,inherited=None):
    table=_make_prototypes(original.train_b,original.train_a,np.asarray(original.train_labels,dtype=int),original.classes,
        original.audit['training_physical_ids'],inherited=inherited)
    return TransportProblem(original,table,_map_context(original.train_b,original.train_a,table),
        _map_context(original.held_b,original.held_a,table))


def _forward(problem,theta,progress=None):
    started=time.perf_counter();p=problem.original;n=len(p.train_labels);c=len(p.classes)
    audit={} if progress is None else progress
    audit.update(problem.audit,head_fit_count=1,factorization_count=0,derivative_triangular_solve_count=0,
        transport_forward_evaluation_count=1)
    b,a,bc=_adapt(problem.train_context,problem.prototypes,theta,derivative_cache=True)
    hb,ha,hc=_adapt(problem.held_context,problem.prototypes,theta,derivative_cache=True)
    identity=bool(np.all(theta[:5]==0)) or problem.prototypes.nu in (None,0) or p.tau0==0
    if identity:distance,cross=p.d0,p.cross_d0
    else:
        distance=_joint_distances(b,a,b,a,p.train_b,p.train_a,p.train_b,p.train_a,symmetric=True,d0=p.d0)
        cross=_joint_distances(hb,ha,b,a,p.held_b,p.held_a,p.train_b,p.train_a,d0=p.cross_d0)
    tau,weights=ch._bandwidth_weights(distance,np.asarray(p.train_labels,dtype=int))
    if p.tau0 is not None and p.tau0>0 and tau<=0:raise FloatingPointError('POSITIVE_JOINT_BANDWIDTH_UNRESOLVED')
    target=np.eye(c)[np.asarray(p.train_labels,dtype=int)]-1/c
    alpha=np.zeros((n,c));reference=mean=np.zeros(n);refself=grand=0.;gamma=None
    kernel=np.zeros((n,n));score=np.zeros((len(p.held_labels),c));tol=128*_EPS*max(n,c)
    audit.update(bandwidth_tau=tau,original_bandwidth_tau=p.tau0,interaction_centered_trace=p.s0,
        radial_centered_trace=None,trace_scale=None,normal_equation_residual=0.,trace_relative_error=0.,
        numerical_tolerance=tol,identity_forward=identity)
    saved={}
    if c>1 and p.s0>0:
        raw=local._radial_minus_one(distance,tau);radial=local._radial(distance,tau)
        sr=float(-2*raw[np.triu_indices(n,1)].sum()/n)
        if sr<=0:raise FloatingPointError('NONPOSITIVE_RADIAL_TRACE')
        gamma=p.s0/sr;center,reference,refself,mean,grand=interaction._center_kernel(raw)
        kernel=gamma*center;matrix=kernel+np.eye(n);audit['factorization_count']=1
        chol=np.linalg.cholesky(matrix)
        alpha=np.ascontiguousarray(solve_triangular(chol.T,solve_triangular(chol,target,lower=True),lower=False))
        crosscenter=ch._cross_center(local._radial_minus_one(cross,tau),reference,refself,mean,grand)
        crosskernel=gamma*crosscenter
        score=np.stack([(gamma*crosscenter[i])@alpha for i in range(len(p.held_labels))]) if len(p.held_labels) else score
        residual=_norm(matrix@alpha-target)/((1+p.s0)*_norm(alpha)+_norm(target))
        terr=abs(float(np.trace(kernel))-p.s0)/p.s0
        if residual>tol or terr>tol:raise FloatingPointError('TRANSPORT_HEAD_RESIDUAL_EXCEEDED')
        audit.update(normal_equation_residual=residual,trace_relative_error=terr,radial_centered_trace=sr,trace_scale=gamma)
        saved.update(chol=chol,crosskernel=crosskernel,radial=radial,sr=sr)
    fit=kernel@alpha;err=fit-target
    audit.update(head_training_loss_data=float(.5*np.sum(err*err)),head_training_loss_ridge=float(.5*np.sum(alpha*fit)),
        sample_weight=1.,ridge_coefficient=1.,status='CLOSED_FORM_SOLVED' if c>1 and p.s0>0 else 'EXACT_ZERO_CLASSIFIER',
        forward_seconds=time.perf_counter()-started)
    audit['head_training_loss_total']=audit['head_training_loss_data']+audit['head_training_loss_ridge']
    _finite(score,alpha)
    return dict(problem=problem,theta=theta.copy(),score=score,b=b,a=a,hb=hb,ha=ha,bc=bc,hc=hc,
        distance=distance,cross=cross,weights=weights,kernel=kernel,alpha=alpha,tau=tau,gamma=gamma,
        parts=(alpha,reference,refself,mean,grand,tau,gamma),audit=audit,**saved)


def _backward(cache,gscore,progress):
    start=time.perf_counter();p=cache['problem'];o=p.original
    if cache['gamma'] is None or cache['tau']==0 or p.prototypes.nu in (None,0) or not len(gscore):
        progress['adjoint_seconds']=time.perf_counter()-start;return np.zeros(10)
    alpha=cache['alpha'];chol=cache['chol'];crosskernel=cache['crosskernel'];kernel=cache['kernel']
    n=len(alpha);gamma=cache['gamma'];sr=cache['sr'];tau=cache['tau']
    barcross=gscore@alpha.T;progress['derivative_triangular_solve_count']+=1
    low=solve_triangular(chol,crosskernel.T@gscore,lower=True);progress['derivative_triangular_solve_count']+=1
    z=solve_triangular(chol.T,low,lower=False);bark=-z@alpha.T;bark=.5*(bark+bark.T)
    barraw=gamma*(bark-bark.mean(axis=0)[None,:]-bark.mean(axis=1)[:,None]+bark.mean())
    crossrow=barcross-barcross.mean(axis=1)[:,None];barraw-=gamma*np.broadcast_to(crossrow.sum(axis=0)/n,(n,n))
    barsr=-(float(np.sum(bark*kernel))+float(np.sum(barcross*crosskernel)))/sr
    barraw[~np.eye(n,dtype=bool)]-=barsr/n
    rd=barraw*cache['radial'];crossrad=local._radial(cache['cross'],tau);rc=gamma*crossrow*crossrad
    dd=np.zeros_like(rd);dc=np.zeros_like(rc);active=cache['radial']>0;activec=crossrad>0
    dd[active]=-rd[active]/tau;dc[activec]=-rc[activec]/tau
    bt=(np.sum(rd[active]*(cache['distance'][active]/tau))+np.sum(rc[activec]*(cache['cross'][activec]/tau)))/tau
    dd+=bt*cache['weights']
    gb,ga,_,_=ch._distance_vjp(cache['b'],cache['a'],cache['b'],cache['a'],.5*dd,symmetric=True)
    ghb,gha,gtb,gta=ch._distance_vjp(cache['hb'],cache['ha'],cache['b'],cache['a'],.5*dc)
    grad=_rotation_vjp(p.train_context,p.prototypes,cache['theta'],gb+gtb,ga+gta,cache['bc'])
    grad+=_rotation_vjp(p.held_context,p.prototypes,cache['theta'],ghb,gha,cache['hc'])
    progress['adjoint_seconds']=time.perf_counter()-start;return grad


@dataclass(frozen=True)
class PrototypeTransportTraining:
    base: ch.ChannelTraining
    problems: tuple
    full_problem: TransportProblem
    audit: Mapping
    def __post_init__(self):object.__setattr__(self,'audit',_freeze(self.audit))
    def __getattr__(self,name):return getattr(self.base,name)
    def audit_dict(self):return ch._safe(_plain(self.audit))


def prepare_prototype_transport_training(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,
        classes,old_classes,inherited=None,context=None,log_callback=None):
    start=time.perf_counter()
    base=ch.prepare_channel_training(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local,
        support_labels=support_labels,support_ids=support_ids,classes=classes,old_classes=old_classes,inherited=inherited,context=context)
    audit=base.audit_dict();audit.pop('channel_preparation_count',None)
    audit.update(transport_preparation_count=1,prototype_construction_count=0,prototype_distance_evaluation_count=0,
        prepared_distance_evaluation_count=2*len(base.problems),completed_stages=[])
    problems=[]
    try:
        for oldproblem in base.problems:
            p=_problem(oldproblem);problems.append(p);audit['completed_stages'].append(p.audit)
            audit['prototype_construction_count']+=1;audit['prototype_distance_evaluation_count']+=3
            event=dict(context or {});event.update(p.audit);ch._emit(log_callback,'TRANSPORT_INNER_PREPARED',event)
        original=ch._make_problem(base.background,base.auxiliary,np.asarray(base.labels,dtype=int),base.ids,base.classes,np.ones(len(base.ids),dtype=bool),None)
        audit['prepared_distance_evaluation_count']+=2
        full=_problem(original,inherited=inherited)
        audit['prototype_construction_count']+=1;audit['prototype_distance_evaluation_count']+=3
        audit.update(inner_folds=[p.audit for p in problems],full_prototypes=_plain(full.prototypes.audit),
            prepare_seconds=time.perf_counter()-start,preparation_seconds=time.perf_counter()-start)
        extra=sum(p.prototypes.p.nbytes+p.prototypes.m.nbytes+8+
            sum(v.nbytes for ctx in (p.train_context,p.held_context) for v in ctx.values()) for p in problems+[full])
        extra+=sum(getattr(original,name).nbytes for name in ('train_b','train_a','held_b','held_a','train_labels','held_labels','d0','cross_d0'))
        audit['prepared_numeric_state_bytes']+=int(extra)
        audit['transient_distance_bytes']+=original.d0.nbytes+original.cross_d0.nbytes
        return PrototypeTransportTraining(base,tuple(problems),full,audit)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc));raise NumericalFailure(str(exc),audit) from exc


@dataclass(frozen=True)
class ObjectiveCache:
    prepared: object
    theta: np.ndarray
    folds: tuple
    def __post_init__(self):object.__setattr__(self,'theta',_readonly(self.theta))


def evaluate_transport_objective(prepared,theta,anchor,*,gradient=True,forward_cache=None):
    theta=_parameter(theta);anchor=_parameter(anchor);start=time.perf_counter();n=len(prepared.ids);c=len(prepared.classes)
    reuse=forward_cache is not None
    if reuse and (forward_cache.prepared is not prepared or not np.array_equal(forward_cache.theta,theta)):
        raise ValueError('Objective cache preparation or theta mismatch')
    cache=[];progress={};audit=dict(inner_objective_evaluation_count=int(not reuse),inner_head_fit_count=0,
        inner_factorization_count=0,derivative_triangular_solve_count=0,backward_evaluation_count=int(gradient),
        transport_forward_evaluation_count=0,
        forward_cache_reused=reuse,inner_folds=[])
    try:
        for i,p in enumerate(prepared.problems):
            progress={}
            result=forward_cache.folds[i] if reuse else _forward(p,theta,progress)
            cache.append(result)
            if not reuse:
                audit['inner_head_fit_count']+=1;audit['inner_factorization_count']+=result['audit']['factorization_count']
                audit['transport_forward_evaluation_count']+=1
            audit['inner_folds'].append(dict(result['audit'],derivative_triangular_solve_count=0,
                head_fit_count=0 if reuse else 1,factorization_count=0 if reuse else result['audit']['factorization_count'],
                transport_forward_evaluation_count=0 if reuse else 1))
            progress={}
        totals=np.zeros(c);counts=np.zeros(c,dtype=int);grads=[]
        for index,result in enumerate(cache):
            labels=np.asarray(result['problem'].original.held_labels,dtype=int)
            _,gs,margin=ch._margin_loss(result['score'],labels)
            sampleloss=.5*np.maximum(0.,1-margin)**2 if c>1 else np.zeros(len(labels))
            sums=np.bincount(labels,weights=sampleloss,minlength=c);nums=np.bincount(labels,minlength=c)
            totals+=sums;counts+=nums
            audit['inner_folds'][index].update(class_margin_loss_sums=sums.tolist(),class_physical_counts=nums.tolist())
            grads.append(gs)
        if np.any(counts==0):raise ValueError('Objective requires physical inner-held coverage for every class')
        losses=totals/counts;data=_norm(losses)/np.sqrt(c);penalty=float(.5*np.sum((theta-anchor)**2)/n)
        weights=np.zeros(c) if data==0 else losses/(c*data*counts)
        grad=np.zeros(10)
        if gradient:
            for i,result in enumerate(cache):
                progress=audit['inner_folds'][i]
                labels=np.asarray(result['problem'].original.held_labels,dtype=int)
                grad+=_backward(result,grads[i]*weights[labels,None],progress)
                progress={}
            grad+=(theta-anchor)/n
        audit['derivative_triangular_solve_count']=sum(f['derivative_triangular_solve_count'] for f in audit['inner_folds'])
        for f,result in zip(audit['inner_folds'],cache):
            labels=np.asarray(result['problem'].original.held_labels,dtype=int)
            loss,_,margin=ch._margin_loss(result['score'],labels)
            f.update(held_margin_loss_sum=loss,held_training_correct_count=int(np.sum(result['score'].argmax(axis=1)==labels)),
                held_training_margin_mean=float(margin.mean()) if len(margin) else None)
        audit.update(loss_data=float(data),loss_proximal=penalty,loss_total=float(data+penalty),
            class_margin_loss_means=losses.tolist(),class_held_counts=counts.tolist(),classes=list(prepared.classes),
            class_loss_values=losses.tolist(),
            gradient=grad.tolist() if gradient else None,loss_scope='CLASS_RMS_INNER_HELD_MARGIN_PLUS_PROXIMAL',
            objective_seconds=time.perf_counter()-start)
        _finite(grad,data,penalty)
        return float(data+penalty),grad,audit,ObjectiveCache(prepared,theta,tuple(cache))
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if progress and 'head_fit_count' in progress and len(cache)<len(prepared.problems):
            audit['inner_head_fit_count']+=progress.get('head_fit_count',0);audit['inner_factorization_count']+=progress.get('factorization_count',0)
            audit['transport_forward_evaluation_count']+=progress.get('transport_forward_evaluation_count',0)
        audit['derivative_triangular_solve_count']=sum(f['derivative_triangular_solve_count'] for f in audit['inner_folds'])
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),current_kernel_progress=deepcopy(progress),theta=theta.tolist(),
            completed_inner_folds=deepcopy(audit['inner_folds']))
        raise NumericalFailure(str(exc),audit) from exc


@dataclass(frozen=True)
class PrototypeTransportState:
    theta: np.ndarray
    base_state: local.BranchLocalRidgeState
    prototypes: PrototypeTable
    original_background: np.ndarray
    original_auxiliary: np.ndarray
    raw: Mapping
    labels: np.ndarray
    ids: tuple
    audit: Mapping
    def __post_init__(self):
        for key in ('theta','original_background','original_auxiliary','labels'):object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'raw',_freeze({k:_readonly(v) for k,v in self.raw.items()}));object.__setattr__(self,'audit',_freeze(self.audit))
    @property
    def u(self):return self.theta
    @property
    def classes(self):return self.base_state.classes
    def audit_dict(self):return ch._safe(_plain(self.audit))
    def score(self,*,z_id,fft,t_emb,f_emb,pa_local):
        b0,a0=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True);s=self.base_state
        if self.audit['identity_forward']:
            return local._score_local(b0,a0,self.original_background,self.original_auxiliary,s.alpha,s.reference_kernel,s.reference_self,s.center_mean,s.center_grand,s.bandwidth_tau,s.trace_scale)
        out=np.empty((len(b0),len(self.classes)))
        for i in range(len(b0)):
            ctx=_map_context(b0[i:i+1],a0[i:i+1],self.prototypes);b,a,_=_adapt(ctx,self.prototypes,self.theta)
            d=_joint_distances(b,a,s.support_background,s.support_auxiliary,b0[i:i+1],a0[i:i+1],self.original_background,self.original_auxiliary)
            center=ch._cross_center(local._radial_minus_one(d,s.bandwidth_tau),s.reference_kernel,s.reference_self,s.center_mean,s.center_grand)
            out[i]=(s.trace_scale*center[0])@s.alpha
        _finite(out);return out
    def predict(self,**features):return np.asarray(self.classes)[np.argmax(self.score(**features),axis=1)]


_COUNTERS=('inner_objective_evaluation_count','inner_head_fit_count','inner_factorization_count','derivative_triangular_solve_count','backward_evaluation_count','transport_forward_evaluation_count')


def _add_cost(audit,objective):
    for k in _COUNTERS:audit[k]+=objective.get(k,0)


def _armijo_tolerance(before,trial,rhs):return float(128*_EPS*max(1.,abs(before),abs(trial),abs(rhs)))


def fit_prototype_transport_local_ridge(prepared,*,mode='B',baseline_state=None,log_callback=None):
    if mode not in ('B','C_seq','C_reset'):raise ValueError('Unknown transport mode')
    if (mode=='B')!=(prepared.inherited is None):raise ValueError('B/C inherited state mismatch')
    if prepared.inherited is not None and set(prepared.classes)==set(prepared.old_classes):return prepared.inherited
    start=time.perf_counter();binding={};baseline_state=ch._canonical_baseline(prepared,baseline_state,binding)
    anchor=prepared.inherited.theta.copy() if mode=='C_seq' else np.zeros(10);theta=anchor.copy()
    audit=dict(mode=mode,classes=list(prepared.classes),old_classes=list(prepared.old_classes),training_physical_ids=list(prepared.ids),
        train_k=prepared.audit['train_k'],transport_stage_count=1,preparation=prepared.audit_dict(),
        no_information=prepared.audit['no_information'],optimizer_steps=0,optimizer_iterations=0,
        accepted_trial_count=0,rejected_trial_count=0,trial_count=0,trial_attempt_count=0,nonzero_projected_update_count=0,
        final_head_fit_count=0,final_factorization_count=0,steps=[],trials=[],gradients=[],completed_stages=[],
        anchor=anchor.tolist(),u_anchor=anchor.tolist(),trainable_parameter_count=10,effective_parameter_count=9,
        optimizer_state_bytes=0,optimizer_state_reset=True,initial_objective=None,source_validation=None,source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        **{k:0 for k in _COUNTERS},**binding)
    current=None;finalprogress={}
    def event(name,record):
        payload=dict(_plain(prepared.audit));payload.update(record,mode=mode);ch._emit(log_callback,name,payload)
    try:
        if audit['no_information']:
            audit['stop_reason']=prepared.audit['no_information_reason'];audit['no_update_reason']=audit['stop_reason'];audit['final_objective']=None
        else:
            loss,_,obj,current=evaluate_transport_objective(prepared,theta,anchor,gradient=False);_add_cost(audit,obj)
            audit['initial_objective']=deepcopy(obj);event('TRANSPORT_INITIAL',dict(u=theta.tolist(),anchor=anchor.tolist(),objective=obj))
            audit['stop_reason']='MAX_ITERATIONS'
            for iteration in range(1,_CONFIG['max_iterations']+1):
                audit['optimizer_iterations']+=1;begin=time.perf_counter()
                loss,g,obj,current=evaluate_transport_objective(prepared,theta,anchor,gradient=True,forward_cache=current);_add_cost(audit,obj)
                gn=_norm(g);gr=dict(iteration=iteration,u=theta.tolist(),gradient=g.tolist(),gradient_norm=gn,objective=deepcopy(obj))
                audit['gradients'].append(gr);event('TRANSPORT_GRADIENT',gr)
                if gn==0:audit['stop_reason']='ZERO_GRADIENT';break
                accepted=False
                for trial in range(1,_CONFIG['max_trials']+1):
                    step_size=_CONFIG['initial_step_size']*_CONFIG['backtrack_factor']**(trial-1)
                    proposed=project_transport(theta-step_size*g/gn);delta=proposed-theta
                    if np.all(delta==0):audit['stop_reason']='ZERO_PROJECTED_STEP';break
                    audit['trial_attempt_count']+=1
                    audit['current_trial']=dict(iteration=iteration,trial=trial,step_size=step_size,u_pre=theta.tolist(),u_trial=proposed.tolist())
                    trial_loss,_,trial_obj,trial_cache=evaluate_transport_objective(prepared,proposed,anchor,gradient=False);_add_cost(audit,trial_obj)
                    rhs=loss+_CONFIG['armijo_coefficient']*float(g@delta);tol=_armijo_tolerance(loss,trial_loss,rhs)
                    accepted=bool(trial_loss<=rhs+tol)
                    tr=dict(iteration=iteration,trial=trial,step_size=step_size,u_pre=theta.tolist(),u_trial=proposed.tolist(),anchor=anchor.tolist(),
                        loss_before=loss,loss_after=trial_loss,armijo_rhs=rhs,armijo_tolerance=tol,accepted=accepted,
                        gradient=g.tolist(),gradient_norm=gn,update_norm=_norm(delta),objective=deepcopy(trial_obj))
                    audit['trials'].append(tr);audit['trial_count']+=1
                    audit['current_trial']=None
                    audit['accepted_trial_count' if accepted else 'rejected_trial_count']+=1;event('TRANSPORT_TRIAL',tr)
                    if accepted:
                        before=theta.copy();theta=proposed;current=trial_cache;loss=trial_loss
                        audit['optimizer_steps']+=1;audit['nonzero_projected_update_count']+=1
                        rec=dict(step=audit['optimizer_steps'],iteration=iteration,trial=trial,u_pre=before.tolist(),u_post=theta.tolist(),anchor=anchor.tolist(),
                            gradient=g.tolist(),gradient_norm=gn,step_size=step_size,learning_rate=step_size,
                            loss_before=tr['loss_before'],loss_after=loss,update_norm=_norm(delta),
                            projection_zero_sum_residual=float(theta[5:].sum()),step_seconds=time.perf_counter()-begin,objective=deepcopy(trial_obj))
                        audit['steps'].append(rec);event('TRANSPORT_STEP',rec);break
                if not accepted:
                    if audit['stop_reason']!='ZERO_PROJECTED_STEP':audit['stop_reason']='ARMIJO_BUDGET_EXHAUSTED'
                    break
            # Reuse the accepted cache; no final re-fit or new objective evaluation.
            _,_,last,_=evaluate_transport_objective(prepared,theta,anchor,gradient=False,forward_cache=current)
            audit['final_objective']=last
        identity=bool(np.all(theta[:5]==0)) or prepared.full_problem.prototypes.nu in (None,0) or prepared.full_problem.original.tau0==0
        if identity and baseline_state is not None:base=baseline_state;finalfit=base.audit_dict()['final_fit']
        else:
            result=_forward(prepared.full_problem,theta,finalprogress)
            audit['final_head_fit_count']=1;audit['final_factorization_count']=result['audit']['factorization_count'];finalfit=result['audit']
            audit['transport_forward_evaluation_count']+=1
            alpha,ref,rs,mean,grand,tau,gamma=result['parts']
            base=local.BranchLocalRidgeState(result['b'],result['a'],alpha,ref,rs,mean,grand,tau,gamma,prepared.classes,finalfit)
        headbytes=sum(getattr(base,k).nbytes for k in ('support_background','support_auxiliary','alpha','reference_kernel','center_mean'))+16+8*(base.bandwidth_tau is not None)+8*(base.trace_scale is not None)
        lineagebytes=prepared.background.nbytes+prepared.auxiliary.nbytes+prepared.labels.nbytes+sum(v.nbytes for v in prepared.raw.values())
        protobytes=prepared.full_problem.prototypes.p.nbytes+prepared.full_problem.prototypes.m.nbytes+8
        audit.update(theta=theta.tolist(),u=theta.tolist(),u_changed_from_anchor=bool(np.any(theta!=anchor)),u_update_norm=_norm(theta-anchor),
            identity_forward=identity or base.trace_scale is None,final_fit=finalfit,adapter_state_bytes=theta.nbytes,
            prototype_state_bytes=protobytes,head_state_bytes=int(headbytes),lineage_state_bytes=int(lineagebytes),
            persistent_state_bytes=int(headbytes+lineagebytes+protobytes+theta.nbytes),
            state_byte_scope='numeric_head_original_adapted_support_raw_labels_theta_prototypes_excludes_Python_metadata',
            factorization_count=audit['inner_factorization_count']+audit['final_factorization_count'],
            status='TRANSPORT_STAGE_COMPLETE',fit_seconds=time.perf_counter()-start,config=deepcopy(_CONFIG))
        event('TRANSPORT_FIT',audit)
        return PrototypeTransportState(theta,base,prepared.full_problem.prototypes,prepared.background,prepared.auxiliary,prepared.raw,prepared.labels,prepared.ids,audit)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if isinstance(exc,NumericalFailure):audit['failed_stage']=exc.audit_dict();_add_cost(audit,audit['failed_stage'])
        if finalprogress and not audit['final_head_fit_count']:
            audit['failed_final_kernel_progress']=finalprogress;audit['final_head_fit_count']=finalprogress.get('head_fit_count',0)
            audit['final_factorization_count']=finalprogress.get('factorization_count',0)
            audit['transport_forward_evaluation_count']+=finalprogress.get('transport_forward_evaluation_count',0)
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),theta=theta.tolist(),u=theta.tolist(),
            fit_seconds=time.perf_counter()-start,factorization_count=audit['inner_factorization_count']+audit['final_factorization_count'])
        raise NumericalFailure(str(exc),audit) from exc
