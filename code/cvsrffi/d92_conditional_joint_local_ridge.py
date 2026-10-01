"""Support-only ConditionalJoint, an independent structural implementation.

B uses the existing analytic affine head. C constrains every residual column
at old support through a conditional affine kernel. No existing release is
patched. Entry/scorer registration and deployment are separate work.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from . import d92_affine_joint_local_ridge as affine
from . import d92_conditional_affine_kernel as conditional
from . import d92_function_coordinate_residual8_local_ridge as fcr
from . import d92_joint_channel_local_ridge as ch
from . import d92_branch_local_ridge as local
from . import d92_branch_interaction as interaction
from .d92_branch_ridge import _freeze, _plain

_EPS = float(np.finfo(np.float64).eps)
_NAMES = fcr._NAMES
V0 = fcr.V0
NumericalFailure = ch.NumericalFailure
latent_coordinates = fcr.latent_coordinates
reconstruct_U = fcr.reconstruct_U
initial_parameters = fcr.initial_parameters
_unique_bytes = affine._unique_bytes
_norm = ch._norm
_finite = ch._finite
_scalar = affine._scalar
_SUFFIXES = ('factorization_count', 'triangular_solve_count', 'triangular_rhs_count',
             'triangular_rhs_element_count', 'triangular_dense_work_unit_count')
_PREFIXES = ('projection', 'residual', 'projection_adjoint', 'residual_adjoint')
_KERNEL_COUNTERS = tuple(p+'_'+s for p in _PREFIXES for s in _SUFFIXES)+('spectral_diagnostic_count', 'completed_factorization_count')
PREPARATION_COUNTERS = affine.PREPARATION_COUNTERS
STAGE_COUNTERS = tuple(dict.fromkeys(affine.STAGE_COUNTERS+_KERNEL_COUNTERS))
AUDIT_COUNTERS = tuple(dict.fromkeys(PREPARATION_COUNTERS+STAGE_COUNTERS))
COUNTERS = STAGE_COUNTERS
_GEOMETRY_COUNTERS = ('raw_distance_evaluation_count','raw_distance_pair_count',
    'reference_distance_evaluation_count','reference_distance_pair_count',
    'kernel_evaluation_count','kernel_pair_count','adapter_physical_evaluation_count')

FROZEN_CONFIG = dict(
    schema='d92_conditional_joint_local_ridge_v1',method='D92-ConditionalJointLocalRidge-v1',
    candidate_status='STRUCTURE_IMPLEMENTATION_NOT_PREREGISTERED_OR_DEPLOYED',
    arms=['local_ridge','conditional_joint_seq'],base_algorithm=deepcopy(local.FROZEN_CONFIG),
    rank=8,input_dim=736,block_dimensions=[160,96,160,160,160],kappa=.25,
    fixed_dictionary='first8_orthonormal_DCT_rows',learn_V=False,
    adapter='fixed_DCT_exact_GELU_function_coordinate_tangent_norm_preserving_residual',
    coordinate='U=anchor_U+Z@W.T;H/sqrtN=P*Sigma*R.T;W=R_retained/Sigma_retained',
    rank_rule='sigma/sigma_max>sqrt(128*eps64*max(N,8))',
    initialization_B='U_zero',initialization_C_seq='exact_actual_current_B_U',
    bandwidth='fixed_original_old_inner_train_R0',trace_scale='fixed_original_old_R0_gamma',
    joint_original_distance_weight=.5,joint_adapted_distance_weight=.5,implicit_feature_dim=123616,
    prior_C='same_run_row_scope_physical_fold_actual_B_function_padded_new_columns_zero',
    target='row_class_centered_onehot_minus_1_over_C',ridge_coefficient=1.,sample_weight=1.,
    B_head='analytic_affine_combined_C_plus_1_RHS',
    C_head='raw_PSD_Gaussian_conditional_affine_old_residual_zero_all_registered_columns',
    C_free_constant='analytic_inside_conditional_kernel_no_added_intercept',
    objective='RMS_across_classes_of_cross_fold_mean_CE_only',temperature=1.,
    proximal_coefficient=0.,coordinate_ball_radius=.5,keep=False,guard=False,
    optimizer='normalized_gradient_projected_first_Armijo_RMSCE_nonincrease',
    max_iterations=4,max_trials=12,initial_step_size=.125,backtrack_factor=.5,
    armijo_coefficient=1e-4,comparison_tolerance_multiplier=128,
    inner_folds='per_class_physical_id_sort_position_mod_min_K_3',
    zero_bandwidth='original_complete_interaction_exact_equivalence_PSD_zero_derivative',
    old_duplicates='exact_complete_kernel_input_equivalence_no_approximate_threshold',
    zero_kernel_C='constraint_forces_g_and_intercept_zero_return_actual_B_prior',
    k1='no_adapter_update_full_support_closed_head',rank0='anchor_U_full_support_closed_head',
    n0='exact_reuse_actual_B_object',tie_break='physical_class_id_lexicographic',
    query_decision_policy='per_sample_all_registered_classes',phase1_frozen=True,
    encoder_backward=False,source_inputs=False,query_fit=False,parameter_search=False,dtype='float64',
    vector_logging='lossless_state_callback_or_records')


def _readonly(value):
    a=np.asarray(value)
    if a.dtype.kind not in 'fibu':raise ValueError('CONDITIONAL_JOINT requires finite numeric state')
    dtype=np.bool_ if a.dtype.kind=='b' else np.int64 if a.dtype.kind in 'iu' else np.float64
    shape=a.shape;a=np.ascontiguousarray(a,dtype=dtype).reshape(shape);_finite(a)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(shape)


class _Recorder:
    def __init__(self, callback):
        self.callback=callback;self.records={};self.keys=set()
    def save(self,key,**arrays):
        if key in self.keys:raise ValueError('Duplicate CONDITIONAL_JOINT record '+key)
        self.keys.add(key);values={name:_readonly(value) for name,value in arrays.items()}
        shapes={name:dict(shape=list(value.shape),dtype=str(value.dtype),nbytes=int(value.nbytes)) for name,value in values.items()}
        if self.callback is None:
            self.records[key]=_freeze(values);ref=dict(storage='in_memory')
        else:
            ref=ch._safe(self.callback(key,values))
            if not isinstance(ref,dict):raise ValueError('state_callback must return mapping')
        return dict(ref,key=key,arrays=shapes)


def _add(target,source,keys):
    for key in keys:target[key]=target.get(key,0)+source.get(key,0)


@dataclass(frozen=True)
class ConditionalJointProblem:
    train_context: Mapping
    held_context: Mapping
    train_labels: np.ndarray
    held_labels: np.ndarray
    classes: tuple
    q: np.ndarray
    old_indices: np.ndarray
    new_indices: np.ndarray
    d0: np.ndarray
    cross_d0: np.ndarray
    tau: float | None
    gamma: float | None
    s0: float
    M_train: np.ndarray
    M_held: np.ndarray
    mode: str
    audit: Mapping
    def __post_init__(self):
        for key in ('train_context','held_context'):
            object.__setattr__(self,key,_freeze({k:_readonly(v) for k,v in getattr(self,key).items()}))
        for key in ('train_labels','held_labels','q','old_indices','new_indices','d0','cross_d0','M_train','M_held'):
            object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'audit',_freeze(self.audit))


def _make_problem(b,a,y,ids,classes,old_classes,keep,H,*,mode,fold=None,prior=None,nuisance=None,progress=None,prior_ref=None):
    keep=np.asarray(keep,dtype=bool);tb,ta=b[keep],a[keep];hb,ha=b[~keep],a[~keep]
    ty,hy=y[keep],y[~keep];old=np.array([classes[int(v)] in old_classes for v in ty])
    oi=np.flatnonzero(old);ni=np.flatnonzero(~old)
    if not len(oi):raise ValueError('Old inner-train reference required')
    nu=nuisance
    if nu is None:
        oy=np.array([old_classes.index(classes[int(v)]) for v in ty[old]],dtype=np.int64)
        nu=affine._nuisance(tb[old],ta[old],oy,len(old_classes))
        if progress is not None:
            _add(progress,dict(prepared_distance_evaluation_count=1,original_distance_pair_count=len(oi)*(len(oi)-1)//2,
                reference_distance_evaluation_count=1,reference_distance_pair_count=len(oi)*(len(oi)-1)//2),PREPARATION_COUNTERS)
    # Reuse the actually computed all-old original geometry; do not bill it twice.
    reuse=mode=='B' and 'old_distance' in nu
    d0=nu['old_distance'] if reuse else local._distances(tb,ta)
    cross=local._distances(hb,ha,tb,ta)
    if progress is not None:
        progress['prepared_distance_evaluation_count']+=int(not reuse)+int(len(hy)>0)
        progress['original_distance_pair_count']+=(0 if reuse else len(ty)*(len(ty)-1)//2)+len(hy)*len(ty)
    mt=np.zeros((len(ty),len(classes)));mh=np.zeros((len(hy),len(classes)))
    if prior is not None:
        pp,pu,pf,pc=prior
        sp={};pt=_score_residual(pp,pf,pu,tb,ta,sp,H[keep]);ph=_score_residual(pp,pf,pu,hb,ha,sp,H[~keep])
        for j,name in enumerate(pc):mt[:,classes.index(name)]=pt[:,j];mh[:,classes.index(name)]=ph[:,j]
        if progress is not None:
            _add(progress,sp,_GEOMETRY_COUNTERS+('dictionary_physical_evaluation_count',))
            progress['prior_intercept_addition_count']+=sp.get('intercept_addition_count',0)
            progress['prior_score_evaluation_count']+=1+int(len(hy)>0);progress['prior_score_physical_count']+=len(y)
            progress['prior_score_seconds']=progress.get('prior_score_seconds',0.)+sp.get('score_seconds',0.)
    tids=tuple(pid for i,pid in enumerate(ids) if keep[i]);hids=tuple(pid for i,pid in enumerate(ids) if not keep[i])
    audit=dict(schema=FROZEN_CONFIG['schema'],method=FROZEN_CONFIG['method'],mode=mode,inner_fold=fold,classes=list(classes),old_classes=list(old_classes),
        training_physical_ids=list(tids),held_physical_ids=list(hids),old_reference_physical_ids=[tids[i] for i in oi],
        train_physical_count=len(ty),held_physical_count=len(hy),old_reference_count=len(oi),
        bandwidth_tau=nu['tau'],trace_scale=nu['gamma'],interaction_centered_trace=nu['s0'],
        prior_ref=prior_ref,prior_source='ZERO' if prior is None else 'FROZEN_CURRENT_ACTUAL_B',
        all_head_statistics_from_old_inner_train_only=True)
    return ConditionalJointProblem(fcr._context(tb,ta,H[keep]),fcr._context(hb,ha,H[~keep]),ty,hy,tuple(classes),
        old.astype(float)/len(oi),oi,ni,d0,cross,nu['tau'],nu['gamma'],nu['s0'],mt,mh,mode,audit)


def _representatives(p,b,a,radial,crossrad):
    """Compress only exact old kernel-input duplicates, never by labels/rank."""
    oi=p.old_indices;representatives=[];groups=[]
    original=p.train_context['original']
    for index in oi:
        group=None
        for j,rep in enumerate(representatives):
            # Complete interaction includes the original linear b/a blocks.
            # Exact input identity proves equivalence for every future cross
            # point, unlike merely matching finite Gram rows or rounded d=0.
            equal=np.array_equal(original[index],original[rep]) and (p.tau==0 or (
                np.array_equal(b[index],b[rep]) and np.array_equal(a[index],a[rep])))
            if equal:
                if not (np.array_equal(radial[index],radial[rep]) and np.array_equal(crossrad[:,index],crossrad[:,rep])):
                    raise FloatingPointError('CONDITIONAL_JOINT_INCONSISTENT_EXACT_KERNEL_GROUP')
                group=j;break
        if group is None:group=len(representatives);representatives.append(int(index))
        groups.append(group)
    return np.asarray(representatives,dtype=np.int64),np.asarray(groups,dtype=np.int64)


def _forward_impl(p,U,progress=None):
    start=time.perf_counter();audit={} if progress is None else progress
    audit.update({key:0 for key in STAGE_COUNTERS});audit.update(_plain(p.audit),head_fit_count=1,factorization_count=0,
        head_triangular_solve_count=0,intercept_fit_count=1,ajlr_forward_evaluation_count=1)
    n=len(p.train_labels);h=len(p.held_labels);c=len(p.classes)
    b,a,bc=fcr._adapt(p.train_context,U,derivative_cache=True);hb,ha,hc=fcr._adapt(p.held_context,U,derivative_cache=True)
    audit['adapter_physical_evaluation_count']=n+h
    identity=not np.any(U) or p.tau==0
    if identity:distance,cross=p.d0,p.cross_d0
    else:
        distance=.5*p.d0+.5*local._distances(b,a);cross=.5*p.cross_d0+.5*local._distances(hb,ha,b,a)
        m=len(p.old_indices)
        audit.update(raw_distance_evaluation_count=1+int(h>0),raw_distance_pair_count=n*(n-1)//2+h*n,
            reference_distance_evaluation_count=1+int(h>0),reference_distance_pair_count=m*(m-1)//2+m*(n-m)+h*m)
    Y=np.eye(c)[np.asarray(p.train_labels,dtype=int)]-1/c;target=Y-p.M_train
    radial=np.zeros((n,n));crossrad=np.zeros((h,n));numeric={};ks=None
    if p.gamma is not None:
        radial=local._radial(distance,p.tau);crossrad=local._radial(cross,p.tau)
        audit.update(kernel_evaluation_count=1+int(h>0),kernel_pair_count=n*n+h*n)
    if p.mode=='B':
        K=np.zeros((n,n));L=np.zeros((h,n));ref=mean=np.zeros(n);rs=grand=0.
        chol=np.empty((0,0));z=np.ones(n);s=float(n);rhs=np.column_stack((target,np.ones(n)))
        intercept=target.mean(axis=0)
        if p.gamma is None and not np.any(p.M_train) and np.all(np.bincount(p.train_labels,minlength=c)==n//c):intercept=np.zeros(c)
        alpha=target-intercept
        if p.gamma is not None:
            center,ref,rs,mean,grand=affine._center_train(local._radial_minus_one(distance,p.tau),p.q)
            K=p.gamma*center;L=p.gamma*affine._center_cross(local._radial_minus_one(cross,p.tau),p.q,ref,rs,mean,grand)
            audit['factorization_count']=audit['residual_factorization_count']=1
            alpha,intercept,z,s,chol,rhs=affine._solve_affine_head(K,target)
            audit['completed_factorization_count']=1
            for suffix,value in zip(_SUFFIXES[1:],(2,2*(c+1),2*n*(c+1),2*n*n*(c+1))):audit['residual_'+suffix]=value
        train=p.M_train+K@alpha+intercept;scores=np.stack([p.M_held[i]+L[i]@alpha+intercept for i in range(h)]) if h else np.empty((0,c))
        numeric=dict(K=K,L=L,alpha=alpha,intercept=intercept,schur_z=z,schur_s=_scalar(s),chol=chol,combined_rhs=rhs,
            reference_kernel=ref,reference_self=_scalar(rs),center_mean=mean,center_grand=_scalar(grand))
        normal=_norm((K+np.eye(n))@alpha+intercept-target)
        denom=(1+float(np.trace(K)))*_norm(alpha)+np.sqrt(n)*_norm(intercept)+_norm(target)
        residual=normal/denom if denom else 0.;tol=128*_EPS*max(n,c)
        if residual>tol or _norm(alpha.sum(axis=0))>tol*max(1.,np.sqrt(n)*_norm(alpha)):
            raise FloatingPointError('CONDITIONAL_JOINT_B_AFFINE_RESIDUAL')
        audit.update(normal_equation_residual=residual,normal_equation_absolute_residual=normal,
            sample_sum_alpha_norm=_norm(alpha.sum(axis=0)),schur_s=s,intercept_norm=_norm(intercept),actual_kernel_trace=float(np.trace(K)),
            head_training_loss_data=float(.5*np.sum((train-Y)**2)),head_training_loss_ridge=float(.5*np.sum(alpha*(K@alpha))))
        representatives=p.old_indices.copy();groups=np.arange(len(representatives),dtype=np.int64)
    else:
        if not len(p.new_indices):raise ValueError('new0 must reuse actual B before forward')
        representatives,groups=_representatives(p,b,a,radial,crossrad)
        oi=representatives;ni=p.new_indices
        if p.gamma is None:
            train=p.M_train.copy();scores=p.M_held.copy()
            numeric=dict(alpha=np.zeros((len(ni),c)),beta=np.zeros((len(oi),c)),v=np.zeros(c),
                K_perp=np.zeros((len(ni),len(ni))),L_perp=np.zeros((h,len(ni))),old_residual=np.zeros((len(oi),c)))
            audit.update(status='ZERO_KERNEL_CONSTRAINT_FORCES_ZERO_RESIDUAL',actual_kernel_trace=0.,constraint_max_abs=0.)
        else:
            raw=p.gamma*radial;rc=p.gamma*crossrad
            blocks=dict(A=raw[np.ix_(oi,oi)],B=raw[np.ix_(oi,ni)],D=raw[np.ix_(ni,ni)],F=rc[:,oi],E=rc[:,ni])
            head_start=time.perf_counter()
            try:
                ks=conditional.fit_conditional_affine(**blocks,M_O=p.M_train[oi],M_N=p.M_train[ni],M_H=p.M_held,R_N=target[ni])
            except conditional.NumericalFailure as exc:
                audit.update(kernel_failure_audit=dict(exc.audit));_add(audit,exc.audit,_KERNEL_COUNTERS)
                audit['factorization_count']=exc.audit.get('factorization_count',0)
                exc.joint_failed_head_arrays=dict(blocks,Y=Y,residual_target=target,M_train=p.M_train,M_held=p.M_held,
                    train_labels=p.train_labels,held_labels=p.held_labels,old_indices=oi,new_indices=ni,
                    original_train=p.train_context['original'],adapted_train=np.column_stack((b,a)),
                    original_held=p.held_context['original'],adapted_held=np.column_stack((hb,ha)),
                    distance=distance,cross_distance=cross,raw_train=radial,raw_cross=crossrad,tau=_scalar(p.tau),gamma=_scalar(p.gamma))
                raise
            audit['conditional_head_seconds']=time.perf_counter()-head_start
            numeric=dict(ks.arrays);_add(audit,ks.audit,_KERNEL_COUNTERS)
            audit['factorization_count']=ks.audit['factorization_count'];audit['conditional_kernel_audit']=dict(ks.audit)
            train=np.stack([ks.expansion.score(k_old=raw[i:i+1,oi],k_new=raw[i:i+1,ni],M=p.M_train[i:i+1])[0] for i in range(n)])
            scores=ks.arrays['held_scores'].copy()
            audit.update(normal_equation_residual=ks.audit['ridge_solve_residual'],actual_kernel_trace=float(np.trace(numeric['K_perp'])),
                constraint_max_abs=float(np.max(np.abs(train[p.old_indices]-p.M_train[p.old_indices]))),
                head_training_loss_data=float(.5*np.sum((train-Y)**2)),
                head_training_loss_ridge=float(.5*np.sum(numeric['alpha']*(numeric['K_perp']@numeric['alpha']))))
    for suffix in _SUFFIXES[1:]:audit['head_'+suffix]=audit['projection_'+suffix]+audit['residual_'+suffix]
    audit['intercept_addition_count']=(n+h)*c
    classerr=max(affine._sum_class_error(x) for x in (Y,target,train,scores))
    tol=128*_EPS*max(n,c);bound=tol*max(1.,_norm(train),_norm(scores))
    if classerr>bound:raise FloatingPointError('CONDITIONAL_JOINT_CLASS_SUM_RESIDUAL')
    if p.mode=='C_seq' and audit['constraint_max_abs']>tol*max(1.,_norm(train),_norm(p.M_train)):
        raise FloatingPointError('CONDITIONAL_JOINT_OLD_CONSTRAINT_RESIDUAL')
    angles=[];normerr=0.
    for ctx,ac in ((p.train_context,bc),(p.held_context,hc)):
        for j,sl in enumerate(fcr._SLICES):
            valid=ctx['rho'][:,j]>0
            chord=np.linalg.norm(ac['y'][valid,sl]-ctx['w'][valid,sl],axis=1)
            angles.extend((2*np.arcsin(np.minimum(1.,chord/2))).tolist())
            if np.any(valid):normerr=max(normerr,float(np.max(np.abs(np.linalg.norm(ac['y'][valid,sl],axis=1)-1))))
    if p.mode=='B':
        fitted_mean=p.q@train;expected_mean=p.q@p.M_train+numeric['intercept']
        reference_error=_norm(p.q@(train-p.M_train)-numeric['intercept'])
    else:
        # C protects physical old points, independently of the affine gauge.
        # The all-column pointwise constraint was checked above; its mean
        # audit must not turn an arbitrary q into an extra C constraint.
        fitted_mean=train[p.old_indices].mean(axis=0)
        expected_mean=p.M_train[p.old_indices].mean(axis=0)
        reference_error=_norm((train-p.M_train)[p.old_indices].mean(axis=0))
    audit.update(class_sum_max_abs=classerr,class_sum_tolerance=bound,old_physical_constraint_count=len(p.old_indices),
        old_projection_representative_count=len(representatives),duplicate_old_constraint_count=len(p.old_indices)-len(representatives),
        old_reference_residual_mean_norm=reference_error,fitted_old_reference_mean=_plain(fitted_mean),
        expected_old_reference_mean=_plain(expected_mean),
        block_angle_radians=fcr._statistics(angles),block_unit_norm_max_error=normerr,
        condition_bound=1+audit['actual_kernel_trace'],gram_min_eigenvalue_lower_bound=1.,
        analytic_intercept_parameter_count=c,analytic_intercept_contrast_count=c-1,
        identity_forward=identity,forward_seconds=time.perf_counter()-start)
    if reference_error>tol*max(1.,_norm(train),_norm(p.M_train)):
        raise FloatingPointError('CONDITIONAL_JOINT_REFERENCE_RESIDUAL_MEAN')
    if 'head_training_loss_data' not in audit:audit['head_training_loss_data']=float(.5*np.sum((train-Y)**2));audit['head_training_loss_ridge']=0.
    audit['head_training_loss_total']=audit['head_training_loss_data']+audit['head_training_loss_ridge']
    audit.setdefault('status','CLOSED_FORM_SOLVED')
    _finite(train,scores)
    return dict(problem=p,U=np.array(U,copy=True),b=b,a=a,hb=hb,ha=ha,bc=bc,hc=hc,distance=distance,cross_distance=cross,
        radial=radial,crossrad=crossrad,Y=Y,residual_target=target,score=scores,train_scores=train,
        representatives=representatives,old_inverse_groups=groups,numeric=numeric,kernel_state=ks,audit=audit)


def _forward(p,U,progress=None):
    audit={} if progress is None else progress
    try:return _forward_impl(p,U,audit)
    except (FloatingPointError,np.linalg.LinAlgError,conditional.NumericalFailure) as exc:
        exc.joint_forward_audit=audit
        raise


def _backward(cache,G,progress):
    p=cache['problem'];start=time.perf_counter()
    if p.gamma is None or p.tau==0 or not len(G):
        progress['adjoint_seconds']=time.perf_counter()-start;return np.zeros((736,8))
    n=len(p.train_labels);c=len(p.classes);h=len(G);num=cache['numeric']
    if p.mode=='B':
        T,eta,g_b,rhs=affine._affine_adjoint(num['chol'],num['L'],G,num['schur_z'],float(num['schur_s']))
        barK=-.5*(T@num['alpha'].T+num['alpha']@T.T);barL=G@num['alpha'].T
        barR,barQ=affine._center_vjp(barK,barL,p.q,p.gamma)
        cache['adjoint_arrays']=dict(T=T,eta=eta,g_b=g_b,rhs=rhs,G=G)
        for suffix,value in zip(_SUFFIXES[1:],(2,2*c,2*n*c,2*n*n*c)):progress['residual_adjoint_'+suffix]=value
        progress['ce_adjoint_solve_count']=1
    else:
        try:adj=conditional.conditional_affine_adjoint(cache['kernel_state'],np.asarray(G,dtype=np.float64))
        except conditional.NumericalFailure as exc:
            _add(progress,exc.audit,_KERNEL_COUNTERS)
            for suffix in _SUFFIXES[1:]:progress['derivative_'+suffix]=progress.get('projection_adjoint_'+suffix,0)+progress.get('residual_adjoint_'+suffix,0)
            progress['ce_adjoint_solve_count']=1
            exc.joint_backward_audit=progress;raise
        cache['adjoint_arrays']=dict(adj.arrays);_add(progress,adj.audit,_KERNEL_COUNTERS)
        oi=cache['representatives'];ni=p.new_indices
        # A and D are symmetric independent blocks; cross B occurs twice in
        # the assembled raw Gram, so split its total derivative in half.
        barR=np.zeros((n,n));barQ=np.zeros((h,n))
        barR[np.ix_(oi,oi)]=adj.arrays['A'];barR[np.ix_(ni,ni)]=adj.arrays['D']
        barR[np.ix_(oi,ni)]=.5*adj.arrays['B'];barR[np.ix_(ni,oi)]=.5*adj.arrays['B'].T
        barQ[:,oi]=adj.arrays['F'];barQ[:,ni]=adj.arrays['E']
        barR*=p.gamma;barQ*=p.gamma;progress['ce_adjoint_solve_count']=1
    for suffix in _SUFFIXES[1:]:
        progress['derivative_'+suffix]=progress.get('projection_adjoint_'+suffix,0)+progress.get('residual_adjoint_'+suffix,0)
    dd=-barR*cache['radial']/p.tau;dc=-barQ*cache['crossrad']/p.tau
    gb,ga,_,_=ch._distance_vjp(cache['b'],cache['a'],cache['b'],cache['a'],.5*dd,symmetric=True)
    ghb,gha,gtb,gta=ch._distance_vjp(cache['hb'],cache['ha'],cache['b'],cache['a'],.5*dc)
    gu=fcr._adapter_vjp(p.train_context,cache['U'],gb+gtb,ga+gta,cache['bc'])
    gu+=fcr._adapter_vjp(p.held_context,cache['U'],ghb,gha,cache['hc'])
    cache['adjoint_arrays'].update(raw_train_upstream=barR,raw_cross_upstream=barQ,adapted_distance_upstream=.5*dd,
        adapted_cross_distance_upstream=.5*dc,g_U=gu)
    progress['adjoint_seconds']=time.perf_counter()-start;return gu


def _head_arrays(cache):
    p=cache['problem']
    arrays=dict(cache['numeric'])
    arrays.update(original_train_b=p.train_context['original'][:,:256],original_train_a=p.train_context['original'][:,256:],
        original_held_b=p.held_context['original'][:,:256],original_held_a=p.held_context['original'][:,256:],
        adapted_train_b=cache['b'],adapted_train_a=cache['a'],adapted_held_b=cache['hb'],adapted_held_a=cache['ha'],
        train_labels=p.train_labels,held_labels=p.held_labels,q=p.q,old_indices=p.old_indices,new_indices=p.new_indices,
        old_representative_indices=cache['representatives'],old_inverse_groups=cache['old_inverse_groups'],
        original_distance=p.d0,original_cross_distance=p.cross_d0,distance=cache['distance'],cross_distance=cache['cross_distance'],
        raw_train=cache['radial'],raw_cross=cache['crossrad'],Y=cache['Y'],residual_target=cache['residual_target'],
        M_train=p.M_train,M_held=p.M_held,train_scores=cache['train_scores'],scores=cache['score'],
        tau=_scalar(p.tau),gamma=_scalar(p.gamma),s0=_scalar(p.s0))
    return arrays


def _score_residual(p,cache,U,b0,a0,progress=None,H=None):
    start=time.perf_counter();n=len(b0);c=len(p.classes);num=cache['numeric'];out=np.empty((n,c))
    if p.gamma is None:
        out[:]=num['intercept'] if p.mode=='B' else 0.
        if progress is not None:
            progress['intercept_addition_count']=progress.get('intercept_addition_count',0)+out.size
            progress['score_seconds']=progress.get('score_seconds',0.)+time.perf_counter()-start
        return out
    original=p.train_context['original'];oi=cache['representatives'];ni=p.new_indices
    for i in range(n):
        d0=local._distances(b0[i:i+1],a0[i:i+1],original[:,:256],original[:,256:])
        if p.tau==0:d=d0
        else:
            ctx=fcr._context(b0[i:i+1],a0[i:i+1],None if H is None else H[i:i+1]);b,a,_=fcr._adapt(ctx,U)
            d=.5*d0+.5*local._distances(b,a,cache['b'],cache['a']) if np.any(U) else d0
        if p.mode=='B':
            centered=affine._center_cross(local._radial_minus_one(d,p.tau),p.q,num['reference_kernel'],float(num['reference_self']),num['center_mean'],float(num['center_grand']))
            out[i]=(p.gamma*centered[0])@num['alpha']+num['intercept']
        else:
            raw=p.gamma*local._radial(d,p.tau)
            out[i]=raw[0,ni]@num['alpha']-raw[0,oi]@num['beta']+num['v']
    if progress is not None:
        width=len(p.train_labels);r=len(p.old_indices);calls=n*(1+int(p.tau!=0 and bool(np.any(U))))
        _add(progress,dict(raw_distance_evaluation_count=calls,raw_distance_pair_count=calls*width,
            reference_distance_evaluation_count=calls,reference_distance_pair_count=calls*r,
            kernel_evaluation_count=n,kernel_pair_count=n*width,
            adapter_physical_evaluation_count=n if p.tau!=0 else 0,
            dictionary_physical_evaluation_count=n if p.tau!=0 and H is None else 0,
            intercept_addition_count=n*c),_GEOMETRY_COUNTERS+('dictionary_physical_evaluation_count','intercept_addition_count'))
        progress['score_seconds']=progress.get('score_seconds',0.)+time.perf_counter()-start
    _finite(out);return out


@dataclass(frozen=True)
class ConditionalJointTraining:
    raw: Mapping
    labels: np.ndarray
    ids: tuple
    classes: tuple
    old_classes: tuple
    H: np.ndarray
    W: np.ndarray
    singular_values: np.ndarray
    r: int
    anchor_U: np.ndarray
    inherited: object
    problems: tuple
    full_problem: ConditionalJointProblem
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        for key in ('labels','H','W','singular_values','anchor_U'):object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'raw',_freeze({k:_readonly(v) for k,v in self.raw.items()}))
        object.__setattr__(self,'records',_freeze(self.records));object.__setattr__(self,'audit',_freeze(self.audit))
    def audit_dict(self):return ch._safe(_plain(self.audit))


def prepare_conditional_joint_training(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes,
        inherited=None,context=None,log_callback=None,state_callback=None):
    start=time.perf_counter();rec=_Recorder(state_callback);audit=dict(context or {});audit.update({k:0 for k in PREPARATION_COUNTERS})
    audit['ajlr_preparation_count']=1
    b,a,y,ids,canonical,_,old,k=interaction._prepare(z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes)
    old=tuple(sorted(old));n=len(ids)
    if not old:raise ValueError('Old support classes required')
    order=np.asarray(sorted(range(n),key=lambda i:support_ids[i]));raw={name:np.asarray(value,dtype=np.float64)[order] for name,value in zip(_NAMES,(z_id,fft,t_emb,f_emb,pa_local))}
    anchor=np.zeros((736,8));mode='B' if inherited is None else 'C_seq'
    if inherited is None:
        if canonical!=old:raise ValueError('C requires actual current CONDITIONAL_JOINT B')
    else:
        if not isinstance(inherited,ConditionalJointState) or inherited.audit['mode']!='B' or inherited.classes!=old:
            raise ValueError('C requires actual old-only CONDITIONAL_JOINT B')
        previous=inherited.audit['preparation']
        for key in ('run_id','row_id','scope'):
            if not audit.get(key) or not previous.get(key):raise ValueError('Actual B lineage requires explicit '+key)
            if audit[key]!=previous[key]:raise ValueError('C lineage crosses '+key)
        for key in ('split_id','fold'):
            if (key in audit)!=(key in previous) or (key in audit and audit[key]!=previous[key]):raise ValueError('C lineage crosses '+key)
        oi=np.array([canonical[int(v)] in old for v in y]);oldids=tuple(pid for i,pid in enumerate(ids) if oi[i])
        if oldids!=inherited.ids:raise ValueError('C old physical IDs differ from B')
        if any(canonical[int(v)]!=inherited.classes[int(w)] for v,w in zip(y[oi],inherited.labels)):
            raise ValueError('C old labels differ from B')
        if any(not np.array_equal(raw[name][oi],inherited.raw[name]) for name in _NAMES):raise ValueError('C old raw features differ from B')
        anchor=inherited.U.copy()
    audit.update(schema=FROZEN_CONFIG['schema'],method=FROZEN_CONFIG['method'],mode=mode,classes=list(canonical),old_classes=list(old),
        training_physical_ids=list(ids),train_physical_count=n,train_k=k,inherited_state=inherited is not None)
    if inherited is not None and canonical==old:
        audit.update(ajlr_preparation_count=0,no_information=True,no_information_reason='N0_REUSE_ACTUAL_B',
            prepared_state_ref=_plain(previous['prepared_state_ref']),inner_folds=[],prior_folds=[],final_problem=_plain(inherited.problem.audit),
            prepared_numeric_state_bytes=0,retained_vector_record_bytes=0,preparation_seconds=time.perf_counter()-start)
        return ConditionalJointTraining(raw,y,ids,canonical,old,inherited.H,inherited.W,inherited.singular_values,
            inherited.W.shape[1],anchor,inherited,(),inherited.problem,{},audit)
    try:
        H=fcr._context(b,a)['h'];audit['dictionary_physical_evaluation_count']=n
        W,s,ci=latent_coordinates(H);audit.update(ci);r=W.shape[1]
        audit['prepared_state_ref']=rec.save('prepared_coordinates',H=H,W=W,singular_values=s,anchor_U=anchor,V0=V0)
        problems=[];priors=[]
        def make(keep,fold,final=False):
            prior=None;nu=None;ref=None
            if inherited is not None:
                if final:
                    pp,pu,pf=inherited.problem,inherited.U,inherited.final_cache;ref=_plain(inherited.audit['final_state_ref'])
                else:
                    oi=np.flatnonzero(np.array([canonical[int(v)] in old for v in y]));oy=np.array([old.index(canonical[int(v)]) for v in y[oi]],dtype=np.int64)
                    pp=_make_problem(b[oi],a[oi],oy,tuple(ids[i] for i in oi),old,old,keep[oi],H[oi],mode='B',fold=fold,progress=audit)
                    pf=_forward(pp,anchor);pu=anchor
                    audit['prior_head_fit_count']+=1;audit['prior_factorization_count']+=pf['audit']['factorization_count']
                    for suffix in _SUFFIXES[1:]:audit['prior_'+suffix]+=pf['audit']['head_'+suffix]
                    audit['prior_intercept_fit_count']+=1;audit['prior_intercept_addition_count']+=pf['audit']['intercept_addition_count']
                    _add(audit,pf['audit'],_GEOMETRY_COUNTERS)
                    ref=rec.save('prior_head_'+str(fold),U=anchor,**_head_arrays(pf))
                    priors.append(dict(_plain(pp.audit),head_state_ref=ref,final_fit=_plain(pf['audit'])))
                prior=(pp,pu,pf,old);nu=dict(tau=pp.tau,gamma=pp.gamma,s0=pp.s0)
            return _make_problem(b,a,y,ids,canonical,old,keep,H,mode=mode,fold=fold,prior=prior,nuisance=nu,progress=audit,prior_ref=ref)
        if k>1:
            folds=min(k,3);assignment=np.empty(n,dtype=np.int64)
            for cls in range(len(canonical)):
                ix=np.flatnonzero(y==cls);assignment[ix]=np.arange(len(ix))%folds
            for fold in range(folds):problems.append(make(assignment!=fold,fold))
        full=make(np.ones(n,dtype=bool),None,True)
        reason=('PHYSICAL_K1' if k==1 else 'ZERO_DICTIONARY_RANK' if r==0 else 'NO_OLD_KERNEL_INFORMATION' if full.gamma is None
            else 'ALL_INNER_GEOMETRY_DEGENERATE' if not any(p.gamma is not None and p.tau is not None and p.tau>0 for p in problems) else None)
        audit.update(no_information=reason is not None,no_information_reason=reason,inner_folds=[_plain(p.audit) for p in problems],
            prior_folds=priors,final_problem=_plain(full.audit),latent_rank=r,preparation_seconds=time.perf_counter()-start,
            prepared_numeric_state_bytes=_unique_bytes((raw,y,H,W,s,anchor,b,a)),retained_vector_record_bytes=_unique_bytes(rec.records),
            source_inputs=False,query_fit=False,reference_distance_scope='subset_of_raw_distance_work_not_additive')
        ch._emit(log_callback,'CONDITIONAL_JOINT_PREPARED',audit)
        return ConditionalJointTraining(raw,y,ids,canonical,old,H,W,s,r,anchor,inherited,tuple(problems),full,rec.records,audit)
    except (FloatingPointError,np.linalg.LinAlgError,conditional.NumericalFailure) as exc:
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),preparation_seconds=time.perf_counter()-start)
        error=NumericalFailure(str(exc),audit);error.records=rec.records;raise error from exc


@dataclass(frozen=True)
class ObjectiveCache:
    prepared: ConditionalJointTraining
    Z: np.ndarray
    folds: tuple
    def __post_init__(self):object.__setattr__(self,'Z',_readonly(self.Z))


def _forward_cost(folds):
    info={key:0 for key in STAGE_COUNTERS}
    for cf in folds:
        fa=cf['audit']
        _add(info,fa,_GEOMETRY_COUNTERS+_KERNEL_COUNTERS+tuple('head_'+s for s in _SUFFIXES[1:])+('intercept_fit_count','intercept_addition_count'))
        info['inner_head_fit_count']+=fa.get('head_fit_count',0)
        info['inner_factorization_count']+=fa.get('factorization_count',0)
        info['ajlr_forward_evaluation_count']+=fa.get('ajlr_forward_evaluation_count',0)
    info['inner_objective_evaluation_count']=1
    return info


def _cache_values(cache):
    if cache is None:return ()
    values=[cache.Z]
    for cf in cache.folds:
        values.append({k:v for k,v in cf.items() if k not in ('problem','kernel_state','audit')})
        values.append({k:v for k,v in vars(cf['problem']).items() if k!='audit'})
        ks=cf['kernel_state']
        if ks is not None:values.extend((ks.arrays,vars(ks.expansion)))
    return values


def evaluate_conditional_joint_objective(prepared,Z,anchor_U=None,*,gradient=True,forward_cache=None):
    start=time.perf_counter();anchor=prepared.anchor_U if anchor_U is None else fcr._parameter(anchor_U)
    if not np.array_equal(anchor,prepared.anchor_U):raise ValueError('Objective actual B anchor differs')
    Z=np.asarray(Z,dtype=np.float64);U=reconstruct_U(Z,anchor,prepared.W);c=len(prepared.classes)
    reused=forward_cache is not None
    if reused and (forward_cache.prepared is not prepared or not np.array_equal(forward_cache.Z,Z)):
        raise ValueError('Objective forward cache binding differs')
    if reused:folds=forward_cache.folds
    else:
        partial=[]
        try:
            for p in prepared.problems:partial.append(_forward(p,U))
        except (FloatingPointError,np.linalg.LinAlgError,conditional.NumericalFailure) as exc:
            billed=partial+[dict(audit=getattr(exc,'joint_forward_audit',{}))]
            exc.joint_objective_audit=_forward_cost(billed)
            exc.joint_partial_folds=tuple(partial)
            raise
        folds=tuple(partial)
    cache=forward_cache if reused else ObjectiveCache(prepared,Z.copy(),folds)
    sums=np.zeros(c);counts=np.zeros(c,dtype=np.int64);ces=[];ds=[]
    for cf in folds:
        labels=cf['problem'].held_labels;ce,d=affine._ce(cf['score'],labels)
        sums+=np.bincount(labels,weights=ce,minlength=c);counts+=np.bincount(labels,minlength=c);ces.append(ce);ds.append(d)
    means=np.divide(sums,counts,out=np.zeros(c),where=counts>0);risk=_norm(means)/np.sqrt(c)
    info={key:0 for key in STAGE_COUNTERS};gU=np.zeros((736,8));fold_audits=[]
    if not reused:info.update(_forward_cost(folds))
    info['backward_evaluation_count']=int(gradient)
    for cf,ce,d in zip(folds,ces,ds):
        labels=cf['problem'].held_labels;fa=deepcopy(cf['audit'])
        fa.update(held_ce_sums=np.bincount(labels,weights=ce,minlength=c).tolist(),held_ce_counts=np.bincount(labels,minlength=c).tolist(),
            held_correct_count=int(np.sum(np.argmax(cf['score'],axis=1)==labels)))
        # A cached gradient is recomputed, so its actual solves are paid again.
        for key in tuple(p+'_'+s for p in ('projection_adjoint','residual_adjoint') for s in _SUFFIXES)+tuple('derivative_'+s for s in _SUFFIXES[1:])+('ce_adjoint_solve_count',):fa[key]=0
        if gradient and risk>0:
            weights=means[labels]/(c*risk*counts[labels])
            try:gU+=_backward(cf,d*weights[:,None],fa)
            except (FloatingPointError,np.linalg.LinAlgError,conditional.NumericalFailure) as exc:
                _add(info,fa,tuple(p+'_'+s for p in ('projection_adjoint','residual_adjoint') for s in _SUFFIXES)+tuple('derivative_'+s for s in _SUFFIXES[1:])+('ce_adjoint_solve_count',))
                exc.joint_objective_audit=info;raise
            _add(info,fa,tuple(p+'_'+s for p in ('projection_adjoint','residual_adjoint') for s in _SUFFIXES)+tuple('derivative_'+s for s in _SUFFIXES[1:])+('ce_adjoint_solve_count',))
        fold_audits.append(fa)
    gZ=gU@prepared.W if gradient else np.zeros_like(Z)
    info.update(loss_ce=float(risk),loss_task=float(risk),RMSCE=float(risk),loss_proximal=0.,loss_total=float(risk),
        class_ce_sums=sums.tolist(),class_ce_counts=counts.tolist(),class_ce_means=means.tolist(),temperature=1.,
        gradient_norm=_norm(gZ) if gradient else None,inner_folds=fold_audits,forward_cache_reused=reused,
        coordinate_ball_radius=.5,coordinate_ball_feasible=bool(_norm(Z)<=.5+128*_EPS),
        forward_cache_bytes=_unique_bytes(_cache_values(cache)),objective_seconds=time.perf_counter()-start,
        **fcr._functional_measurement(prepared,U,anchor,Z))
    _finite(risk,gZ);return float(risk),gZ,info,cache


def project_coordinates(Z):
    Z=np.asarray(Z,dtype=np.float64);_finite(Z);norm=_norm(Z)
    return Z.copy() if norm<=.5 else Z*(.5/norm)


def _trial_acceptance(before,after,dot):
    return affine._trial_acceptance(before,after,dot)


@dataclass(frozen=True)
class ConditionalJointState:
    U: np.ndarray
    Z: np.ndarray
    W: np.ndarray
    anchor_U: np.ndarray
    H: np.ndarray
    singular_values: np.ndarray
    raw: Mapping
    labels: np.ndarray
    ids: tuple
    classes: tuple
    old_classes: tuple
    problem: ConditionalJointProblem
    final_cache: Mapping
    prior: object
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        for key in ('U','Z','W','anchor_U','H','singular_values','labels'):object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'raw',_freeze({k:_readonly(v) for k,v in self.raw.items()}))
        def freeze(value):
            if isinstance(value,np.ndarray):return _readonly(value)
            if isinstance(value,Mapping):return _freeze({k:freeze(v) for k,v in value.items()})
            return value
        object.__setattr__(self,'final_cache',freeze(self.final_cache));object.__setattr__(self,'records',_freeze(self.records));object.__setattr__(self,'audit',_freeze(self.audit))
    @property
    def V(self):return V0
    @property
    def V0(self):return V0
    def audit_dict(self):return ch._safe(_plain(self.audit))
    def state_records(self):return {key:{k:np.array(v,copy=True) for k,v in record.items()} for key,record in self.records.items()}
    def _score_geometry(self,b,a,progress=None):
        residual={};prior={};out=_score_residual(self.problem,self.final_cache,self.U,b,a,residual)
        if self.prior is not None:
            old=_score_residual(self.prior.problem,self.prior.final_cache,self.prior.U,b,a,prior)
            for j,name in enumerate(self.prior.classes):out[:,self.classes.index(name)]+=old[:,j]
        if progress is not None:
            progress.update(residual=residual,prior=prior)
            for key in set(residual)|set(prior):progress[key]=residual.get(key,0)+prior.get(key,0)
        _finite(out);return out
    def score(self,*,z_id,fft,t_emb,f_emb,pa_local):
        return self._score_geometry(*interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True))
    def score_with_audit(self,*,z_id,fft,t_emb,f_emb,pa_local):
        start=time.perf_counter();b,a=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True);p={}
        out=self._score_geometry(b,a,p);p.update(score_physical_count=len(b),score_seconds=time.perf_counter()-start,
            reference_distance_scope='subset_of_raw_distance_work_not_additive');return out,p
    def predict(self,**features):
        order=np.asarray(sorted(range(len(self.classes)),key=lambda i:self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(self.score(**features)[:,order],axis=1)]]
    def to_arrays(self):
        arrays=_head_arrays(dict(self.final_cache,problem=self.problem))
        arrays.update(self.raw,U=self.U,Z=self.Z,W=self.W,anchor_U=self.anchor_U,H=self.H,singular_values=self.singular_values,V0=V0)
        if self.prior is not None:
            arrays.update({'prior_B_'+key:value for key,value in self.prior.to_arrays().items()})
            arrays['prior_old_class_indices']=np.array([self.classes.index(name) for name in self.prior.classes],dtype=np.int64)
        return arrays
    def _resident_values(self):
        values=(self.U,self.Z,self.W,self.anchor_U,self.H,self.singular_values,self.raw,self.labels,V0,
            {k:v for k,v in vars(self.problem).items() if k!='audit'},self.final_cache)
        return values if self.prior is None else values+(self.prior._resident_values(),)
    def _deployment_values(self):
        num=self.final_cache['numeric'];names=('alpha','intercept','reference_kernel','reference_self','center_mean','center_grand') if self.problem.mode=='B' else ('alpha','beta','v')
        values=(self.U,V0,self.raw,self.labels,self.problem.q,self.problem.train_context['original'],self.final_cache['b'],self.final_cache['a'],
            self.final_cache['representatives'],self.problem.new_indices,{k:num[k] for k in names})
        return values if self.prior is None else values+(self.prior._deployment_values(),)


def _cache_for_state(cache):
    return {key:value for key,value in cache.items() if key not in ('problem','bc','hc','kernel_state','adjoint_arrays')}


def fit_conditional_joint_local_ridge(prepared,*,mode='B',baseline_state=None,log_callback=None,state_callback=None):
    if mode not in ('B','C_seq'):raise ValueError('Unknown CONDITIONAL_JOINT mode')
    if (mode=='B')!=(prepared.inherited is None):raise ValueError('Mode does not match actual B inheritance')
    if prepared.audit.get('no_information_reason')=='N0_REUSE_ACTUAL_B':return prepared.inherited
    start=time.perf_counter();rec=_Recorder(state_callback);rec.records.update(dict(prepared.records))
    audit=dict(schema=FROZEN_CONFIG['schema'],method=FROZEN_CONFIG['method'],mode=mode,config=deepcopy(FROZEN_CONFIG),
        preparation=_plain(prepared.audit),no_information=prepared.audit['no_information'],gradients=[],trials=[],steps=[])
    audit.update({key:0 for key in STAGE_COUNTERS});audit['ajlr_stage_count']=1
    Z=np.zeros((736,prepared.r));U=prepared.anchor_U.copy();cache=None;finalprogress={}
    def cost(info):
        _add(audit,info,STAGE_COUNTERS)
        audit['max_forward_cache_bytes']=max(audit.get('max_forward_cache_bytes',0),info.get('forward_cache_bytes',0))
    def event(name,payload):ch._emit(log_callback,'CONDITIONAL_JOINT_'+name,dict(mode=mode,**payload))
    def save_objective(key,obj,oc,**extra):
        arrays=dict(Z=Z,U=U,W=prepared.W,anchor_U=prepared.anchor_U,RMSCE=_scalar(obj.get('RMSCE')),
            class_ce_sums=np.asarray(obj.get('class_ce_sums',[])),class_ce_counts=np.asarray(obj.get('class_ce_counts',[]),dtype=np.int64),
            class_ce_means=np.asarray(obj.get('class_ce_means',[])),prox=_scalar(0.),**extra)
        if oc is not None:
            for i,cf in enumerate(oc.folds):
                fa=obj['inner_folds'][i];head=rec.save(key+'_fold_'+str(i),U=U,**_head_arrays(cf),
                    **{'adjoint_'+k:v for k,v in cf.get('adjoint_arrays',{}).items()})
                fa['head_state_ref']=head
        return rec.save(key,**arrays)
    try:
        if audit['no_information']:
            audit['stop_reason']=prepared.audit['no_information_reason'];audit['initialization_state_ref']=save_objective('initial',{},None)
            event('INITIAL',dict(state_ref=audit['initialization_state_ref'],no_update_reason=audit['stop_reason']))
        else:
            loss,_,obj,cache=evaluate_conditional_joint_objective(prepared,Z,gradient=False);cost(obj)
            audit['initialization_state_ref']=save_objective('initial',obj,cache);audit['initial_objective']=deepcopy(obj)
            event('INITIAL',dict(objective=obj,state_ref=audit['initialization_state_ref']));audit['stop_reason']='MAX_ITERATIONS'
            for iteration in range(1,5):
                audit['optimizer_iterations']+=1
                loss,g,obj,cache=evaluate_conditional_joint_objective(prepared,Z,gradient=True,forward_cache=cache);cost(obj)
                norm=_norm(g);direction=np.zeros_like(g) if norm==0 else -g/norm
                ref=save_objective('gradient_'+str(iteration),obj,cache,g_Z=g,d_Z=direction)
                gr=dict(iteration=iteration,state_ref=ref,gradient_norm=norm,direction_norm=_norm(direction),objective=deepcopy(obj));audit['gradients'].append(gr);event('GRADIENT',gr)
                if norm==0:audit['stop_reason']='ZERO_GRADIENT';break
                accepted=False
                for trial in range(1,13):
                    step=.125*.5**(trial-1);tz=project_coordinates(Z+step*direction);delta=tz-Z
                    if not np.any(delta):audit['stop_reason']='ZERO_FEASIBLE_DISPLACEMENT';break
                    audit['trial_attempt_count']+=1
                    try:tl,_,to,tc=evaluate_conditional_joint_objective(prepared,tz,gradient=False)
                    except (FloatingPointError,np.linalg.LinAlgError,conditional.NumericalFailure) as exc:
                        exc.joint_attempt_Z=tz.copy();exc.joint_attempt_U=reconstruct_U(tz,prepared.anchor_U,prepared.W);raise
                    cost(to)
                    audit['max_simultaneous_forward_cache_bytes']=max(audit.get('max_simultaneous_forward_cache_bytes',0),
                        _unique_bytes((_cache_values(cache),_cache_values(tc))))
                    acceptance=_trial_acceptance(loss,tl,float(np.sum(g*delta)))
                    oldz,oldu=Z,U;Z,U=tz,reconstruct_U(tz,prepared.anchor_U,prepared.W)
                    tref=save_objective('trial_'+str(iteration)+'_'+str(trial),to,tc,delta_Z=delta,d_Z=direction);Z,U=oldz,oldu
                    tr=dict(iteration=iteration,trial=trial,step_size=step,state_ref=tref,gradient_state_ref=ref,
                        loss_before=loss,loss_after=tl,update_norm=_norm(delta),objective=deepcopy(to),**acceptance)
                    audit['trial_count']+=1;audit['trials'].append(tr);accepted=acceptance['accepted']
                    audit['accepted_trial_count' if accepted else 'rejected_trial_count']+=1;event('TRIAL',tr)
                    if accepted:
                        Z,U,cache,loss=tz,reconstruct_U(tz,prepared.anchor_U,prepared.W),tc,tl;audit['optimizer_steps']+=1
                        sr=dict(step=audit['optimizer_steps'],iteration=iteration,trial=trial,state_ref=tref,
                            loss_before=tr['loss_before'],loss_after=tl,learning_rate=step,update_norm=_norm(delta),objective=deepcopy(to))
                        audit['steps'].append(sr);event('STEP',sr);break
                if not accepted:
                    if audit['stop_reason']!='ZERO_FEASIBLE_DISPLACEMENT':audit['stop_reason']='TRIAL_BUDGET_EXHAUSTED'
                    break
            _,_,final_obj,_=evaluate_conditional_joint_objective(prepared,Z,gradient=False,forward_cache=cache)
            audit['final_objective_state_ref']=save_objective('final_objective',final_obj,cache);audit['final_objective']=final_obj
        final=_forward(prepared.full_problem,U,finalprogress)
        audit['final_head_fit_count']=1;audit['final_factorization_count']=finalprogress['factorization_count']
        _add(audit,finalprogress,_GEOMETRY_COUNTERS+_KERNEL_COUNTERS+tuple('head_'+s for s in _SUFFIXES[1:])+('intercept_fit_count','intercept_addition_count','ajlr_forward_evaluation_count'))
        arrays=_head_arrays(final);arrays.update(prepared.raw,U=U,Z=Z,W=prepared.W,anchor_U=prepared.anchor_U,H=prepared.H,singular_values=prepared.singular_values,V0=V0)
        if prepared.inherited is not None:
            arrays.update({'prior_B_'+key:value for key,value in prepared.inherited.to_arrays().items()})
            arrays['prior_old_class_indices']=np.asarray([prepared.classes.index(name) for name in prepared.inherited.classes],dtype=np.int64)
        audit['final_state_ref']=rec.save('final',**arrays);audit.update(final_fit=_plain(finalprogress),coordinate_norm=_norm(Z),
            parameter_U_norm=_norm(U),parameter_changed_from_anchor=bool(np.any(U!=prepared.anchor_U)),u_changed_from_anchor=bool(np.any(U!=prepared.anchor_U)),
            declared_coordinate_scalar_count=736*prepared.r,trainable_parameter_count=0 if audit['no_information'] else 736*prepared.r,
            active_parameter_count=736*prepared.r if audit['optimizer_steps'] else 0,
            trained_parameter_count=736*prepared.r if audit['optimizer_steps'] else 0,optimizer_parameter_count=0 if audit['no_information'] else 736*prepared.r,
            analytic_head_scalar_count=sum(v.size for k,v in final['numeric'].items() if k in ('alpha','beta','v','intercept')),
            inherited_from_B=prepared.inherited is not None,fit_seconds=time.perf_counter()-start,hardware_dtype='CPU_float64',status='COMPLETED')
        state=ConditionalJointState(U,Z,prepared.W,prepared.anchor_U,prepared.H,prepared.singular_values,prepared.raw,prepared.labels,prepared.ids,
            prepared.classes,prepared.old_classes,prepared.full_problem,_cache_for_state(final),prepared.inherited,rec.records,audit)
        audit.update(resident_numeric_state_bytes=_unique_bytes(state._resident_values()),deployment_numeric_state_bytes=_unique_bytes(state._deployment_values()),
            retained_vector_record_bytes=_unique_bytes(rec.records),prepared_numeric_state_bytes=prepared.audit['prepared_numeric_state_bytes'])
        state=ConditionalJointState(U,Z,prepared.W,prepared.anchor_U,prepared.H,prepared.singular_values,prepared.raw,prepared.labels,prepared.ids,
            prepared.classes,prepared.old_classes,prepared.full_problem,_cache_for_state(final),prepared.inherited,rec.records,audit)
        event('FINAL',dict(state_ref=audit['final_state_ref'],audit=state.audit_dict()));return state
    except (FloatingPointError,np.linalg.LinAlgError,conditional.NumericalFailure) as exc:
        if hasattr(exc,'joint_objective_audit'):cost(exc.joint_objective_audit)
        elif finalprogress:
            audit['final_head_fit_count']=finalprogress.get('head_fit_count',0);audit['final_factorization_count']=finalprogress.get('factorization_count',0)
            _add(audit,finalprogress,_GEOMETRY_COUNTERS+_KERNEL_COUNTERS+tuple('head_'+s for s in _SUFFIXES[1:])+('intercept_fit_count','intercept_addition_count','ajlr_forward_evaluation_count'))
        attemptZ=getattr(exc,'joint_attempt_Z',Z);attemptU=getattr(exc,'joint_attempt_U',U)
        audit['failure_completed_head_refs']=[rec.save('failure_completed_fold_'+str(i),U=attemptU,**_head_arrays(cf))
            for i,cf in enumerate(getattr(exc,'joint_partial_folds',()))]
        audit['failure_head_audit']=_plain(getattr(exc,'joint_forward_audit',{}))
        audit['failure_state_ref']=rec.save('failure',Z=attemptZ,U=attemptU,last_accepted_Z=Z,last_accepted_U=U,
            W=prepared.W,anchor_U=prepared.anchor_U,**getattr(exc,'joint_failed_head_arrays',{}))
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),partial_final_forward=_plain(finalprogress),fit_seconds=time.perf_counter()-start)
        error=NumericalFailure(str(exc),audit);error.records=rec.records;raise error from exc


def predict_conditional_joint_local_ridge(state,**features):
    if not isinstance(state,ConditionalJointState):raise ValueError('CONDITIONAL_JOINT state required')
    return state.predict(**features)
