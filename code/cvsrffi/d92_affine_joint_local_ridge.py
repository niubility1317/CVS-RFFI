"""AffineJoint: support-only functional anchoring with an analytic class intercept.

The fixed old physical reference measure is re-embedded on every forward.
The unpenalized intercept is fitted on legal support; it is not query calibration.
This independent structural candidate is a draft, not a published experiment.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import solve_triangular
from . import d92_function_coordinate_residual8_local_ridge as fcr
from . import d92_joint_channel_local_ridge as ch
from . import d92_branch_local_ridge as local
from . import d92_branch_interaction as interaction
from .d92_branch_ridge import _freeze, _plain

_EPS = float(np.finfo(np.float64).eps)
_NAMES = fcr._NAMES
V0 = fcr.V0
NumericalFailure = ch.NumericalFailure
_finite = ch._finite
_norm = ch._norm
latent_coordinates = fcr.latent_coordinates
reconstruct_U = fcr.reconstruct_U
initial_parameters = fcr.initial_parameters


def _readonly(value):
    """Immutable numeric storage; labels/indices/counts retain integer dtype."""
    a=np.asarray(value)
    if a.dtype.kind not in 'fibu':raise ValueError('AFFINE_JOINT state requires numeric arrays')
    dtype=np.bool_ if a.dtype.kind=='b' else np.int64 if a.dtype.kind in 'iu' else np.float64
    shape=a.shape
    a=np.ascontiguousarray(a,dtype=dtype).reshape(shape);_finite(a)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


class _Recorder:
    def __init__(self,callback):self.callback=callback;self.records={};self.keys=set()
    def save(self,key,**arrays):
        if key in self.keys:raise ValueError('Duplicate AFFINE_JOINT state record key '+key)
        self.keys.add(key);values={k:_readonly(v) for k,v in arrays.items()}
        metadata={k:dict(shape=list(v.shape),dtype=str(v.dtype),nbytes=int(v.nbytes)) for k,v in values.items()}
        if self.callback is None:
            self.records[key]=_freeze(values);ref=dict(storage='in_memory',key=key,arrays=metadata)
        else:
            ref=ch._safe(self.callback(key,values))
            if not isinstance(ref,dict):raise ValueError('state_callback must return JSON mapping')
            ref=dict(ref,key=key,arrays=metadata)
        return ref

FROZEN_CONFIG = dict(
    schema='d92_affine_joint_local_ridge_v1', method='D92-AffineJointLocalRidge-v1',
    base_algorithm=deepcopy(local.FROZEN_CONFIG), arms=['local_ridge', 'affine_joint_seq'],
    channel=dict(route='residual', mode='post_sync', equalization_enabled=False, fs_hz=25000000),
    allowed_scenarios=['practical_high', 'practical_mid', 'practical_low_urban'],
    rank=8, input_dim=736, block_dimensions=[160,96,160,160,160], kappa=.25,
    fixed_dictionary='first8_orthonormal_DCT_rows', learn_V=False,
    adapter='fixed_DCT_exact_GELU_function_coordinate_tangent_norm_preserving_residual',
    coordinate='U=anchor_U+Z@W.T;H/sqrtN=P*Sigma*R.T;W=R_retained/Sigma_retained',
    rank_rule='sigma/sigma_max>sqrt(128*eps64*max(N,8))',
    trainable_parameter_count=5888, active_parameter_count='736*retained_latent_rank_if_updated',
    initialization_B='U_zero', initialization_C_seq='exact_actual_current_B_U',
    joint_original_distance_weight=.5, joint_adapted_distance_weight=.5,
    implicit_feature_dim=123616, bandwidth='fixed_original_old_inner_train_R0',
    trace_scale='fixed_actual_B_original_old_R0_gamma',
    centering='fixed_physical_old_reference_measure_reembedded_at_current_U',
    prior_B='zero', prior_C='actual_B_function_old_columns_padded_new_columns_zero',
    target='row_class_centered_onehot_minus_1_over_C', residual_target='Y-M_no_sample_centering',
    candidate_status='STRUCTURE_CANDIDATE_DRAFT_NOT_PUBLISHED',
    free_intercept=True, intercept_penalty=0., intercept_fit='analytic_support_Schur_combined_C_plus_1_RHS',
    prior_lineage='same_new_run_row_scope_physical_fold_actual_B_only',
    ridge_coefficient=1., sample_weight=1.,
    objective='RMS_across_classes_of_cross_fold_mean_CE+0.5*Z_frobenius_squared',
    temperature=1., keep=False, guard=False,
    optimizer='normalized_gradient_first_Armijo_nonincrease_trial',
    max_iterations=4, max_trials=12, initial_step_size=.125, backtrack_factor=.5,
    armijo_coefficient=1e-4, comparison_tolerance_multiplier=128,
    inner_folds='per_class_physical_id_sort_position_mod_min_K_3',
    zero_bandwidth='original_complete_interaction_equivalence_PSD_zero_derivative',
    missing_old_scale='zero_residual_kernel_intercept_mean_E_alpha_E_minus_intercept',
    k1='no_adapter_update_full_support_closed_head', rank0='anchor_U_full_support_closed_head',
    n0='exact_reuse_actual_B', tie_break='physical_class_id_lexicographic',
    query_decision_policy='per_sample_all_registered_classes', phase1_frozen=True,
    encoder_backward=False, source_inputs=False, query_fit=False, parameter_search=False,
    dtype='float64', vector_logging='lossless_state_callback_or_records')

PREPARATION_COUNTERS = (
    'ajlr_preparation_count', 'latent_svd_count', 'dictionary_physical_evaluation_count',
    'prepared_distance_evaluation_count', 'original_distance_pair_count',
    'prior_head_fit_count', 'prior_factorization_count', 'prior_triangular_solve_count',
    'prior_triangular_rhs_count','prior_triangular_rhs_element_count','prior_triangular_dense_work_unit_count',
    'prior_intercept_fit_count','prior_intercept_addition_count',
    'prior_score_evaluation_count', 'prior_score_physical_count',
    'reference_distance_evaluation_count', 'reference_distance_pair_count',
    'raw_distance_evaluation_count', 'raw_distance_pair_count',
    'kernel_evaluation_count', 'kernel_pair_count','adapter_physical_evaluation_count')
STAGE_COUNTERS = (
    'ajlr_stage_count', 'optimizer_steps', 'optimizer_iterations',
    'inner_objective_evaluation_count', 'inner_head_fit_count', 'inner_factorization_count',
    'final_head_fit_count', 'final_factorization_count', 'head_triangular_solve_count',
    'head_triangular_rhs_count','head_triangular_rhs_element_count','head_triangular_dense_work_unit_count',
    'intercept_fit_count','intercept_addition_count',
    'derivative_triangular_solve_count', 'ce_adjoint_solve_count',
    'derivative_triangular_rhs_count','derivative_triangular_rhs_element_count','derivative_triangular_dense_work_unit_count',
    'backward_evaluation_count', 'accepted_trial_count', 'rejected_trial_count',
    'trial_count', 'trial_attempt_count', 'ajlr_forward_evaluation_count',
    'reference_distance_evaluation_count', 'reference_distance_pair_count',
    'raw_distance_evaluation_count', 'raw_distance_pair_count',
    'kernel_evaluation_count', 'kernel_pair_count', 'adapter_physical_evaluation_count')
AUDIT_COUNTERS = tuple(dict.fromkeys(PREPARATION_COUNTERS + STAGE_COUNTERS))
COUNTERS = STAGE_COUNTERS


def _scalar(value):
    return np.empty(0, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)


def _sum_class_error(a):
    return float(np.max(np.abs(a.sum(axis=1)))) if len(a) else 0.


def _unique_bytes(values):
    """Count actual root ndarray buffers once (views share their root)."""
    seen = set(); total = 0
    def visit(value):
        nonlocal total
        if isinstance(value, np.ndarray):
            root = value
            while isinstance(root.base, np.ndarray): root = root.base
            if id(root) not in seen: seen.add(id(root)); total += root.nbytes
        elif isinstance(value, Mapping):
            for x in value.values(): visit(x)
        elif isinstance(value, (tuple,list)):
            for x in value: visit(x)
    visit(values)
    return int(total)


def _nuisance(b, a, labels, class_count):
    d, s0, tau, _, reason = local._geometry(b,a,labels,class_count)
    gamma = None; sr = None
    if tau is not None and s0 > 0:
        raw = local._radial_minus_one(d,tau)
        sr = float(-2*raw[np.triu_indices(len(b),1)].sum()/len(b))
        if sr <= 0: raise FloatingPointError('AFFINE_JOINT_NONPOSITIVE_OLD_RADIAL_TRACE')
        gamma = s0/sr; _finite(gamma)
    return dict(tau=tau, gamma=gamma, s0=s0, original_radial_trace=sr,
                degeneracy_reason=reason, old_distance=d)


def _center_train(raw, q):
    """Reference differences followed by rank-one measure centering, O(N²)."""
    j = int(np.flatnonzero(q)[0]); reference = raw[j].copy(); rs = float(raw[j,j])
    diff = (raw-raw[:,j:j+1])-reference[None,:]+rs
    if np.all(q == q[0]):
        mean = diff.mean(axis=0); grand = float(mean.mean())
        center = diff-diff.mean(axis=1,keepdims=True)-mean[None,:]+grand
    else:
        mean = q@diff; grand = float(mean@q)
        center = diff-(diff@q)[:,None]-mean[None,:]+grand
    return .5*(center+center.T), reference, rs, mean, grand


def _center_cross(raw, q, reference, rs, mean, grand):
    j = int(np.flatnonzero(q)[0]); out = np.empty_like(raw)
    for i in range(len(raw)):
        diff = (raw[i]-raw[i,j])-reference+rs
        avg = float(diff.mean()) if np.all(q == q[0]) else float(diff@q)
        out[i] = diff-avg-mean+grand
    return out


def _center_vjp(barK, barL, q, gamma):
    """Complete Pq VJP, including both moving old-reference endpoints."""
    lp = barL-barL.sum(axis=1,keepdims=True)*q[None,:]
    kr = (barK-q[:,None]*barK.sum(axis=0)[None,:]
          -barK.sum(axis=1)[:,None]*q[None,:]
          +float(barK.sum())*q[:,None]*q[None,:])
    rr = gamma*(kr-q[:,None]*lp.sum(axis=0)[None,:])
    return .5*(rr+rr.T), gamma*lp


def _solve_affine_head(K,E):
    """One SPD factor and two combined-RHS solves; no inverse or jitter."""
    n,c=E.shape;rhs=np.column_stack((E,np.ones(n)))
    chol=np.linalg.cholesky(K+np.eye(n))
    solved=solve_triangular(chol.T,solve_triangular(chol,rhs,lower=True),lower=False)
    F=solved[:,:c];z=solved[:,c];s=float(z.sum())
    if not np.isfinite(s) or s<=0:raise FloatingPointError('AFFINE_JOINT_NONPOSITIVE_SCHUR')
    intercept=F.sum(axis=0)/s;alpha=np.ascontiguousarray(F-z[:,None]*intercept[None,:])
    _finite(alpha,intercept,z,s)
    return alpha,intercept,z,s,chol,rhs


def _affine_adjoint(chol,L,gscore,z,s):
    """Full saddle adjoint; e^T T is g_b, not generally zero."""
    rhs=L.T@gscore
    V=solve_triangular(chol.T,solve_triangular(chol,rhs,lower=True),lower=False)
    g_b=gscore.sum(axis=0);eta=(V.sum(axis=0)-g_b)/s
    T=V-z[:,None]*eta[None,:];_finite(T,eta,g_b)
    return T,eta,g_b,rhs


@dataclass(frozen=True)
class AffineJointProblem:
    train_context: Mapping
    held_context: Mapping
    train_labels: np.ndarray
    held_labels: np.ndarray
    classes: tuple
    q: np.ndarray
    d0: np.ndarray
    cross_d0: np.ndarray
    tau: float | None
    gamma: float | None
    s0: float
    M_train: np.ndarray
    M_held: np.ndarray
    audit: Mapping
    def __post_init__(self):
        for key in ('train_context','held_context'):
            object.__setattr__(self,key,_freeze({k:_readonly(v) for k,v in getattr(self,key).items()}))
        for key in ('train_labels','held_labels','q','d0','cross_d0','M_train','M_held'):
            object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'audit',_freeze(self.audit))


def _make_problem(b,a,labels,ids,classes,old_classes,keep,H,*,fold=None,
                  prior=None,nuisance=None,prior_score_progress=None,prior_ref=None):
    keep=np.asarray(keep,dtype=bool); ty=labels[keep]; hy=labels[~keep]
    train_ids=tuple(x for i,x in enumerate(ids) if keep[i]); held_ids=tuple(x for i,x in enumerate(ids) if not keep[i])
    oldix=np.array([classes[int(y)] in old_classes for y in ty],dtype=bool)
    if not np.any(oldix): raise ValueError('Old inner-train reference required')
    q=oldix.astype(float)/int(oldix.sum())
    old_labels=np.array([old_classes.index(classes[int(y)]) for y in ty[oldix]])
    nu=_nuisance(b[keep][oldix],a[keep][oldix],old_labels,len(old_classes)) if nuisance is None else nuisance
    d0=local._distances(b[keep],a[keep]); cross=local._distances(b[~keep],a[~keep],b[keep],a[keep])
    mt=np.zeros((len(ty),len(classes))); mh=np.zeros((len(hy),len(classes)))
    if prior is not None:
        if isinstance(prior,AffineJointState):
            pt=_score_head_geometry(prior.problem,prior.final_cache,prior.U,b[keep],a[keep],prior_score_progress,H[keep])
            ph=_score_head_geometry(prior.problem,prior.final_cache,prior.U,b[~keep],a[~keep],prior_score_progress,H[~keep])
            pc=prior.classes
        else:
            pp,pu,pf=prior
            pt=_score_head_geometry(pp,pf,pu,b[keep],a[keep],prior_score_progress,H[keep])
            ph=_score_head_geometry(pp,pf,pu,b[~keep],a[~keep],prior_score_progress,H[~keep]);pc=pp.classes
        for j,name in enumerate(pc):
            mt[:,classes.index(name)]=pt[:,j]; mh[:,classes.index(name)]=ph[:,j]
    audit=dict(inner_fold=fold,classes=list(classes),old_classes=list(old_classes),
        training_physical_ids=list(train_ids),held_physical_ids=list(held_ids),
        old_reference_physical_ids=[train_ids[i] for i in np.flatnonzero(oldix)],
        train_physical_count=len(ty),held_physical_count=len(hy),old_reference_count=int(oldix.sum()),
        bandwidth_tau=nu['tau'],trace_scale=nu['gamma'],interaction_centered_trace=nu['s0'],
        prior_ref=prior_ref,
        all_head_statistics_from_old_inner_train_only=True,
        prior_source='ZERO' if prior is None else ('ACTUAL_FINAL_B' if isinstance(prior,AffineJointState) else 'FROZEN_ACTUAL_B_U_OLD_INNER_TRAIN_HEAD'))
    return AffineJointProblem(fcr._context(b[keep],a[keep],H[keep]),fcr._context(b[~keep],a[~keep],H[~keep]),
        ty,hy,tuple(classes),q,d0,cross,nu['tau'],nu['gamma'],nu['s0'],mt,mh,audit)


def _forward(problem,U,progress=None):
    start=time.perf_counter(); p=problem; n=len(p.train_labels); c=len(p.classes); h=len(p.held_labels)
    audit={} if progress is None else progress
    audit.update(_plain(p.audit),head_fit_count=1,factorization_count=0,head_triangular_solve_count=0,
        head_triangular_rhs_count=0,head_triangular_rhs_element_count=0,head_triangular_dense_work_unit_count=0,
        intercept_fit_count=1,intercept_addition_count=(n+h)*c,
        ajlr_forward_evaluation_count=1,reference_distance_evaluation_count=0,reference_distance_pair_count=0,
        raw_distance_evaluation_count=0,raw_distance_pair_count=0,kernel_evaluation_count=0,kernel_pair_count=0,
        adapter_physical_evaluation_count=n+h)
    b,a,bc=fcr._adapt(p.train_context,U,derivative_cache=True)
    hb,ha,hc=fcr._adapt(p.held_context,U,derivative_cache=True)
    # The zero-bandwidth branch uses ORIGINAL complete feature equivalence.
    identity=bool(np.all(U==0)) or p.tau==0
    if identity: distance,cross=p.d0,p.cross_d0
    else:
        ad=local._distances(b,a); ac=local._distances(hb,ha,b,a)
        distance=.5*p.d0+.5*ad; cross=.5*p.cross_d0+.5*ac
        m=int(np.count_nonzero(p.q))
        audit.update(raw_distance_evaluation_count=1+int(h>0),raw_distance_pair_count=n*(n-1)//2+h*n,
                     reference_distance_evaluation_count=1+int(h>0),
                     reference_distance_pair_count=m*(m-1)//2+m*(n-m)+h*m)
    Y=np.eye(c)[np.asarray(p.train_labels,dtype=int)]-1/c; E=Y-p.M_train
    K=np.zeros((n,n)); L=np.zeros((h,n)); raw=np.zeros((n,n)); crossraw=np.zeros((h,n))
    radial=np.zeros((n,n)); crossrad=np.zeros((h,n)); ref=mean=np.zeros(n); rs=grand=0.
    chol=np.empty((0,0));rhs=np.column_stack((E,np.ones(n)));z=np.ones(n);s=float(n)
    intercept=E.mean(axis=0)
    # The mathematical zero intercept for balanced zero-prior heads is exact.
    # Preserve the specified exact tie in the missing-old-information branch.
    if p.gamma is None and not np.any(p.M_train) and np.all(np.bincount(np.asarray(p.train_labels,dtype=int),minlength=c)==n//c):
        intercept=np.zeros(c)
    alpha=E-intercept[None,:]
    if p.gamma is not None:
        raw=local._radial_minus_one(distance,p.tau); crossraw=local._radial_minus_one(cross,p.tau)
        radial=local._radial(distance,p.tau); crossrad=local._radial(cross,p.tau)
        center,ref,rs,mean,grand=_center_train(raw,p.q); K=p.gamma*center
        L=p.gamma*_center_cross(crossraw,p.q,ref,rs,mean,grand)
        audit.update(kernel_evaluation_count=1+int(h>0),kernel_pair_count=n*n+h*n,
                     factorization_count=1,head_triangular_solve_count=2,
                     head_triangular_rhs_count=2*(c+1),head_triangular_rhs_element_count=2*n*(c+1),
                     head_triangular_dense_work_unit_count=2*n*n*(c+1))
        alpha,intercept,z,s,chol,rhs=_solve_affine_head(K,E)
    train_scores=p.M_train+K@alpha+intercept[None,:]
    scores=np.stack([p.M_held[i]+L[i]@alpha+intercept for i in range(h)]) if h else np.empty((0,c))
    actual_trace=float(np.trace(K)); denom=(1+actual_trace)*_norm(alpha)+np.sqrt(n)*_norm(intercept)+_norm(E)
    normal=_norm((K+np.eye(n))@alpha+intercept[None,:]-E); residual=normal/denom if denom else 0.
    tol=128*_EPS*max(n,c)
    if residual>tol: raise FloatingPointError('AFFINE_JOINT_HEAD_RESIDUAL_EXCEEDED')
    sample_sum=_norm(alpha.sum(axis=0));sample_denom=np.sqrt(n)*_norm(alpha)
    sample_residual=sample_sum/sample_denom if sample_denom else 0.
    if sample_residual>tol:raise FloatingPointError('AFFINE_JOINT_ALPHA_SAMPLE_SUM_EXCEEDED')
    schur_tol=tol*max(1.,float(n),abs(s));schur_lower=n/(1+actual_trace)
    if not np.isfinite(s) or s<=0 or s<schur_lower-schur_tol or s>n+schur_tol:
        raise FloatingPointError('AFFINE_JOINT_SCHUR_BOUND_EXCEEDED')
    classerr=max(_sum_class_error(x) for x in (Y,p.M_train,p.M_held,E,alpha,intercept[None,:],train_scores,scores))
    classbound=tol*max(1.,_norm(alpha),_norm(train_scores),_norm(scores))
    if classerr>classbound: raise FloatingPointError('AFFINE_JOINT_CLASS_SUM_RESIDUAL_EXCEEDED')
    reference_mean=p.q@(K@alpha); reference_error=_norm(reference_mean)
    if reference_error>tol*max(1.,_norm(K)*_norm(alpha)):
        raise FloatingPointError('AFFINE_JOINT_REFERENCE_MEAN_RESIDUAL_EXCEEDED')
    fit=K@alpha; error=train_scores-Y
    angles=[]; norm_error=0.
    for ctx,cache in ((p.train_context,bc),(p.held_context,hc)):
        for j,sl in enumerate(fcr._SLICES):
            valid=ctx['rho'][:,j]>0
            chord=np.linalg.norm(cache['y'][valid,sl]-ctx['w'][valid,sl],axis=1)
            angles.extend((2*np.arcsin(np.minimum(1.,chord/2))).tolist())
            original=ctx['original'][:,sl]
            # Norm preservation is already checked blockwise inside the adapter.
            if len(original): norm_error=max(norm_error,float(np.max(np.abs(np.linalg.norm(cache['y'][:,sl],axis=1)[valid]-1))) if np.any(valid) else 0.)
    audit.update(actual_kernel_trace=actual_trace,normal_equation_residual=residual,
        normal_equation_absolute_residual=normal,residual_denominator='(1+actual_trace_K)*norm(alpha)+sqrt(n)*norm(intercept)+norm(E)',
        sample_sum_alpha_norm=sample_sum,sample_sum_alpha_residual=sample_residual,
        schur_s=s,schur_s_lower_bound=schur_lower,schur_s_upper_bound=float(n),schur_s_tolerance=schur_tol,
        intercept_norm=_norm(intercept),analytic_intercept_parameter_count=c,analytic_intercept_contrast_count=c-1,
        numerical_tolerance=tol,condition_bound=1+actual_trace,gram_min_eigenvalue_lower_bound=1.,
        class_sum_max_abs=classerr,class_sum_tolerance=classbound,old_reference_residual_mean_norm=reference_error,
        residual_sample_mean=_plain(E.mean(axis=0)),fitted_old_reference_mean=_plain(p.q@train_scores),
        expected_old_reference_mean=_plain(p.q@p.M_train+intercept),
        head_training_loss_data=float(.5*np.sum(error*error)),head_training_loss_ridge=float(.5*np.sum(alpha*fit)),
        block_angle_radians=fcr._statistics(angles),block_unit_norm_max_error=norm_error,
        kernel_frobenius_norm=_norm(K),identity_forward=identity,
        status='NO_OLD_KERNEL_INFORMATION' if p.gamma is None else 'CLOSED_FORM_SOLVED',
        forward_seconds=time.perf_counter()-start)
    audit['head_training_loss_total']=audit['head_training_loss_data']+audit['head_training_loss_ridge']
    _finite(K,L,alpha,intercept,z,s,scores,train_scores)
    return dict(problem=p,U=np.array(U,copy=True),b=b,a=a,hb=hb,ha=ha,bc=bc,hc=hc,
        distance=distance,cross_distance=cross,raw=raw,crossraw=crossraw,radial=radial,crossrad=crossrad,
        K=K,L=L,chol=chol,alpha=alpha,Y=Y,E=E,score=scores,train_scores=train_scores,
        intercept=intercept,schur_z=z,schur_s=s,combined_rhs=rhs,
        reference=ref,reference_self=rs,mean=mean,grand=grand,audit=audit)


def _backward(cache,gscore,progress):
    start=time.perf_counter(); p=cache['problem']
    if p.gamma is None or p.tau==0 or not len(gscore):
        progress['adjoint_seconds']=time.perf_counter()-start
        return np.zeros((736,8))
    alpha=cache['alpha']; chol=cache['chol']; L=cache['L']
    T,eta,g_b,rhs=_affine_adjoint(chol,L,gscore,cache['schur_z'],cache['schur_s'])
    progress['derivative_triangular_solve_count']+=2; progress['ce_adjoint_solve_count']+=1
    n,c=alpha.shape
    for key,value in dict(derivative_triangular_rhs_count=2*c,derivative_triangular_rhs_element_count=2*n*c,
                         derivative_triangular_dense_work_unit_count=2*n*n*c).items():progress[key]=progress.get(key,0)+value
    progress.update(adjoint_g_b_norm=_norm(g_b),adjoint_sample_sum_residual=_norm(T.sum(axis=0)-g_b))
    cache['_adjoint_arrays']=dict(adjoint_T=T,adjoint_eta=eta,adjoint_g_b=g_b,adjoint_rhs=rhs)
    barL=gscore@alpha.T; barK=-(T@alpha.T); barK=.5*(barK+barK.T)
    barR,barQ=_center_vjp(barK,barL,p.q,p.gamma)
    dd=np.zeros_like(barR); dc=np.zeros_like(barQ)
    active=cache['radial']>0; activec=cache['crossrad']>0
    dd[active]=-barR[active]*cache['radial'][active]/p.tau
    dc[activec]=-barQ[activec]*cache['crossrad'][activec]/p.tau
    gb,ga,_,_=ch._distance_vjp(cache['b'],cache['a'],cache['b'],cache['a'],.5*dd,symmetric=True)
    ghb,gha,gtb,gta=ch._distance_vjp(cache['hb'],cache['ha'],cache['b'],cache['a'],.5*dc)
    gu=fcr._adapter_vjp(p.train_context,cache['U'],gb+gtb,ga+gta,cache['bc'])
    gu+=fcr._adapter_vjp(p.held_context,cache['U'],ghb,gha,cache['hc'])
    progress['adjoint_seconds']=time.perf_counter()-start
    return gu


def _head_arrays(cache):
    p=cache['problem']
    return dict(original_train_b=p.train_context['original'][:,:256],original_train_a=p.train_context['original'][:,256:],
        original_held_b=p.held_context['original'][:,:256],original_held_a=p.held_context['original'][:,256:],
        adapted_train_b=cache['b'],adapted_train_a=cache['a'],adapted_held_b=cache['hb'],adapted_held_a=cache['ha'],
        train_labels=p.train_labels,held_labels=p.held_labels,q=p.q,raw_train=cache['radial'],raw_cross=cache['crossrad'],
        raw_train_minus_one=cache['raw'],raw_cross_minus_one=cache['crossraw'],distance=cache['distance'],cross_distance=cache['cross_distance'],
        K=cache['K'],L=cache['L'],chol=cache['chol'],alpha=cache['alpha'],Y=cache['Y'],M_train=p.M_train,M_held=p.M_held,E=cache['E'],
        intercept=cache['intercept'],schur_z=cache['schur_z'],schur_s=_scalar(cache['schur_s']),combined_rhs=cache['combined_rhs'],
        train_scores=cache['train_scores'],scores=cache['score'],reference_kernel=cache['reference'],reference_self=_scalar(cache['reference_self']),
        center_mean=cache['mean'],center_grand=_scalar(cache['grand']),tau=_scalar(p.tau),gamma=_scalar(p.gamma),s0=_scalar(p.s0),
        actual_trace=_scalar(cache['audit']['actual_kernel_trace']))


def _score_head_geometry(problem,cache,U,b0,a0,progress=None,H=None):
    start=time.perf_counter()
    out=np.broadcast_to(cache['intercept'],(len(b0),len(problem.classes))).copy()
    if problem.gamma is None:
        if progress is not None:progress['intercept_addition_count']=progress.get('intercept_addition_count',0)+out.size
        return out
    for i in range(len(b0)):
        if problem.tau==0:
            d=local._distances(b0[i:i+1],a0[i:i+1],problem.train_context['original'][:,:256],problem.train_context['original'][:,256:])
        else:
            b,a,_=fcr._adapt(fcr._context(b0[i:i+1],a0[i:i+1],None if H is None else H[i:i+1]),U)
            d0=local._distances(b0[i:i+1],a0[i:i+1],problem.train_context['original'][:,:256],problem.train_context['original'][:,256:])
            d=.5*d0+.5*local._distances(b,a,cache['b'],cache['a']) if np.any(U) else d0
        cross=_center_cross(local._radial_minus_one(d,problem.tau),problem.q,cache['reference'],cache['reference_self'],cache['mean'],cache['grand'])
        out[i]=(problem.gamma*cross[0])@cache['alpha']+cache['intercept']
    if progress is not None:
        n=len(b0);m=len(problem.train_labels);r=int(np.count_nonzero(problem.q))
        dc=n*(1+int(problem.tau!=0 and bool(np.any(U))))
        for key,value in dict(raw_distance_evaluation_count=dc,raw_distance_pair_count=dc*m,
            reference_distance_evaluation_count=dc,reference_distance_pair_count=dc*r,
            kernel_evaluation_count=n,kernel_pair_count=n*m,
            intercept_addition_count=n*len(problem.classes),
            adapter_physical_evaluation_count=n if problem.tau!=0 else 0,
            dictionary_physical_evaluation_count=n if problem.tau!=0 and H is None else 0).items():
            progress[key]=progress.get(key,0)+value
        progress['prior_score_seconds']=progress.get('prior_score_seconds',0.)+time.perf_counter()-start
    _finite(out);return out


@dataclass(frozen=True)
class AffineJointTraining:
    raw: Mapping
    labels: np.ndarray
    ids: tuple
    classes: tuple
    old_classes: tuple
    background: np.ndarray
    auxiliary: np.ndarray
    H: np.ndarray
    W: np.ndarray
    singular_values: np.ndarray
    r: int
    anchor_U: np.ndarray
    inherited: object
    problems: tuple
    full_problem: AffineJointProblem
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        for key in ('labels','background','auxiliary','H','W','singular_values','anchor_U'):
            object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'raw',_freeze({k:_readonly(v) for k,v in self.raw.items()}))
        object.__setattr__(self,'records',_freeze(self.records));object.__setattr__(self,'audit',_freeze(self.audit))
    def audit_dict(self):return ch._safe(_plain(self.audit))
    def state_records(self):return {k:{n:np.array(v,copy=True) for n,v in a.items()} for k,a in self.records.items()}


def prepare_affine_joint_training(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,
        classes,old_classes,inherited=None,context=None,log_callback=None,state_callback=None):
    start=time.perf_counter();rec=_Recorder(state_callback);audit=dict(context or {})
    audit.update({key:0 for key in PREPARATION_COUNTERS});audit['ajlr_preparation_count']=1
    b,a,y,ids,canonical,_,old,k=interaction._prepare(z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes)
    old=tuple(sorted(old));n=len(ids)
    if not old:raise ValueError('Old support classes required')
    order=np.asarray(sorted(range(n),key=lambda i:support_ids[i]))
    raw={key:np.asarray(v,dtype=np.float64)[order] for key,v in zip(_NAMES,(z_id,fft,t_emb,f_emb,pa_local))}
    if inherited is None:
        if set(canonical)!=set(old):raise ValueError('C requires actual current B state')
        anchor=np.zeros((736,8))
    else:
        if not isinstance(inherited,AffineJointState) or inherited.audit['mode']!='B' or inherited.classes!=old:
            raise ValueError('C requires actual old-only AFFINE_JOINT B state')
        previous=inherited.audit['preparation']
        for key in ('run_id','row_id','scope'):
            if key not in audit or key not in previous or not audit[key] or not previous[key]:
                raise ValueError('C actual B lineage requires explicit '+key)
            if audit[key]!=previous[key]:raise ValueError('C lineage crosses '+key)
        for key in ('split_id','fold'):
            if (key in audit)!=(key in previous) or (key in audit and audit[key]!=previous[key]):
                raise ValueError('C lineage crosses '+key)
        ix=[i for i,v in enumerate(y) if canonical[int(v)] in old]
        if tuple(ids[i] for i in ix)!=inherited.ids:raise ValueError('C old physical IDs differ from B')
        if any(canonical[int(y[i])]!=inherited.classes[int(inherited.labels[j])] for j,i in enumerate(ix)):
            raise ValueError('C old labels differ from B')
        if any(not np.array_equal(raw[key][ix],inherited.raw[key]) for key in _NAMES):
            raise ValueError('C old raw features differ from B')
        anchor=inherited.U.copy()
    audit.update(training_physical_ids=list(ids),classes=list(canonical),old_classes=list(old),train_k=k,
                 train_physical_count=n,inherited_state=inherited is not None)
    if inherited is not None and canonical==old:
        audit.update(ajlr_preparation_count=0,no_information=True,no_information_reason='N0_REUSE_ACTUAL_B',
            latent_rank=inherited.W.shape[1],preparation_seconds=time.perf_counter()-start,
            prepared_state_ref=_plain(inherited.audit['preparation']['prepared_state_ref']),
            inner_folds=[],prior_folds=[],final_problem=_plain(inherited.problem.audit),
            prepared_numeric_state_bytes=0,retained_vector_record_bytes=0)
        return AffineJointTraining(raw,y,ids,canonical,old,b,a,inherited.H,inherited.W,
            inherited.singular_values,inherited.W.shape[1],anchor,inherited,(),inherited.problem,{},audit)
    try:
        H=fcr._context(b,a)['h']; audit['dictionary_physical_evaluation_count']=n
        W,s,ci=latent_coordinates(H);audit.update(ci);r=W.shape[1]
        audit['prepared_state_ref']=rec.save('prepared_coordinates',H=H,W=W,singular_values=s,anchor_U=anchor,V0=V0)
        problems=[];prior_folds=[]
        n0=inherited is not None and canonical==old
        def make(keep,fold,final=False):
            prior=None;nu=None;prior_ref=None;score_progress={}
            if inherited is not None:
                if final:
                    prior=inherited;nu=dict(tau=inherited.problem.tau,gamma=inherited.problem.gamma,s0=inherited.problem.s0)
                    prior_ref=_plain(inherited.audit['final_state_ref'])
                else:
                    oldmask=np.array([canonical[int(v)] in old for v in y])
                    oi=np.flatnonzero(oldmask);old_y=np.array([old.index(canonical[int(v)]) for v in y[oi]])
                    op=_make_problem(b[oi],a[oi],old_y,tuple(ids[i] for i in oi),old,old,keep[oi],H[oi],fold=fold)
                    pf=_forward(op,anchor);prior=(op,anchor,pf)
                    audit['prior_head_fit_count']+=1;audit['prior_factorization_count']+=pf['audit']['factorization_count']
                    audit['prior_triangular_solve_count']+=pf['audit']['head_triangular_solve_count']
                    for suffix in ('rhs_count','rhs_element_count','dense_work_unit_count'):
                        audit['prior_triangular_'+suffix]+=pf['audit']['head_triangular_'+suffix]
                    audit['prior_intercept_fit_count']+=pf['audit']['intercept_fit_count']
                    audit['prior_intercept_addition_count']+=pf['audit']['intercept_addition_count']
                    for key in ('raw_distance_evaluation_count','raw_distance_pair_count','reference_distance_evaluation_count','reference_distance_pair_count','kernel_evaluation_count','kernel_pair_count','adapter_physical_evaluation_count'):
                        audit[key]+=pf['audit'].get(key,0)
                    ref=rec.save('prior_head_'+str(fold),U=anchor,**_head_arrays(pf))
                    prior_ref=ref
                    prior_folds.append(dict(op.audit,head_ref=ref,head_state_ref=ref,final_fit=pf['audit']))
                    nu=dict(tau=op.tau,gamma=op.gamma,s0=op.s0)
                    m=int(keep[oi].sum());audit['prepared_distance_evaluation_count']+=3
                    audit['original_distance_pair_count']+=m*(m-1)+(len(oi)-m)*m
                audit['prior_score_evaluation_count']+=1+int(np.any(~keep));audit['prior_score_physical_count']+=n
            p=_make_problem(b,a,y,ids,canonical,old,keep,H,fold=fold,prior=prior,nuisance=nu,
                            prior_score_progress=score_progress,prior_ref=prior_ref)
            for key,value in score_progress.items():
                target='prior_intercept_addition_count' if key=='intercept_addition_count' else key
                audit[target]=audit.get(target,0)+value
            nt=int(keep.sum());m=int(np.count_nonzero(p.q))
            audit['prepared_distance_evaluation_count']+=2+(1 if nu is None else 0)
            audit['original_distance_pair_count']+=nt*(nt-1)//2+(n-nt)*nt+(m*(m-1)//2 if nu is None else 0)
            audit['reference_distance_evaluation_count']+=1 if nu is None else 0
            audit['reference_distance_pair_count']+=m*(m-1)//2 if nu is None else 0
            return p
        if not n0 and k>1:
            assignment=np.empty(n,dtype=int);folds=min(k,3)
            for cls in range(len(canonical)):
                ix=np.flatnonzero(y==cls);assignment[ix]=np.arange(len(ix))%folds
            for fold in range(folds):problems.append(make(assignment!=fold,fold))
        if n0:
            full=inherited.problem
        else:full=make(np.ones(n,dtype=bool),None,final=True)
        reason=('N0_REUSE_ACTUAL_B' if n0 else 'PHYSICAL_K1' if k==1 else 'ZERO_DICTIONARY_RANK' if r==0
                else 'NO_OLD_KERNEL_INFORMATION' if full.gamma is None else 'ALL_INNER_GEOMETRY_DEGENERATE'
                if not any(p.gamma is not None and p.tau is not None and p.tau>0 for p in problems) else None)
        audit.update(no_information=reason is not None,no_information_reason=reason,
            inner_folds=[_plain(p.audit) for p in problems],prior_folds=prior_folds,
            final_problem=_plain(full.audit),latent_rank=r,preparation_seconds=time.perf_counter()-start,
            prepared_numeric_state_bytes=_unique_bytes((raw,b,a,y,H,W,s,anchor)),
            retained_vector_record_bytes=_unique_bytes(rec.records),source_inputs=False,query_fit=False,
            reference_distance_scope='subset_of_raw_distance_work_not_additive')
        ch._emit(log_callback,'AFFINE_JOINT_PREPARED',audit)
        return AffineJointTraining(raw,y,ids,canonical,old,b,a,H,W,s,r,anchor,inherited,tuple(problems),full,rec.records,audit)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),preparation_seconds=time.perf_counter()-start)
        error=NumericalFailure(str(exc),audit);error.records=rec.records;raise error from exc


@dataclass(frozen=True)
class ObjectiveCache:
    prepared: AffineJointTraining
    Z: np.ndarray
    anchor_U: np.ndarray
    folds: tuple


def _ce(scores,labels):
    if not len(scores):return np.empty(0),np.empty_like(scores)
    top=scores.max(axis=1,keepdims=True);ex=np.exp(scores-top);den=ex.sum(axis=1,keepdims=True)
    values=np.log(den[:,0])+top[:,0]-scores[np.arange(len(scores)),labels]
    grad=ex/den;grad[np.arange(len(scores)),labels]-=1
    _finite(values,grad);return values,grad


def evaluate_affine_joint_objective(prepared,Z,anchor_U=None,*,gradient=True,forward_cache=None):
    start=time.perf_counter();anchor=prepared.anchor_U if anchor_U is None else fcr._parameter(anchor_U)
    if not np.array_equal(anchor,prepared.anchor_U):raise ValueError('AFFINE_JOINT objective anchor binding differs from actual B')
    Z=np.asarray(Z,dtype=np.float64);U=reconstruct_U(Z,anchor,prepared.W);c=len(prepared.classes)
    reused=forward_cache is not None
    if reused:
        if (forward_cache.prepared is not prepared or not np.array_equal(forward_cache.Z,Z)
                or not np.array_equal(forward_cache.anchor_U,anchor)):
            raise ValueError('AFFINE_JOINT objective cache binding mismatch')
        folds=forward_cache.folds
    else:folds=tuple(_forward(p,U) for p in prepared.problems)
    cache=forward_cache if reused else ObjectiveCache(prepared,Z.copy(),anchor.copy(),folds)
    sums=np.zeros(c);counts=np.zeros(c,dtype=np.int64);values=[];derivatives=[]
    for cf in folds:
        labels=np.asarray(cf['problem'].held_labels,dtype=int);ce,g=_ce(cf['score'],labels)
        sums+=np.bincount(labels,weights=ce,minlength=c);counts+=np.bincount(labels,minlength=c)
        values.append(ce);derivatives.append(g)
    means=np.divide(sums,counts,out=np.zeros(c),where=counts>0);risk=float(_norm(means)/np.sqrt(c))
    prox=float(.5*np.sum(Z*Z));gU=np.zeros((736,8))
    info={key:0 for key in STAGE_COUNTERS}
    info.update(inner_objective_evaluation_count=0 if reused else 1,
        inner_head_fit_count=0 if reused else len(folds),
        inner_factorization_count=0 if reused else sum(x['audit']['factorization_count'] for x in folds),
        head_triangular_solve_count=0 if reused else sum(x['audit']['head_triangular_solve_count'] for x in folds),
        ajlr_forward_evaluation_count=0 if reused else len(folds),
        backward_evaluation_count=int(gradient),forward_cache_reused=reused)
    for key in ('raw_distance_evaluation_count','raw_distance_pair_count','reference_distance_evaluation_count','reference_distance_pair_count','kernel_evaluation_count','kernel_pair_count','adapter_physical_evaluation_count'):
        info[key]=0 if reused else sum(x['audit'].get(key,0) for x in folds)
    for key in ('head_triangular_rhs_count','head_triangular_rhs_element_count','head_triangular_dense_work_unit_count',
                'intercept_fit_count','intercept_addition_count'):
        info[key]=0 if reused else sum(x['audit'].get(key,0) for x in folds)
    fold_audits=[]
    for cf,ce,d in zip(folds,values,derivatives):
        labels=np.asarray(cf['problem'].held_labels,dtype=int);fa=deepcopy(cf['audit'])
        fs=np.bincount(labels,weights=ce,minlength=c);fc=np.bincount(labels,minlength=c)
        fa.update(held_ce_sums=fs.tolist(),held_ce_counts=fc.tolist(),
            held_correct_count=int(np.sum(cf['score'].argmax(axis=1)==labels)),
            derivative_triangular_solve_count=0,ce_adjoint_solve_count=0)
        if gradient and risk>0:
            weight=means[labels]/(c*risk*counts[labels]);gU+=_backward(cf,d*weight[:,None],fa)
            info['derivative_triangular_solve_count']+=fa['derivative_triangular_solve_count']
            info['ce_adjoint_solve_count']+=fa['ce_adjoint_solve_count']
            for suffix in ('rhs_count','rhs_element_count','dense_work_unit_count'):
                key='derivative_triangular_'+suffix;info[key]+=fa.get(key,0)
        fold_audits.append(fa)
    gZ=gU@prepared.W+Z if gradient else np.zeros_like(Z)
    measurement=fcr._functional_measurement(prepared,U,anchor,Z)
    info.update(loss_ce=risk,loss_task=risk,RMSCE=risk,loss_proximal=prox,loss_total=risk+prox,
        class_ce_sums=sums.tolist(),class_ce_counts=counts.tolist(),class_ce_means=means.tolist(),
        gradient_norm=_norm(gZ) if gradient else None,inner_folds=fold_audits,
        forward_cache_bytes=_unique_bytes([dict((k,v) for k,v in x.items() if isinstance(v,(np.ndarray,Mapping))) for x in folds]),
        objective_seconds=time.perf_counter()-start,temperature=1.,**measurement)
    _finite(risk,prox,gZ)
    return risk+prox,gZ,info,cache


def _trial_acceptance(before,after,dot):
    tol=128*_EPS*max(1.,abs(before),abs(after),abs(before+1e-4*dot))
    armijo=after<=before+1e-4*dot+tol;nonincrease=after<=before+tol
    return dict(accepted=bool(armijo and nonincrease),armijo_pass=bool(armijo),
                objective_nonincrease_pass=bool(nonincrease),comparison_tolerance=tol,gradient_dot_delta=dot)


@dataclass(frozen=True)
class AffineJointState:
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
    problem: AffineJointProblem
    final_cache: Mapping
    prior: object
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        for key in ('U','Z','W','anchor_U','H','singular_values','labels'):
            object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'raw',_freeze({k:_readonly(v) for k,v in self.raw.items()}))
        # Deployment cache contains only immutable numeric head state, no derivative cache.
        object.__setattr__(self,'final_cache',_freeze({k:_readonly(v) if isinstance(v,np.ndarray) else v for k,v in self.final_cache.items()}))
        object.__setattr__(self,'records',_freeze(self.records));object.__setattr__(self,'audit',_freeze(self.audit))
    @property
    def V(self):return V0
    @property
    def V0(self):return V0
    def audit_dict(self):return ch._safe(_plain(self.audit))
    def state_records(self):return {k:{n:np.array(v,copy=True) for n,v in a.items()} for k,a in self.records.items()}
    def _score_geometry(self,b,a,progress=None):
        residual={};prior_progress={}
        out=_score_head_geometry(self.problem,self.final_cache,self.U,b,a,residual)
        if self.prior is not None:
            old=_score_head_geometry(self.prior.problem,self.prior.final_cache,self.prior.U,b,a,prior_progress)
            for j,name in enumerate(self.prior.classes):out[:,self.classes.index(name)]+=old[:,j]
        if progress is not None:
            progress.update(residual=residual,prior=prior_progress)
            for key in set(residual)|set(prior_progress):progress[key]=residual.get(key,0)+prior_progress.get(key,0)
        _finite(out);return out
    def score(self,*,z_id,fft,t_emb,f_emb,pa_local):
        b,a=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True)
        return self._score_geometry(b,a)
    def score_with_audit(self,*,z_id,fft,t_emb,f_emb,pa_local):
        start=time.perf_counter()
        b,a=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True)
        feature_seconds=time.perf_counter()-start;progress={}
        out=self._score_geometry(b,a,progress)
        progress.update(score_physical_count=len(b),feature_geometry_seconds=feature_seconds,
                        score_seconds=time.perf_counter()-start,
                        reference_distance_scope='subset_of_raw_distance_work_not_additive')
        return out,progress
    def predict(self,**features):
        order=np.asarray(sorted(range(len(self.classes)),key=lambda i:self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(self.score(**features)[:,order],axis=1)]]
    def to_arrays(self):
        arrays=_head_arrays(dict(self.final_cache,problem=self.problem,U=self.U))
        arrays.update(self.raw,U=self.U,Z=self.Z,W=self.W,anchor_U=self.anchor_U,V0=V0,H=self.H,singular_values=self.singular_values)
        if self.prior is not None:
            arrays.update({'prior_B_'+k:v for k,v in self.prior.to_arrays().items()})
            arrays['prior_old_class_indices']=np.asarray([self.classes.index(name) for name in self.prior.classes],dtype=np.int64)
        return arrays
    def _resident_values(self):
        values=(self.U,self.Z,self.W,self.anchor_U,self.H,self.singular_values,self.raw,self.labels,V0,
                {k:v for k,v in vars(self.problem).items() if k!='audit'},
                {k:v for k,v in self.final_cache.items() if k!='audit'})
        return values if self.prior is None else values+(self.prior._resident_values(),)
    def _deployment_values(self):
        values=(self.U,V0,self.raw,self.labels,self.problem.q,self.problem.train_context['original'],
                self.final_cache['b'],self.final_cache['a'],self.final_cache['alpha'],
                self.final_cache['reference'],self.final_cache['mean'],self.final_cache['intercept'])
        return values if self.prior is None else values+(self.prior._deployment_values(),)


def _cache_for_state(cache):
    names=('U','b','a','hb','ha','distance','cross_distance','raw','crossraw','radial','crossrad',
           'K','L','chol','alpha','Y','E','intercept','schur_z','schur_s','combined_rhs',
           'score','train_scores','reference','reference_self','mean','grand','audit')
    return {key:cache[key] for key in names}


def fit_affine_joint_local_ridge(prepared,*,mode='B',baseline_state=None,log_callback=None,state_callback=None):
    if mode not in ('B','C_seq'):raise ValueError('Only B and C_seq are defined')
    if (mode=='B')!=(prepared.inherited is None):raise ValueError('B/C AFFINE_JOINT lineage mismatch')
    if prepared.inherited is not None and prepared.classes==prepared.old_classes:return prepared.inherited
    start=time.perf_counter();rec=_Recorder(state_callback);Z=np.zeros((736,prepared.r));U=prepared.anchor_U.copy()
    audit={key:0 for key in STAGE_COUNTERS}
    audit.update(ajlr_stage_count=1,mode=mode,classes=list(prepared.classes),old_classes=list(prepared.old_classes),
        training_physical_ids=list(prepared.ids),train_k=prepared.audit['train_k'],preparation=prepared.audit_dict(),
        no_information=prepared.audit['no_information'],steps=[],trials=[],gradients=[],initial_objective=None,final_objective=None,
        maximum_trainable_parameter_count=5888,coordinate_parameter_count=736*prepared.r,
        trainable_parameter_count=0 if prepared.audit['no_information'] else 736*prepared.r,
        latent_rank=prepared.r,source_validation=None,source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        optimizer_state_reset=True,optimizer_state_bytes=0,baseline_reused=False,
        hardware_dtype='CPU_float64',config=deepcopy(FROZEN_CONFIG))
    initial_cache=None;cache=None;finalprogress={}
    def cost(obj):
        for key in STAGE_COUNTERS:audit[key]+=obj.get(key,0)
    def save_objective(key,obj,oc,**extra):
        refs=[]
        if oc is not None:
            for j,cf in enumerate(oc.folds):
                if '_head_ref' not in cf:
                    cf['_head_ref']=rec.save(key+'_head_'+str(j),U=cf['U'],**_head_arrays(cf))
                refs.append(cf['_head_ref'])
            scores=np.concatenate([f['score'] for f in oc.folds]) if oc.folds else np.empty((0,len(prepared.classes)))
            labels=np.concatenate([f['problem'].held_labels for f in oc.folds]) if oc.folds else np.empty(0,dtype=np.int64)
        else:scores=np.empty((0,len(prepared.classes)));labels=np.empty(0,dtype=np.int64)
        arrays=dict(Z=Z,U=U,anchor_U=prepared.anchor_U,W=prepared.W,singular_values=prepared.singular_values,
            scores=scores,labels=labels,class_ce_sums=np.asarray(obj.get('class_ce_sums',[])),
            class_ce_counts=np.asarray(obj.get('class_ce_counts',[]),dtype=np.int64),class_ce_means=np.asarray(obj.get('class_ce_means',[])),
            RMSCE=_scalar(obj.get('RMSCE')),prox=_scalar(obj.get('loss_proximal')))
        if 'g_Z' in extra and oc is not None:
            for j,cf in enumerate(oc.folds):
                for name,value in cf.get('_adjoint_arrays',{}).items():arrays['fold_'+str(j)+'_'+name]=value
        arrays.update(extra);ref=rec.save(key,**arrays)
        obj['head_refs']=refs
        for fa,hr in zip(obj.get('inner_folds',[]),refs):fa['head_ref']=hr
        return ref
    def event(name,payload):
        record=dict(_plain(prepared.audit));record.update(payload);record['mode']=mode
        ch._emit(log_callback,'AFFINE_JOINT_'+name,record)
    try:
        if audit['no_information']:
            audit['stop_reason']=prepared.audit['no_information_reason']
            audit['initialization_state_ref']=save_objective('initial',{},None)
            event('INITIAL',dict(state_ref=audit['initialization_state_ref'],no_update_reason=audit['stop_reason']))
        else:
            loss,_,obj,cache=evaluate_affine_joint_objective(prepared,Z,gradient=False);cost(obj);initial_cache=cache
            audit['initialization_state_ref']=save_objective('initial',obj,cache)
            audit['initial_objective']=deepcopy(obj);event('INITIAL',dict(objective=obj,state_ref=audit['initialization_state_ref']))
            audit['stop_reason']='MAX_ITERATIONS'
            for iteration in range(1,5):
                begin=time.perf_counter();audit['optimizer_iterations']+=1
                loss,g,obj,cache=evaluate_affine_joint_objective(prepared,Z,gradient=True,forward_cache=cache);cost(obj)
                norm=_norm(g);direction=np.zeros_like(g) if norm==0 else -g/norm
                ref=save_objective('gradient_'+str(iteration),obj,cache,g_Z=g,d_Z=direction)
                gr=dict(iteration=iteration,state_ref=ref,objective=deepcopy(obj),gradient_norm=norm,direction_norm=_norm(direction))
                audit['gradients'].append(gr);event('GRADIENT',gr)
                if norm==0:audit['stop_reason']='ZERO_GRADIENT';break
                accepted=False
                for trial in range(1,13):
                    step=.125*.5**(trial-1);tz=Z+step*direction;tu=reconstruct_U(tz,prepared.anchor_U,prepared.W)
                    delta=tz-Z;audit['trial_attempt_count']+=1
                    tl,_,to,tc=evaluate_affine_joint_objective(prepared,tz,gradient=False);cost(to)
                    acceptance=_trial_acceptance(loss,tl,float(np.sum(g*delta)))
                    # Record actual trial coordinates while retaining last accepted cache.
                    oldz,oldu=Z,U;Z,U=tz,tu
                    tref=save_objective('trial_'+str(iteration)+'_'+str(trial),to,tc,delta_Z=delta,d_Z=direction)
                    Z,U=oldz,oldu
                    tr=dict(iteration=iteration,trial=trial,step_size=step,state_ref=tref,gradient_state_ref=ref,
                        loss_before=loss,loss_after=tl,update_norm=_norm(delta),objective=deepcopy(to),**acceptance)
                    audit['trial_count']+=1;audit['trials'].append(tr);accepted=acceptance['accepted']
                    audit['accepted_trial_count' if accepted else 'rejected_trial_count']+=1;event('TRIAL',tr)
                    if accepted:
                        Z,U,cache,loss=tz,tu,tc,tl;audit['optimizer_steps']+=1
                        sr=dict(step=audit['optimizer_steps'],iteration=iteration,trial=trial,state_ref=tref,
                            gradient_state_ref=ref,learning_rate=step,step_size=step,loss_before=tr['loss_before'],loss_after=tl,
                            update_norm=_norm(delta),step_seconds=time.perf_counter()-begin,objective=deepcopy(to))
                        audit['steps'].append(sr);event('STEP',sr);break
                if not accepted:audit['stop_reason']='TRIAL_BUDGET_EXHAUSTED';break
            _,_,final_obj,_=evaluate_affine_joint_objective(prepared,Z,gradient=False,forward_cache=cache)
            audit['final_objective_state_ref']=save_objective('final_objective',final_obj,cache)
            audit['final_objective']=final_obj
        final=_forward(prepared.full_problem,U,finalprogress)
        audit['final_head_fit_count']=1;audit['final_factorization_count']=finalprogress['factorization_count']
        for key in ('head_triangular_solve_count','head_triangular_rhs_count','head_triangular_rhs_element_count',
                    'head_triangular_dense_work_unit_count','intercept_fit_count','intercept_addition_count',
                    'ajlr_forward_evaluation_count','reference_distance_evaluation_count',
                    'reference_distance_pair_count','raw_distance_evaluation_count','raw_distance_pair_count',
                    'kernel_evaluation_count','kernel_pair_count','adapter_physical_evaluation_count'):
            audit[key]+=finalprogress.get(key,0)
        arrays=_head_arrays(final);arrays.update(prepared.raw,U=U,Z=Z,W=prepared.W,anchor_U=prepared.anchor_U,
            V0=V0,H=prepared.H,singular_values=prepared.singular_values)
        if prepared.inherited is not None:
            arrays.update({'prior_B_'+k:v for k,v in prepared.inherited.to_arrays().items()})
            arrays['prior_old_class_indices']=np.asarray([prepared.classes.index(name) for name in prepared.inherited.classes],dtype=np.int64)
        audit['final_state_ref']=rec.save('final',**arrays)
        audit.update(final_fit=deepcopy(finalprogress),identity_forward=bool(np.all(U==0)) or prepared.full_problem.tau==0,
            parameter_U_norm=_norm(U),coordinate_norm=_norm(Z),parameter_changed_from_anchor=bool(np.any(U!=prepared.anchor_U)),
            u_changed_from_anchor=bool(np.any(U!=prepared.anchor_U)),u_update_norm=_norm(U-prepared.anchor_U),
            trained_parameter_count=736*prepared.r if audit['optimizer_steps'] else 0,
            analytic_intercept_parameter_count=len(prepared.classes),analytic_intercept_contrast_count=len(prepared.classes)-1,
            analytic_coefficient_parameter_count=len(prepared.ids)*len(prepared.classes),
            analytic_head_parameter_count=(len(prepared.ids)+1)*len(prepared.classes),
            factorization_count=audit['inner_factorization_count']+audit['final_factorization_count'],
            fit_seconds=time.perf_counter()-start,status='AFFINE_JOINT_STAGE_COMPLETE',
            retained_vector_record_bytes=_unique_bytes(rec.records),
            forward_cache_bytes=_unique_bytes(cache.folds) if cache is not None else 0,
            training_coordinate_state_bytes=_unique_bytes((Z,prepared.W,prepared.H,prepared.singular_values,prepared.anchor_U)),
            adapter_state_bytes=int(U.nbytes+V0.nbytes),prior_state_bytes=0 if prepared.inherited is None else prepared.inherited.audit['persistent_state_bytes'],
            head_state_bytes=_unique_bytes({k:v for k,v in final.items() if k in ('b','a','alpha','intercept','reference','mean')}),
            **fcr._functional_measurement(prepared,U,prepared.anchor_U,Z))
        deployed=(U,V0,prepared.raw,prepared.labels,prepared.full_problem.q,
                  prepared.full_problem.train_context['original'],final['b'],final['a'],final['alpha'],final['reference'],final['mean'])
        # Measure the immutable objects actually retained by the returned state.
        # V0 shared between C and B is charged once, not once per stage.
        state=AffineJointState(U,Z,prepared.W,prepared.anchor_U,prepared.H,prepared.singular_values,prepared.raw,prepared.labels,
            prepared.ids,prepared.classes,prepared.old_classes,prepared.full_problem,_cache_for_state(final),prepared.inherited,rec.records,audit)
        audit['persistent_state_bytes']=_unique_bytes(state._resident_values())
        audit['deployment_numeric_state_bytes']=_unique_bytes(state._deployment_values())
        audit['state_byte_scope']='unique_actual_retained_numeric_state_with_coordinates_head_cache_and_actual_B_excludes_records_Python'
        audit['serialized_deployment_bytes']=None;audit['incremental_transmission_bytes']=None
        audit['gpu_memory_bytes']=None;audit['peak_rss_bytes']=None
        object.__setattr__(state,'audit',_freeze(audit))
        event('FINAL',audit)
        return state
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),failed_final_progress=finalprogress,
            fit_seconds=time.perf_counter()-start,accepted_state_ref=rec.save('failure_accepted',Z=Z,U=U,anchor_U=prepared.anchor_U,W=prepared.W))
        error=NumericalFailure(str(exc),audit);error.records=rec.records;raise error from exc


def predict_affine_joint_local_ridge(state,**features):
    if not isinstance(state,AffineJointState):raise TypeError('AffineJoint prediction requires an AffineJointState')
    return state.predict(**features)
