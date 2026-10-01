"""Independent support-only margin joint draft; no experiment is launched here.

B: analytic affine ridge. C: frozen actual-B prior and full physical-row
margin QP. Both use CE-only function-coordinate projected adapter updates.
Caller-supplied QP resource limits are mandatory, not experiment defaults.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np

from . import d92_affine_joint_local_ridge as affine
from . import d92_conditional_joint_local_ridge as projected
from . import d92_margin_qp_head as qp
from . import d92_function_coordinate_residual8_local_ridge as fcr
from . import d92_joint_channel_local_ridge as ch
from . import d92_branch_local_ridge as local
from . import d92_branch_interaction as interaction
from .d92_branch_ridge import _freeze, _plain

_EPS = float(np.finfo(np.float64).eps)
_NAMES = fcr._NAMES
V0 = fcr.V0
NumericalFailure = ch.NumericalFailure
_norm, _finite = ch._norm, ch._finite
_readonly, _scalar = affine._readonly, affine._scalar
latent_coordinates, reconstruct_U = fcr.latent_coordinates, fcr.reconstruct_U
initial_parameters = fcr.initial_parameters
project_coordinates = projected.project_coordinates
_trial_acceptance = affine._trial_acceptance

QP_WORK_KEYS = (
    'transitions', 'full_constraint_scans', 'spectral_checks',
    'compact_snapshot_rebuilds', 'compact_snapshot_dense_work_units',
    'spectral_cubic_dimension_units', 'independence_checks',
    'factorization_attempts', 'factorizations_completed',
    'condition_estimation_calls', 'triangular_calls', 'triangular_rhs_columns',
    'triangular_rhs_elements', 'triangular_dense_work_units',
)
PREPARATION_COUNTERS = affine.PREPARATION_COUNTERS
STAGE_COUNTERS = tuple(dict.fromkeys(affine.STAGE_COUNTERS +
    ('completed_factorization_count',) + tuple(
        'margin_qp_' + operation + '_' + key
        for operation in ('forward', 'adjoint') for key in QP_WORK_KEYS)))
AUDIT_COUNTERS = tuple(dict.fromkeys(PREPARATION_COUNTERS + STAGE_COUNTERS))
COUNTERS = STAGE_COUNTERS
_GEOMETRY_COUNTERS = ('raw_distance_evaluation_count', 'raw_distance_pair_count',
    'reference_distance_evaluation_count', 'reference_distance_pair_count',
    'kernel_evaluation_count', 'kernel_pair_count', 'adapter_physical_evaluation_count')
_HEAD_COUNTERS = ('head_triangular_solve_count', 'head_triangular_rhs_count',
    'head_triangular_rhs_element_count', 'head_triangular_dense_work_unit_count',
    'intercept_fit_count', 'intercept_addition_count', 'ajlr_forward_evaluation_count')
_ADJOINT_COUNTERS = tuple('margin_qp_adjoint_' + key for key in QP_WORK_KEYS) + (
    'derivative_triangular_solve_count', 'derivative_triangular_rhs_count',
    'derivative_triangular_rhs_element_count', 'derivative_triangular_dense_work_unit_count',
    'ce_adjoint_solve_count')
_FAILURES = (FloatingPointError, np.linalg.LinAlgError, qp.MarginQPFailure)

FROZEN_CONFIG = dict(
    schema='d92_margin_joint_local_ridge_v1', method='D92-MarginJointLocalRidge-v1',
    candidate_status='STRUCTURE_IMPLEMENTATION_DRAFT_NOT_PREREGISTERED_OR_LAUNCHED',
    base_algorithm=deepcopy(local.FROZEN_CONFIG), arms=['local_ridge', 'margin_joint_seq'],
    rank=8, input_dim=736, block_dimensions=[160, 96, 160, 160, 160], kappa=.25,
    fixed_dictionary='first8_orthonormal_DCT_rows', learn_V=False,
    coordinate='U=anchor_U+Z@W.T;H/sqrtN=P*Sigma*R.T;W=R_retained/Sigma_retained',
    rank_rule='sigma/sigma_max>sqrt(128*eps64*max(N,8))',
    initialization_B='U_zero', initialization_C_seq='exact_actual_current_B_U',
    bandwidth='fixed_original_old_inner_train_R0', trace_scale='fixed_original_old_R0_gamma',
    joint_original_distance_weight=.5, joint_adapted_distance_weight=.5, implicit_feature_dim=123616,
    B_head='analytic_affine_combined_C_plus_1_RHS',
    C_head='full_physical_train_raw_PSD_kernel_margin_QP_free_intercept',
    prior_C='same_run_row_scope_physical_fold_actual_B_padded_new_columns_zero',
    margin='original_min_true_vs_all_registered_competitors_no_clamp',
    target='row_class_centered_onehot_minus_1_over_C', ridge_coefficient=1., sample_weight=1.,
    objective='RMS_across_classes_of_cross_fold_mean_CE_only', temperature=1.,
    proximal_coefficient=0., coordinate_ball_radius=.5, keep=False, guard=False,
    optimizer='normalized_gradient_projected_first_Armijo_RMSCE_nonincrease',
    max_iterations=4, max_trials=12, initial_step_size=.125, backtrack_factor=.5,
    armijo_coefficient=1e-4, comparison_tolerance_multiplier=128,
    qp_max_transitions='EXPLICIT_CALLER_POSITIVE_INTEGER_REQUIRED',
    qp_max_factor_buffer_bytes='EXPLICIT_CALLER_POSITIVE_INTEGER_REQUIRED',
    qp_limit_scope='per_actual_C_head_call_not_whole_experiment_budget',
    unsupported_jacobian='TECHNICAL_FAILURE_no_unconstrained_or_zero_fallback',
    inner_folds='per_class_physical_id_sort_position_mod_min_K_3',
    zero_bandwidth='original_complete_interaction_exact_equivalence_no_continuous_kernel_gradient',
    duplicate_rows='retain_all_physical_SSE_and_all_label_constraints_no_compression',
    zero_kernel='full_constrained_free_constant_head_no_adapter_update',
    k1='no_adapter_update_full_support_head', rank0='anchor_U_full_support_head',
    n0='exact_reuse_actual_B_object_before_QP', tie_break='physical_class_id_lexicographic',
    query_decision_policy='per_sample_all_registered_classes', phase1_frozen=True,
    encoder_backward=False, source_inputs=False, query_fit=False, parameter_search=False,
    dtype='float64', vector_logging='lossless_state_callback_or_records',
)


def _unique_bytes(values):
    """Physical numeric backing buffers, including MappingProxyType, once each."""
    seen = set()
    total = 0
    def visit(value):
        nonlocal total
        if isinstance(value, np.ndarray):
            root = value
            while isinstance(root.base, np.ndarray):
                root = root.base
            backing = root.base if isinstance(root.base, (bytes, bytearray)) else root
            if id(backing) not in seen:
                seen.add(id(backing))
                total += len(backing) if isinstance(backing, (bytes, bytearray)) else root.nbytes
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(values)
    return int(total)


def _add(target, source, keys):
    for key in keys:
        target[key] = target.get(key, 0) + source.get(key, 0)


def _qp_account(target, source, operation):
    for key in QP_WORK_KEYS:
        name = 'margin_qp_' + operation + '_' + key
        target[name] = target.get(name, 0) + source.get(key, 0)
    stem = 'head' if operation == 'forward' else 'derivative'
    for suffix, key in (('solve_count', 'triangular_calls'), ('rhs_count', 'triangular_rhs_columns'),
                        ('rhs_element_count', 'triangular_rhs_elements'),
                        ('dense_work_unit_count', 'triangular_dense_work_units')):
        name = stem + '_triangular_' + suffix
        target[name] = target.get(name, 0) + source.get(key, 0)
    if operation == 'forward':
        target['factorization_count'] = source.get('factorization_attempts', 0)
        target['completed_factorization_count'] = source.get('factorizations_completed', 0)
    for name in ('peak_factor_buffer_bytes', 'peak_explicit_solve_temporary_bytes'):
        target['margin_qp_' + name] = max(target.get('margin_qp_' + name, 0), source.get(name, 0))


def compact_training_record(record):
    """Scalar JSON/CSV surface; lossless arrays remain in state_callback archives."""
    objective = record.get('objective', {})
    audit = record.get('audit', record)
    result = {key:record.get(key) for key in ('event', 'mode', 'run_id', 'row_id', 'split_id',
        'scope', 'fold', 'iteration', 'trial', 'step', 'step_size', 'learning_rate',
        'loss_before', 'loss_after', 'gradient_norm', 'update_norm', 'accepted', 'text')}
    for key in ('RMSCE', 'loss_ce', 'loss_total', 'loss_proximal', 'objective_seconds'):
        result[key] = objective.get(key, audit.get(key))
    event = str(record.get('event',''))
    cumulative = event.endswith(('FINAL','FAILURE'))
    preparation = event.endswith('PREPARED')
    owner = audit if cumulative or preparation else objective
    result['counter_ownership'] = 'PREPARATION' if preparation else 'STAGE_CUMULATIVE' if cumulative else 'OBJECTIVE_INCREMENT'
    for key in AUDIT_COUNTERS:
        if key in owner:
            result[key] = owner[key]
    result.update(source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
                  peak_rss_bytes=None, peak_gpu_memory_bytes=None)
    return ch._safe(result)


def _emit(callback, event, payload, context=None):
    record = dict(context or {})
    record.update(payload)
    objective = record.get('objective', {})
    text = ('MARGIN_JOINT ' + event + ' mode=' + str(record.get('mode')) +
            ' RMSCE=' + str(objective.get('RMSCE')) + ' prox=0' +
            ' lr=' + str(record.get('step_size', record.get('learning_rate'))) +
            ' grad=' + str(record.get('gradient_norm')) +
            ' update=' + str(record.get('update_norm')) +
            ' accepted=' + str(record.get('accepted')) +
            ' heads=' + str(objective.get('inner_head_fit_count')) +
            ' factors=' + str(objective.get('inner_factorization_count')) +
            ' qp_transitions=' + str(objective.get('margin_qp_forward_transitions')) +
            ' head_rhs=' + str(objective.get('head_triangular_rhs_count')) +
            ' adjoint_rhs=' + str(objective.get('derivative_triangular_rhs_count')) +
            ' seconds=' + str(objective.get('objective_seconds')) + ' source_validation=N/A')
    if event=='PREPARED':
        text += (' rank=' + str(record.get('latent_rank')) + ' dictionary=DCT8 kappa=.25 ball=.5'
                 ' max_updates=4 max_trials=12 initial_step=.125 temperature=1'
                 ' qp_max_transitions=' + str(record.get('qp_max_transitions')) +
                 ' qp_max_factor_buffer_bytes=' + str(record.get('qp_max_factor_buffer_bytes')))
    record['text'] = text
    ch._emit(callback, 'MARGIN_JOINT_' + event, record)


class _Recorder:
    def __init__(self, callback):
        self.callback, self.records, self.keys = callback, {}, set()
    def save(self, key, **arrays):
        if key in self.keys:
            raise ValueError('Duplicate MARGIN_JOINT record ' + key)
        self.keys.add(key)
        values = {name:_readonly(value) for name, value in arrays.items()}
        metadata = {name:dict(shape=list(value.shape), dtype=str(value.dtype), nbytes=int(value.nbytes))
                    for name, value in values.items()}
        if self.callback is None:
            self.records[key] = _freeze(values)
            ref = dict(storage='in_memory')
        else:
            ref = ch._safe(self.callback(key, values))
            if not isinstance(ref, dict):
                raise ValueError('state_callback must return a JSON mapping')
        return dict(ref, key=key, arrays=metadata)


def _actual_config(max_transitions,max_factor_buffer_bytes):
    result = deepcopy(FROZEN_CONFIG)
    result.update(qp_max_transitions=max_transitions,qp_max_factor_buffer_bytes=max_factor_buffer_bytes)
    return result


def _adapter_audit(cache):
    angles,norm_error = [],0.
    for context,derivative in ((cache['problem'].train_context,cache['bc']),
                               (cache['problem'].held_context,cache['hc'])):
        for index,sl in enumerate(fcr._SLICES):
            valid = context['rho'][:,index]>0
            if np.any(valid):
                chord = np.linalg.norm(derivative['y'][valid,sl]-context['w'][valid,sl],axis=1)
                angles.extend((2*np.arcsin(np.minimum(1.,chord/2))).tolist())
                norm_error = max(norm_error,float(np.max(np.abs(np.linalg.norm(derivative['y'][valid,sl],axis=1)-1))))
    return dict(block_angle_radians=fcr._statistics(angles),block_unit_norm_max_error=norm_error)


@dataclass(frozen=True)
class MarginJointProblem:
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
    max_transitions: int
    max_factor_buffer_bytes: int
    audit: Mapping
    def __post_init__(self):
        for key in ('train_context', 'held_context'):
            object.__setattr__(self, key, _freeze({k:_readonly(v) for k,v in getattr(self,key).items()}))
        for key in ('train_labels','held_labels','q','old_indices','new_indices','d0','cross_d0','M_train','M_held'):
            object.__setattr__(self, key, _readonly(getattr(self,key)))
        object.__setattr__(self, 'audit', _freeze(self.audit))


def _make_problem(b,a,y,ids,classes,old_classes,keep,H,*,mode,max_transitions,max_factor_buffer_bytes,
                  fold=None,prior=None,nuisance=None,progress=None,prior_ref=None):
    keep = np.asarray(keep, dtype=bool)
    tb, ta, hb, ha, ty, hy = b[keep], a[keep], b[~keep], a[~keep], y[keep], y[~keep]
    old = np.array([classes[int(label)] in old_classes for label in ty])
    oi, ni = np.flatnonzero(old), np.flatnonzero(~old)
    if not len(oi):
        raise ValueError('Old inner-train reference required')
    nu = nuisance
    if nu is None:
        if progress is not None:
            progress['prepared_distance_evaluation_count'] += 1
            progress['original_distance_pair_count'] += len(oi)*(len(oi)-1)//2
            progress['reference_distance_evaluation_count'] += 1
            progress['reference_distance_pair_count'] += len(oi)*(len(oi)-1)//2
        old_y = np.array([old_classes.index(classes[int(label)]) for label in ty[old]], dtype=np.int64)
        nu = affine._nuisance(tb[old],ta[old],old_y,len(old_classes))
    reuse = mode == 'B' and 'old_distance' in nu
    if progress is not None:
        progress['prepared_distance_evaluation_count'] += int(not reuse)+int(len(hy)>0)
        progress['original_distance_pair_count'] += (0 if reuse else len(ty)*(len(ty)-1)//2)+len(hy)*len(ty)
    d0 = nu['old_distance'] if reuse else local._distances(tb,ta)
    cross = local._distances(hb,ha,tb,ta)
    mt, mh = np.zeros((len(ty),len(classes))), np.zeros((len(hy),len(classes)))
    if prior is not None:
        pp, pu, pf, pc = prior
        cost = {}
        pt = _score_residual(pp,pf,pu,tb,ta,cost,H[keep])
        ph = _score_residual(pp,pf,pu,hb,ha,cost,H[~keep])
        for col, name in enumerate(pc):
            mt[:,classes.index(name)], mh[:,classes.index(name)] = pt[:,col], ph[:,col]
        if progress is not None:
            _add(progress,cost,_GEOMETRY_COUNTERS+('dictionary_physical_evaluation_count',))
            progress['prior_intercept_addition_count'] += cost.get('intercept_addition_count',0)
            progress['prior_score_evaluation_count'] += 1+int(len(hy)>0)
            progress['prior_score_physical_count'] += len(y)
            progress['prior_score_seconds'] = progress.get('prior_score_seconds',0.)+cost.get('score_seconds',0.)
    tids = tuple(pid for i,pid in enumerate(ids) if keep[i])
    hids = tuple(pid for i,pid in enumerate(ids) if not keep[i])
    audit = dict(schema=FROZEN_CONFIG['schema'],method=FROZEN_CONFIG['method'],mode=mode,inner_fold=fold,
        classes=list(classes),old_classes=list(old_classes),training_physical_ids=list(tids),held_physical_ids=list(hids),
        old_reference_physical_ids=[tids[i] for i in oi],train_physical_count=len(ty),held_physical_count=len(hy),
        old_reference_count=len(oi),bandwidth_tau=nu['tau'],trace_scale=nu['gamma'],interaction_centered_trace=nu['s0'],
        prior_ref=prior_ref,prior_source='ZERO' if prior is None else 'FROZEN_CURRENT_ACTUAL_B',
        all_head_statistics_from_old_inner_train_only=True,qp_max_transitions=max_transitions,
        qp_max_factor_buffer_bytes=max_factor_buffer_bytes)
    return MarginJointProblem(fcr._context(tb,ta,H[keep]),fcr._context(hb,ha,H[~keep]),ty,hy,tuple(classes),
        old.astype(float)/len(oi),oi,ni,d0,cross,nu['tau'],nu['gamma'],nu['s0'],mt,mh,mode,
        max_transitions,max_factor_buffer_bytes,audit)


def _forward(p,U,progress=None):
    start = time.perf_counter()
    audit = {} if progress is None else progress
    audit.update({key:0 for key in STAGE_COUNTERS})
    audit.update(_plain(p.audit),head_fit_count=1,factorization_count=0,intercept_fit_count=1,
                 ajlr_forward_evaluation_count=1)
    cache = None
    try:
        if p.mode == 'B':
            cache = affine._forward(p,U,audit)
            names = ('K','L','chol','alpha','intercept','schur_z','schur_s','combined_rhs',
                     'reference','reference_self','mean','grand')
            cache.update(numeric={key:cache[key] for key in names},qp_state=None,
                         residual_target=cache['E'])
            audit['completed_factorization_count'] = audit['factorization_count']
            return cache
        if not len(p.new_indices):
            raise ValueError('new0 must reuse actual B before any QP')
        n,h,c = len(p.train_labels),len(p.held_labels),len(p.classes)
        b,a,bc = fcr._adapt(p.train_context,U,derivative_cache=True)
        hb,ha,hc = fcr._adapt(p.held_context,U,derivative_cache=True)
        audit['adapter_physical_evaluation_count'] = n+h
        identity = not np.any(U) or p.tau == 0
        if identity:
            distance,cross = p.d0,p.cross_d0
        else:
            distance = .5*p.d0+.5*local._distances(b,a)
            cross = .5*p.cross_d0+.5*local._distances(hb,ha,b,a)
            m = len(p.old_indices)
            audit.update(raw_distance_evaluation_count=1+int(h>0),raw_distance_pair_count=n*(n-1)//2+h*n,
                reference_distance_evaluation_count=1+int(h>0),reference_distance_pair_count=m*(m-1)//2+m*(n-m)+h*m)
        radial,crossrad = np.zeros((n,n)),np.zeros((h,n))
        if p.gamma is not None:
            radial,crossrad = local._radial(distance,p.tau),local._radial(cross,p.tau)
            audit.update(kernel_evaluation_count=1+int(h>0),kernel_pair_count=n*n+h*n)
        K,L = (np.zeros((n,n)),np.zeros((h,n))) if p.gamma is None else (p.gamma*radial,p.gamma*crossrad)
        Y = np.eye(c)[p.train_labels]-1/c
        cache = dict(problem=p,U=np.array(U,copy=True),b=b,a=a,hb=hb,ha=ha,bc=bc,hc=hc,
            distance=distance,cross_distance=cross,radial=radial,crossrad=crossrad,K=K,L=L,Y=Y,
            residual_target=Y-p.M_train,audit=audit)
        head_start = time.perf_counter()
        try:
            head = qp.fit_margin_qp_head(K=K,M=p.M_train,labels=p.train_labels,old_indices=p.old_indices,
                max_transitions=p.max_transitions,max_factor_buffer_bytes=p.max_factor_buffer_bytes)
        except qp.MarginQPFailure as exc:
            _qp_account(audit,exc.audit,'forward')
            audit['margin_qp_audit'] = exc.audit
            exc.joint_failed_head_arrays = dict(_head_arrays_partial(cache),
                **{'qp_failure_'+key:value for key,value in exc.arrays.items()})
            raise
        _qp_account(audit,head.audit,'forward')
        score = qp.predict_margin_qp_head(head,L=L,M=p.M_held)
        cache.update(numeric=dict(head.arrays),qp_state=head,score=score,train_scores=head.train_scores)
        final = head.audit['final_residuals']
        audit.update(margin_qp_audit=head.audit,margin_qp_head_seconds=time.perf_counter()-head_start,
            actual_kernel_trace=float(np.trace(K)),normal_equation_residual=final['stationarity'],
            margin_minimum_slack=final['minimum_slack'],margin_feasibility_tolerance=final['residual_tolerance'],
            old_physical_constraint_count=len(p.old_indices)*(c-1),physical_train_count=n,
            head_training_loss_data=float(.5*np.sum((head.train_scores-Y)**2)),
            head_training_loss_ridge=float(.5*final['regularizer']),intercept_addition_count=(n+h)*c,
            intercept_norm=_norm(head.b),analytic_intercept_parameter_count=c,analytic_intercept_contrast_count=c-1,
            identity_forward=identity,status='NUMERIC_KKT_CERTIFIED')
        audit['head_training_loss_total'] = audit['head_training_loss_data']+audit['head_training_loss_ridge']
        audit['class_sum_max_abs'] = max(affine._sum_class_error(value) for value in (Y,head.train_scores,score))
        audit['forward_seconds'] = time.perf_counter()-start
        audit.update(_adapter_audit(cache))
        _finite(head.train_scores,score)
        return cache
    except _FAILURES as exc:
        exc.joint_forward_audit = audit
        if not hasattr(exc,'joint_failed_head_arrays'):
            exc.joint_failed_head_arrays = _head_arrays_partial(cache if cache is not None else dict(problem=p))
        raise


def _backward(cache,G,progress):
    p = cache['problem']
    start = time.perf_counter()
    if p.gamma is None or p.gamma == 0 or p.tau == 0 or not len(G):
        progress['kernel_gradient_status'] = 'NOT_REQUESTED_NO_CONTINUOUS_ADAPTER_KERNEL_INFORMATION'
        progress['adjoint_seconds'] = time.perf_counter()-start
        return np.zeros((736,8))
    n,c = len(p.train_labels),len(p.classes)
    if p.mode == 'B':
        num = cache['numeric']
        T,t,g_b,rhs = affine._affine_adjoint(num['chol'],num['L'],G,num['schur_z'],float(num['schur_s']))
        barK = -.5*(T@num['alpha'].T+num['alpha']@T.T)
        barL = G@num['alpha'].T
        rawbar,crossbar = affine._center_vjp(barK,barL,p.q,p.gamma)
        progress['derivative_triangular_solve_count'] += 2
        progress['derivative_triangular_rhs_count'] += 2*c
        progress['derivative_triangular_rhs_element_count'] += 2*n*c
        progress['derivative_triangular_dense_work_unit_count'] += 2*n*n*c
        adjarrays = dict(T_G=T,t_G=t,g_b=g_b,rhs=rhs,G=G,K=barK,L=barL)
    else:
        try:
            adj = qp.margin_qp_head_vjp(cache['qp_state'],L=cache['L'],G=G)
        except qp.MarginQPFailure as exc:
            _qp_account(progress,exc.audit,'adjoint')
            progress.update(margin_qp_adjoint_audit=exc.audit,ce_adjoint_solve_count=1,
                            kernel_gradient_status='UNSUPPORTED_OR_TECHNICAL_FAILURE')
            exc.joint_backward_audit = progress
            exc.joint_failed_head_arrays = dict(_head_arrays(cache),G=G,
                **{'qp_failure_'+key:value for key,value in exc.arrays.items()})
            raise
        _qp_account(progress,adj.audit,'adjoint')
        rawbar,crossbar = p.gamma*adj.arrays['K'],p.gamma*adj.arrays['L']
        progress['margin_qp_adjoint_audit'] = adj.audit
        adjarrays = dict(adj.arrays,G=G)
    progress['ce_adjoint_solve_count'] = 1
    # Raw Gaussian exp[-(d_original+d_adapted)/(2*tau)]. Full Gram
    # Frobenius upstream; the symmetric distance helper accounts for both ends.
    dd,dc = -rawbar*cache['radial']/p.tau,-crossbar*cache['crossrad']/p.tau
    gb,ga,_,_ = ch._distance_vjp(cache['b'],cache['a'],cache['b'],cache['a'],.5*dd,symmetric=True)
    ghb,gha,gtb,gta = ch._distance_vjp(cache['hb'],cache['ha'],cache['b'],cache['a'],.5*dc)
    gu = fcr._adapter_vjp(p.train_context,cache['U'],gb+gtb,ga+gta,cache['bc'])
    gu += fcr._adapter_vjp(p.held_context,cache['U'],ghb,gha,cache['hc'])
    cache['adjoint_arrays'] = dict(adjarrays,raw_train_upstream=rawbar,raw_cross_upstream=crossbar,
        adapted_distance_upstream=.5*dd,adapted_cross_distance_upstream=.5*dc,g_U=gu)
    progress.update(kernel_gradient_status='REGULAR_COMPLETE_JACOBIAN',adjoint_seconds=time.perf_counter()-start)
    return gu


def _head_arrays_partial(cache):
    p = cache['problem']
    arrays = dict(original_train_b=p.train_context['original'][:,:256],original_train_a=p.train_context['original'][:,256:],
        original_held_b=p.held_context['original'][:,:256],original_held_a=p.held_context['original'][:,256:],
        train_labels=p.train_labels,held_labels=p.held_labels,q=p.q,old_indices=p.old_indices,new_indices=p.new_indices,
        original_distance=p.d0,original_cross_distance=p.cross_d0,M_train=p.M_train,M_held=p.M_held,
        tau=_scalar(p.tau),gamma=_scalar(p.gamma))
    # The QP head owns vector s0 (all old-vs-registered constraint slack).
    # Keep the independent geometric scale under a distinct C-stage name.
    arrays['s0' if p.mode == 'B' else 'reference_s0'] = _scalar(p.s0)
    for key in ('distance','cross_distance','radial','crossrad','K','L','Y','residual_target','score','train_scores'):
        if key in cache:
            arrays[{'radial':'raw_train','crossrad':'raw_cross','score':'scores'}.get(key,key)] = cache[key]
    for key,name in (('b','adapted_train_b'),('a','adapted_train_a'),('hb','adapted_held_b'),('ha','adapted_held_a')):
        if key in cache:
            arrays[name] = cache[key]
    return arrays


def _head_arrays(cache):
    return dict(cache['numeric'],**_head_arrays_partial(cache))


def _score_residual(p,cache,U,b0,a0,progress=None,H=None):
    start = time.perf_counter()
    num,n,c = cache['numeric'],len(b0),len(p.classes)
    intercept = num['intercept'] if p.mode == 'B' else num['b']
    out = np.broadcast_to(intercept,(n,c)).copy()
    if p.gamma is not None:
        original = p.train_context['original']
        for i in range(n):
            d0 = local._distances(b0[i:i+1],a0[i:i+1],original[:,:256],original[:,256:])
            if p.tau == 0:
                distance = d0
            else:
                b,a,_ = fcr._adapt(fcr._context(b0[i:i+1],a0[i:i+1],None if H is None else H[i:i+1]),U)
                distance = .5*d0+.5*local._distances(b,a,cache['b'],cache['a']) if np.any(U) else d0
            if p.mode == 'B':
                cross = affine._center_cross(local._radial_minus_one(distance,p.tau),p.q,num['reference'],
                    float(num['reference_self']),num['mean'],float(num['grand']))
            else:
                cross = local._radial(distance,p.tau)
            out[i] = (p.gamma*cross[0])@num['alpha']+intercept
        if progress is not None:
            width,m = len(p.train_labels),len(p.old_indices)
            calls = n*(1+int(p.tau!=0 and bool(np.any(U))))
            _add(progress,dict(raw_distance_evaluation_count=calls,raw_distance_pair_count=calls*width,
                reference_distance_evaluation_count=calls,reference_distance_pair_count=calls*m,
                kernel_evaluation_count=n,kernel_pair_count=n*width,adapter_physical_evaluation_count=n if p.tau!=0 else 0,
                dictionary_physical_evaluation_count=n if p.tau!=0 and H is None else 0),
                _GEOMETRY_COUNTERS+('dictionary_physical_evaluation_count',))
    if progress is not None:
        progress['intercept_addition_count'] = progress.get('intercept_addition_count',0)+n*c
        progress['score_seconds'] = progress.get('score_seconds',0.)+time.perf_counter()-start
    _finite(out)
    return out


@dataclass(frozen=True)
class MarginJointTraining:
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
    full_problem: MarginJointProblem
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        for key in ('labels','H','W','singular_values','anchor_U'):
            object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'raw',_freeze({k:_readonly(v) for k,v in self.raw.items()}))
        object.__setattr__(self,'records',_freeze(self.records))
        object.__setattr__(self,'audit',_freeze(self.audit))
    def audit_dict(self):
        return ch._safe(_plain(self.audit))
    def state_records(self):
        return {key:{name:np.array(value,copy=True) for name,value in record.items()} for key,record in self.records.items()}


def _problem_values(p):
    return {key:value for key,value in vars(p).items() if key!='audit'}


def prepare_margin_joint_training(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes,
        max_transitions,max_factor_buffer_bytes,inherited=None,context=None,log_callback=None,state_callback=None):
    for name,value in (('max_transitions',max_transitions),('max_factor_buffer_bytes',max_factor_buffer_bytes)):
        if type(value) is not int or value<=0:
            raise ValueError(name+' requires an explicit positive integer')
    start = time.perf_counter()
    rec = _Recorder(state_callback)
    audit = dict(context or {})
    audit.update({key:0 for key in PREPARATION_COUNTERS})
    audit['ajlr_preparation_count'] = 1
    b,a,y,ids,canonical,_,old,k = interaction._prepare(z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes)
    old = tuple(sorted(old))
    if not old:
        raise ValueError('Old support classes required')
    order = np.asarray(sorted(range(len(ids)),key=lambda i:support_ids[i]))
    raw = {name:np.asarray(value,dtype=np.float64)[order] for name,value in zip(_NAMES,(z_id,fft,t_emb,f_emb,pa_local))}
    mode,anchor = ('B' if inherited is None else 'C_seq'),np.zeros((736,8))
    if inherited is None:
        if canonical!=old:
            raise ValueError('C requires actual current MARGIN_JOINT B')
    else:
        if not isinstance(inherited,MarginJointState) or inherited.audit['mode']!='B' or inherited.classes!=old:
            raise ValueError('C requires actual old-only MARGIN_JOINT B')
        previous = inherited.audit['preparation']
        for key in ('run_id','row_id','scope'):
            if not audit.get(key) or not previous.get(key):
                raise ValueError('Actual B lineage requires explicit '+key)
            if audit[key]!=previous[key]:
                raise ValueError('C lineage crosses '+key)
        for key in ('split_id','fold'):
            if (key in audit)!=(key in previous) or (key in audit and audit[key]!=previous[key]):
                raise ValueError('C lineage crosses '+key)
        oi = np.array([canonical[int(label)] in old for label in y])
        if tuple(pid for i,pid in enumerate(ids) if oi[i])!=inherited.ids:
            raise ValueError('C old physical IDs differ from B')
        if any(canonical[int(label)]!=inherited.classes[int(before)] for label,before in zip(y[oi],inherited.labels)):
            raise ValueError('C old labels differ from B')
        if any(not np.array_equal(raw[name][oi],inherited.raw[name]) for name in _NAMES):
            raise ValueError('C old raw features differ from B')
        anchor = inherited.U.copy()
    audit.update(schema=FROZEN_CONFIG['schema'],method=FROZEN_CONFIG['method'],mode=mode,classes=list(canonical),old_classes=list(old),
        training_physical_ids=list(ids),train_physical_count=len(ids),train_k=k,inherited_state=inherited is not None,
        qp_max_transitions=max_transitions,qp_max_factor_buffer_bytes=max_factor_buffer_bytes,
        actual_parameters=_actual_config(max_transitions,max_factor_buffer_bytes))
    if inherited is not None and canonical==old:
        audit.update(ajlr_preparation_count=0,no_information=True,no_information_reason='N0_REUSE_ACTUAL_B',
            prepared_state_ref=_plain(previous['prepared_state_ref']),inner_folds=[],prior_folds=[],
            final_problem=_plain(inherited.problem.audit),latent_rank=inherited.W.shape[1],
            retained_vector_record_bytes=0,preparation_seconds=time.perf_counter()-start)
        result = MarginJointTraining(raw,y,ids,canonical,old,inherited.H,inherited.W,inherited.singular_values,
            inherited.W.shape[1],anchor,inherited,(),inherited.problem,{},audit)
    else:
        try:
            H = fcr._context(b,a)['h']
            audit['dictionary_physical_evaluation_count'] = len(y)
            W,s,ci = latent_coordinates(H)
            audit.update(ci)
            audit['prepared_state_ref'] = rec.save('prepared_coordinates',H=H,W=W,singular_values=s,anchor_U=anchor,V0=V0)
            problems,priors = [],[]
            def make(keep,fold,final=False):
                prior,nu,ref = None,None,None
                if inherited is not None:
                    if final:
                        pp,pu,pf = inherited.problem,inherited.U,inherited.final_cache
                        ref = _plain(inherited.audit['final_state_ref'])
                    else:
                        oi = np.flatnonzero(np.array([canonical[int(label)] in old for label in y]))
                        oy = np.array([old.index(canonical[int(label)]) for label in y[oi]],dtype=np.int64)
                        pp = _make_problem(b[oi],a[oi],oy,tuple(ids[i] for i in oi),old,old,keep[oi],H[oi],mode='B',
                            max_transitions=max_transitions,max_factor_buffer_bytes=max_factor_buffer_bytes,fold=fold,progress=audit)
                        prior_progress = {}
                        try:
                            pf = _forward(pp,anchor,prior_progress)
                        finally:
                            audit['prior_head_fit_count'] += prior_progress.get('head_fit_count',0)
                            audit['prior_factorization_count'] += prior_progress.get('factorization_count',0)
                            for suffix in ('solve_count','rhs_count','rhs_element_count','dense_work_unit_count'):
                                audit['prior_triangular_'+suffix] += prior_progress.get('head_triangular_'+suffix,0)
                            audit['prior_intercept_fit_count'] += prior_progress.get('intercept_fit_count',0)
                            audit['prior_intercept_addition_count'] += prior_progress.get('intercept_addition_count',0)
                            _add(audit,prior_progress,_GEOMETRY_COUNTERS)
                        pu = anchor
                        ref = rec.save('prior_head_'+str(fold),U=anchor,**_head_arrays(pf))
                        priors.append(dict(_plain(pp.audit),head_state_ref=ref,final_fit=_plain(pf['audit'])))
                    prior = (pp,pu,pf,old)
                    nu = dict(tau=pp.tau,gamma=pp.gamma,s0=pp.s0)
                return _make_problem(b,a,y,ids,canonical,old,keep,H,mode=mode,max_transitions=max_transitions,
                    max_factor_buffer_bytes=max_factor_buffer_bytes,fold=fold,prior=prior,nuisance=nu,progress=audit,prior_ref=ref)
            if k>1:
                folds = min(k,3)
                assignment = np.empty(len(y),dtype=np.int64)
                for cls in range(len(canonical)):
                    ix = np.flatnonzero(y==cls)
                    assignment[ix] = np.arange(len(ix))%folds
                for fold in range(folds):
                    problems.append(make(assignment!=fold,fold))
            full = make(np.ones(len(y),dtype=bool),None,True)
            reason = ('PHYSICAL_K1' if k==1 else 'ZERO_DICTIONARY_RANK' if W.shape[1]==0 else
                'NO_OLD_KERNEL_INFORMATION' if full.gamma is None or full.gamma == 0 else 'ALL_INNER_GEOMETRY_DEGENERATE'
                if not any(p.gamma is not None and p.gamma>0 and p.tau is not None and p.tau>0 and len(p.held_labels) for p in problems) else None)
            audit.update(no_information=reason is not None,no_information_reason=reason,inner_folds=[_plain(p.audit) for p in problems],
                prior_folds=priors,final_problem=_plain(full.audit),latent_rank=W.shape[1],preparation_seconds=time.perf_counter()-start,
                retained_vector_record_bytes=_unique_bytes(rec.records),source_inputs=False,query_fit=False,
                reference_distance_scope='subset_of_raw_distance_work_not_additive')
            result = MarginJointTraining(raw,y,ids,canonical,old,H,W,s,W.shape[1],anchor,inherited,tuple(problems),full,rec.records,audit)
        except _FAILURES as exc:
            audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),preparation_seconds=time.perf_counter()-start)
            failure_arrays = getattr(exc,'joint_failed_head_arrays',{})
            if hasattr(exc,'joint_failed_head_arrays'):
                audit['failure_state_ref'] = rec.save('preparation_failure',**failure_arrays)
            error = NumericalFailure(str(exc),audit)
            error.records = rec.records
            error.arrays = _freeze({key:_readonly(value) for key,value in failure_arrays.items()})
            raise error from exc
    values = (result.raw,result.labels,result.H,result.W,result.singular_values,result.anchor_U,
              [_problem_values(p) for p in result.problems],_problem_values(result.full_problem))
    sealed = result.audit_dict()
    sealed['prepared_numeric_state_bytes'] = _unique_bytes(values)
    object.__setattr__(result,'audit',_freeze(sealed))
    _emit(log_callback,'PREPARED',result.audit_dict(),context)
    return result


@dataclass(frozen=True)
class ObjectiveCache:
    prepared: MarginJointTraining
    Z: np.ndarray
    folds: tuple
    def __post_init__(self):
        object.__setattr__(self,'Z',_readonly(self.Z))


def _cache_values(cache):
    if cache is None:
        return ()
    return (cache.Z,[({key:value for key,value in cf.items() if key not in ('problem','qp_state','audit')},
                      _problem_values(cf['problem']),None if cf.get('qp_state') is None else cf['qp_state'].arrays)
                     for cf in cache.folds])


def _forward_cost(folds):
    result = {key:0 for key in STAGE_COUNTERS}
    for cache in folds:
        audit = cache['audit']
        _add(result,audit,_GEOMETRY_COUNTERS+_HEAD_COUNTERS+('completed_factorization_count',)+
             tuple('margin_qp_forward_'+key for key in QP_WORK_KEYS))
        result['inner_head_fit_count'] += audit.get('head_fit_count',0)
        result['inner_factorization_count'] += audit.get('factorization_count',0)
        for key in ('margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes'):
            result[key] = max(result.get(key,0),audit.get(key,0))
    result['inner_objective_evaluation_count'] = 1
    return result


def evaluate_margin_joint_objective(prepared,Z,anchor_U=None,*,gradient=True,forward_cache=None):
    start = time.perf_counter()
    anchor = prepared.anchor_U if anchor_U is None else fcr._parameter(anchor_U)
    if not np.array_equal(anchor,prepared.anchor_U):
        raise ValueError('Objective anchor differs from actual B')
    Z = np.asarray(Z,dtype=np.float64)
    U = reconstruct_U(Z,anchor,prepared.W)
    reused = forward_cache is not None
    if reused and (forward_cache.prepared is not prepared or not np.array_equal(forward_cache.Z,Z)):
        raise ValueError('Objective forward cache binding differs')
    if reused:
        folds = forward_cache.folds
    else:
        partial = []
        try:
            for p in prepared.problems:
                partial.append(_forward(p,U))
        except _FAILURES as exc:
            exc.joint_objective_audit = _forward_cost(partial+[dict(audit=getattr(exc,'joint_forward_audit',{}))])
            exc.joint_partial_folds = tuple(partial)
            raise
        folds = tuple(partial)
    cache = forward_cache if reused else ObjectiveCache(prepared,Z,folds)
    c = len(prepared.classes)
    sums,counts,ces,ds = np.zeros(c),np.zeros(c,dtype=np.int64),[],[]
    for cf in folds:
        labels = cf['problem'].held_labels
        ce,d = affine._ce(cf['score'],labels)
        sums += np.bincount(labels,weights=ce,minlength=c)
        counts += np.bincount(labels,minlength=c)
        ces.append(ce)
        ds.append(d)
    means = np.divide(sums,counts,out=np.zeros(c),where=counts>0)
    risk = _norm(means)/np.sqrt(c)
    info = {key:0 for key in STAGE_COUNTERS}
    if not reused:
        info.update(_forward_cost(folds))
    info['backward_evaluation_count'] = int(gradient)
    gU,fold_audits = np.zeros((736,8)),[]
    for cf,ce,d in zip(folds,ces,ds):
        labels = cf['problem'].held_labels
        fa = deepcopy(cf['audit'])
        fa.update(held_ce_sums=np.bincount(labels,weights=ce,minlength=c).tolist(),
            held_ce_counts=np.bincount(labels,minlength=c).tolist(),held_correct_count=int(np.sum(np.argmax(cf['score'],axis=1)==labels)))
        fa.update({key:0 for key in _ADJOINT_COUNTERS})
        if gradient and risk>0:
            weights = means[labels]/(c*risk*counts[labels])
            try:
                gU += _backward(cf,d*weights[:,None],fa)
            except _FAILURES as exc:
                _add(info,fa,_ADJOINT_COUNTERS)
                for key in ('margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes'):
                    info[key] = max(info.get(key,0),fa.get(key,0))
                exc.joint_objective_audit = info
                exc.joint_partial_folds = folds
                raise
            _add(info,fa,_ADJOINT_COUNTERS)
            for key in ('margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes'):
                info[key] = max(info.get(key,0),fa.get(key,0))
        fold_audits.append(fa)
    gZ = gU@prepared.W if gradient else np.zeros_like(Z)
    info.update(RMSCE=float(risk),loss_ce=float(risk),loss_task=float(risk),loss_proximal=0.,loss_total=float(risk),
        class_ce_sums=sums.tolist(),class_ce_counts=counts.tolist(),class_ce_means=means.tolist(),temperature=1.,
        gradient_norm=_norm(gZ) if gradient else None,inner_folds=fold_audits,forward_cache_reused=reused,
        coordinate_ball_radius=.5,coordinate_ball_feasible=bool(_norm(Z)<=.5+128*_EPS),
        forward_cache_bytes=_unique_bytes(_cache_values(cache)),objective_seconds=time.perf_counter()-start,
        **fcr._functional_measurement(prepared,U,anchor,Z))
    _finite(risk,gZ)
    return float(risk),gZ,info,cache


def _capture_single_query(z_id,fft,t_emb,f_emb,pa_local):
    captured = {}
    for name,value,width in zip(_NAMES,(z_id,fft,t_emb,f_emb,pa_local),(160,96,160,160,160)):
        array = np.asarray(value,dtype=np.float64)
        if array.shape != (1,width):
            raise ValueError('Prior score reuse requires exactly one complete sample: '+name)
        captured[name] = _readonly(array)
    return _freeze(captured)


def _immutable_array_binding(array,*,capture_values=True,allow_snapshot=False):
    """No hashes: sealed buffers need metadata checks; other buffers need values."""
    if not isinstance(array,np.ndarray):
        raise ValueError('Prior scoring state must contain numeric arrays')
    root = array
    while isinstance(root.base,np.ndarray):
        root = root.base
    sealed = isinstance(root.base,bytes)
    if not sealed and not allow_snapshot:
        raise ValueError('Prior score reuse requires sealed B numeric storage')
    buffer = root.base if root.base is not None else root
    return (array,array.shape,array.dtype.str,array.strides,
            int(array.__array_interface__['data'][0]),buffer,
            None if sealed or not capture_values else _readonly(array))


def _check_array_binding(now,old,*,scope):
    array,shape,dtype,strides,address,buffer,snapshot = old
    if (now[0] is not array or now[1:5] != (shape,dtype,strides,address) or
            now[5] is not buffer):
        raise ValueError(scope+': numeric storage changed')
    if snapshot is not None and not np.array_equal(array,snapshot):
        raise ValueError(scope+': numeric values changed')


def _scoring_numeric_binding(value,*,capture_values=True,allow_snapshot=False):
    if isinstance(value,np.ndarray):
        return ('array',_immutable_array_binding(value,capture_values=capture_values,allow_snapshot=allow_snapshot))
    if type(value) in (int,float) or isinstance(value,(np.integer,np.floating)):
        if not np.isfinite(value):
            raise ValueError('Prior scoring scalar must be finite')
        # Preserve the actual scalar type and value, including signed zero.
        # Production centering stores reference_self/grand as Python floats.
        return ('scalar',type(value),value,bool(np.signbit(value)) if value==0 else None)
    raise ValueError('Prior scoring state must contain numeric arrays or immutable real scalars')


def _check_scoring_numeric_binding(now,old,*,scope):
    if now[0] != old[0]:
        raise ValueError(scope+': numeric storage changed')
    if old[0]=='array':
        _check_array_binding(now[1],old[1],scope=scope)
    elif now[1:] != old[1:]:
        raise ValueError(scope+': numeric scalar type or value changed')


def _prior_scoring_binding(state,*,capture_values=True):
    if state.prior is not None or state.problem.mode != 'B':
        raise ValueError('Reusable prior scores require the actual old-only B state')
    if tuple(state.classes) != tuple(state.problem.classes) or len(set(state.classes)) != len(state.classes):
        raise ValueError('Invalid B class column order')
    p,cache = state.problem,state.final_cache
    num = cache['numeric']
    arrays = [state.U,p.train_labels,p.old_indices,num['intercept']]
    dictionary = None
    if p.gamma is not None:
        arrays += [p.train_context['original'],p.q,num['reference'],num['reference_self'],
                   num['mean'],num['grand'],num['alpha']]
        if p.tau != 0:
            # _context's fixed dictionary can be ordinary read-only NumPy storage;
            # capture values as well rather than trusting writeable=False alone.
            dictionary = fcr._V0
            if np.any(state.U):
                arrays += [cache['b'],cache['a']]
    bindings = tuple(_scoring_numeric_binding(value,capture_values=capture_values) for value in arrays)
    if dictionary is not None:
        bindings += (_scoring_numeric_binding(dictionary,capture_values=capture_values,allow_snapshot=True),)
    return ((p,cache,num,state.U,p.train_context,state.audit,p.audit),
            (tuple(state.classes),tuple(p.classes),tuple(state.ids),tuple(state.old_classes),p.mode,p.tau,p.gamma),bindings)


def _check_prior_scoring_binding(state,binding):
    try:
        current = _prior_scoring_binding(state,capture_values=False)
    except ValueError as exc:
        raise ValueError('Expired prior score cache: '+str(exc)) from exc
    if (any(x is not y for x,y in zip(current[0],binding[0])) or
            current[1] != binding[1] or len(current[2]) != len(binding[2])):
        raise ValueError('Expired prior score cache: B scoring state changed')
    for now,old in zip(current[2],binding[2]):
        _check_scoring_numeric_binding(now,old,scope='Expired prior score cache: B')


@dataclass(frozen=True)
class _SingleQueryPriorCache:
    """Ephemeral single-input value cache, not a fitted/query-statistic state."""
    prior: object
    features: Mapping
    background: np.ndarray
    auxiliary: np.ndarray
    scores: np.ndarray
    classes: tuple
    binding: tuple
    def __post_init__(self):
        object.__setattr__(self,'features',_freeze({name:_readonly(value) for name,value in self.features.items()}))
        for name in ('background','auxiliary','scores'):
            object.__setattr__(self,name,_readonly(getattr(self,name)))
        arrays = tuple(self.features[name] for name in _NAMES)+(self.background,self.auxiliary,self.scores)
        object.__setattr__(self,'_array_bindings',tuple(_immutable_array_binding(array) for array in arrays))


@dataclass(frozen=True)
class MarginJointState:
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
    problem: MarginJointProblem
    final_cache: Mapping
    prior: object
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        for key in ('U','Z','W','anchor_U','H','singular_values','labels'):
            object.__setattr__(self,key,_readonly(getattr(self,key)))
        def freeze(value):
            if isinstance(value,np.ndarray):
                return _readonly(value)
            if isinstance(value,Mapping):
                return _freeze({key:freeze(item) for key,item in value.items()})
            return value
        object.__setattr__(self,'raw',freeze(self.raw))
        object.__setattr__(self,'final_cache',freeze(self.final_cache))
        object.__setattr__(self,'records',_freeze(self.records))
        object.__setattr__(self,'audit',_freeze(self.audit))
    @property
    def V(self):
        return V0
    @property
    def V0(self):
        return V0
    def audit_dict(self):
        return ch._safe(_plain(self.audit))
    def state_records(self):
        return {key:{name:np.array(value,copy=True) for name,value in record.items()} for key,record in self.records.items()}
    def _score_geometry(self,b,a,progress=None):
        residual,prior = {},{}
        out = _score_residual(self.problem,self.final_cache,self.U,b,a,residual)
        if self.prior is not None:
            old = _score_residual(self.prior.problem,self.prior.final_cache,self.prior.U,b,a,prior)
            for col,name in enumerate(self.prior.classes):
                out[:,self.classes.index(name)] += old[:,col]
        if progress is not None:
            progress.update(residual=residual,prior=prior)
            for key in set(residual)|set(prior):
                progress[key] = residual.get(key,0)+prior.get(key,0)
        _finite(out)
        return out
    def score(self,*,z_id,fft,t_emb,f_emb,pa_local):
        return self._score_geometry(*interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True))
    def score_with_audit(self,*,z_id,fft,t_emb,f_emb,pa_local):
        start = time.perf_counter()
        b,a = interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True)
        progress = {}
        scores = self._score_geometry(b,a,progress)
        progress.update(score_physical_count=len(b),score_seconds=time.perf_counter()-start,
            reference_distance_scope='subset_of_raw_distance_work_not_additive')
        return scores,progress
    def score_single_for_reuse(self,*,z_id,fft,t_emb,f_emb,pa_local):
        """Score actual B once; return scores, its normal audit, and a bound packet.

        Optional API only. Existing score_with_audit and benchmark callers do not
        use this path. Capture float64 inputs before scoring; never retain caller
        views or trust a caller assertion about which sample produced the prior.
        """
        start = time.perf_counter()
        captured = _capture_single_query(z_id,fft,t_emb,f_emb,pa_local)
        binding = _prior_scoring_binding(self)
        before_score = time.perf_counter()
        scores,audit = self.score_with_audit(**captured)
        after_score = time.perf_counter()
        _check_prior_scoring_binding(self,binding)
        if scores.dtype != np.float64 or scores.shape != (1,len(self.classes)):
            raise ValueError('B scores must retain original single-sample float64 columns')
        b,a = interaction._blocks(**captured,allow_empty=False)
        packet = _SingleQueryPriorCache(self,captured,b,a,scores,tuple(self.classes),binding)
        audit = dict(audit,packet_preparation_seconds=(before_score-start)+(time.perf_counter()-after_score),
                     single_query_api_seconds=time.perf_counter()-start)
        return scores,audit,packet
    def score_single_with_reused_prior(self,packet,*,z_id,fft,t_emb,f_emb,pa_local):
        """Reuse only this actual B's same-input scores; evaluate all C residuals.

        Packet production work belongs to the earlier B call, not this audit.
        This cache provides no cross-sample statistics and never changes a head.
        """
        start = time.perf_counter()
        if not isinstance(packet,_SingleQueryPriorCache):
            raise ValueError('Single-query prior score packet required')
        if self is not packet.prior and self.prior is not packet.prior:
            raise ValueError('Prior score packet does not belong to this actual B')
        arrays = tuple(packet.features[name] for name in _NAMES)+(packet.background,packet.auxiliary,packet.scores)
        for array,binding in zip(arrays,packet._array_bindings):
            _check_array_binding(_immutable_array_binding(array),binding,scope='Invalid prior packet column shape or view')
        captured = _capture_single_query(z_id,fft,t_emb,f_emb,pa_local)
        for name in _NAMES:
            if (packet.features[name].shape != captured[name].shape or
                    packet.features[name].dtype != np.float64 or
                    not np.array_equal(packet.features[name].view(np.uint64),captured[name].view(np.uint64))):
                raise ValueError('Prior score packet belongs to a different single input')
        _check_prior_scoring_binding(packet.prior,packet.binding)
        if (packet.classes != tuple(packet.prior.classes) or packet.scores.dtype != np.float64 or
                packet.scores.shape != (1,len(packet.classes)) or
                packet.background.shape != (1,256) or packet.auxiliary.shape != (1,480)):
            raise ValueError('Invalid prior packet geometry or score column shape')
        if tuple(self.classes) != tuple(self.problem.classes) or len(set(self.classes)) != len(self.classes):
            raise ValueError('Invalid registered class column order')
        if not set(packet.classes).issubset(self.classes):
            raise ValueError('Prior class columns are missing from registered classes')
        columns = [self.classes.index(name) for name in packet.classes]
        validation_seconds = time.perf_counter()-start
        residual = {}
        if self is packet.prior:
            # Existing new0 semantics: C is literally B, hence no residual call.
            out = packet.scores.copy()
        else:
            try:
                out = _score_residual(self.problem,self.final_cache,self.U,
                                      packet.background,packet.auxiliary,residual)
            except Exception as exc:
                # Work reported before the failure is known; missing operations
                # and completion counts remain unknown, never fabricated zeros.
                exc.single_query_reuse_audit = dict(residual=residual,prior={},status='TECHNICAL_FAILURE',
                    prior_reused=True,prior_work_executed=False,score_physical_count=None,
                    score_seconds=time.perf_counter()-start,cache_validation_seconds=validation_seconds)
                raise
            if out.shape != (1,len(self.classes)):
                raise ValueError('Invalid C residual score columns')
            for col,target in enumerate(columns):
                out[:,target] += packet.scores[:,col]
        _finite(out)
        progress = dict(residual)
        progress.update(residual=residual,prior={},score_physical_count=1,
            score_seconds=time.perf_counter()-start,cache_validation_seconds=validation_seconds,
            prior_reused=True,prior_work_executed=False,prior_score_evaluation_count=0,
            prior_score_physical_count=0,reused_prior_physical_count=1,prior_score_seconds=None,
            composition_addition_count=0 if self is packet.prior else len(columns),
            prior_score_seconds_reason='NOT_EXECUTED_THIS_CALL_ALREADY_PAID_BY_B_SCORE',
            prior_column_indices=columns,new0_exact_B_reuse=self is packet.prior,
            reference_distance_scope='subset_of_raw_distance_work_not_additive')
        return out,progress
    def predict(self,**features):
        order = np.asarray(sorted(range(len(self.classes)),key=lambda i:self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(self.score(**features)[:,order],axis=1)]]
    def to_arrays(self):
        arrays = _head_arrays(dict(self.final_cache,problem=self.problem))
        arrays.update(self.raw,U=self.U,Z=self.Z,W=self.W,anchor_U=self.anchor_U,H=self.H,singular_values=self.singular_values,V0=V0)
        if self.prior is not None:
            arrays.update({'prior_B_'+key:value for key,value in self.prior.to_arrays().items()})
            arrays['prior_old_class_indices'] = np.array([self.classes.index(name) for name in self.prior.classes],dtype=np.int64)
        return arrays
    def _resident_values(self):
        values = (self.U,self.Z,self.W,self.anchor_U,self.H,self.singular_values,self.raw,self.labels,V0,
                  _problem_values(self.problem),self.final_cache,self.records)
        return values if self.prior is None else values+(self.prior._resident_values(),)
    def _deployment_values(self):
        num = self.final_cache['numeric']
        keys = ('alpha','intercept','reference','reference_self','mean','grand') if self.problem.mode=='B' else ('alpha','b')
        if self.problem.gamma is None:
            values = (num['intercept'] if self.problem.mode=='B' else num['b'],)
        else:
            values = (self.problem.train_context['original'],{key:num[key] for key in keys})
            if self.problem.mode=='B':
                values += (self.problem.q,)
            if self.problem.tau!=0:
                values += (self.U,V0,self.final_cache['b'],self.final_cache['a'])
        return values if self.prior is None else values+(self.prior._deployment_values(),)


def _cache_for_state(cache):
    return {key:value for key,value in cache.items() if key not in ('problem','bc','hc','qp_state','adjoint_arrays')}


def fit_margin_joint_local_ridge(prepared,*,mode='B',baseline_state=None,log_callback=None,state_callback=None):
    if mode not in ('B','C_seq') or (mode=='B')!=(prepared.inherited is None):
        raise ValueError('Mode does not match actual B inheritance')
    if prepared.audit.get('no_information_reason')=='N0_REUSE_ACTUAL_B':
        return prepared.inherited
    start = time.perf_counter()
    rec = _Recorder(state_callback)
    rec.records.update(dict(prepared.records))
    audit = {key:0 for key in STAGE_COUNTERS}
    audit.update(schema=FROZEN_CONFIG['schema'],method=FROZEN_CONFIG['method'],mode=mode,
        config=_actual_config(prepared.full_problem.max_transitions,prepared.full_problem.max_factor_buffer_bytes),
        preparation=prepared.audit_dict(),ajlr_stage_count=1,no_information=prepared.audit['no_information'],
        gradients=[],trials=[],steps=[],source_validation=None,source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        peak_rss_bytes=None,peak_gpu_memory_bytes=None,serialized_deployment_bytes=None,incremental_transmission_bytes=None,
        hardware_dtype='CPU_float64',device_model=None,device_model_reason='NOT_MEASURED',
        qp_max_transitions=prepared.full_problem.max_transitions,qp_max_factor_buffer_bytes=prepared.full_problem.max_factor_buffer_bytes)
    Z,U,cache,finalprogress = np.zeros((736,prepared.r)),prepared.anchor_U.copy(),None,{}
    def cost(info):
        _add(audit,info,STAGE_COUNTERS)
        for key in ('forward_cache_bytes','margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes'):
            target = 'max_'+key if key=='forward_cache_bytes' else key
            audit[target] = max(audit.get(target,0),info.get(key,0))
    def event(name,payload):
        _emit(log_callback,name,dict(mode=mode,**payload),prepared.audit)
    def save_objective(key,obj,oc,**extra):
        scores = np.concatenate([cf['score'] for cf in oc.folds]) if oc is not None and oc.folds else np.empty((0,len(prepared.classes)))
        labels = np.concatenate([cf['problem'].held_labels for cf in oc.folds]) if oc is not None and oc.folds else np.empty(0,dtype=np.int64)
        if oc is not None:
            for index,cf in enumerate(oc.folds):
                ref = rec.save(key+'_fold_'+str(index),U=U,**_head_arrays(cf),
                    **{'adjoint_'+name:value for name,value in cf.get('adjoint_arrays',{}).items()})
                obj['inner_folds'][index]['head_state_ref'] = ref
        return rec.save(key,Z=Z,U=U,W=prepared.W,anchor_U=prepared.anchor_U,scores=scores,labels=labels,
            RMSCE=_scalar(obj.get('RMSCE')),prox=_scalar(0.),class_ce_sums=np.asarray(obj.get('class_ce_sums',[])),
            class_ce_counts=np.asarray(obj.get('class_ce_counts',[]),dtype=np.int64),class_ce_means=np.asarray(obj.get('class_ce_means',[])),**extra)
    try:
        if audit['no_information']:
            audit['stop_reason'] = prepared.audit['no_information_reason']
            audit['initialization_state_ref'] = save_objective('initial',{},None)
            event('INITIAL',dict(state_ref=audit['initialization_state_ref'],no_update_reason=audit['stop_reason']))
        else:
            loss,_,obj,cache = evaluate_margin_joint_objective(prepared,Z,gradient=False)
            cost(obj)
            audit['initialization_state_ref'] = save_objective('initial',obj,cache)
            audit['initial_objective'],audit['stop_reason'] = deepcopy(obj),'MAX_ITERATIONS'
            event('INITIAL',dict(objective=obj,state_ref=audit['initialization_state_ref']))
            for iteration in range(1,5):
                audit['optimizer_iterations'] += 1
                loss,g,obj,cache = evaluate_margin_joint_objective(prepared,Z,gradient=True,forward_cache=cache)
                cost(obj)
                norm = _norm(g)
                direction = np.zeros_like(g) if norm==0 else -g/norm
                ref = save_objective('gradient_'+str(iteration),obj,cache,g_Z=g,d_Z=direction)
                gr = dict(iteration=iteration,state_ref=ref,gradient_norm=norm,direction_norm=_norm(direction),objective=deepcopy(obj))
                audit['gradients'].append(gr)
                event('GRADIENT',gr)
                if norm==0:
                    audit['stop_reason'] = 'ZERO_GRADIENT'
                    break
                accepted = False
                for trial in range(1,13):
                    step = .125*.5**(trial-1)
                    tz = project_coordinates(Z+step*direction)
                    delta = tz-Z
                    if not np.any(delta):
                        audit['stop_reason'] = 'ZERO_FEASIBLE_DISPLACEMENT'
                        break
                    audit['trial_attempt_count'] += 1
                    try:
                        tl,_,to,tc = evaluate_margin_joint_objective(prepared,tz,gradient=False)
                    except _FAILURES as exc:
                        exc.joint_attempt_Z,exc.joint_attempt_U = tz.copy(),reconstruct_U(tz,prepared.anchor_U,prepared.W)
                        raise
                    cost(to)
                    audit['max_simultaneous_forward_cache_bytes'] = max(audit.get('max_simultaneous_forward_cache_bytes',0),
                        _unique_bytes((_cache_values(cache),_cache_values(tc))))
                    acceptance = _trial_acceptance(loss,tl,float(np.sum(g*delta)))
                    oldz,oldu = Z,U
                    Z,U = tz,reconstruct_U(tz,prepared.anchor_U,prepared.W)
                    tref = save_objective('trial_'+str(iteration)+'_'+str(trial),to,tc,delta_Z=delta,d_Z=direction)
                    Z,U = oldz,oldu
                    tr = dict(iteration=iteration,trial=trial,step_size=step,state_ref=tref,gradient_state_ref=ref,
                        loss_before=loss,loss_after=tl,update_norm=_norm(delta),objective=deepcopy(to),**acceptance)
                    audit['trial_count'] += 1
                    audit['trials'].append(tr)
                    accepted = acceptance['accepted']
                    audit['accepted_trial_count' if accepted else 'rejected_trial_count'] += 1
                    event('TRIAL',tr)
                    if accepted:
                        Z,U,cache,loss = tz,reconstruct_U(tz,prepared.anchor_U,prepared.W),tc,tl
                        audit['optimizer_steps'] += 1
                        sr = dict(step=audit['optimizer_steps'],iteration=iteration,trial=trial,state_ref=tref,
                            loss_before=tr['loss_before'],loss_after=tl,learning_rate=step,update_norm=_norm(delta),objective=deepcopy(to))
                        audit['steps'].append(sr)
                        event('STEP',sr)
                        break
                if not accepted:
                    if audit['stop_reason']!='ZERO_FEASIBLE_DISPLACEMENT':
                        audit['stop_reason'] = 'TRIAL_BUDGET_EXHAUSTED'
                    break
            _,_,final_obj,_ = evaluate_margin_joint_objective(prepared,Z,gradient=False,forward_cache=cache)
            audit['final_objective_state_ref'] = save_objective('final_objective',final_obj,cache)
            audit['final_objective'] = final_obj
        final = _forward(prepared.full_problem,U,finalprogress)
        audit['final_head_fit_count'],audit['final_factorization_count'] = 1,finalprogress['factorization_count']
        _add(audit,finalprogress,_GEOMETRY_COUNTERS+_HEAD_COUNTERS+('completed_factorization_count',)+
             tuple('margin_qp_forward_'+key for key in QP_WORK_KEYS))
        for key in ('margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes'):
            audit[key] = max(audit.get(key,0),finalprogress.get(key,0))
        arrays = _head_arrays(final)
        arrays.update(prepared.raw,U=U,Z=Z,W=prepared.W,anchor_U=prepared.anchor_U,H=prepared.H,singular_values=prepared.singular_values,V0=V0)
        if prepared.inherited is not None:
            arrays.update({'prior_B_'+key:value for key,value in prepared.inherited.to_arrays().items()})
            arrays['prior_old_class_indices'] = np.array([prepared.classes.index(name) for name in prepared.inherited.classes],dtype=np.int64)
        audit['final_state_ref'] = rec.save('final',**arrays)
        trained = 736*prepared.r if audit['optimizer_steps'] else 0
        audit.update(final_fit=_plain(finalprogress),coordinate_norm=_norm(Z),parameter_U_norm=_norm(U),
            parameter_changed_from_anchor=bool(np.any(U!=prepared.anchor_U)),u_changed_from_anchor=bool(np.any(U!=prepared.anchor_U)),
            declared_coordinate_scalar_count=736*prepared.r,trainable_parameter_count=0 if audit['no_information'] else 736*prepared.r,
            active_parameter_count=trained,trained_parameter_count=trained,optimizer_parameter_count=0 if audit['no_information'] else 736*prepared.r,
            changed_coordinate_scalar_count=int(np.count_nonzero(Z)),
            analytic_coefficient_parameter_count=len(prepared.ids)*len(prepared.classes),analytic_intercept_parameter_count=len(prepared.classes),
            analytic_head_parameter_count=(len(prepared.ids)+1)*len(prepared.classes),inherited_from_B=prepared.inherited is not None,
            fit_seconds=time.perf_counter()-start,status='COMPLETED',retained_vector_record_bytes=_unique_bytes(rec.records),
            prepared_numeric_state_bytes=prepared.audit['prepared_numeric_state_bytes'])
        state = MarginJointState(U,Z,prepared.W,prepared.anchor_U,prepared.H,prepared.singular_values,prepared.raw,prepared.labels,
            prepared.ids,prepared.classes,prepared.old_classes,prepared.full_problem,_cache_for_state(final),prepared.inherited,rec.records,audit)
        audit.update(resident_numeric_state_bytes=_unique_bytes(state._resident_values()),deployment_numeric_state_bytes=_unique_bytes(state._deployment_values()),
            state_byte_scope='physical_unique_retained_numeric_buffers_includes_in_memory_records_excludes_Python_library_workspace',
            deployment_byte_scope='numeric_expansion_and_geometry_and_actual_B_only_not_process_peak_or_serialized_packet')
        object.__setattr__(state,'audit',_freeze(audit))
        event('FINAL',dict(state_ref=audit['final_state_ref'],audit=state.audit_dict()))
        return state
    except _FAILURES as exc:
        if hasattr(exc,'joint_objective_audit'):
            cost(exc.joint_objective_audit)
        elif finalprogress:
            audit['final_head_fit_count'] = finalprogress.get('head_fit_count',0)
            audit['final_factorization_count'] = finalprogress.get('factorization_count',0)
            _add(audit,finalprogress,_GEOMETRY_COUNTERS+_HEAD_COUNTERS+('completed_factorization_count',)+
                 tuple('margin_qp_forward_'+key for key in QP_WORK_KEYS))
            for key in ('margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes'):
                audit[key] = max(audit.get(key,0),finalprogress.get(key,0))
        attemptZ,attemptU = getattr(exc,'joint_attempt_Z',Z),getattr(exc,'joint_attempt_U',U)
        audit['failure_completed_head_refs'] = [rec.save('failure_completed_fold_'+str(index),U=attemptU,**_head_arrays(cf))
            for index,cf in enumerate(getattr(exc,'joint_partial_folds',()))]
        failure_arrays = dict(Z=attemptZ,U=attemptU,last_accepted_Z=Z,last_accepted_U=U,
            W=prepared.W,anchor_U=prepared.anchor_U,**getattr(exc,'joint_failed_head_arrays',{}))
        audit['failure_state_ref'] = rec.save('failure',**failure_arrays)
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),failure_code=getattr(exc,'code',None),
            failure_head_audit=_plain(getattr(exc,'joint_forward_audit',getattr(exc,'joint_backward_audit',{}))),
            qp_failure_audit=exc.audit if isinstance(exc,qp.MarginQPFailure) else None,fit_seconds=time.perf_counter()-start)
        event('FAILURE',dict(audit=audit,state_ref=audit['failure_state_ref']))
        error = NumericalFailure(str(exc),audit)
        error.records = rec.records
        error.arrays = _freeze({key:_readonly(value) for key,value in failure_arrays.items()})
        raise error from exc


def predict_margin_joint_local_ridge(state,**features):
    if not isinstance(state,MarginJointState):
        raise ValueError('MARGIN_JOINT state required')
    return state.predict(**features)
