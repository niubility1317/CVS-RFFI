"""Independent support-only all-pair barrier joint draft; no experiment I/O.

B delegates the existing CE-only MarginJoint B implementation unchanged. C freezes actual B,
fits a new-only analytic affine ridge and a full-support Bernoulli barrier gate,
and updates only the whitened DCT8 coordinates using CE-only projected Armijo.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import time

import numpy as np
from scipy.linalg import solve_triangular

from . import d92_affine_joint_local_ridge as affine
from . import d92_conditional_joint_local_ridge as projected
from . import d92_group_barrier_gate as gate
from . import d92_margin_joint_local_ridge as inherited_B
from . import d92_function_coordinate_residual8_local_ridge as fcr
from . import d92_joint_channel_local_ridge as ch
from . import d92_branch_local_ridge as local
from . import d92_branch_interaction as interaction
from .d92_branch_ridge import _freeze, _plain

_NAMES = fcr._NAMES
V0 = fcr.V0
_readonly = affine._readonly
_finite = ch._finite
_norm = ch._norm
NumericalFailure = ch.NumericalFailure
reconstruct_U = fcr.reconstruct_U
project_coordinates = projected.project_coordinates
_FAILURES = (gate.GroupBarrierFailure, FloatingPointError, np.linalg.LinAlgError, OverflowError)

FROZEN_CONFIG = dict(
    schema='d92_group_barrier_joint_local_ridge_v1', method='D92-GroupBarrierJointLocalRidge-v1',
    candidate_status='SOURCE_ONLY_DRAFT_NOT_PUBLISHED', B='unchanged_CE_only_hard_ball_MarginJoint_B',
    C='new_only_affine_ridge_plus_all_pair_fixed_positive_barrier_Bernoulli_gate',
    B_objective=inherited_B.FROZEN_CONFIG['objective'], C_objective='RMS_class_mean_CE_only',
    rank=8, input_dim=736, dictionary='first8_orthonormal_DCT_rows', learn_V=False,
    initialization_C='actual_current_B_U_no_reset', coordinate_ball_radius=.5,
    proximal_coefficient=0., ridge_coefficient=1., temperature=1.,
    barrier_average_original_objective_gap=gate.AVERAGE_DUAL_GAP,
    barrier_law='zeta=N*1e-4/(old_physical_count*new_class_count)_fixed_per_head',
    bandwidth='fixed_original_old_inner_train', trace_scale='fixed_actual_old_prior',
    max_iterations=4, max_trials=12, initial_step_size=.125, backtrack_factor=.5,
    free_new_intercept=True, free_gate_intercept=True, max_coordinates=5888,
    query_decision_policy='per_physical_sample_all_registered_classes',
    tie_break='physical_class_id_lexicographic', phase1_frozen=True,
    source_inputs=False, query_fit=False, encoder_backward=False, parameter_search=False,
    extra_ground_data_stat_payload_bytes=0, code_wire_payload_bytes=None, dtype='float64')

PREPARATION_COUNTERS = inherited_B.PREPARATION_COUNTERS
_GEOMETRY = ('raw_distance_evaluation_count', 'raw_distance_pair_count',
    'reference_distance_evaluation_count', 'reference_distance_pair_count',
    'kernel_evaluation_count', 'kernel_pair_count', 'adapter_physical_evaluation_count')
_GATE_WORK = ('factorization_attempts', 'factorizations_completed', 'condition_estimation_calls',
    'triangular_calls', 'triangular_rhs_columns', 'triangular_rhs_elements', 'triangular_dense_work_units',
    'spectral_checks', 'spectral_cubic_dimension_units', 'objective_evaluations',
    'logistic_record_evaluations', 'barrier_constraint_evaluations', 'line_search_trials',
    'newton_iterations', 'accepted_steps', 'factorization_seconds', 'triangular_seconds',
    'spectral_seconds', 'objective_seconds', 'wall_seconds')
_RIDGE_WORK = ('new_ridge_fit_count', 'new_ridge_factorization_count', 'new_ridge_factorization_attempts',
    'new_ridge_factorizations_completed', 'new_ridge_triangular_calls', 'new_ridge_triangular_completed_calls',
    'new_ridge_triangular_rhs_columns', 'new_ridge_triangular_rhs_elements',
    'new_ridge_triangular_dense_work_units', 'new_ridge_adjoint_calls',
    'new_ridge_adjoint_triangular_calls', 'new_ridge_adjoint_triangular_completed_calls', 'new_ridge_adjoint_rhs_columns',
    'new_ridge_adjoint_rhs_elements', 'new_ridge_adjoint_dense_work_units', 'new_ridge_seconds',
    'new_ridge_adjoint_seconds')
STAGE_COUNTERS = tuple(dict.fromkeys(
    ('ajlr_stage_count', 'optimizer_steps', 'optimizer_iterations', 'trial_attempt_count', 'trial_count',
     'accepted_trial_count', 'rejected_trial_count', 'inner_objective_evaluation_count',
     'inner_head_fit_count', 'final_head_fit_count', 'backward_evaluation_count') + _GEOMETRY +
    _RIDGE_WORK + tuple('group_gate_'+scope+'_'+name for scope in ('forward', 'adjoint') for name in _GATE_WORK)))
AUDIT_COUNTERS = tuple(dict.fromkeys(PREPARATION_COUNTERS + STAGE_COUNTERS))


def _unique_bytes(values):
    seen = set()
    def visit(value):
        if isinstance(value, np.ndarray):
            root = value
            while isinstance(root.base, np.ndarray): root = root.base
            backing = root.base if isinstance(root.base, (bytes, bytearray)) else root
            if id(backing) in seen: return 0
            seen.add(id(backing))
            return len(backing) if isinstance(backing, (bytes, bytearray)) else root.nbytes
        if isinstance(value, Mapping): return sum(visit(x) for x in value.values())
        if isinstance(value, (tuple, list)): return sum(visit(x) for x in value)
        if isinstance(value, gate.GroupBarrierState): return visit(value.arrays)
        return 0
    return visit(values)


def _failure_readonly(value):
    """Seal original failure numbers, including NaN/Inf; never repair values."""
    array=np.asarray(value)
    if array.dtype.kind not in 'fibu': raise ValueError('Failure state requires numeric arrays')
    dtype=np.bool_ if array.dtype.kind=='b' else np.int64 if array.dtype.kind in 'iu' else np.float64
    array=np.ascontiguousarray(array,dtype=dtype).reshape(array.shape)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


class _Recorder(affine._Recorder):
    """Success stays strict-finite; failure archives preserve numerical evidence."""
    def save(self,key,**arrays):
        if not key.startswith('failure'): return super().save(key,**arrays)
        if key in self.keys: raise ValueError('Duplicate GROUP_BARRIER failure state key '+key)
        self.keys.add(key)
        values={name:_failure_readonly(value) for name,value in arrays.items()}
        metadata={name:dict(shape=list(value.shape),dtype=str(value.dtype),nbytes=int(value.nbytes),
            nonfinite_count=int(np.count_nonzero(~np.isfinite(value)))) for name,value in values.items()}
        if self.callback is None:
            self.records[key]=_freeze(values); ref=dict(storage='in_memory',key=key,arrays=metadata)
        else:
            ref=ch._safe(self.callback(key,values))
            if not isinstance(ref,dict): raise ValueError('state_callback must return JSON mapping')
            ref=dict(ref,key=key,arrays=metadata)
        return ref


def _add(target, source, names=STAGE_COUNTERS):
    for name in names: target[name] = target.get(name, 0) + source.get(name, 0)
    for name in ('gate_peak_factor_buffer_bytes', 'gate_peak_explicit_temporary_bytes', 'forward_cache_bytes'):
        target[name] = max(target.get(name, 0), source.get(name, 0))


def _gate_account(target, source, scope):
    for name in _GATE_WORK:
        key = 'group_gate_'+scope+'_'+name
        target[key] = target.get(key, 0) + source.get(name, 0)
    for name, origin in (('gate_peak_factor_buffer_bytes', 'peak_factor_buffer_bytes'),
                         ('gate_peak_explicit_temporary_bytes', 'peak_explicit_temporary_bytes')):
        target[name] = max(target.get(name, 0), source.get(origin, 0))


def _limits(max_newton_iterations, max_line_search_trials, max_factor_buffer_bytes):
    values = dict(max_newton_iterations=max_newton_iterations, max_line_search_trials=max_line_search_trials,
                  max_factor_buffer_bytes=max_factor_buffer_bytes)
    for name, value in values.items(): gate._positive_int(value, name)
    return values


def _ridge_triangular(chol,rhs,info,*,adjoint=False,transpose=False):
    """Count an actual attempted call before execution; completed is separate."""
    n,columns=rhs.shape
    if adjoint:
        updates=dict(new_ridge_adjoint_triangular_calls=1,new_ridge_adjoint_rhs_columns=columns,
            new_ridge_adjoint_rhs_elements=n*columns,new_ridge_adjoint_dense_work_units=n*n*columns)
        completed='new_ridge_adjoint_triangular_completed_calls'
    else:
        updates=dict(new_ridge_triangular_calls=1,new_ridge_triangular_rhs_columns=columns,
            new_ridge_triangular_rhs_elements=n*columns,new_ridge_triangular_dense_work_units=n*n*columns)
        completed='new_ridge_triangular_completed_calls'
    for key,value in updates.items(): info[key]=info.get(key,0)+value
    result=solve_triangular(chol.T if transpose else chol,rhs,lower=not transpose)
    info[completed]=info.get(completed,0)+1
    return result


def _new_ridge_fit(K,Y,info,partial):
    """Same raw affine ridge operations, with per-call failure-safe ledger."""
    started=time.perf_counter(); info['new_ridge_fit_count']=info.get('new_ridge_fit_count',0)+1
    try:
        n=Y.shape[0]; rhs=np.column_stack((Y,np.ones(n)))
        factor_input=K+np.eye(n)
        partial.update(new_rhs=rhs,new_factor_input=factor_input)
        info['new_ridge_factorization_count']=info.get('new_ridge_factorization_count',0)+1
        info['new_ridge_factorization_attempts']=info.get('new_ridge_factorization_attempts',0)+1
        chol=np.linalg.cholesky(factor_input)
        info['new_ridge_factorizations_completed']=info.get('new_ridge_factorizations_completed',0)+1
        partial['new_chol']=chol
        lower=_ridge_triangular(chol,rhs,info)
        partial['new_lower_solution']=lower
        solved=_ridge_triangular(chol,lower,info,transpose=True)
        partial['new_solved_rhs']=solved
        F,z=solved[:,:Y.shape[1]],solved[:,Y.shape[1]]; s=float(z.sum())
        partial['new_schur_z']=z
        if not np.isfinite(s) or s<=0: raise FloatingPointError('GROUP_NEW_RIDGE_NONPOSITIVE_SCHUR')
        partial['new_schur_s']=s
        intercept=F.sum(axis=0)/s; alpha=np.ascontiguousarray(F-z[:,None]*intercept[None,:])
        _finite(alpha,intercept,z,s)
        partial.update(new_alpha=alpha,new_intercept=intercept)
        return alpha,intercept,z,s,chol,rhs
    finally:
        info['new_ridge_seconds']=info.get('new_ridge_seconds',0.)+time.perf_counter()-started


def _new_ridge_adjoint(chol,L,G,z,s,info,partial):
    """Original free-intercept saddle adjoint with actual partial solve work."""
    started=time.perf_counter(); info['new_ridge_adjoint_calls']=info.get('new_ridge_adjoint_calls',0)+1
    try:
        rhs=L.T@G; gb=G.sum(axis=0)
        partial.update(new_adjoint_rhs=rhs,new_g_b=gb)
        lower=_ridge_triangular(chol,rhs,info,adjoint=True)
        partial['new_adjoint_lower_solution']=lower
        V=_ridge_triangular(chol,lower,info,adjoint=True,transpose=True)
        partial['new_adjoint_solved_rhs']=V
        eta=(V.sum(axis=0)-gb)/s; T=V-z[:,None]*eta[None,:]
        _finite(T,eta,gb)
        partial.update(new_T=T,new_intercept_adjoint=eta)
        return T,eta,gb,rhs
    finally:
        info['new_ridge_adjoint_seconds']=info.get('new_ridge_adjoint_seconds',0.)+time.perf_counter()-started


def _logsoftmax(values):
    if not len(values): return np.empty_like(values)
    shift = values-values.max(axis=1, keepdims=True)
    out = shift-np.log(np.exp(shift).sum(axis=1, keepdims=True))
    _finite(out)
    return out


def _sigmoid_residual(g, old):
    e = np.exp(-np.abs(g))
    positive = np.where(g >= 0, 1/(1+e), e/(1+e))
    negative = np.where(g >= 0, e/(1+e), 1/(1+e))
    return np.where(old, -negative, positive)


def _compose(p, B, h, g):
    oldclasses = tuple(p.audit['old_classes'])
    oldcols = np.array([p.classes.index(name) for name in oldclasses], dtype=np.int64)
    newcols = np.array([i for i, name in enumerate(p.classes) if name not in oldclasses], dtype=np.int64)
    logold, lognew = _logsoftmax(B), _logsoftmax(h)
    out = np.empty((len(g), len(p.classes)))
    out[:, oldcols] = -np.logaddexp(0., -g)[:, None]+logold
    out[:, newcols] = -np.logaddexp(0., g)[:, None]+lognew
    _finite(out)
    return out, logold, lognew


def compact_training_record(record):
    obj = record.get('objective', record.get('audit', record))
    result = {name:record.get(name) for name in ('event', 'mode', 'iteration', 'trial', 'step',
        'step_size', 'learning_rate', 'gradient_norm', 'update_norm', 'accepted', 'text')}
    for name in ('RMSCE', 'loss_total', 'loss_ce', 'loss_proximal', 'objective_seconds'):
        result[name] = obj.get(name)
    for name in AUDIT_COUNTERS:
        if name in obj: result[name] = obj[name]
    result.update(source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        counter_ownership='STAGE_CUMULATIVE' if str(record.get('event', '')).endswith(('FINAL','FAILURE')) else 'OBJECTIVE_INCREMENT')
    return ch._safe(result)


def _emit(callback, name, payload, context=None):
    record = dict(context or {}); record.update(payload)
    obj = record.get('objective', record.get('audit', record))
    record['text'] = ('GROUP_BARRIER '+name+' RMSCE='+str(obj.get('RMSCE'))+' prox=0'+
        ' lr='+str(record.get('step_size', record.get('learning_rate')))+
        ' grad='+str(record.get('gradient_norm'))+' update='+str(record.get('update_norm'))+
        ' accepted='+str(record.get('accepted'))+' gate_Newton='+str(obj.get('group_gate_forward_newton_iterations'))+
        ' gate_logistic='+str(obj.get('gate_logistic_loss_sum'))+' gate_ridge='+str(obj.get('gate_ridge_penalty'))+
        ' gate_barrier='+str(obj.get('gate_barrier_term'))+' new_ridge_SSE='+str(obj.get('new_ridge_training_sse'))+
        ' new_free_b='+str(obj.get('new_free_intercept_norm'))+
        ' seconds='+str(obj.get('objective_seconds', obj.get('fit_seconds')))+' source_validation=N/A')
    if name == 'PREPARED': record['text'] += ' actual_parameters='+str(FROZEN_CONFIG)
    ch._emit(callback, 'GROUP_BARRIER_'+name, record)


def prepare_group_barrier_joint_training(*,max_newton_iterations,max_line_search_trials,max_factor_buffer_bytes,**kwargs):
    """Reuse validated Margin B preparation; no C QP forward is called here."""
    limits=_limits(max_newton_iterations,max_line_search_trials,max_factor_buffer_bytes)
    prepared=inherited_B.prepare_margin_joint_training(**kwargs,max_transitions=max_newton_iterations,
                                                       max_factor_buffer_bytes=max_factor_buffer_bytes)
    audit=prepared.audit_dict()
    audit.update(group_barrier_resources=limits,
        legacy_qp_metadata_scope='unused_for_group_C;preparation_only_fits_B_inner_teachers;no_C_QP_call')
    prepared=replace(prepared,audit=audit)
    _emit(kwargs.get('log_callback'),'PREPARED',dict(audit=prepared.audit_dict(),actual_parameters=dict(FROZEN_CONFIG,**limits)))
    return prepared


def _forward(p, U, limits, progress=None):
    started = time.perf_counter(); info = {} if progress is None else progress
    info.update({name:0 for name in STAGE_COUNTERS})
    oldclasses = tuple(p.audit['old_classes'])
    oldcols = np.array([p.classes.index(name) for name in oldclasses], dtype=np.int64)
    newcols = np.array([i for i,name in enumerate(p.classes) if name not in oldclasses], dtype=np.int64)
    oldmask = np.isin(p.train_labels, oldcols)
    oi, ni = np.flatnonzero(oldmask), np.flatnonzero(~oldmask)
    if not len(oi) or not len(ni) or not len(newcols): raise ValueError('Both legal train groups and new classes required')
    n, h = len(p.train_labels), len(p.held_labels)
    cache = dict(problem=p, U=np.array(U, copy=True), old_indices=oi, new_indices=ni,
                 old_columns=oldcols, new_columns=newcols, audit=info)
    try:
        b, a, bc = fcr._adapt(p.train_context, U, derivative_cache=True)
        hb, ha, hc = fcr._adapt(p.held_context, U, derivative_cache=True)
        info['adapter_physical_evaluation_count'] = n+h
        identity = not np.any(U) or p.tau == 0
        distance, cross = p.d0, p.cross_d0
        if not identity:
            distance = .5*p.d0+.5*local._distances(b,a)
            cross = .5*p.cross_d0+.5*local._distances(hb,ha,b,a)
            info.update(raw_distance_evaluation_count=1+int(h>0), raw_distance_pair_count=n*(n-1)//2+h*n,
                reference_distance_evaluation_count=1+int(h>0),
                reference_distance_pair_count=len(oi)*(len(oi)-1)//2+len(oi)*len(ni)+h*len(oi))
        radial, crossrad = np.zeros((n,n)), np.zeros((h,n))
        if p.gamma is not None and p.gamma != 0:
            radial, crossrad = local._radial(distance,p.tau), local._radial(cross,p.tau)
            info.update(kernel_evaluation_count=1+int(h>0), kernel_pair_count=n*n+h*n)
        K, L = p.gamma*radial if p.gamma is not None else radial, p.gamma*crossrad if p.gamma is not None else crossrad
        # The gate requires exact symmetry; averaging only removes roundoff.
        K = .5*(K+K.T)
        newlabels = np.array([np.flatnonzero(newcols==p.train_labels[i])[0] for i in ni], dtype=np.int64)
        Y = np.eye(len(newcols))[newlabels]-1/len(newcols)
        cache.update(b=b,a=a,hb=hb,ha=ha,bc=bc,hc=hc,distance=distance,cross_distance=cross,
                     radial=radial,crossrad=crossrad,K=K,L=L,Ynew=Y,new_labels=newlabels)
        alpha,intercept,z,s,chol,rhs=_new_ridge_fit(K[np.ix_(ni,ni)],Y,info,cache)
        htrain, hheld = K[:,ni]@alpha+intercept, L[:,ni]@alpha+intercept
        lognew_train = _logsoftmax(htrain)
        prior = p.M_train[:,oldcols]
        own = np.array([np.flatnonzero(oldcols==p.train_labels[i])[0] for i in oi],dtype=np.int64)
        competitor = prior[oi].copy(); competitor[np.arange(len(oi)),own] = -np.inf
        maximum = np.maximum(0.,competitor.max(axis=1))
        rawmargin = prior[oi,own]-maximum
        top = prior[oi].max(axis=1)
        lse = top+np.log(np.exp(prior[oi]-top[:,None]).sum(axis=1))
        bounds = (lse-maximum)[:,None]+lognew_train[oi]
        _finite(rawmargin,bounds)
        cache.update(new_alpha=alpha,new_intercept=intercept,new_schur_z=z,new_schur_s=s,
            new_chol=chol,new_rhs=rhs,htrain=htrain,hheld=hheld,margin=rawmargin,bounds=bounds)
        gs = gate.fit_group_barrier_gate(K=K,targets=oldmask.astype(float),old_indices=oi,lower_bounds=bounds,**limits)
        _gate_account(info,gs.audit,'forward'); cache['gate'] = gs
        gh = gate.predict_group_barrier_gate(gs,L=L)
        score, logold, lognew = _compose(p,p.M_held[:,oldcols],hheld,gh)
        train_scores, _, _ = _compose(p,prior,htrain,gs.f)
        cache.update(gheld=gh,logold=logold,lognew=lognew,score=score,train_scores=train_scores)
        info.update(identity_forward=identity,negative_original_margin_count=int(np.sum(rawmargin<0)),
            zero_original_margin_count=int(np.sum(rawmargin==0)),minimum_original_margin=float(rawmargin.min()),
            gate_fit_audit=gs.audit_dict(),gate_zeta=gs.zeta,new_class_count=len(newcols),
            new_ridge_training_sse=float(.5*np.sum((htrain[ni]-Y)**2)),
            new_ridge_regularizer=float(.5*np.sum(alpha*(K[np.ix_(ni,ni)]@alpha))),
            new_free_intercept_norm=_norm(intercept),gate_free_intercept=gs.b,
            gate_logistic_loss_sum=gs.audit['logistic_loss_sum'],gate_barrier_term=gs.audit['barrier_term'],
            gate_ridge_penalty=gs.audit['ridge_penalty'],gate_minimum_slack=gs.audit['minimum_slack'],
            head_seconds=time.perf_counter()-started,status='NUMERIC_EQUILIBRIUM_CHECKED')
        _finite(score,train_scores)
        return cache
    except _FAILURES as exc:
        if isinstance(exc,gate.GroupBarrierFailure): _gate_account(info,exc.audit,'forward')
        exc.group_head_audit = info
        exc.group_failed_arrays = _head_arrays(cache)
        if isinstance(exc,gate.GroupBarrierFailure):
            exc.group_failed_arrays.update({'gate_failure_'+key:value for key,value in exc.arrays.items()})
        raise


def _head_arrays(cache):
    p = cache['problem']
    out = dict(train_labels=p.train_labels,held_labels=p.held_labels,M_train=p.M_train,M_held=p.M_held,
        tau=affine._scalar(p.tau),gamma=affine._scalar(p.gamma),reference_s0=affine._scalar(p.s0),
        original_train=p.train_context['original'],original_held=p.held_context['original'])
    for name,value in cache.items():
        if isinstance(value,np.ndarray): out[name] = value
    if 'new_schur_s' in cache: out['new_schur_s'] = np.asarray(cache['new_schur_s'])
    if 'gate' in cache:
        out.update({'gate_'+key:value for key,value in cache['gate'].arrays.items()})
        out.update(gate_b=np.asarray(cache['gate'].b),gate_zeta=np.asarray(cache['gate'].zeta))
    out.update(cache.get('adjoint_arrays',{}))
    return out


def _backward(cache,v,Fheld,info):
    p = cache['problem']; tick = time.perf_counter()
    # A reused legal forward cache may have a previous successful VJP. A new
    # attempt archives only its own completed work, including gate failures.
    cache.pop('adjoint_arrays',None)
    for name in ('new_adjoint_rhs','new_g_b','new_adjoint_lower_solution','new_adjoint_solved_rhs',
                 'new_T','new_intercept_adjoint'):
        cache.pop(name,None)
    if p.gamma is None or p.gamma == 0 or p.tau == 0 or not len(v):
        info['kernel_gradient_status']='NO_CONTINUOUS_ADAPTER_KERNEL_INFORMATION'
        return np.zeros((736,8))
    try:
        adj = gate.group_barrier_gate_vjp(cache['gate'],L=cache['L'],G=v)
    except gate.GroupBarrierFailure as exc:
        _gate_account(info,exc.audit,'adjoint')
        exc.group_head_audit=info; exc.group_failed_arrays=_head_arrays(cache)
        exc.group_failed_arrays.update({'gate_failure_'+key:value for key,value in exc.arrays.items()})
        raise
    _gate_account(info,adj.audit,'adjoint')
    oi,ni = cache['old_indices'],cache['new_indices']
    F_old = adj.gradlower_bounds-adj.gradlower_bounds.sum(axis=1,keepdims=True)*np.exp(_logsoftmax(cache['htrain'][oi]))
    F = np.concatenate((F_old,Fheld))
    Ln = np.concatenate((cache['K'][np.ix_(oi,ni)],cache['L'][:,ni]))
    try:
        T,t,gb,rhs=_new_ridge_adjoint(cache['new_chol'],Ln,F,cache['new_schur_z'],cache['new_schur_s'],info,cache)
    except _FAILURES as exc:
        exc.group_head_audit=info; exc.group_failed_arrays=_head_arrays(cache)
        raise
    Kbar,Lbar = np.array(adj.gradK,copy=True),np.array(adj.gradL,copy=True)
    kn = -T@cache['new_alpha'].T; kn=.5*(kn+kn.T)
    ln = F@cache['new_alpha'].T
    Kbar[np.ix_(ni,ni)] += kn
    Kbar[np.ix_(oi,ni)] += .5*ln[:len(oi)]
    Kbar[np.ix_(ni,oi)] += .5*ln[:len(oi)].T
    Lbar[:,ni] += ln[len(oi):]
    info.update(gate_adjoint_audit=dict(adj.audit))
    dd=-p.gamma*Kbar*cache['radial']/p.tau
    dc=-p.gamma*Lbar*cache['crossrad']/p.tau
    gb0,ga0,_,_=ch._distance_vjp(cache['b'],cache['a'],cache['b'],cache['a'],.5*dd,symmetric=True)
    ghb,gha,gtb,gta=ch._distance_vjp(cache['hb'],cache['ha'],cache['b'],cache['a'],.5*dc)
    gu=fcr._adapter_vjp(p.train_context,cache['U'],gb0+gtb,ga0+gta,cache['bc'])
    gu+=fcr._adapter_vjp(p.held_context,cache['U'],ghb,gha,cache['hc'])
    cache['adjoint_arrays']=dict(gate_G=v,new_held_G=Fheld,new_old_G=F_old,
        gate_K_upstream=adj.gradK,gate_L_upstream=adj.gradL,bounds_upstream=adj.gradlower_bounds,
        new_T=T,new_intercept_adjoint=t,new_g_b=gb,new_adjoint_rhs=rhs,
        K_upstream=Kbar,L_upstream=Lbar,adapted_distance_upstream=.5*dd,
        adapted_cross_distance_upstream=.5*dc,g_U=gu)
    info.update(kernel_gradient_status='COMPLETE_SMOOTH_FIXED_BARRIER_JACOBIAN',adjoint_seconds=time.perf_counter()-tick)
    return gu


@dataclass(frozen=True)
class ObjectiveCache:
    prepared: object
    Z: np.ndarray
    limits: tuple
    folds: tuple


def evaluate_group_barrier_joint_objective(prepared,Z,*,gradient=True,forward_cache=None,
        max_newton_iterations,max_line_search_trials,max_factor_buffer_bytes):
    limits=_limits(max_newton_iterations,max_line_search_trials,max_factor_buffer_bytes)
    if prepared.inherited is None: raise ValueError('C objective requires actual B')
    Z=np.asarray(Z,dtype=np.float64)
    if Z.shape!=(736,prepared.r): raise ValueError('Whitened coordinate shape mismatch')
    _finite(Z)
    if _norm(Z)>.5+128*np.finfo(np.float64).eps: raise ValueError('Coordinates outside fixed hard ball')
    start=time.perf_counter(); U=reconstruct_U(Z,prepared.anchor_U,prepared.W)
    binding=tuple(limits.items()); reused=forward_cache is not None
    info={name:0 for name in STAGE_COUNTERS}; folds=[]
    try:
        if reused:
            if (not isinstance(forward_cache,ObjectiveCache) or forward_cache.prepared is not prepared or
                forward_cache.limits!=binding or not np.array_equal(forward_cache.Z,Z)):
                raise ValueError('Forward cache belongs to a different exact objective')
            folds=list(forward_cache.folds)
        else:
            info['inner_objective_evaluation_count']=1
            for problem in prepared.problems:
                ci={}
                try: cf=_forward(problem,U,limits,ci)
                except _FAILURES:
                    _add(info,ci); raise
                _add(info,ci); info['inner_head_fit_count']+=1; folds.append(cf)
        c=len(prepared.classes); sums=np.zeros(c); counts=np.zeros(c,dtype=np.int64); losses=[]
        for cf in folds:
            labels=cf['problem'].held_labels; old=np.isin(labels,cf['old_columns'])
            positions={int(col):j for j,col in enumerate(cf['old_columns'])}
            newpos={int(col):j for j,col in enumerate(cf['new_columns'])}
            ce=np.logaddexp(0.,np.where(old,-cf['gheld'],cf['gheld']))
            for i,label in enumerate(labels): ce[i]-=cf['logold'][i,positions[int(label)]] if old[i] else cf['lognew'][i,newpos[int(label)]]
            sums+=np.bincount(labels,weights=ce,minlength=c); counts+=np.bincount(labels,minlength=c)
            losses.append((ce,old,newpos))
        means=np.divide(sums,counts,out=np.zeros(c),where=counts>0); risk=_norm(means)/np.sqrt(c)
        gu=np.zeros((736,8)); fas=[]
        for cf,(ce,old,newpos) in zip(folds,losses):
            fa=deepcopy(cf['audit']); labels=cf['problem'].held_labels
            if gradient and risk>0:
                weight=means[labels]/(c*risk*counts[labels])
                v=weight*_sigmoid_residual(cf['gheld'],old)
                F=np.zeros_like(cf['hheld'])
                for i,label in enumerate(labels):
                    if not old[i]:
                        F[i]=np.exp(cf['lognew'][i]); F[i,newpos[int(label)]]-=1; F[i]*=weight[i]
                bi={}
                try: gu+=_backward(cf,v,F,bi)
                except _FAILURES:
                    _add(info,bi); raise
                _add(info,bi); fa.update(bi)
            fa.update(held_ce_sums=np.bincount(labels,weights=ce,minlength=c).tolist(),
                held_ce_counts=np.bincount(labels,minlength=c).tolist()); fas.append(fa)
        gz=gu@prepared.W if gradient else np.zeros_like(Z)
        if gradient and folds and risk==0: raise FloatingPointError('RMSCE_ZERO_UNSUPPORTED_JACOBIAN')
        cache=forward_cache if reused else ObjectiveCache(prepared,_readonly(Z),binding,tuple(folds))
        info.update(RMSCE=float(risk),loss_ce=float(risk),loss_task=float(risk),loss_total=float(risk),
            loss_proximal=0.,class_ce_sums=sums.tolist(),class_ce_counts=counts.tolist(),class_ce_means=means.tolist(),
            backward_evaluation_count=int(gradient),inner_folds=fas,forward_cache_reused=reused,
            forward_cache_bytes=_unique_bytes([_head_arrays(cf) for cf in folds]),
            gradient_norm=_norm(gz) if gradient else None,objective_seconds=time.perf_counter()-start,
            **fcr._functional_measurement(prepared,U,prepared.anchor_U,Z))
        for name in ('gate_logistic_loss_sum','gate_ridge_penalty','gate_barrier_term','new_ridge_training_sse',
                     'new_ridge_regularizer','new_free_intercept_norm'):
            info[name]=sum(cf['audit'].get(name,0.) for cf in folds)
        _finite(risk,gz)
        return float(risk),gz,info,cache
    except _FAILURES as exc:
        exc.group_objective_audit=info; exc.group_partial_folds=tuple(folds)
        raise


@dataclass(frozen=True)
class GroupBarrierJointState:
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
    problem: object
    final_cache: Mapping
    prior: object
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        for name in ('U','Z','W','anchor_U','H','singular_values','labels'):
            object.__setattr__(self,name,_readonly(getattr(self,name)))
        object.__setattr__(self,'raw',_freeze({key:_readonly(value) for key,value in self.raw.items()}))
        object.__setattr__(self,'final_cache',_freeze({key:_readonly(value) if isinstance(value,np.ndarray) else value for key,value in self.final_cache.items()}))
        object.__setattr__(self,'records',_freeze(self.records)); object.__setattr__(self,'audit',_freeze(self.audit))
    @property
    def V(self): return V0
    def audit_dict(self): return ch._safe(_plain(self.audit))
    def state_records(self): return {key:{name:np.array(value,copy=True) for name,value in arrays.items()} for key,arrays in self.records.items()}
    def to_arrays(self):
        out=_head_arrays(dict(self.final_cache,problem=self.problem))
        out.update(self.raw,U=self.U,Z=self.Z,W=self.W,anchor_U=self.anchor_U,H=self.H,singular_values=self.singular_values,V0=V0)
        out.update({'prior_B_'+key:value for key,value in self.prior.to_arrays().items()})
        return out
    def _resident_values(self):
        return (self.U,self.Z,self.W,self.anchor_U,self.H,self.singular_values,self.raw,self.labels,
                vars(self.problem),self.final_cache,self.records,self.prior._resident_values())
    def _deployment_values(self):
        num=self.final_cache
        values=(num['new_alpha'],num['new_intercept'],num['gate'].alpha,np.asarray(num['gate'].b),self.prior._deployment_values())
        if self.problem.gamma is not None and self.problem.gamma!=0:
            values+=(self.problem.train_context['original'],)
            if self.problem.tau!=0: values+=(self.U,V0,num['b'],num['a'])
        return values
    def _score_geometry(self,b0,a0,progress):
        p,num=self.problem,self.final_cache; n=len(b0); hnew=[]; gates=[]; residual={}; prior={}
        B=inherited_B._score_residual(self.prior.problem,self.prior.final_cache,self.prior.U,b0,a0,prior)
        for i in range(n):
            L=np.zeros((1,len(p.train_labels)))
            if p.gamma is not None and p.gamma!=0:
                tb,ta=p.train_context['original'][:,:256],p.train_context['original'][:,256:]
                d0=local._distances(b0[i:i+1],a0[i:i+1],tb,ta)
                dist=d0; calls=1
                if p.tau!=0 and np.any(self.U):
                    bb,aa,_=fcr._adapt(fcr._context(b0[i:i+1],a0[i:i+1]),self.U)
                    dist=.5*d0+.5*local._distances(bb,aa,num['b'],num['a']); calls+=1
                    residual['adapter_physical_evaluation_count']=residual.get('adapter_physical_evaluation_count',0)+1
                    residual['dictionary_physical_evaluation_count']=residual.get('dictionary_physical_evaluation_count',0)+1
                L=p.gamma*local._radial(dist,p.tau)
                for key,value in dict(raw_distance_evaluation_count=calls,raw_distance_pair_count=calls*len(p.train_labels),
                    kernel_evaluation_count=1,kernel_pair_count=len(p.train_labels)).items(): residual[key]=residual.get(key,0)+value
            hnew.append((L[:,num['new_indices']]@num['new_alpha']+num['new_intercept'])[0])
            gates.append(float(gate.predict_group_barrier_gate(num['gate'],L=L)[0]))
        h=np.asarray(hnew,dtype=np.float64).reshape(n,len(num['new_columns'])); g=np.asarray(gates)
        score,lo,ln=_compose(p,B,h,g)
        progress.update(residual=residual,prior=prior,score_physical_count=n,
            prediction_policy='all_registered_classes_both_groups_evaluated',
            numerical_argmax_policy='within_old_use_frozen_raw_B;compare_group_maxima_with_gate_plus_logprob')
        for key in set(residual)|set(prior):
            if isinstance(residual.get(key,0),(int,float)) and isinstance(prior.get(key,0),(int,float)):
                progress[key]=residual.get(key,0)+prior.get(key,0)
        return score,B,h,g,lo,ln
    def score_with_audit(self,*,z_id,fft,t_emb,f_emb,pa_local):
        start=time.perf_counter(); b,a=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True)
        info={}; score,*_ = self._score_geometry(b,a,info)
        info['score_seconds']=time.perf_counter()-start
        return score,info
    def score(self,**features): return self.score_with_audit(**features)[0]
    def predict(self,*,z_id,fft,t_emb,f_emb,pa_local):
        b,a=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True)
        _,B,h,g,lo,ln=self._score_geometry(b,a,{})
        old,new=self.final_cache['old_columns'],self.final_cache['new_columns']; result=[]
        # Canonical class order makes each within-group tie lexicographic.
        for i in range(len(b)):
            bo,bn=int(np.argmax(B[i])),int(np.argmax(h[i]))
            gap=g[i]+lo[i,bo]-ln[i,bn]
            oo,nn=self.classes[old[bo]],self.classes[new[bn]]
            result.append(oo if gap>0 else nn if gap<0 else min(oo,nn))
        return np.asarray(result,dtype=str)


def fit_group_barrier_joint_local_ridge(prepared,*,mode='B',max_newton_iterations,
        max_line_search_trials,max_factor_buffer_bytes,log_callback=None,state_callback=None):
    limits=_limits(max_newton_iterations,max_line_search_trials,max_factor_buffer_bytes)
    if mode not in ('B','C_seq') or (mode=='B')!=(prepared.inherited is None): raise ValueError('B/C lineage mode mismatch')
    if prepared.audit.get('group_barrier_resources') is not None and _plain(prepared.audit['group_barrier_resources'])!=limits:
        raise ValueError('Solver resources differ from frozen preparation')
    if mode=='B':
        return inherited_B.fit_margin_joint_local_ridge(prepared,mode='B',log_callback=log_callback,state_callback=state_callback)
    if prepared.classes==prepared.old_classes: return prepared.inherited
    start=time.perf_counter(); rec=_Recorder(state_callback); rec.records.update(dict(prepared.records))
    Z=np.zeros((736,prepared.r)); U=prepared.anchor_U.copy(); cache=None; finalprogress={}
    audit={name:0 for name in STAGE_COUNTERS}
    config=dict(deepcopy(FROZEN_CONFIG),**limits)
    audit.update(schema=config['schema'],method=config['method'],mode=mode,config=config,
        preparation=prepared.audit_dict(),ajlr_stage_count=1,gradients=[],trials=[],steps=[],
        no_information=prepared.audit['no_information'],source_validation=None,source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        encoder_backward=False,hardware_dtype='CPU_float64',device_model=None,peak_rss_bytes=None,peak_gpu_memory_bytes=None,
        extra_ground_data_stat_payload_bytes=0,code_wire_payload_bytes=None)
    def save(key,obj,oc,z=Z,u=U,**extra):
        if oc is not None:
            for j,cf in enumerate(oc.folds):
                ref=rec.save(key+'_fold_'+str(j),**_head_arrays(cf)); obj['inner_folds'][j]['head_state_ref']=ref
        return rec.save(key,Z=z,U=u,W=prepared.W,anchor_U=prepared.anchor_U,
            RMSCE=affine._scalar(obj.get('RMSCE')),class_ce_sums=np.asarray(obj.get('class_ce_sums',[])),
            class_ce_counts=np.asarray(obj.get('class_ce_counts',[]),dtype=np.int64),**extra)
    def event(name,payload): _emit(log_callback,name,dict(mode=mode,**payload),prepared.audit)
    try:
        if audit['no_information']:
            audit['stop_reason']=prepared.audit['no_information_reason']
            audit['initialization_state_ref']=save('initial',{},None,z=Z,u=U)
            event('INITIAL',dict(state_ref=audit['initialization_state_ref'],no_update_reason=audit['stop_reason']))
        else:
            loss,_,obj,cache=evaluate_group_barrier_joint_objective(prepared,Z,gradient=False,**limits); _add(audit,obj)
            audit['initialization_state_ref']=save('initial',obj,cache,z=Z,u=U)
            audit['initial_objective']=deepcopy(obj); event('INITIAL',dict(objective=obj,state_ref=audit['initialization_state_ref']))
            audit['stop_reason']='MAX_ITERATIONS'
            for iteration in range(1,5):
                audit['optimizer_iterations']+=1
                loss,grad,obj,cache=evaluate_group_barrier_joint_objective(prepared,Z,gradient=True,forward_cache=cache,**limits); _add(audit,obj)
                norm=_norm(grad); direction=np.zeros_like(grad) if norm==0 else -grad/norm
                ref=save('gradient_'+str(iteration),obj,cache,z=Z,u=U,g_Z=grad,d_Z=direction)
                gr=dict(iteration=iteration,gradient_norm=norm,objective=deepcopy(obj),state_ref=ref)
                audit['gradients'].append(gr); event('GRADIENT',gr)
                if norm==0: audit['stop_reason']='ZERO_GRADIENT'; break
                accepted=False
                for trial in range(1,13):
                    step=.125*.5**(trial-1); tz=project_coordinates(Z+step*direction); delta=tz-Z
                    if not np.any(delta): audit['stop_reason']='ZERO_FEASIBLE_DISPLACEMENT'; break
                    audit['trial_attempt_count']+=1
                    try: tl,_,to,tc=evaluate_group_barrier_joint_objective(prepared,tz,gradient=False,**limits)
                    except _FAILURES as exc:
                        exc.group_attempt_Z=tz; exc.group_attempt_U=reconstruct_U(tz,prepared.anchor_U,prepared.W); raise
                    _add(audit,to)
                    tu=reconstruct_U(tz,prepared.anchor_U,prepared.W)
                    acceptance=affine._trial_acceptance(loss,tl,float(np.sum(grad*delta)))
                    tref=save('trial_'+str(iteration)+'_'+str(trial),to,tc,z=tz,u=tu,delta_Z=delta,d_Z=direction)
                    tr=dict(iteration=iteration,trial=trial,step_size=step,loss_before=loss,loss_after=tl,
                        update_norm=_norm(delta),state_ref=tref,objective=deepcopy(to),**acceptance)
                    audit['trial_count']+=1; audit['trials'].append(tr); accepted=acceptance['accepted']
                    audit['accepted_trial_count' if accepted else 'rejected_trial_count']+=1; event('TRIAL',tr)
                    if accepted:
                        Z,U,cache,loss=tz,tu,tc,tl; audit['optimizer_steps']+=1
                        sr=dict(step=audit['optimizer_steps'],iteration=iteration,trial=trial,learning_rate=step,
                            loss_before=tr['loss_before'],loss_after=tl,update_norm=_norm(delta),state_ref=tref)
                        audit['steps'].append(sr); event('STEP',sr); break
                if not accepted:
                    if audit['stop_reason']!='ZERO_FEASIBLE_DISPLACEMENT': audit['stop_reason']='TRIAL_BUDGET_EXHAUSTED'
                    break
            _,_,obj,_=evaluate_group_barrier_joint_objective(prepared,Z,gradient=False,forward_cache=cache,**limits)
            audit['final_objective_state_ref']=save('final_objective',obj,cache,z=Z,u=U); audit['final_objective']=obj
        final=_forward(prepared.full_problem,U,limits,finalprogress); _add(audit,finalprogress); audit['final_head_fit_count']=1
        arrays=_head_arrays(final); arrays.update(prepared.raw,U=U,Z=Z,W=prepared.W,anchor_U=prepared.anchor_U,
            H=prepared.H,singular_values=prepared.singular_values,V0=V0)
        arrays.update({'prior_B_'+name:value for name,value in prepared.inherited.to_arrays().items()})
        audit['final_state_ref']=rec.save('final',**arrays)
        audit.update(status='COMPLETED',fit_seconds=time.perf_counter()-start,final_fit=deepcopy(finalprogress),
            coordinate_norm=_norm(Z),parameter_U_norm=_norm(U),inherited_from_B=True,
            trainable_parameter_count=0 if audit['no_information'] else 736*prepared.r,
            active_parameter_count=736*prepared.r if audit['optimizer_steps'] else 0,
            declared_coordinate_scalar_count=736*prepared.r,parameter_changed_from_anchor=bool(np.any(U!=prepared.anchor_U)),
            analytic_new_coefficient_count=len(final['new_indices'])*len(final['new_columns']),
            analytic_new_intercept_count=len(final['new_columns']),analytic_gate_coefficient_count=len(prepared.ids),
            analytic_gate_intercept_count=1,retained_vector_record_bytes=_unique_bytes(rec.records),
            serialized_deployment_bytes=None,incremental_transmission_bytes=None)
        names=('U','b','a','distance','cross_distance','radial','crossrad','K','L','Ynew','new_labels','old_indices',
            'new_indices','old_columns','new_columns','new_alpha','new_intercept','new_schur_z','new_schur_s',
            'new_chol','new_rhs','htrain','hheld','margin','bounds','gate','gheld','logold','lognew','score','train_scores')
        state=GroupBarrierJointState(U,Z,prepared.W,prepared.anchor_U,prepared.H,prepared.singular_values,
            prepared.raw,prepared.labels,prepared.ids,prepared.classes,prepared.old_classes,prepared.full_problem,
            {name:final[name] for name in names},prepared.inherited,rec.records,audit)
        audit.update(resident_numeric_state_bytes=_unique_bytes(state._resident_values()),
            deployment_numeric_state_bytes=_unique_bytes(state._deployment_values()),
            state_byte_scope='unique_retained_numeric_backings_including_records_actual_B_excludes_Python_library_workspace')
        object.__setattr__(state,'audit',_freeze(audit)); event('FINAL',dict(audit=state.audit_dict()))
        return state
    except _FAILURES as exc:
        if hasattr(exc,'group_objective_audit'): _add(audit,exc.group_objective_audit)
        elif finalprogress: _add(audit,finalprogress)
        failed=dict(getattr(exc,'group_failed_arrays',{}))
        failed.update(Z=getattr(exc,'group_attempt_Z',Z),U=getattr(exc,'group_attempt_U',U),
            last_accepted_Z=Z,last_accepted_U=U,W=prepared.W,anchor_U=prepared.anchor_U)
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),failure_code=getattr(exc,'code',None),
            fit_seconds=time.perf_counter()-start,failure_state_ref=None,
            failed_head_audit=getattr(exc,'group_head_audit',{}))
        audit['completed_failure_head_refs']=[]
        try:
            for i,cf in enumerate(getattr(exc,'group_partial_folds',())):
                audit['completed_failure_head_refs'].append(rec.save('failure_completed_'+str(i),**_head_arrays(cf)))
            audit['failure_state_ref']=rec.save('failure',**failed)
            audit['failure_archive_status']='CALLBACK_RETURNED' if state_callback is not None else 'RETAINED_IN_MEMORY'
        except Exception as archive_error:
            # Reporting/archival failures are separate evidence. They must not
            # replace the originating numerical failure or destroy its arrays.
            audit.update(failure_archive_status='FAILED',failure_archive_error=str(archive_error))
        audit['fit_seconds']=time.perf_counter()-start
        try: event('FAILURE',dict(audit=audit))
        except Exception as log_error: audit['failure_log_error']=str(log_error)
        error=NumericalFailure(str(exc),audit)
        error.code=getattr(exc,'code',None)
        error.records=rec.records; error.arrays=_freeze({name:_failure_readonly(value) for name,value in failed.items()})
        error.original_failure_audit=_freeze(_plain(getattr(exc,'audit',{})))
        error.original_failure_arrays=_freeze({name:_failure_readonly(value) for name,value in getattr(exc,'arrays',{}).items()})
        raise error from exc


def predict_group_barrier_joint_local_ridge(state,**features):
    if not isinstance(state,(GroupBarrierJointState,inherited_B.MarginJointState)):
        raise TypeError('GroupBarrierJoint or exact inherited B state required')
    return state.predict(**features)
