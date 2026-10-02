"""Group/class-balanced finite barrier gate on current legal train labels.

Only the Bernoulli supervision changes: w=N/(2*C_group*n_class). The full
old-row by new-class barrier, free intercept, zeta and explicit budgets stay
unchanged. No query, held labels, prototype teacher or file I/O is present.
Pure scaled SPD algebra is reused without rebinding any original module global.
"""
from dataclasses import dataclass
from types import MappingProxyType
import time
import numpy as np
from . import d92_group_barrier_gate as original

EPS=original.EPS
AVERAGE_DUAL_GAP=original.AVERAGE_DUAL_GAP
ARMIJO=original.ARMIJO
SCHEMA='d92_group_balanced_barrier_gate_v1'
WEIGHT_RULE='N/(2*C_group*n_class)'
WEIGHT_SCOPE='CURRENT_FOLD_LEGAL_TRAIN_LABELS_ONLY'
WEIGHT_SUM_KEYS=('weight_construction_attempts','weight_constructions_completed',
    'weight_physical_record_count','weight_class_count','weighted_logistic_record_evaluations',
    'weighted_entropy_record_evaluations','weight_construction_seconds')
_seal=original._seal
_audit=original._audit
_positive_int=original._positive_int
_float_array=original._float_array
_factor=original._factor
_solve=original._solve
_residuals=original._residuals


def _ledger():
    return dict(original._ledger(),**dict.fromkeys(WEIGHT_SUM_KEYS,0))


class GroupBalancedBarrierFailure(original.GroupBarrierFailure):
    pass


@dataclass(frozen=True)
class GroupBalancedBarrierState(original.GroupBarrierState):
    labels: np.ndarray
    weights: np.ndarray
    class_labels: np.ndarray
    class_counts: np.ndarray
    class_targets: np.ndarray
    @property
    def arrays(self):
        return MappingProxyType(dict(super().arrays,labels=self.labels,weights=self.weights,
            class_labels=self.class_labels,class_counts=self.class_counts,class_targets=self.class_targets))


@dataclass(frozen=True)
class GroupBalancedBarrierJVP:
    score_jacobian: np.ndarray
    arrays: object
    audit: object
    def audit_dict(self):return dict(self.audit)


def _weights(labels,targets,audit):
    tick=time.perf_counter();audit['weight_construction_attempts']+=1
    audit['weight_physical_record_count']+=len(labels)
    try:
        values,first,inverse,counts=np.unique(labels,return_index=True,return_inverse=True,return_counts=True)
        audit['weight_class_count']+=len(values)
        groups=targets[first]
        if not np.array_equal(targets,groups[inverse]):
            raise ValueError('One physical class cannot belong to both groups')
        group_classes=np.asarray([np.count_nonzero(groups==0),np.count_nonzero(groups==1)])
        if np.any(group_classes==0):raise ValueError('Both train class groups required')
        weights=len(labels)/(2*group_classes[targets.astype(np.int64)]*counts[inverse])
        if not np.isfinite(weights).all() or np.any(weights<=0):
            raise ArithmeticError('INVALID_GROUP_BALANCED_WEIGHTS')
        audit['weight_constructions_completed']+=1
        audit.update(weight_rule=WEIGHT_RULE,weight_scope=WEIGHT_SCOPE,
            old_class_count=int(group_classes[1]),new_class_count=int(group_classes[0]),
            old_supervision_mass=float(weights[targets==1].sum()),
            new_supervision_mass=float(weights[targets==0].sum()),
            total_supervision_mass=float(weights.sum()),weight_numeric_bytes=weights.nbytes,
            weight_derivative_scope='FIXED_CURRENT_TRAIN_LABELS_AND_COUNTS; dw=0')
        return weights,values,counts,groups
    finally:audit['weight_construction_seconds']+=time.perf_counter()-tick


def _evaluate(K, alpha, b, targets, old, bounds, zeta, weights, audit):
    tick = time.perf_counter()
    audit['objective_evaluations'] += 1
    audit['logistic_record_evaluations'] += len(targets)
    audit['weighted_logistic_record_evaluations'] += len(targets)
    audit['barrier_constraint_evaluations'] += bounds.size
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            Ka = K @ alpha
            f = Ka + b
            slacks = f[old, None] - bounds
            if not np.isfinite(f).all() or not np.isfinite(slacks).all():
                raise ArithmeticError('NONFINITE_LOGITS_OR_SLACKS')
            if np.any(slacks <= 0):
                raise ArithmeticError('NONSTRICT_BARRIER_FEASIBILITY')
            e = np.exp(-np.abs(f))
            if np.any(e == 0):
                raise ArithmeticError('LOGISTIC_CURVATURE_UNDERFLOW')
            # Both label-aware residuals avoid subtracting 1 from a rounded p.
            positive = np.where(f >= 0, 1 / (1 + e), e / (1 + e))
            negative = np.where(f >= 0, e / (1 + e), 1 / (1 + e))
            q = weights * np.where(targets == 1, -negative, positive)
            curvature = weights * (e / ((1 + e) ** 2))
            multipliers = zeta / slacks
            barrier_D = multipliers / slacks
            q[old] -= multipliers.sum(axis=1)
            curvature[old] += barrier_D.sum(axis=1)
            if np.any(curvature <= 0) or not np.isfinite(curvature).all():
                raise ArithmeticError('INVALID_EFFECTIVE_CURVATURE')
            losses = np.logaddexp(0., np.where(targets == 1, -f, f))
            ridge = .5 * float(alpha @ Ka)
            loss_sum = float(weights @ losses)
            log_slacks = np.log(slacks)
            barrier = -zeta * float(log_slacks.sum())
            primal = ridge + loss_sum
            objective = primal + barrier
            if not np.isfinite(objective):
                raise ArithmeticError('NONFINITE_OBJECTIVE')
            # Propagate floating-point dot/subtraction error into q. This is
            # material when a strictly positive slack is small relative to f,a.
            f_error = 8*EPS*max(1,len(targets))*(np.abs(K)@np.abs(alpha)+abs(b))
            q_error = curvature*f_error
            q_error[old] += np.sum(barrier_D*(4*EPS*(np.abs(f[old,None])+np.abs(bounds))),axis=1)
            q_error += 8*EPS*(np.abs(q)+weights)
            # Absolute objective evaluation error model: dot/reduction gamma_k,
            # elementary-function rounding (8 eps per value), logistic's unit
            # Lipschitz constant, and log(slack)'s finite perturbation bound.
            # This decides when objective differences cannot resolve a step;
            # it does not relax stationarity or modify the mathematical loss.
            operations = max(3, 2*len(targets)+2, bounds.size+2)
            if operations*EPS >= 1:
                raise ArithmeticError('OBJECTIVE_ROUNDOFF_MODEL_UNRESOLVED')
            gamma = operations*EPS/(1-operations*EPS)
            slack_error = f_error[old,None]+4*EPS*(np.abs(f[old,None])+np.abs(bounds))
            relative_slack_error = slack_error/slacks
            if np.any(relative_slack_error >= 1):
                raise ArithmeticError('OBJECTIVE_SLACK_ROUNDOFF_UNRESOLVED')
            ridge_error = .5*gamma*float(np.abs(alpha)@(np.abs(K)@np.abs(alpha)))
            logistic_error = float(weights @ f_error)+(gamma+8*EPS)*float(weights @ np.abs(losses))
            barrier_error = zeta*float((-np.log1p(-relative_slack_error)).sum())
            barrier_error += zeta*(gamma+8*EPS)*float(np.abs(log_slacks).sum())
            objective_error = ridge_error+logistic_error+barrier_error
            objective_error += gamma*(abs(ridge)+abs(loss_sum)+abs(barrier))
            if not np.isfinite(objective_error):
                raise ArithmeticError('NONFINITE_OBJECTIVE_ROUNDOFF_BOUND')
            return dict(f=f, slacks=slacks, q=q, D=curvature, objective=objective,
                primal=primal, multipliers=multipliers, q_roundoff=float(np.max(q_error)),
                logistic_loss_sum=loss_sum,ridge_penalty=ridge,barrier_term=barrier,
                objective_roundoff_bound=float(objective_error))
    finally:
        audit['objective_seconds'] += time.perf_counter() - tick


def fit_group_balanced_barrier_gate(*, K, targets, labels, old_indices, lower_bounds,
                           max_newton_iterations, max_line_search_trials, max_factor_buffer_bytes):
    for name, value in [('max_newton_iterations', max_newton_iterations),
                        ('max_line_search_trials', max_line_search_trials),
                        ('max_factor_buffer_bytes', max_factor_buffer_bytes)]:
        _positive_int(value, name)
    K = _float_array(K, 'K', 2)
    t = _float_array(targets, 'targets', 1)
    bounds = _float_array(lower_bounds, 'lower_bounds', 2)
    old_raw = np.asarray(old_indices)
    if old_raw.ndim != 1 or old_raw.dtype.kind not in 'iu':
        raise ValueError('old_indices must be an integer vector')
    old = np.array(old_raw, dtype=np.int64, copy=True)
    n = len(t)
    if n < 2 or K.shape != (n, n) or not np.array_equal(K, K.T):
        raise ValueError('K must be raw exactly symmetric N by N')
    if not np.isin(t, (0., 1.)).all() or not np.any(t == 0) or not np.any(t == 1):
        raise ValueError('targets must contain both physical Bernoulli groups')
    if len(set(old.tolist())) != len(old) or not np.array_equal(np.sort(old), np.flatnonzero(t == 1)):
        raise ValueError('old_indices must match all targets=1 exactly')
    if bounds.shape[0] != len(old) or bounds.shape[1] < 1:
        raise ValueError('lower_bounds must retain every old by new constraint')
    label_input=np.asarray(labels)
    if label_input.shape!=(n,) or label_input.dtype.kind not in 'iu' or np.any(label_input<0) or np.any(label_input>np.iinfo(np.int64).max):
        raise ValueError('labels must be a current-train nonnegative integer vector')
    labels=np.array(label_input,dtype=np.int64,copy=True)
    audit = _ledger()
    audit.update(max_newton_iterations=max_newton_iterations,max_line_search_trials=max_line_search_trials,
        max_factor_buffer_bytes=max_factor_buffer_bytes,ridge_coefficient=1.,armijo_coefficient=ARMIJO)
    started = time.perf_counter()
    alpha = np.zeros(n)
    b = 0.
    zeta = n * AVERAGE_DUAL_GAP / bounds.size
    value = None
    weights=np.empty(0);class_labels=np.empty(0,dtype=np.int64)
    class_counts=np.empty(0,dtype=np.int64);class_targets=np.empty(0)
    try:
        weights,class_labels,class_counts,class_targets=_weights(labels,t,audit)
        if np.count_nonzero(class_targets==0)!=bounds.shape[1]:
            raise ValueError("Every physical new class must retain one bound column")
        tick = time.perf_counter()
        audit['spectral_checks'] += 1
        audit['spectral_cubic_dimension_units'] += n ** 3
        try:
            eigenvalues = np.linalg.eigvalsh(K)
        finally:
            audit['spectral_seconds'] += time.perf_counter() - tick
        psd_tolerance = 64 * EPS * n * max(1., float(np.max(np.abs(eigenvalues))))
        audit.update(minimum_kernel_eigenvalue=float(eigenvalues[0]), kernel_psd_tolerance=psd_tolerance)
        if eigenvalues[0] < -psd_tolerance:
            raise ArithmeticError('K_NOT_PSD')
        maximum = float(np.max(bounds))
        b = maximum + max(1., abs(maximum) * np.sqrt(EPS))
        if not np.isfinite(b) or not b > maximum:
            raise ArithmeticError('NO_REPRESENTABLE_STRICT_INITIALIZATION')
        value = _evaluate(K, alpha, b, t, old, bounds, zeta, weights, audit)
        for iteration in range(max_newton_iterations + 1):
            h, tolerance, residuals = _residuals(K, alpha, value)
            audit.update(residuals)
            if (residuals['canonical_residual'] <= tolerance and
                    residuals['sum_alpha_residual'] <= n*tolerance and
                    residuals['intercept_residual'] <= n*tolerance):
                break
            if iteration == max_newton_iterations:
                raise ArithmeticError('NEWTON_ITERATION_LIMIT')
            audit['newton_iterations'] += 1
            root, H, chol = _factor(K, value['D'], max_factor_buffer_bytes, audit)
            Kh = K @ h
            solved = _solve(H, chol, np.column_stack((root * Kh, root)), audit)
            del H, chol  # Do not retain a prior factor during the next factor call.
            Bh = h - root * solved[:, 0]
            beta = root * solved[:, 1]
            s = float(root @ solved[:, 1])
            if not np.isfinite(s) or s <= 0:
                raise ArithmeticError('INTERCEPT_SCHUR_NONPOSITIVE')
            db = -float(Bh.sum()) / s
            da = -Bh - beta * db
            df = K @ da + db
            slope = float((K @ h) @ da + value['q'].sum() * db)
            if not np.isfinite(slope) or slope >= 0:
                raise ArithmeticError('NONDESCENT_NEWTON_DIRECTION')
            rate = 1.
            decreasing = np.broadcast_to(df[old, None], bounds.shape) < 0
            if np.any(decreasing):
                full_df = np.broadcast_to(df[old, None], bounds.shape)
                rate = min(rate, .99 * float(np.min(-value['slacks'][decreasing] / full_df[decreasing])))
            accepted = False
            for trial in range(max_line_search_trials):
                audit['line_search_trials'] += 1
                next_alpha, next_b = alpha + rate * da, b + rate * db
                try:
                    candidate = _evaluate(K, next_alpha, next_b, t, old, bounds, zeta, weights, audit)
                except ArithmeticError as exc:
                    # A Newton direction from a valid current state can place
                    # an unaccepted trial outside binary64's finite-curvature
                    # domain. Reject that trial; never clamp logits/curvature
                    # or relax current/final-state checks. Its objective and
                    # physical-record work was already charged by _evaluate.
                    recoverable = isinstance(exc, FloatingPointError) or str(exc) in (
                        'NONFINITE_LOGITS_OR_SLACKS', 'NONSTRICT_BARRIER_FEASIBILITY',
                        'LOGISTIC_CURVATURE_UNDERFLOW', 'INVALID_EFFECTIVE_CURVATURE',
                        'NONFINITE_OBJECTIVE', 'OBJECTIVE_SLACK_ROUNDOFF_UNRESOLVED',
                        'NONFINITE_OBJECTIVE_ROUNDOFF_BOUND')
                    if not recoverable:
                        raise
                    audit.update(last_invalid_trial_code=str(exc),
                        last_invalid_trial_step_size=float(rate))
                    rate *= .5
                    continue
                bound = value['objective'] + ARMIJO * rate * slope
                predicted_decrease = -rate*slope
                error_bound = value['objective_roundoff_bound']+candidate['objective_roundoff_bound']
                increase = candidate['objective']-value['objective']
                _, _, candidate_residuals = _residuals(K,next_alpha,candidate)
                # Compare both states using the SAME current residual scales.
                # All physical constraints were independently checked above.
                def merit(measured):
                    return max(measured['canonical_residual']/tolerance,
                        measured['intercept_residual']/residuals['intercept_tolerance'],
                        measured['primal_gradient_residual']/residuals['primal_gradient_tolerance'],
                        measured['sum_alpha_residual']/residuals['intercept_tolerance'])
                before_merit, after_merit = merit(residuals), merit(candidate_residuals)
                roundoff_accept = (0 < predicted_decrease <= error_bound and
                    increase <= error_bound and after_merit < before_merit and
                    candidate_residuals['canonical_residual'] < residuals['canonical_residual'])
                # If the decrease is unresolvable, even floating Armijo equality
                # is insufficient without a decreasing residual merit.
                resolvable = predicted_decrease > error_bound
                armijo_accept = resolvable and candidate['objective'] <= bound
                audit.update(last_step_size=float(rate), last_directional_derivative=slope,
                    last_armijo_bound=float(bound), last_trial_objective=candidate['objective'],
                    last_objective_roundoff_bound=float(error_bound),
                    last_objective_increase=float(increase),last_objective_increased=bool(increase>0),
                    last_predicted_decrease=float(predicted_decrease),
                    last_residual_merit_before=float(before_merit),last_residual_merit_after=float(after_merit))
                if armijo_accept or roundoff_accept:
                    audit['last_acceptance_rule'] = 'ROUND_OFF_RESIDUAL_ACCEPTANCE' if roundoff_accept else 'OBJECTIVE_ARMIJO'
                    audit['round_off_residual_acceptances'] += int(roundoff_accept)
                    alpha, b, value = next_alpha, next_b, candidate
                    accepted = True
                    audit['accepted_steps'] += 1
                    break
                rate *= .5
            if not accepted:
                raise ArithmeticError('LINE_SEARCH_LIMIT')
        # Independent original equations are read back after convergence.
        value = _evaluate(K, alpha, b, t, old, bounds, zeta, weights, audit)
        _, tolerance, residuals = _residuals(K, alpha, value)
        audit.update(residuals)
        if residuals['canonical_residual'] > tolerance or residuals['intercept_residual'] > n*tolerance:
            raise ArithmeticError('FINAL_EQUILIBRIUM_READBACK')
        audit['logistic_record_evaluations'] += n
        audit['weighted_entropy_record_evaluations'] += n
        e = np.exp(-np.abs(value['f']))
        p = np.where(value['f'] >= 0, 1/(1+e), e/(1+e))
        one_minus_p = np.where(value['f'] >= 0, e/(1+e), 1/(1+e))
        entropy = -p*np.logaddexp(0.,-value['f'])-one_minus_p*np.logaddexp(0.,value['f'])
        candidate_dual_alpha = -value['q']
        dual = -.5*float(candidate_dual_alpha@(K@candidate_dual_alpha))-float(weights @ entropy)+float(np.sum(value['multipliers']*bounds))
        audit.update(status='COMPLETE', zeta=zeta, objective=value['objective'], primal_objective=value['primal'],
            logistic_loss_sum=value['logistic_loss_sum'],ridge_penalty=value['ridge_penalty'],
            barrier_term=value['barrier_term'],
            objective_roundoff_bound=value['objective_roundoff_bound'],
            objective_roundoff_scope='gamma_k reductions/dots, 8eps elementary values, propagated log-slack/input errors; not stationarity relaxation',
            minimum_slack=float(value['slacks'].min()), constraint_count=bounds.size,
            theoretical_center_path_gap=float(bounds.size*zeta), theoretical_average_dual_gap=AVERAGE_DUAL_GAP,
            complementarity_gap=float(np.sum(value['multipliers']*value['slacks'])),
            dual_nonnegative=bool(np.all(value['multipliers'] > 0)),
            candidate_dual_objective=dual, candidate_primal_dual_gap=value['primal']-dual,
            candidate_dual_box_feasible=bool(np.all((p>=0)&(p<=1))),
            candidate_dual_equality_exact=bool(float(value['q'].sum())==0.),
            dual_intercept_feasibility_residual=residuals['intercept_residual'],
            dual_certificate_scope='approximate equilibrium only; candidate dual value is not a certified feasible bound unless its exact equality holds',
            complementarity_residual=float(np.max(np.abs(value['multipliers']*value['slacks']-zeta))),
            K_numeric_bytes=K.nbytes, deployment_numeric_bytes=alpha.nbytes+8,
            resident_numeric_bytes=sum(v.nbytes for v in (alpha,value['slacks'],t,old,bounds,K,value['f'],value['q'],value['D'],labels,weights,class_labels,class_counts,class_targets))+16,
            process_peak_bytes=None, factor_buffer_scope='explicit H plus Cholesky; excludes LAPACK workspace and other arrays',
            explicit_temporary_scope='three explicit solve RHS/solution buffers; not all process temporaries',
            wall_seconds=time.perf_counter()-started)
        return GroupBalancedBarrierState(*[_seal(v) if isinstance(v, np.ndarray) else v for v in
            (alpha, float(b), value['slacks'], zeta, t, old, bounds, K, value['f'], value['q'], value['D'])],
            _audit(dict(audit,schema=SCHEMA)), max_factor_buffer_bytes,
            *[_seal(v) for v in (labels,weights,class_labels,class_counts,class_targets)])
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
        audit.update(status='TECHNICAL_FAILURE', wall_seconds=time.perf_counter()-started)
        # Always rebuild slacks from the same accepted state, even after a failed trial.
        arrays = dict(K=K, alpha=alpha, b=np.asarray(b), f=K@alpha+b,labels=labels,weights=weights,
            class_labels=class_labels,class_counts=class_counts,class_targets=class_targets,
            slacks=(K@alpha+b)[old, None]-bounds, targets=t, old_indices=old, lower_bounds=bounds)
        raise GroupBalancedBarrierFailure(str(exc), dict(audit,schema=SCHEMA), arrays) from exc


def predict_group_balanced_barrier_gate(state,*,L):
    if not isinstance(state,GroupBalancedBarrierState):raise TypeError('Group-balanced gate state required')
    return original.predict_group_barrier_gate(state,L=L)


def group_balanced_barrier_gate_vjp(state,*,L,G):
    """Pure existing adjoint uses the new weighted D_eff, qeff and slacks.

    Weights and class counts are fixed data. No derivative of their labels is
    introduced; all symmetric K, L, bounds and free-b terms remain present.
    """
    if not isinstance(state,GroupBalancedBarrierState):raise TypeError('Group-balanced gate state required')
    try:
        result=original.group_barrier_gate_vjp(state,L=L,G=G)
        audit=dict(_ledger(),**result.audit, schema=SCHEMA,weights_fixed=True)
        return original.GroupBarrierVJP(result.gradK,result.gradL,result.gradlower_bounds,_audit(audit))
    except original.GroupBarrierFailure as exc:
        raise GroupBalancedBarrierFailure(exc.code,dict(_ledger(),**exc.audit,schema=SCHEMA),
            dict(exc.arrays,labels=state.labels,weights=state.weights)) from exc


def group_balanced_barrier_gate_jvp(state,*,K_jacobian,L,L_jacobian,bounds_jacobian):
    if not isinstance(state,GroupBalancedBarrierState):raise TypeError('Group-balanced gate state required')
    dk=_float_array(K_jacobian,'K_jacobian',3);L=_float_array(L,'L',2)
    dl=_float_array(L_jacobian,'L_jacobian',3);dbounds=_float_array(bounds_jacobian,'bounds_jacobian',3)
    n=len(state.alpha);r=dk.shape[-1]
    if r<1 or dk.shape!=(n,n,r) or not np.array_equal(dk,dk.swapaxes(0,1)) or L.shape[1]!=n or dl.shape!=L.shape+(r,) or dbounds.shape!=state.lower_bounds.shape+(r,):
        raise ValueError('Full symmetric K/L/all-bounds JVP shapes required')
    audit=_ledger();started=time.perf_counter();partial=dict(labels=state.labels,weights=state.weights)
    try:
        root,H,chol=_factor(state.K,state.D_eff,state.max_factor_buffer_bytes,audit)
        partial.update(gate_jvp_H=H,gate_jvp_chol=chol)
        dkalpha=np.einsum('ijp,j->ip',dk,state.alpha)
        ratio=((state.zeta/state.slacks)/state.slacks)/state.D_eff[state.old_indices,None]
        if np.any(ratio<0) or np.any(ratio>1+16*EPS):raise ArithmeticError('INVALID_GATE_JVP_CURVATURE_RATIO')
        scaled=np.zeros((n,r));scaled[state.old_indices]=root[state.old_indices,None]*np.einsum('ij,ijp->ip',ratio,dbounds)
        rhs=-root[:,None]*dkalpha+scaled
        partial['gate_jvp_rhs']=np.column_stack((rhs,root))
        solved=_solve(H,chol,partial['gate_jvp_rhs'],audit);partial['gate_jvp_solved']=solved
        u=solved[:,-1];s=float(root@u)
        if not np.isfinite(s) or s<=0:raise ArithmeticError('GATE_JVP_INTERCEPT_SCHUR_NONPOSITIVE')
        db=(root@solved[:,:r])/s;x=solved[:,:r]-u[:,None]*db;da=root[:,None]*x
        residual=float(np.max(np.abs(H@x+root[:,None]*db-rhs)))
        equality=float(np.max(np.abs(da.sum(axis=0))))
        scale=float(np.max(np.abs(H)@np.abs(x)+root[:,None]*np.abs(db)+np.abs(rhs)))
        tol=128*EPS*max(1,n)*max(1.,scale)
        eqtol=128*EPS*max(1,n)*max(1.,float(np.max(np.sum(np.abs(da),axis=0))))
        dg=np.einsum('ijp,j->ip',dl,state.alpha)+L@da+db
        if not all(np.isfinite(v).all() for v in (da,db,dg)):raise ArithmeticError('NONFINITE_GATE_JVP')
        audit.update(status='COMPLETE',jvp_scaled_residual=residual,jvp_scaled_tolerance=tol,
            jvp_intercept_residual=equality,jvp_intercept_tolerance=eqtol,weights_fixed=True,schema=SCHEMA)
        if residual>tol or equality>eqtol:raise ArithmeticError('GATE_JVP_READBACK_FAILED')
        partial.update(gate_alpha_jacobian=da,gate_b_jacobian=db,gate_score_jacobian=dg)
        audit['wall_seconds']=time.perf_counter()-started
        arrays=MappingProxyType({k:_seal(v) for k,v in partial.items()})
        return GroupBalancedBarrierJVP(arrays['gate_score_jacobian'],arrays,_audit(audit))
    except Exception as exc:
        audit.update(status='TECHNICAL_FAILURE',wall_seconds=time.perf_counter()-started,schema=SCHEMA)
        raise GroupBalancedBarrierFailure(getattr(exc,'code',str(exc)),audit,partial) from exc
