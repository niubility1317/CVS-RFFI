"""Finite-barrier support gate. No file I/O, training route or query statistics.

The fixed barrier is a distinct smooth candidate, not an exact hard-QP solver.
All dense numerical state is copied and read-only; audit values are scalars.
"""
from dataclasses import dataclass
from types import MappingProxyType
import time

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dpocon


EPS = np.finfo(np.float64).eps
AVERAGE_DUAL_GAP = 1e-4
ARMIJO = 1e-4


def _seal(value):
    array = np.array(value, copy=True)
    # Backing immutable bytes also prevents callers re-enabling WRITEABLE.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _audit(values):
    return MappingProxyType({key: value.item() if isinstance(value,np.generic) else value
                             for key,value in values.items()})


class GroupBarrierFailure(RuntimeError):
    def __init__(self, code, audit, arrays):
        super().__init__(code)
        self.code = code
        self.audit = _audit(dict(audit, failure_code=code))
        self.arrays = MappingProxyType({k: _seal(v) for k, v in arrays.items()})

    def audit_dict(self):
        return dict(self.audit)


@dataclass(frozen=True)
class GroupBarrierState:
    alpha: np.ndarray
    b: float
    slacks: np.ndarray
    zeta: float
    targets: np.ndarray
    old_indices: np.ndarray
    lower_bounds: np.ndarray
    K: np.ndarray
    f: np.ndarray
    qeff: np.ndarray
    D_eff: np.ndarray
    audit: object
    max_factor_buffer_bytes: int

    @property
    def arrays(self):
        return MappingProxyType({name:getattr(self,name) for name in
            ('alpha','slacks','targets','old_indices','lower_bounds','K','f','qeff','D_eff')})

    def audit_dict(self):
        return dict(self.audit)


@dataclass(frozen=True)
class GroupBarrierVJP:
    gradK: np.ndarray
    gradL: np.ndarray
    gradlower_bounds: np.ndarray
    audit: object


def _ledger():
    return dict(factorization_attempts=0, factorizations_completed=0,
        condition_estimation_calls=0, triangular_calls=0, triangular_rhs_columns=0,
        triangular_rhs_elements=0, triangular_dense_work_units=0,
        spectral_checks=0, spectral_cubic_dimension_units=0,
        objective_evaluations=0, logistic_record_evaluations=0,
        barrier_constraint_evaluations=0, line_search_trials=0,
        newton_iterations=0, accepted_steps=0, round_off_residual_acceptances=0, peak_factor_buffer_bytes=0,
        peak_explicit_temporary_bytes=0, factorization_seconds=0.,
        triangular_seconds=0., spectral_seconds=0., objective_seconds=0.)


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be an explicit positive integer')


def _float_array(value, name, ndim):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(name + ' must be finite with ndim=' + str(ndim))
    return result


def _evaluate(K, alpha, b, targets, old, bounds, zeta, audit):
    tick = time.perf_counter()
    audit['objective_evaluations'] += 1
    audit['logistic_record_evaluations'] += len(targets)
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
            q = np.where(targets == 1, -negative, positive)
            curvature = e / ((1 + e) ** 2)
            multipliers = zeta / slacks
            barrier_D = multipliers / slacks
            q[old] -= multipliers.sum(axis=1)
            curvature[old] += barrier_D.sum(axis=1)
            if np.any(curvature <= 0) or not np.isfinite(curvature).all():
                raise ArithmeticError('INVALID_EFFECTIVE_CURVATURE')
            losses = np.logaddexp(0., np.where(targets == 1, -f, f))
            ridge = .5 * float(alpha @ Ka)
            loss_sum = float(losses.sum())
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
            q_error += 8*EPS*(np.abs(q)+1)
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
            logistic_error = float(f_error.sum())+(gamma+8*EPS)*float(np.abs(losses).sum())
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


def _factor(K, D, limit, audit):
    n = len(D)
    # Explicit factor input and returned Cholesky coexist. LAPACK workspace is
    # not claimed as measured process memory. Check before constructing H.
    required = 2 * n * n * 8
    if required > limit:
        raise ArithmeticError('FACTOR_BUFFER_LIMIT')
    root = np.sqrt(D)
    H = (root[:, None] * K) * root[None, :]
    H.flat[::n+1] += 1.
    audit['peak_factor_buffer_bytes'] = max(audit['peak_factor_buffer_bytes'], required)
    tick = time.perf_counter()
    audit['factorization_attempts'] += 1
    try:
        chol = cholesky(H, lower=True, check_finite=True, overwrite_a=False)
        audit['factorizations_completed'] += 1
    finally:
        audit['factorization_seconds'] += time.perf_counter() - tick
    audit['condition_estimation_calls'] += 1
    rcond, info = dpocon(chol, float(np.linalg.norm(H, 1)), uplo='L')
    if info != 0 or not np.isfinite(rcond) or rcond <= 64 * EPS * max(1, n):
        raise ArithmeticError('NEWTON_SYSTEM_CONDITION_UNRESOLVED')
    audit['last_condition_estimate'] = float(1 / rcond)
    audit['max_condition_estimate'] = max(audit.get('max_condition_estimate', 1.), float(1 / rcond))
    return root, H, chol


def _solve(H, chol, rhs, audit):
    n, columns = rhs.shape
    audit['peak_explicit_temporary_bytes'] = max(audit['peak_explicit_temporary_bytes'], 3 * rhs.nbytes)
    value = rhs
    tick = time.perf_counter()
    try:
        for transpose in (False, True):
            audit['triangular_calls'] += 1
            audit['triangular_rhs_columns'] += columns
            audit['triangular_rhs_elements'] += n * columns
            audit['triangular_dense_work_units'] += n * n * columns
            value = solve_triangular(chol, value, lower=True, trans='T' if transpose else 'N', check_finite=True)
    finally:
        audit['triangular_seconds'] += time.perf_counter() - tick
    residual = float(np.linalg.norm(H @ value - rhs, np.inf))
    scale = float(np.linalg.norm(H, np.inf) * np.linalg.norm(value, np.inf) + np.linalg.norm(rhs, np.inf))
    tolerance = 128 * EPS * max(1, n) * max(1., scale)
    audit['last_solve_residual'] = residual
    audit['last_solve_tolerance'] = tolerance
    if residual > tolerance or not np.isfinite(value).all():
        raise ArithmeticError('NEWTON_SOLVE_RESIDUAL')
    return value


def _residuals(K, alpha, value):
    h = alpha + value['q']
    scale = max(1., float(np.linalg.norm(alpha, np.inf)), float(np.linalg.norm(value['q'], np.inf)))
    tolerance = max(512 * EPS * max(1, len(alpha)) * scale, value['q_roundoff'])
    return h, tolerance, dict(canonical_residual=float(np.linalg.norm(h, np.inf)),
        intercept_residual=abs(float(value['q'].sum())),
        sum_alpha_residual=abs(float(alpha.sum())),
        primal_gradient_residual=float(np.linalg.norm(K @ h, np.inf)),
        stationarity_tolerance=tolerance,
        evaluated_gradient_roundoff_bound=value['q_roundoff'],
        intercept_tolerance=len(alpha) * tolerance,
        primal_gradient_tolerance=max(1., float(np.linalg.norm(K, np.inf))) * tolerance)


def fit_group_barrier_gate(*, K, targets, old_indices, lower_bounds,
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
    audit = _ledger()
    audit.update(max_newton_iterations=max_newton_iterations,max_line_search_trials=max_line_search_trials,
        max_factor_buffer_bytes=max_factor_buffer_bytes,ridge_coefficient=1.,armijo_coefficient=ARMIJO)
    started = time.perf_counter()
    alpha = np.zeros(n)
    b = 0.
    zeta = n * AVERAGE_DUAL_GAP / bounds.size
    value = None
    try:
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
        value = _evaluate(K, alpha, b, t, old, bounds, zeta, audit)
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
                candidate = _evaluate(K, next_alpha, next_b, t, old, bounds, zeta, audit)
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
        value = _evaluate(K, alpha, b, t, old, bounds, zeta, audit)
        _, tolerance, residuals = _residuals(K, alpha, value)
        audit.update(residuals)
        if residuals['canonical_residual'] > tolerance or residuals['intercept_residual'] > n*tolerance:
            raise ArithmeticError('FINAL_EQUILIBRIUM_READBACK')
        audit['logistic_record_evaluations'] += n
        e = np.exp(-np.abs(value['f']))
        p = np.where(value['f'] >= 0, 1/(1+e), e/(1+e))
        one_minus_p = np.where(value['f'] >= 0, e/(1+e), 1/(1+e))
        entropy = -p*np.logaddexp(0.,-value['f'])-one_minus_p*np.logaddexp(0.,value['f'])
        candidate_dual_alpha = -value['q']
        dual = -.5*float(candidate_dual_alpha@(K@candidate_dual_alpha))-float(entropy.sum())+float(np.sum(value['multipliers']*bounds))
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
            resident_numeric_bytes=sum(v.nbytes for v in (alpha,value['slacks'],t,old,bounds,K,value['f'],value['q'],value['D']))+16,
            process_peak_bytes=None, factor_buffer_scope='explicit H plus Cholesky; excludes LAPACK workspace and other arrays',
            explicit_temporary_scope='three explicit solve RHS/solution buffers; not all process temporaries',
            wall_seconds=time.perf_counter()-started)
        return GroupBarrierState(*[_seal(v) if isinstance(v, np.ndarray) else v for v in
            (alpha, float(b), value['slacks'], zeta, t, old, bounds, K, value['f'], value['q'], value['D'])],
            _audit(audit), max_factor_buffer_bytes)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
        audit.update(status='TECHNICAL_FAILURE', wall_seconds=time.perf_counter()-started)
        # Always rebuild slacks from the same accepted state, even after a failed trial.
        arrays = dict(K=K, alpha=alpha, b=np.asarray(b), f=K@alpha+b,
            slacks=(K@alpha+b)[old, None]-bounds, targets=t, old_indices=old, lower_bounds=bounds)
        raise GroupBarrierFailure(str(exc), audit, arrays) from exc


def predict_group_barrier_gate(state, *, L):
    L = _float_array(L, 'L', 2)
    if L.shape[1] != len(state.alpha):
        raise ValueError('L must have one column per physical training record')
    result = L @ state.alpha + state.b
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite prediction')
    return _seal(result)


def group_barrier_gate_vjp(state, *, L, G):
    L = _float_array(L, 'L', 2)
    G = _float_array(G, 'G', 1)
    if L.shape != (len(G), len(state.alpha)):
        raise ValueError('L/G endpoint shapes mismatch')
    audit = _ledger()
    tick = time.perf_counter()
    try:
        K, D, alpha = state.K, state.D_eff, state.alpha
        root, H, chol = _factor(K, D, state.max_factor_buffer_bytes, audit)
        covalpha, covb = L.T @ G, float(G.sum())
        solved = _solve(H, chol, np.column_stack((root*covalpha, root)), audit)
        s = float(root @ solved[:, 1])
        if not np.isfinite(s) or s <= 0:
            raise ArithmeticError('ADJOINT_INTERCEPT_SCHUR_NONPOSITIVE')
        c = float(root @ solved[:, 0]-covb)/s
        # Solve in sqrt(D)-scaled coordinates. Forming r by subtraction and
        # subsequently multiplying by D loses the adjoint on rank-one kernels
        # with high barrier curvature. Here y=sqrt(D)*r and vD=D*r are obtained
        # directly from the SPD solutions; neither gradient uses a lost r.
        y = solved[:, 0]-solved[:, 1]*c
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            vD = root*y
            r = y/root
            consistency = np.abs(D*r-vD)
            consistency_bound = 16*EPS*(np.abs(D*r)+np.abs(vD))
        if not all(np.isfinite(v).all() for v in (r,vD,consistency,consistency_bound)):
            raise ArithmeticError('NONFINITE_SCALED_ADJOINT')
        equation_residual = float(np.max(np.abs(r+K@vD+c-covalpha)))
        intercept_residual = abs(float(vD.sum())-covb)
        residual = max(equation_residual,intercept_residual)
        scale = max(1.,float(np.linalg.norm(covalpha,np.inf)),abs(covb),
                    float(np.linalg.norm(r,np.inf))*(1+float(np.linalg.norm(K,np.inf))*float(np.max(D))),abs(c))
        tolerance = 512*EPS*len(alpha)*scale
        audit.update(adjoint_residual=residual,adjoint_tolerance=tolerance,
            adjoint_equation_residual=equation_residual,adjoint_intercept_residual=intercept_residual,
            scaled_adjoint_consistency_residual=float(np.max(consistency)),
            scaled_adjoint_consistency_roundoff_bound=float(np.max(consistency_bound)))
        if np.any(consistency>consistency_bound):
            raise ArithmeticError('SCALED_ADJOINT_CONSISTENCY_RESIDUAL')
        if residual > tolerance:
            raise ArithmeticError('ADJOINT_READBACK_RESIDUAL')
        raw = -np.outer(vD, alpha)
        gradK = .5*(raw+raw.T)
        gradL = np.outer(G, alpha)
        # Each pair's positive curvature is a summand of the corresponding D.
        # Use its bounded ratio; do not recover D*r from a subtracted r.
        ratio = ((state.zeta/state.slacks)/state.slacks)/D[state.old_indices,None]
        if not np.isfinite(ratio).all() or np.any(ratio<0) or np.any(ratio>1+16*EPS):
            raise ArithmeticError('INVALID_BARRIER_CURVATURE_RATIO')
        gradbounds = vD[state.old_indices,None]*ratio
        if not all(np.isfinite(v).all() for v in (gradK,gradL,gradbounds)):
            raise ArithmeticError('NONFINITE_ADJOINT_OUTPUT')
        audit.update(status='COMPLETE', adjoint_residual=residual, adjoint_tolerance=tolerance,
            intercept_adjoint=c, wall_seconds=time.perf_counter()-tick,
            gradient_scope='smooth fixed-zeta full physical bounds; K varied symmetrically')
        return GroupBarrierVJP(_seal(gradK), _seal(gradL), _seal(gradbounds), _audit(audit))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
        audit.update(status='TECHNICAL_FAILURE', wall_seconds=time.perf_counter()-tick)
        raise GroupBarrierFailure(str(exc), audit, dict(alpha=state.alpha,b=np.asarray(state.b),
            slacks=state.slacks,L=L,G=G)) from exc
