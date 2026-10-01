"""Bounded margin-constrained affine ridge head; not a complete D92 method.

Only train labels enter fit. The prior is an explicit frozen numeric input.
No files, checkpoint, encoder, geometry, registration router, or query state.
All returned arrays own immutable byte-backed storage; audit properties return
independent mutable copies. No jitter, pseudoinverse or unconstrained fallback.
"""
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.linalg import cholesky, eigvalsh, solve_triangular
from scipy.linalg.lapack import dpocon

SCHEMA = "d92_margin_qp_head_component_v1"
_EPS = np.finfo(np.float64).eps


def _array(value, name, ndim):
    if np.iscomplexobj(value):
        raise ValueError(name + ": real finite array required")
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(name + ": finite array with required dimension expected")
    return result


def _indices(value, name, n, *, nonempty=True):
    result = np.asarray(value)
    if (result.ndim != 1 or result.dtype.kind not in "iu"
            or (nonempty and not len(result)) or np.any(result < 0) or np.any(result >= n)):
        raise ValueError(name + ": explicit integer indices required")
    return np.array(result, dtype=np.int64, copy=True)


def _gamma(operations):
    product = int(operations) * _EPS
    if product >= .01:
        raise ValueError("Dimension exceeds supported floating-point error model")
    return product / (1.0 - product)


def _maxabs(value):
    return float(np.max(np.abs(value), initial=0.0))


def _frozen(arrays):
    result = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        shape = value.shape
        contiguous = np.ascontiguousarray(value)
        result[name] = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(shape)
    return MappingProxyType(result)


class MarginQPFailure(RuntimeError):
    """Technical failure, never a claim that this known-feasible QP is infeasible."""
    def __init__(self, code, message, audit, arrays):
        super().__init__(code + ": " + message)
        self.code = code
        self._audit = deepcopy(audit)
        self._arrays = _frozen(arrays)

    @property
    def audit(self):
        return deepcopy(self._audit)

    @property
    def arrays(self):
        return self._arrays


class UnsupportedJacobian(MarginQPFailure):
    pass


@dataclass(frozen=True)
class MarginQPHead:
    _arrays: object
    _audit: dict

    @property
    def arrays(self):
        return self._arrays

    @property
    def audit(self):
        return deepcopy(self._audit)

    @property
    def alpha(self):
        return self._arrays["alpha"]

    @property
    def b(self):
        return self._arrays["b"]

    @property
    def train_scores(self):
        return self._arrays["train_scores"]


@dataclass(frozen=True)
class MarginQPVJP:
    _arrays: object
    _audit: dict

    @property
    def arrays(self):
        return self._arrays

    @property
    def audit(self):
        return deepcopy(self._audit)


class _Ledger:
    def __init__(self, max_transitions, max_factor_buffer_bytes):
        self.data = dict(
            schema=SCHEMA, status="IN_PROGRESS", lambda_ridge=1.0,
            max_transitions=max_transitions, max_factor_buffer_bytes=max_factor_buffer_bytes,
            transitions=0, full_constraint_scans=0, spectral_checks=0,
            compact_snapshot_rebuilds=0, compact_snapshot_dense_work_units=0,
            compact_snapshot_work_scope="m*m*C dimension units per explicit P_old@V; not measured FLOPs or time",
            spectral_cubic_dimension_units=0, independence_checks=0,
            factorization_attempts=0, factorizations_completed=0,
            condition_estimation_calls=0, triangular_calls=0,
            triangular_rhs_columns=0, triangular_rhs_elements=0,
            triangular_dense_work_units=0, solves=[], factorizations=[], events=[],
            peak_factor_buffer_bytes=0, peak_explicit_solve_temporary_bytes=0,
            factor_buffer_scope="Owned factor input/output arrays; excludes kernel/state arrays and unmeasured library workspace",
            temporary_scope="Two explicit triangular-solve outputs; excludes caller RHS, BLAS workspace and other live arrays",
            process_peak_memory_bytes=None, wall_seconds=None, deployment_bytes=None,
            no_full_dual_hessian_preallocation=True, training_method_complete=False,
        )
        self.snapshot = {}

    def fail(self, code, message, **details):
        self.data.update(status="TECHNICAL_FAILURE", failure_code=code, failure_details=details)
        raise MarginQPFailure(code, message, self.data, self.snapshot)

    def reserve(self, bytes_needed):
        if bytes_needed > self.data["max_factor_buffer_bytes"]:
            self.fail("FACTOR_BUFFER_LIMIT", "Required owned factor buffers exceed caller limit",
                      required_bytes=int(bytes_needed))

    def observe_factor_buffers(self, bytes_owned):
        self.data["peak_factor_buffer_bytes"] = max(
            self.data["peak_factor_buffer_bytes"], int(bytes_owned))

    def factor(self, matrix, system, *, retained_bytes=0):
        self.reserve(retained_bytes + 2 * matrix.nbytes)
        self.observe_factor_buffers(retained_bytes + matrix.nbytes)
        record = dict(system=system, dimension=len(matrix), status="ATTEMPTED")
        self.data["factorizations"].append(record)
        self.data["factorization_attempts"] += 1
        try:
            factor = cholesky(matrix, lower=True, overwrite_a=False, check_finite=False)
        except np.linalg.LinAlgError as error:
            record["status"] = "FAILED"
            self.fail("WORKING_RANK_UNRESOLVED" if system == "working" else "AFFINE_FACTOR_FAILED",
                      str(error))
        record["status"] = "COMPLETED"
        self.data["factorizations_completed"] += 1
        self.observe_factor_buffers(retained_bytes + matrix.nbytes + factor.nbytes)
        self.data["condition_estimation_calls"] += 1
        rcond, info = dpocon(factor, float(np.linalg.norm(matrix, 1)), uplo="L")
        record.update(rcond=float(rcond), condition_estimation_info=int(info))
        if info != 0 or not np.isfinite(rcond) or rcond <= _gamma(32 * (len(matrix) + 1)):
            self.fail("WORKING_RANK_UNRESOLVED" if system == "working" else "AFFINE_CONDITION_UNRESOLVED",
                      "Condition estimate cannot resolve an invertible system", rcond=float(rcond))
        return factor, float(rcond)

    def solve(self, factor, rhs, system):
        rhs = np.asarray(rhs, dtype=np.float64)
        width = 1 if rhs.ndim == 1 else rhs.shape[1]
        n = len(factor)
        self.data["peak_explicit_solve_temporary_bytes"] = max(
            self.data["peak_explicit_solve_temporary_bytes"], int(2 * rhs.nbytes))
        value = rhs
        for transpose in (False, True):
            record = dict(system=system, dimension=n, rhs_columns=width,
                          transpose=transpose, status="ATTEMPTED")
            self.data["solves"].append(record)
            self.data["triangular_calls"] += 1
            self.data["triangular_rhs_columns"] += width
            self.data["triangular_rhs_elements"] += n * width
            self.data["triangular_dense_work_units"] += n * n * width
            try:
                value = solve_triangular(factor, value, lower=True, trans="T" if transpose else "N",
                                         check_finite=False, overwrite_b=False)
            except (np.linalg.LinAlgError, ValueError) as error:
                record["status"] = "FAILED"
                self.fail("TRIANGULAR_SOLVE_FAILED", str(error))
            record["status"] = "COMPLETED"
            if not np.isfinite(value).all():
                self.fail("NONFINITE_SOLVE", "Triangular solve returned nonfinite values")
        return value


def _pairs(labels, old, classes):
    rows = np.repeat(np.arange(len(old)), classes - 1)
    truths = np.repeat(labels[old], classes - 1)
    others = np.concatenate([np.delete(np.arange(classes), labels[i]) for i in old])
    return rows, truths, others


def _difference(scores, rows, truths, others):
    return scores[rows, truths] - scores[rows, others]


def _scatter(values, rows, truths, others, m, classes):
    table = np.zeros((m, classes))
    np.add.at(table, (rows, truths), values)
    np.add.at(table, (rows, others), -values)
    return table


def _working_matrix(P_old, rows, truths, others, working):
    w = np.asarray(working, dtype=np.int64)
    y, j = truths[w], others[w]
    inner = ((y[:, None] == y[None, :]).astype(np.float64)
             - (y[:, None] == j[None, :])
             - (j[:, None] == y[None, :])
             + (j[:, None] == j[None, :]))
    return P_old[np.ix_(rows[w], rows[w])] * inner


def _affine(ledger, factor, z, s, rhs, system):
    solved = ledger.solve(factor, rhs, system)
    b = (z @ rhs) / s
    return solved - z[:, None] * b, b


def fit_margin_qp_head(*, K, M, labels, old_indices,
                       max_transitions, max_factor_buffer_bytes):
    """Fit exactly this head's fixed lambda=1 problem, or preserve technical failure.

    K: symmetric PSD float64[n,n]; M: frozen prior[n,C]. All labels are
    legal TRAIN labels. old_indices defines constrained train rows only.
    Resource limits are explicit positive integers, with no hidden retry.
    This function does not implement new0 reuse or B/adapter preparation.
    """
    for name, value in (("max_transitions", max_transitions),
                        ("max_factor_buffer_bytes", max_factor_buffer_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + ": explicit finite positive integer required")
    K = _array(K, "K", 2)
    M = _array(M, "M", 2)
    n, classes = M.shape
    if n < 1 or classes < 2 or K.shape != (n, n) or not np.array_equal(K, K.T):
        raise ValueError("Exact symmetric K[n,n], nonempty train and >=2 classes required")
    labels = _indices(labels, "labels", classes)
    old = _indices(old_indices, "old_indices", n)
    if labels.shape != (n,) or len(np.unique(old)) != len(old):
        raise ValueError("Train label count or unique old train rows mismatch")
    old = np.sort(old)
    m = len(old)
    rows, truths, others = _pairs(labels, old, classes)
    q = len(rows)
    ledger = _Ledger(max_transitions, max_factor_buffer_bytes)
    ledger.snapshot = dict(K=K, M=M, labels=labels, old_indices=old,
                           rho=np.asarray(0.0), V=np.zeros((m, classes)), working_set=np.empty(0, dtype=np.int64))
    # 32 is a fixed arithmetic safety envelope for the composed dense operations,
    # not a performance parameter. Actual norms/backward error/condition enter below.
    base = _gamma(32 * (n + classes + q + 1))
    ledger.data.update(n=n, m=m, classes=classes, constraint_count=q, machine_epsilon=_EPS,
                       arithmetic_gamma=base, tolerance_rule="gamma_32(n+C+q+1) times actual norm, backward-error and condition scales")
    ledger.reserve(16 * n * n)
    ledger.data["spectral_checks"] += 1
    ledger.data["spectral_cubic_dimension_units"] += n ** 3
    try:
        spectrum = eigvalsh(K, check_finite=False)
    except np.linalg.LinAlgError as error:
        ledger.fail("KERNEL_SPECTRUM_UNRESOLVED", str(error))
    spectral_tol = _gamma(16 * (n + 1)) * max(1.0, float(np.linalg.norm(K, 1)))
    ledger.data.update(kernel_min_eigenvalue=float(spectrum[0]), kernel_spectral_tolerance=spectral_tol)
    if spectrum[0] < -spectral_tol:
        ledger.fail("NON_PSD_KERNEL", "Kernel violates PSD beyond arithmetic uncertainty")
    if spectrum[0] < 0:
        ledger.data["kernel_psd_roundoff_uncertainty"] = float(-spectrum[0])
    A = K + np.eye(n)
    chol_A, rcond_A = ledger.factor(A, "affine")
    z = ledger.solve(chol_A, np.ones(n), "affine_constant")
    s = float(np.sum(z))
    if not np.isfinite(s) or s <= 0:
        ledger.fail("AFFINE_INTERCEPT_UNRESOLVED", "Nonpositive affine constant denominator")
    R = np.eye(classes)[labels] - 1.0 / classes - M
    alpha0, b0 = _affine(ledger, chol_A, z, s, R, "affine_base")
    residual0 = K @ alpha0 + b0
    base_error = _maxabs(A @ alpha0 + b0 - R) + _maxabs(np.sum(alpha0, axis=0))
    selection = np.zeros((n, m))
    selection[old, np.arange(m)] = 1.0
    selected_solve = ledger.solve(chol_A, selection, "old_response_columns")
    P_old_raw = K[old] @ selected_solve + np.outer(z[old], z[old]) / s
    p_sym_error = _maxabs(P_old_raw - P_old_raw.T)
    p_scale = max(1.0, _maxabs(P_old_raw))
    if p_sym_error > base * p_scale / rcond_A:
        ledger.fail("OLD_RESPONSE_ASYMMETRY", "Old response symmetry is numerically unresolved")
    # Roundoff symmetrization only; no eigenvalue alteration or diagonal addition.
    P_old = (P_old_raw + P_old_raw.T) * .5
    response_error = _maxabs(A @ selected_solve - selection)
    arithmetic = base / rcond_A + base_error + response_error + p_sym_error
    base_old = residual0[old]
    kappa = float(np.sum(R * residual0))
    baseline_objective = float(.5 * np.sum((residual0 - R) ** 2) + .5 * np.sum(alpha0 * (K @ alpha0)))
    margins = np.array([min(M[i, labels[i]] - M[i, j] for j in range(classes) if j != labels[i]) for i in old])
    delta = margins[rows]
    s0 = _difference(M[old] + base_old, rows, truths, others) - delta
    ledger.data.update(affine_rcond=rcond_A, affine_base_residual=base_error, old_response_solve_residual=response_error,
                       old_response_symmetry_error=p_sym_error, arithmetic_bound=arithmetic,
                       A_input_numeric_bytes=A.nbytes, A_factor_numeric_bytes=chol_A.nbytes,
                       P_old_numeric_bytes=P_old.nbytes, old_response_temporary_numeric_bytes=selection.nbytes + selected_solve.nbytes + P_old_raw.nbytes)
    del selection, selected_solve, P_old_raw, spectrum
    rho = 0.0
    V = np.zeros((m, classes))
    working = []
    seen = set()
    QW = chol_W = None
    rcond_W = 1.0
    accepted_mu = None

    def snapshot_current_primal():
        # Rebuild from the current raw compact state BEFORE any operation that
        # can fail. This also covers the last MOVE exhausting the loop budget.
        # A previous working solve's direction/multiplier is not current-state
        # evidence after rho/V/W change; its event remains in the audit trail.
        current_old = rho * base_old + P_old @ V
        slack = _difference(M[old] + current_old, rows, truths, others) - delta
        ledger.data["full_constraint_scans"] += 1
        ledger.data["compact_snapshot_rebuilds"] += 1
        ledger.data["compact_snapshot_dense_work_units"] += m * m * classes
        ledger.snapshot.pop("last_working_multipliers", None)
        ledger.snapshot.pop("direction_energy", None)
        ledger.snapshot.update(rho=np.asarray(rho), V=V, working_set=np.asarray(working, dtype=np.int64),
                               margins=margins, P_old=P_old, slack=slack)
        return current_old, slack

    for transition in range(max_transitions):
        ledger.data["transitions"] += 1
        current_old, slack = snapshot_current_primal()
        signature = (tuple(working), rho.hex(), V.tobytes())
        if signature in seen:
            ledger.fail("WORKING_SET_CYCLE", "Exact compact primal/working state repeated")
        seen.add(signature)
        ledger.data["cycle_snapshot_numeric_bytes"] = sum(len(item[2]) for item in seen)
        a = len(working)
        QW = chol_W = None  # do not retain a previous working factor while rebuilding
        ledger.reserve(16 * n * n + 16 * a * a)
        mu = np.zeros(q)
        if a:
            QW = _working_matrix(P_old, rows, truths, others, working)
            chol_W, rcond_W = ledger.factor(QW, "working", retained_bytes=A.nbytes + chol_A.nbytes)
            mu[working] = ledger.solve(chol_W, -s0[working], "working_multiplier")
            work_error = _maxabs(QW @ mu[working] + s0[working])
            if work_error > base * max(1.0, _maxabs(QW) * _maxabs(mu), _maxabs(s0)) / rcond_W:
                ledger.fail("WORKING_SOLVE_RESIDUAL", "Working equality solve failed residual check")
        else:
            rcond_W = 1.0
        Vstar = _scatter(mu, rows, truths, others, m, classes)
        drho = 1.0 - rho
        dV = Vstar - V
        PdV = P_old @ dV
        direction = drho * base_old + PdV
        energy_terms = (drho * drho * kappa, 2 * drho * float(np.sum(dV * base_old)), float(np.sum(dV * PdV)))
        energy = float(sum(energy_terms))
        energy_tol = (arithmetic + base / rcond_W) * max(1.0, sum(abs(term) for term in energy_terms))
        slope = _difference(direction, rows, truths, others)
        slack_tol = (arithmetic + base / rcond_W) * max(1.0, _maxabs(M), _maxabs(current_old), _maxabs(delta))
        slope_tol = (arithmetic + base / rcond_W) * max(1.0, _maxabs(direction))
        event = dict(transition=transition, working=working.copy(), rho=rho, direction_energy=energy,
                     direction_energy_tolerance=energy_tol, minimum_slack=float(np.min(slack)),
                     slack_tolerance=slack_tol, working_rcond=rcond_W)
        ledger.data["events"].append(event)
        ledger.snapshot.update(slack=slack, last_working_multipliers=mu, direction_energy=np.asarray(energy))
        if np.min(slack) < -slack_tol:
            ledger.fail("PRIMAL_FEASIBILITY_LOST", "Full constraint scan failed", minimum_slack=float(np.min(slack)))
        if a and _maxabs(slack[working]) > slack_tol:
            ledger.fail("WORKING_EQUALITY_LOST", "Current primal escaped working equalities")
        if not np.isfinite(energy) or energy < -energy_tol:
            ledger.fail("DIRECTION_ENERGY_UNRESOLVED", "Compact primal energy is inconsistent")
        if energy <= energy_tol:
            dual_tol = (arithmetic + base / rcond_W) * max(1.0, _maxabs(mu))
            negative = [index for index in working if mu[index] < -dual_tol]
            if negative:
                removed = min(negative)
                working.remove(removed)
                event.update(action="REMOVE_NEGATIVE_MULTIPLIER", removed=removed, multiplier=float(mu[removed]))
                continue
            event["action"] = "FINAL_KKT_CANDIDATE"
            accepted_mu = mu
            break
        inactive = np.ones(q, dtype=bool)
        inactive[working] = False
        candidates = np.flatnonzero(inactive & (slope < -slope_tol))
        ratios = slack[candidates] / (-slope[candidates])
        theta = 1.0
        blocker = None
        if len(ratios):
            smallest = float(np.min(ratios))
            if smallest < 1.0:
                # Exact ratio ties use the original pair index. No top-k filtering.
                blocker = int(np.min(candidates[ratios == smallest]))
                theta = max(0.0, smallest)
                event["blocking_raw_ratio"] = smallest
        event.update(action="MOVE", step=theta, blocker=blocker,
                     numerically_zero_negative_slope_count=int(np.sum(inactive & (slope < 0) & (slope >= -slope_tol))),
                     predicted_objective_change=-theta * (1 - .5 * theta) * energy)
        if blocker is not None:
            # Check the next row against the CURRENT independent set, without
            # forming or factorizing a dependent augmented working matrix.
            diagonal = 2.0 * P_old[rows[blocker], rows[blocker]]
            projection = 0.0
            if a:
                w = np.asarray(working, dtype=np.int64)
                inner = ((truths[w] == truths[blocker]).astype(float)
                         - (truths[w] == others[blocker])
                         - (others[w] == truths[blocker])
                         + (others[w] == others[blocker]))
                cross = P_old[rows[w], rows[blocker]] * inner
                response = ledger.solve(chol_W, cross, "working_independence")
                projection = float(cross @ response)
            schur = diagonal - projection
            rank_tol = (arithmetic + base / rcond_W) * max(1.0, abs(diagonal), abs(projection))
            ledger.data["independence_checks"] += 1
            event.update(blocker_schur=schur, blocker_rank_tolerance=rank_tol)
            if not np.isfinite(schur) or schur <= rank_tol:
                ledger.fail("WORKING_RANK_UNRESOLVED", "Blocking row independence cannot be certified",
                            blocker=blocker, schur=schur, rank_tolerance=rank_tol)
        rho += theta * drho
        V = V + theta * dV
        if blocker is not None:
            working.append(blocker)
            working.sort()
    if accepted_mu is None:
        snapshot_current_primal()
        ledger.fail("TRANSITION_LIMIT", "Caller transition budget exhausted; final optimality not certified")
    # Independent reconstruction from original K, R and all inequalities.
    mu = accepted_mu
    V_old = _scatter(mu, rows, truths, others, m, classes)
    V_train = np.zeros_like(R)
    V_train[old] = V_old
    alpha, b = _affine(ledger, chol_A, z, s, R + V_train, "affine_final_readback")
    residual = K @ alpha + b
    scores = M + residual
    slack = _difference(scores[old], rows, truths, others) - delta
    ledger.data["full_constraint_scans"] += 1
    stationarity = _maxabs(A @ alpha + b - R - V_train)
    intercept = _maxabs(np.sum(alpha, axis=0))
    compact_difference = _maxabs(residual[old] - (rho * base_old + P_old @ V))
    regularizer = float(np.sum(alpha * (K @ alpha)))
    objective = float(.5 * np.sum((residual - R) ** 2) + .5 * regularizer)
    Qmu = _difference(P_old @ V_old, rows, truths, others)
    dual_value = float(baseline_objective - s0 @ mu - .5 * mu @ Qmu)
    gap = objective - dual_value
    factor_error = base / rcond_W
    final_scale = max(1.0, _maxabs(A) * n * _maxabs(alpha), _maxabs(R + V_train), _maxabs(scores), _maxabs(delta))
    final_tol = (arithmetic + factor_error) * final_scale
    dual_tol = (arithmetic + factor_error) * max(1.0, _maxabs(mu))
    gap_tol = (arithmetic + factor_error) * max(1.0, abs(objective), abs(dual_value),
                                               float(np.sum(np.abs(mu * slack))), float(np.sum(R * R)))
    final = dict(stationarity=stationarity, intercept=intercept, compact_readback_difference=compact_difference,
                 minimum_slack=float(np.min(slack)), minimum_multiplier=float(np.min(mu)),
                 complementarity=_maxabs(mu * slack), primal_objective=objective, dual_objective=dual_value,
                 primal_dual_gap=gap, gap_identity_error=abs(gap - float(mu @ slack)), regularizer=regularizer,
                 residual_tolerance=final_tol, dual_tolerance=dual_tol, gap_tolerance=gap_tol)
    ledger.data["final_residuals"] = final
    ledger.snapshot.update(alpha=alpha, b=b, train_scores=scores, slack=slack, multipliers=mu)
    if (not np.isfinite(list(final.values())).all() or stationarity > final_tol or intercept > final_tol
            or compact_difference > np.sqrt(max(energy_tol, 0.0)) + final_tol
            or np.min(slack) < -final_tol or np.min(mu) < -dual_tol
            or _maxabs(mu * slack) > gap_tol or abs(gap) > gap_tol
            or final["gap_identity_error"] > gap_tol or regularizer < -gap_tol):
        ledger.fail("FINAL_KKT_READBACK_FAILED", "Original equations/all constraints/gap did not certify the head")
    ledger.data.update(status="NUMERIC_KKT_CERTIFIED", working_rcond=rcond_W,
                       active_count=len(working), condition_scope="LAPACK 1-norm estimates, not spectral condition numbers")
    arrays = dict(K=K, M=M, labels=labels, old_indices=old, pair_rows=rows, pair_truth=truths, pair_other=others,
                  margins=margins, delta=delta, R=R, z=z, s=np.asarray(s), P_old=P_old, s0=s0,
                  alpha=alpha, b=b, train_scores=scores, slack=slack, multipliers=mu,
                  working_set=np.asarray(working, dtype=np.int64), chol_A=chol_A,
                  chol_working=np.empty((0, 0)) if chol_W is None else chol_W)
    # Release factor INPUTS before immutable factor copies, so the two-copy
    # reservation also covers return-state isolation. No factor input is archived.
    del A, QW
    frozen = _frozen(arrays)
    ledger.data.update(returned_numeric_state_bytes=sum(value.nbytes for value in frozen.values()),
                       head_expansion_numeric_bytes=alpha.nbytes + b.nbytes,
                       head_expansion_scope="alpha+b only; excludes frozen actual B, kernel training coordinates and adapter",
                       returned_audit_policy="Independent mutable deep copy on each access",
                       returned_array_policy="Independent immutable byte-backed arrays")
    return MarginQPHead(frozen, deepcopy(ledger.data))


def predict_margin_qp_head(state, *, L, M):
    """Fixed all-column inference. No labels, role, quota or fit/update arguments."""
    L = _array(L, "L", 2)
    M = _array(M, "M", 2)
    if L.shape[1] != len(state.alpha) or M.shape != (len(L), state.alpha.shape[1]):
        raise ValueError("Cross kernel/frozen prior prediction shape mismatch")
    result = M + L @ state.alpha + state.b
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite fixed prediction")
    return _frozen(dict(scores=result))["scores"]


def margin_qp_head_vjp(state, *, L, G):
    """Full K/L VJP only for an independently certified regular active region.

    Actual B, labels, thresholds, old indices and class mapping are frozen.
    Unsupported regions raise rather than supplying an arbitrary generalized
    derivative or an unconstrained/zero fallback.
    """
    values = state.arrays
    n, classes = values["alpha"].shape
    L = _array(L, "L", 2)
    G = _array(G, "G", 2)
    if L.shape[1] != n or G.shape != (len(L), classes):
        raise ValueError("Held cross kernel/upstream shape mismatch")
    forward = state.audit
    residuals = forward["final_residuals"]
    working = values["working_set"]
    tight = np.flatnonzero(np.abs(values["slack"]) <= residuals["residual_tolerance"])
    regular = (forward["status"] == "NUMERIC_KKT_CERTIFIED"
               and np.array_equal(tight, working)
               and np.all(values["multipliers"][working] > residuals["dual_tolerance"]))
    if not regular:
        raise UnsupportedJacobian("UNSUPPORTED_ACTIVE_JACOBIAN",
                                  "All tight constraints must equal an independent strictly complementary working set",
                                  dict(status="UNSUPPORTED_JACOBIAN", forward=forward, tight=tight.tolist()), values)
    ledger = _Ledger(forward["max_transitions"], forward["max_factor_buffer_bytes"])
    ledger.data.update(operation="REGULAR_VJP", forward_factorizations_reused=True,
                       forward_numeric_state_bytes=forward["returned_numeric_state_bytes"])
    ledger.snapshot = dict(working_set=working)
    z, s = values["z"], float(values["s"])
    gb = np.sum(G, axis=0)
    BG = L.T @ G
    solved = ledger.solve(values["chol_A"], BG, "adjoint_affine")
    tg = (z @ BG - gb) / s
    TG = solved - z[:, None] * tg
    B = np.zeros((n, classes))
    eta = np.empty(0)
    if len(working):
        rhs = _difference(TG[values["old_indices"]], values["pair_rows"], values["pair_truth"], values["pair_other"])[working]
        eta = ledger.solve(values["chol_working"], rhs, "adjoint_working")
        all_eta = np.zeros(len(values["slack"]))
        all_eta[working] = eta
        B[values["old_indices"]] = _scatter(all_eta, values["pair_rows"], values["pair_truth"],
                                             values["pair_other"], len(values["old_indices"]), classes)
        WB, wb_b = _affine(ledger, values["chol_A"], z, s, B, "adjoint_multiplier_response")
    else:
        WB = np.zeros_like(B)
        wb_b = np.zeros(classes)
    # Certify the two affine adjoints and the active response using the
    # original matrices. No extra factorization or full dual Hessian is made.
    adjoint_error = _maxabs(values["K"] @ TG + TG + tg - BG)
    constant_error = _maxabs(np.sum(TG, axis=0) - gb)
    multiplier_error = _maxabs(values["K"] @ WB + WB + wb_b - B)
    multiplier_constant_error = _maxabs(np.sum(WB, axis=0))
    response_error = 0.0
    if len(working):
        old_response = B[values["old_indices"]] - WB[values["old_indices"]]
        response = _difference(old_response, values["pair_rows"], values["pair_truth"], values["pair_other"])[working]
        response_error = _maxabs(response - rhs)
    scale = max(1.0, n * (_maxabs(values["K"]) + 1.0) * max(_maxabs(TG), _maxabs(WB)),
                _maxabs(BG), _maxabs(B), _maxabs(gb), _maxabs(tg), _maxabs(wb_b))
    tolerance = (forward["arithmetic_bound"] + forward["arithmetic_gamma"] / forward["working_rcond"]) * scale
    actual = dict(affine=adjoint_error, constant=constant_error,
                  multiplier_affine=multiplier_error, multiplier_constant=multiplier_constant_error,
                  active_response=response_error, tolerance=tolerance)
    ledger.data["adjoint_residuals"] = actual
    ledger.snapshot.update(T_G=TG, t_G=tg, g_b=gb, eta=eta, W_eta=WB)
    if not np.isfinite(list(actual.values())).all() or max(value for key, value in actual.items() if key != "tolerance") > tolerance:
        ledger.fail("ADJOINT_READBACK_FAILED", "Original adjoint equations failed their residual checks")
    Kbar = -(TG + WB) @ values["alpha"].T
    Kbar = .5 * (Kbar + Kbar.T)
    Lbar = G @ values["alpha"].T
    if not np.isfinite(Kbar).all() or not np.isfinite(Lbar).all():
        ledger.fail("NONFINITE_ADJOINT", "Regular VJP produced nonfinite values")
    ledger.data.update(status="REGULAR_JACOBIAN", active_count=len(working),
                       factor_buffer_scope="No new factorization; immutable forward factors reused",
                       reused_factor_bytes=values["chol_A"].nbytes + values["chol_working"].nbytes)
    arrays = _frozen(dict(K=Kbar, L=Lbar, T_G=TG, t_G=tg, g_b=gb, eta=eta, W_eta=WB))
    ledger.data["returned_numeric_state_bytes"] = sum(value.nbytes for value in arrays.values())
    return MarginQPVJP(arrays, deepcopy(ledger.data))
