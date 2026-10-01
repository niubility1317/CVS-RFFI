"""Independent r <= 5 support prediction-Fisher and metric-ball diagnostics.

This module performs no I/O and owns no data-role decision.  The caller supplies
legal support labels and complete physical score JVPs, including all heads.
It changes no actual loss, core, state, or experiment.  No jitter, pseudoinverse,
rank truncation, negative-curvature clipping, or complete-head certificate.
"""
from copy import deepcopy
from collections.abc import Mapping
from time import perf_counter
from types import MappingProxyType

import numpy as np

SCHEMA = 'd92_support_metric_step_v1'
EVIDENCE_LEVEL = 'FLOAT64_SUBPROBLEM_DIAGNOSTIC_NOT_COMPLETE_HEAD_CERTIFICATE'
RADIUS = 0.5
MAX_DIMENSION = 5
MAX_SECULAR_ITERATIONS = 128
_EPS = np.finfo(np.float64).eps
_COUNT_KEYS = (
    'fisher_construction_attempts', 'fisher_constructions_completed',
    'physical_support_count', 'score_jvp_scalar_count',
    'symmetry_check_count', 'spectral_check_attempts', 'spectral_checks_completed',
    'factorization_attempts', 'factorizations_completed',
    'triangular_calls', 'triangular_calls_completed', 'triangular_rhs_columns',
    'triangular_rhs_elements', 'triangular_dense_work_units',
    'secular_evaluation_count', 'secular_iteration_count', 'inward_rescale_count',
    'diagnostic_readback_count',
)
AUDIT_SUM_KEYS = _COUNT_KEYS + (
    'fisher_seconds', 'spectral_seconds', 'factor_seconds',
    'triangular_seconds', 'diagnostic_seconds', 'seconds',
)
AUDIT_MAX_KEYS = ('peak_single_factor_input_output_bytes',)
AUDIT_COUNTERS = _COUNT_KEYS


def _array(value, name, *, shape=None, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu':
        raise ValueError(name + ' must be real numeric')
    out = np.asarray(raw, dtype=np.float64)
    if (shape is not None and out.shape != shape) or (ndim is not None and out.ndim != ndim):
        raise ValueError(name + ' has an invalid shape')
    if not np.isfinite(out).all():
        raise ValueError(name + ' must be finite')
    return out


def _readonly(value):
    a = np.asarray(value)
    return np.frombuffer(np.ascontiguousarray(a).tobytes(), dtype=a.dtype).reshape(a.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return deepcopy(value)


def _norm(value):
    a = np.asarray(value).ravel()
    if a.size == 0:
        return 0.0
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.sum((a / scale) ** 2)))


def _finite(*values):
    if any(not np.isfinite(v).all() for v in values):
        raise FloatingPointError('Nonfinite float64 intermediate')


class SupportMetricFailure(FloatingPointError):
    """Actual attempted work and immutable partial arrays, including nonfinite values."""
    def __init__(self, code, message, audit, arrays):
        super().__init__(message)
        self.code = code
        self._audit = deepcopy(audit)
        self.arrays = MappingProxyType({k: _readonly(v) for k, v in arrays.items()})
        self._audit['partial_numeric_state_bytes'] = sum(a.nbytes for a in self.arrays.values())

    @property
    def audit(self):
        return deepcopy(self._audit)

    def audit_dict(self):
        return self.audit


class _Result:
    def __init__(self, arrays, audit):
        _finite(*arrays.values())
        frozen = MappingProxyType({k: _readonly(v) for k, v in arrays.items()})
        metadata = deepcopy(audit)
        metadata['returned_numeric_state_bytes'] = sum(v.nbytes for v in frozen.values())
        object.__setattr__(self, 'arrays', frozen)
        object.__setattr__(self, '_audit', _freeze(metadata))

    def __setattr__(self, name, value):
        raise AttributeError('Support metric results are immutable')

    @property
    def audit(self):
        return _plain(self._audit)

    def audit_dict(self):
        return self.audit


class FisherResult(_Result):
    @property
    def fisher(self): return self.arrays['fisher']
    @property
    def centered_jvp(self): return self.arrays['centered_jvp']
    @property
    def factors(self): return self.arrays['fisher_factors']
    @property
    def sample_weights(self): return self.arrays['sample_weights']


class MetricStep(_Result):
    @property
    def direction(self): return self.arrays['direction']
    @property
    def fisher(self): return self.arrays['fisher']
    @property
    def M(self): return self.arrays['metric']
    @property
    def H(self): return self.arrays['hessian']
    @property
    def multiplier(self): return float(self.arrays['multiplier'])


class _Ledger:
    def __init__(self, operation):
        self.start = perf_counter()
        self.data = {k: 0 for k in _COUNT_KEYS}
        self.data.update({k: 0.0 for k in AUDIT_SUM_KEYS if k.endswith('seconds')})
        self.data.update({k: 0 for k in AUDIT_MAX_KEYS})
        self.data.update(schema=SCHEMA, operation=operation, evidence_level=EVIDENCE_LEVEL,
            input_permissions='CALLER_OWNS_LEGAL_SUPPORT_AND_COMPLETE_PHYSICAL_SCORE_JVP',
            complete_head_error_bound_available=False, rigorous_interval_certificate=False,
            physical_rank_and_theta_lift_processed=False, actual_proximal_loss_added=False,
            jitter_added=False, pseudoinverse_used=False, curvature_clipped=False,
            factor_memory_scope='MAX_SINGLE_EXPLICIT_FACTOR_INPUT_AND_RETURNED_CHOL_ONLY',
            process_peak_memory_bytes=None, lapack_workspace_bytes=None,
            dense_work_units_scope='N_SQUARED_TIMES_ACTUAL_RHS_COLUMNS_NOT_MEASURED_FLOP')
        self.snapshot = {}

    def __enter__(self): return self

    def __exit__(self, kind, error, traceback):
        if isinstance(error, SupportMetricFailure):
            # A timed helper's finally block runs after fail() constructed the
            # exception.  Refresh its actual ledger after all helper timers.
            self.data['seconds'] = perf_counter() - self.start
            error._audit = deepcopy(self.data)
            error._audit['partial_numeric_state_bytes'] = sum(
                a.nbytes for a in error.arrays.values())
        elif error is not None:
            if isinstance(error, (FloatingPointError, np.linalg.LinAlgError)):
                self.fail('NUMERICAL_FAILURE', str(error))
        return False

    def fail(self, code, message):
        self.data.update(status='TECHNICAL_FAILURE', code=code,
            seconds=perf_counter() - self.start)
        raise SupportMetricFailure(code, message, self.data, self.snapshot)

    def result(self, cls, arrays, **metadata):
        _finite(*arrays.values())
        self.data.update(metadata, status='COMPLETED', seconds=perf_counter() - self.start)
        return cls(arrays, self.data)


def _validate_support(score_jvp, probabilities, labels):
    J = _array(score_jvp, 'score_jvp', ndim=3)
    n, c, r = J.shape
    if n < 1 or c < 1 or r > MAX_DIMENSION:
        raise ValueError('Nonempty support/classes and 0 <= r <= 5 required')
    p = _array(probabilities, 'probabilities', shape=(n, c))
    y = np.asarray(labels)
    if y.shape != (n,) or y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y >= c):
        raise ValueError('labels must contain legal integer support class indices')
    y = y.astype(np.int64)
    counts = np.bincount(y, minlength=c)
    if np.any(counts == 0):
        raise ValueError('Every registered support class must be represented')
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError('probabilities must lie in [0,1]')
    totals = np.sum(p, axis=1)
    p_tol = 128 * _EPS * max(1, c)
    if np.any(np.abs(totals - 1) > p_tol):
        raise ValueError('probability rows must sum to one at binary64 arithmetic scale')
    return J, p, y, counts, float(np.max(np.abs(totals - 1))), p_tol


def _fisher(J, p, y, counts, probability_error, probability_tolerance, ledger):
    start = perf_counter()
    ledger.data['fisher_construction_attempts'] += 1
    n, c, r = J.shape
    ledger.data.update(physical_support_count=n, score_jvp_scalar_count=J.size,
        dimension=r, class_count=c, physical_count=n,
        probability_row_sum_max_abs_error=probability_error,
        probability_diagnostic_tolerance=probability_tolerance)
    ledger.snapshot.update(score_jvp=J, probabilities=p, labels=y, class_counts=counts)
    try:
        weights = 1.0 / (c * counts[y].astype(np.float64))
        means = np.einsum('nc,ncr->nr', p, J)
        centered = J - means[:, None, :]
        weighted_probabilities = weights[:, None] * p
        if r > 0 and np.any((p > 0) & (weighted_probabilities == 0)):
            ledger.snapshot['weighted_probabilities'] = weighted_probabilities
            ledger.fail('FISHER_WEIGHT_UNDERFLOW', 'A positive probability weight underflowed')
        factors = np.sqrt(weighted_probabilities)[:, :, None] * centered
        ledger.snapshot.update(sample_weights=weights, centered_jvp=centered,
            fisher_factors=factors)
        # n*c by r, never c by c.  Explicit dimensions also support r=0.
        flat = factors.reshape(n * c, r)
        F = flat.T @ flat
        ledger.snapshot['fisher'] = F
        _finite(weights, centered, factors, F)
        ledger.data['fisher_constructions_completed'] += 1
        return dict(fisher=F, centered_jvp=centered, fisher_factors=factors,
            sample_weights=weights, class_counts=counts), dict(
            class_count=c, physical_count=n, dimension=r,
            fisher_scope='CLASS_BALANCED_EXPECTED_PREDICTION_FISHER_NOT_OBSERVED_LABEL_OUTER_PRODUCT',
            probability_row_sum_max_abs_error=probability_error,
            probability_diagnostic_tolerance=probability_tolerance,
            probabilities_renormalized=False, dense_class_fisher_materialized=False,
            probability_zero_count=int(np.count_nonzero(p == 0)),
            upstream_probability_underflow_excluded=False)
    finally:
        ledger.data['fisher_seconds'] += perf_counter() - start


def build_class_balanced_prediction_fisher(*, score_jvp, probabilities, labels):
    """Labels choose input-class balance; Fisher expectation uses all output probabilities.

    Stored zero probabilities are retained, not given a floor.  This module
    cannot determine whether a caller's zeros came from upstream underflow.
    """
    inputs = _validate_support(score_jvp, probabilities, labels)
    with _Ledger('build_class_balanced_prediction_fisher') as ledger, np.errstate(
            over='raise', invalid='raise', divide='raise', under='ignore'):
        arrays, metadata = _fisher(*inputs, ledger)
        return ledger.result(FisherResult, arrays, **metadata)


def _symmetric(value, name, ledger):
    ledger.data['symmetry_check_count'] += 1
    if not np.array_equal(value, value.T):
        ledger.fail('NONSYMMETRIC_MATRIX', name + ' must be exactly symmetric')


def _psd(value, name, ledger, tolerance):
    start = perf_counter()
    ledger.data['spectral_check_attempts'] += 1
    try:
        values = np.linalg.eigvalsh(value)
        ledger.snapshot[name + '_eigenvalues'] = values
        _finite(values)
        ledger.data['spectral_checks_completed'] += 1
        if values[0] < -tolerance:
            ledger.fail('NON_PSD_INPUT', name + ' failed the binary64 PSD diagnostic')
        # No eigenvalue is clipped, lifted, or used to truncate a direction.
        return float(values[0])
    finally:
        ledger.data['spectral_seconds'] += perf_counter() - start


def _chol(matrix, name, ledger):
    ledger.data['factorization_attempts'] += 1
    ledger.snapshot.pop(name + '_chol', None)
    ledger.snapshot[name + '_factor_input'] = matrix
    ledger.data['peak_single_factor_input_output_bytes'] = max(
        ledger.data['peak_single_factor_input_output_bytes'], matrix.nbytes)
    start = perf_counter()
    try:
        factor = np.linalg.cholesky(matrix)
        ledger.snapshot[name + '_chol'] = factor
        _finite(factor)
        ledger.data['factorizations_completed'] += 1
        ledger.data['peak_single_factor_input_output_bytes'] = max(
            ledger.data['peak_single_factor_input_output_bytes'], matrix.nbytes + factor.nbytes)
        return factor
    except np.linalg.LinAlgError as error:
        ledger.fail('NON_SPD_FACTOR', name + ': ' + str(error))
    finally:
        ledger.data['factor_seconds'] += perf_counter() - start


def _triangular(chol, rhs, ledger, *, transpose=False):
    """True triangular substitutions; no implicit refactorization by NumPy solve."""
    n, columns = rhs.shape
    ledger.data['triangular_calls'] += 1
    ledger.data['triangular_rhs_columns'] += columns
    ledger.data['triangular_rhs_elements'] += n * columns
    ledger.data['triangular_dense_work_units'] += n * n * columns
    start = perf_counter()
    value = np.empty_like(rhs)
    try:
        if transpose:
            for i in range(n - 1, -1, -1):
                value[i] = (rhs[i] - chol[i + 1:, i] @ value[i + 1:]) / chol[i, i]
        else:
            for i in range(n):
                value[i] = (rhs[i] - chol[i, :i] @ value[:i]) / chol[i, i]
        _finite(value)
        ledger.data['triangular_calls_completed'] += 1
        return value
    finally:
        ledger.data['triangular_seconds'] += perf_counter() - start


def _evaluate_secular(A, c, multiplier, ledger):
    # Failure archives describe only this attempt.  A new input/multiplier
    # must never be paired with a previous attempt's factor or direction.
    for key in tuple(ledger.snapshot):
        if key.startswith('secular_'):
            ledger.snapshot.pop(key)
    ledger.data['secular_evaluation_count'] += 1
    ledger.snapshot['secular_multiplier'] = np.asarray(multiplier)
    rhs = -c[:, None]
    ledger.snapshot['secular_rhs'] = rhs
    ledger.data['secular_attempt_phase'] = 'FACTOR_ATTEMPT'
    matrix = A + multiplier * np.eye(len(c))
    factor = _chol(matrix, 'secular', ledger)
    ledger.data['secular_attempt_phase'] = 'FORWARD_TRIANGULAR_ATTEMPT'
    forward = _triangular(factor, rhs, ledger)
    ledger.snapshot['secular_forward'] = forward
    ledger.data['secular_attempt_phase'] = 'BACKWARD_TRIANGULAR_ATTEMPT'
    solution = _triangular(factor, forward, ledger, transpose=True)[:, 0]
    ledger.snapshot['secular_direction'] = solution
    length = _norm(solution)
    _finite(np.asarray(length))
    ledger.data['secular_attempt_phase'] = 'COMPLETED'
    return solution, length


def _feasible_direction(M, direction, tolerance, ledger):
    """Original-coordinate readback and at most two arithmetic inward changes."""
    Md = M @ direction
    metric_value = float(direction @ Md)
    _finite(np.asarray(metric_value))
    for _ in range(2):
        if metric_value <= RADIUS ** 2:
            break
        if metric_value - RADIUS ** 2 > tolerance * max(RADIUS ** 2, abs(metric_value)):
            ledger.fail('METRIC_BALL_FEASIBILITY', 'Original-coordinate ball readback failed')
        ratio = np.nextafter(RADIUS / np.sqrt(metric_value), 0.0)
        direction = direction * ratio
        ledger.data['inward_rescale_count'] += 1
        Md = M @ direction
        metric_value = float(direction @ Md)
        _finite(np.asarray(metric_value))
    if metric_value > RADIUS ** 2 or metric_value < 0:
        ledger.fail('METRIC_BALL_FEASIBILITY', 'Feasible-side reconstruction unresolved')
    return direction, Md, metric_value


def _metric_secular_ready(M, L_inverse, z, multiplier, tolerance, ledger):
    """Use the final diagnostic's physical complementarity scale to stop.

    A norm gap <= tau*rho only bounds the squared ball slack by roughly
    2*tau*rho**2.  Instead, read the actual reconstructed direction and its
    lambda*(rho**2-d.T@M@d), including the same inward arithmetic adjustment
    used by the final diagnostic.  The final tau is unchanged.
    """
    start = perf_counter()
    ledger.data['diagnostic_readback_count'] += 1
    try:
        direction, _, value = _feasible_direction(M, L_inverse.T @ z, tolerance, ledger)
        complementarity = abs(multiplier * (RADIUS ** 2 - value))
        scale = max(1.0, abs(multiplier) * RADIUS ** 2)
        _finite(np.asarray([complementarity, scale]))
        return direction, bool(complementarity / scale <= tolerance)
    finally:
        ledger.data['diagnostic_seconds'] += perf_counter() - start


def _diagnostics(g, G, B, F, M, H, direction, multiplier, tolerance, ledger):
    start = perf_counter()
    ledger.data['diagnostic_readback_count'] += 1
    try:
        direction, Md, metric_value = _feasible_direction(M, direction, tolerance, ledger)
        ledger.snapshot.update(direction=direction, multiplier=np.asarray(multiplier))
        residual = H @ direction + g + multiplier * Md
        residual_norm = _norm(residual)
        scale = max(1.0, _norm(g), _norm(H) * _norm(direction),
            abs(multiplier) * _norm(Md))
        slack = RADIUS ** 2 - metric_value
        complementarity = abs(multiplier * slack)
        comp_scale = max(1.0, abs(multiplier) * RADIUS ** 2)
        derivative = float(g @ direction)
        objective = float(derivative + 0.5 * direction @ H @ direction)
        _finite(np.asarray([metric_value, residual_norm, scale, slack,
            complementarity, comp_scale, derivative, objective]))
        ledger.snapshot['kkt_residual'] = residual
        relative = residual_norm / scale
        comp_relative = complementarity / comp_scale
        if relative > tolerance:
            ledger.fail('KKT_RESIDUAL_EXCEEDED', 'Original-coordinate KKT diagnostic failed')
        if comp_relative > tolerance:
            ledger.fail('COMPLEMENTARITY_RESIDUAL_EXCEEDED', 'Metric ball complementarity diagnostic failed')
        return direction, dict(
            metric_ball_value=metric_value, metric_ball_slack=slack,
            metric_ball_violation=max(0.0, metric_value - RADIUS ** 2),
            metric_direction_norm=float(np.sqrt(metric_value)),
            stationarity_residual_norm=residual_norm, stationarity_residual_scale=scale,
            stationarity_relative_residual=relative,
            complementarity_residual=complementarity,
            complementarity_relative_residual=comp_relative,
            directional_derivative=derivative, quadratic_value=objective,
            floating_gradient_is_zero=bool(not np.any(g)), gradient_zero_certified=False,
            direction_error_bound=None, precise_original_problem_certificate=False,
            original_coordinate_readback=True)
    finally:
        ledger.data['diagnostic_seconds'] += perf_counter() - start


def solve_support_metric_step(*, gradient, ggn, physical_gram, score_jvp,
                              probabilities, labels, max_secular_iterations=128):
    """Solve the independent diagnostic subproblem, fixed radius .5 and M=B+F.

    Symmetric inputs are required.  PSD eigenspectrum diagnostics allow only
    binary64 uncertainty, never clipping the actual matrix.  Cholesky of M and
    every whitened quadratic system must succeed with no added diagonal term.
    Returned residuals are float64 readbacks, not directed-rounding certificates.
    """
    inputs = _validate_support(score_jvp, probabilities, labels)
    J, p, y, counts, p_error, p_tol = inputs
    r = J.shape[2]
    g = _array(gradient, 'gradient', shape=(r,))
    G = _array(ggn, 'ggn', shape=(r, r))
    B = _array(physical_gram, 'physical_gram', shape=(r, r))
    if isinstance(max_secular_iterations, (bool, np.bool_)) or not isinstance(
            max_secular_iterations, (int, np.integer)) or not 1 <= max_secular_iterations <= 128:
        raise ValueError('max_secular_iterations must be an integer in [1,128]')
    with _Ledger('solve_support_metric_step') as ledger, np.errstate(
            over='raise', invalid='raise', divide='raise', under='ignore'):
        ledger.data.update(dimension=r, radius=RADIUS,
            max_secular_iterations=int(max_secular_iterations),
            numerical_diagnostic_relative_tolerance=128 * _EPS * max(1, r))
        ledger.snapshot.update(gradient=g, ggn=G, physical_gram=B)
        _symmetric(G, 'ggn', ledger)
        _symmetric(B, 'physical_gram', ledger)
        arrays, metadata = _fisher(*inputs, ledger)
        F = arrays['fisher']
        M = B + F
        H = G + M
        ledger.snapshot.update(metric=M, hessian=H)
        tolerance = 128 * _EPS * max(1, r)
        metadata.update(radius=RADIUS, max_secular_iterations=int(max_secular_iterations),
            numerical_diagnostic_relative_tolerance=tolerance,
            diagnostic_tolerance_scope='BINARY64_ARITHMETIC_ONLY_NOT_SCIENTIFIC_OBJECTIVE_OR_RANK',
            metric_definition='CALLER_PHYSICAL_GRAM_PLUS_CLASS_BALANCED_PREDICTION_FISHER',
            hessian_definition='CALLER_RMSCE_GGN_PLUS_METRIC',
            physical_gram_replaced_by_identity=False)
        arrays.update(gradient=g, ggn=G, physical_gram=B, metric=M, hessian=H)
        if r == 0:
            arrays.update(direction=np.zeros(0), multiplier=np.asarray(0.0),
                kkt_residual=np.zeros(0))
            return ledger.result(MetricStep, arrays, **metadata,
                active=False, zero_dimensional_physical_space=True,
                floating_gradient_is_zero=True, gradient_zero_certified=False,
                metric_ball_value=0.0, metric_ball_slack=RADIUS ** 2,
                metric_ball_violation=0.0, metric_direction_norm=0.0,
                stationarity_residual_norm=0.0, stationarity_residual_scale=1.0,
                stationarity_relative_residual=0.0, complementarity_residual=0.0,
                complementarity_relative_residual=0.0, directional_derivative=0.0,
                quadratic_value=0.0, direction_error_bound=None,
                precise_original_problem_certificate=False, original_coordinate_readback=True)
        B_tol = tolerance * max(1.0, _norm(B))
        G_tol = tolerance * max(1.0, _norm(G))
        metadata.update(physical_gram_min_eigenvalue=_psd(B, 'physical_gram', ledger, B_tol),
            ggn_min_eigenvalue=_psd(G, 'ggn', ledger, G_tol),
            physical_gram_psd_diagnostic_tolerance=B_tol, ggn_psd_diagnostic_tolerance=G_tol,
            psd_diagnostic_not_exact_rank_certificate=True)
        metric_chol = _chol(M, 'metric', ledger)
        # One metric factor, one stacked triangular call for gradient plus r
        # basis RHS.  No dense inverse API and no factor per coordinate.
        whiten_rhs = np.column_stack((g, np.eye(r)))
        ledger.snapshot['whitening_rhs'] = whiten_rhs
        solved = _triangular(metric_chol, whiten_rhs, ledger)
        c, L_inverse = solved[:, 0], solved[:, 1:]
        white_raw = L_inverse @ H @ L_inverse.T
        white_asymmetry = _norm(white_raw - white_raw.T)
        white_scale = max(1.0, _norm(white_raw))
        _finite(np.asarray(white_asymmetry), np.asarray(white_scale))
        ledger.data['symmetry_check_count'] += 1
        if white_asymmetry > tolerance * white_scale:
            ledger.fail('WHITENING_ASYMMETRY', 'Whitened quadratic exceeded arithmetic symmetry scale')
        # This is the symmetric part of the same quadratic, no spectrum edit.
        A = 0.5 * white_raw + 0.5 * white_raw.T
        ledger.snapshot.update(metric_chol=metric_chol, whitening_inverse=L_inverse,
            whitened_gradient=c, whitened_hessian=A)
        arrays.update(metric_chol=metric_chol, whitening_inverse=L_inverse,
            whitened_gradient=c, whitened_hessian=A)
        identity_residual = _norm(L_inverse @ M @ L_inverse.T - np.eye(r))
        _finite(np.asarray(identity_residual))
        metadata.update(whitening_asymmetry_norm=white_asymmetry,
            whitening_symmetrization_norm=_norm(A - white_raw),
            whitening_identity_residual=identity_residual,
            metric_factor_reused_for_stacked_rhs=True, whitening_rhs_columns=r + 1)
        z, length = _evaluate_secular(A, c, 0.0, ledger)
        active, multiplier = bool(length > RADIUS), 0.0
        direction = L_inverse.T @ z
        if active:
            lower = 0.0
            upper = _norm(c) / RADIUS
            _finite(np.asarray(upper))
            if upper <= 0:
                ledger.fail('SECULAR_BRACKET_FAILURE', 'Positive finite bracket cannot be represented')
            high_z, high_length = _evaluate_secular(A, c, upper, ledger)
            if high_length > RADIUS:
                ledger.fail('SECULAR_BRACKET_FAILURE', 'SPD upper bracket was not numerically feasible')
            high_direction, ready = _metric_secular_ready(
                M, L_inverse, high_z, upper, tolerance, ledger)
            for _ in range(int(max_secular_iterations)):
                if ready:
                    break
                middle = 0.5 * lower + 0.5 * upper
                ledger.data['secular_iteration_count'] += 1
                trial_z, trial_length = _evaluate_secular(A, c, middle, ledger)
                if trial_length > RADIUS:
                    lower = middle
                else:
                    upper, high_z, high_length = middle, trial_z, trial_length
                    high_direction, ready = _metric_secular_ready(
                        M, L_inverse, high_z, upper, tolerance, ledger)
            if not ready:
                ledger.fail('SECULAR_BUDGET_EXHAUSTED', 'Feasible-side secular root did not converge')
            z, multiplier = high_z, upper
            direction = high_direction
        direction, diagnostics = _diagnostics(g, G, B, F, M, H,
            direction, multiplier, tolerance, ledger)
        arrays.update(direction=direction, multiplier=np.asarray(multiplier),
            whitened_direction=metric_chol.T @ direction, secular_direction=z,
            kkt_residual=ledger.snapshot['kkt_residual'])
        return ledger.result(MetricStep, arrays, **metadata, **diagnostics,
            active=active, zero_dimensional_physical_space=False,
            unique_numerically_spd_subproblem=True)
