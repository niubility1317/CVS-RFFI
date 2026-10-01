"""Independent positive-kernel head certificates using the full symmetric KKT.

This analysis oracle never fits an adapter, reads labels, or calls a candidate
head. Its dense indefinite solve is deliberately independent of the training
implementation's Schur reduction. Priors and R_N are frozen inputs.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


EPS = float(np.finfo(np.float64).eps)
ROUNDING_MULTIPLIER = 64.0


class MathCertificateFailure(ValueError):
    """An algebraic or numerical certificate failed; audit preserves evidence."""

    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = dict(audit)


def _eta(n):
    return ROUNDING_MULTIPLIER * EPS * max(1, int(n))


def _array(value, name, ndim=2):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float64
            or value.ndim != ndim):
        raise ValueError(name + ' must be a NumPy float64 array with ndim=' + str(ndim))
    if not np.isfinite(value).all():
        raise ValueError(name + ' must be finite')
    return np.array(value, copy=True)


def _readonly(value):
    value = np.asarray(value, dtype=np.float64)
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


def _norm(value):
    return float(np.linalg.norm(value))


def _require(condition, message, audit):
    if not condition:
        raise MathCertificateFailure(message, audit)


def _finite(value, name, audit):
    _require(np.isfinite(value).all(), 'Nonfinite ' + name, audit)


def _sym(value):
    return 0.5 * (value + value.T)


def _relative(error, *scale_terms):
    return _norm(error) / max(1.0, sum(float(term) for term in scale_terms))


def _base_audit(m, p, h, c):
    q = m + p + 1
    return dict(m=m, p=p, h=h, class_count=c, kkt_order=q, phase='input',
                rounding_multiplier=ROUNDING_MULTIPLIER,
                relative_tolerance=_eta(max(q, h, c)),
                general_solve_count=0, completed_general_solve_count=0,
                general_rhs_column_count=0, general_rhs_element_count=0,
                general_dense_cubic_work_unit_count=0,
                general_dense_rhs_work_unit_count=0,
                spectral_diagnostic_count=0, positive_kernel_checked=False,
                optimizer_steps=0, jitter=0.0, frozen_prior=True, frozen_R_N=True,
                cost_scope='dense_work_proxies_not_measured_flops_or_time')


def _inputs(*, A, B, D, F, E, M_O, M_N, M_H, R_N):
    data = {name: _array(value, name) for name, value in
            dict(A=A, B=B, D=D, F=F, E=E, M_O=M_O, M_N=M_N,
                 M_H=M_H, R_N=R_N).items()}
    m, p, h, c = len(data['A']), len(data['D']), len(data['F']), data['R_N'].shape[1]
    expected = dict(A=(m, m), B=(m, p), D=(p, p), F=(h, m), E=(h, p),
                    M_O=(m, c), M_N=(p, c), M_H=(h, c), R_N=(p, c))
    if min(m, p, c) <= 0 or any(data[name].shape != shape for name, shape in expected.items()):
        raise ValueError('Positive-kernel certificate requires m,p,C>0 and consistent block shapes')
    audit = _base_audit(m, p, h, c)
    for name in ('A', 'D'):
        matrix = data[name]
        residual = _relative(matrix - matrix.T, 2.0 * _norm(matrix))
        audit[name + '_symmetry_residual'] = residual
        _require(np.isfinite(residual) and residual <= audit['relative_tolerance'],
                 name + ' is not symmetric at the fixed rounding scale', audit)
        data['raw_' + name] = matrix.copy()
        data[name] = _sym(matrix)
    return data, audit


def _system(data):
    A, B, D = (data[name] for name in ('A', 'B', 'D'))
    m, p = len(A), len(D)
    # All signs are from the original first-order conditions, not a Schur head.
    Q = np.block([[A, -B, -np.ones((m, 1))],
                  [-B.T, D + np.eye(p), np.ones((p, 1))],
                  [-np.ones((1, m)), np.ones((1, p)), np.zeros((1, 1))]])
    rhs = np.vstack((np.zeros((m, data['R_N'].shape[1])), data['R_N'],
                     np.zeros((1, data['R_N'].shape[1]))))
    return Q, rhs


def _general_solve(Q, rhs, audit, phase):
    audit['phase'] = phase
    n, c = rhs.shape
    audit['general_solve_count'] += 1
    audit['general_rhs_column_count'] += c
    audit['general_rhs_element_count'] += n * c
    audit['general_dense_cubic_work_unit_count'] += n ** 3
    audit['general_dense_rhs_work_unit_count'] += n * n * c
    result = np.linalg.solve(Q, rhs)
    audit['completed_general_solve_count'] += 1
    _finite(result, phase + ' solution', audit)
    return result


def _check_positive_kernel(data, audit):
    """Independent spectral checks; their cubic cost is reported separately."""
    audit['phase'] = 'positive_kernel_spectrum'
    audit['spectral_diagnostic_count'] += 1
    old_spectrum = np.linalg.eigvalsh(data['A'])
    _finite(old_spectrum, 'old Gram spectrum', audit)
    audit['old_minimum_eigenvalue'] = float(old_spectrum[0])
    audit['old_maximum_eigenvalue'] = float(old_spectrum[-1])
    _require(old_spectrum[0] > 0.0, 'Raw old Gram must be strictly positive definite; no jitter', audit)
    raw = np.block([[data['A'], data['B']], [data['B'].T, data['D']]])
    audit['spectral_diagnostic_count'] += 1
    spectrum = np.linalg.eigvalsh(raw)
    _finite(spectrum, 'joint raw Gram spectrum', audit)
    tolerance = _eta(len(raw)) * max(1.0, _norm(raw))
    audit['raw_minimum_eigenvalue'] = float(spectrum[0])
    audit['raw_psd_absolute_tolerance'] = tolerance
    audit['spectral_dense_cubic_work_unit_count'] = len(data['A']) ** 3 + len(raw) ** 3
    _require(spectrum[0] >= -tolerance, 'Joint raw Gram is not PSD at the fixed rounding scale', audit)
    audit['positive_kernel_checked'] = True


@dataclass(frozen=True)
class PositiveKernelHeadCertificate:
    arrays: object
    audit: object

    def __post_init__(self):
        object.__setattr__(self, 'arrays', MappingProxyType(
            {name: _readonly(value) for name, value in self.arrays.items()}))
        object.__setattr__(self, 'audit', MappingProxyType(dict(self.audit)))


@dataclass(frozen=True)
class PositiveKernelHeadVJP:
    arrays: object
    audit: object

    def __post_init__(self):
        object.__setattr__(self, 'arrays', MappingProxyType(
            {name: _readonly(value) for name, value in self.arrays.items()}))
        object.__setattr__(self, 'audit', MappingProxyType(dict(self.audit)))


def _certificate(data, audit, *, alpha, beta, v, old_scores, train_new_scores, held_scores):
    m, p, h, c = (audit[name] for name in ('m', 'p', 'h', 'class_count'))
    claims = {name: _array(value, name, 1 if name == 'v' else 2) for name, value in
              dict(alpha=alpha, beta=beta, v=v, old_scores=old_scores,
                   train_new_scores=train_new_scores, held_scores=held_scores).items()}
    expected = dict(alpha=(p, c), beta=(m, c), v=(c,), old_scores=(m, c),
                    train_new_scores=(p, c), held_scores=(h, c))
    if any(claims[name].shape != shape for name, shape in expected.items()):
        raise ValueError('Claimed coefficients/scores do not match every declared row and class column')
    alpha, beta, v = (claims[name] for name in ('alpha', 'beta', 'v'))
    A, B, D, F, E = (data[name] for name in ('A', 'B', 'D', 'F', 'E'))
    Q, rhs = _system(data)
    theta = np.vstack((beta, alpha, v[None, :]))
    _finite(Q, 'KKT matrix', audit)
    equation_error = Q @ theta - rhs
    old_residual = B @ alpha - A @ beta + v
    new_residual = D @ alpha - B.T @ beta + v
    held_residual = E @ alpha - F @ beta + v
    computed = dict(old_scores=data['M_O'] + old_residual,
                    train_new_scores=data['M_N'] + new_residual,
                    held_scores=data['M_H'] + held_residual)
    audit['phase'] = 'coefficient_certificate'
    audit['kkt_backward_residual'] = _relative(equation_error, _norm(Q) * _norm(theta), _norm(rhs))
    audit['old_constraint_backward_residual'] = _relative(
        old_residual, _norm(B) * _norm(alpha), _norm(A) * _norm(beta), np.sqrt(m) * _norm(v))
    audit['new_stationarity_backward_residual'] = _relative(
        new_residual + alpha - data['R_N'], _norm(D) * _norm(alpha),
        _norm(B) * _norm(beta), np.sqrt(p) * _norm(v), _norm(alpha), _norm(data['R_N']))
    audit['coefficient_sum_backward_residual'] = _relative(
        beta.sum(0) - alpha.sum(0), np.sqrt(m) * _norm(beta), np.sqrt(p) * _norm(alpha))
    audit['old_residual_max_abs'] = float(np.max(np.abs(old_residual)))
    audit['old_residual_norm'] = _norm(old_residual)
    checks = ('kkt_backward_residual', 'old_constraint_backward_residual',
              'new_stationarity_backward_residual', 'coefficient_sum_backward_residual')
    for name in checks:
        _require(np.isfinite(audit[name]) and audit[name] <= audit['relative_tolerance'],
                 'Failed ' + name, audit)
    audit['phase'] = 'score_certificate'
    for name, prior, left, right, count in (
            ('old_scores', data['M_O'], B, A, m),
            ('train_new_scores', data['M_N'], D, B.T, p),
            ('held_scores', data['M_H'], E, F, h)):
        error = computed[name] - claims[name]
        residual = _relative(error, _norm(prior), _norm(left) * _norm(alpha),
                             _norm(right) * _norm(beta), np.sqrt(count) * _norm(v), _norm(claims[name]))
        audit[name + '_backward_residual'] = residual
        _require(np.isfinite(residual) and residual <= audit['relative_tolerance'],
                 'Failed ' + name + ' reconstruction certificate', audit)
    aa, bb, cross = float(np.sum(alpha * (D @ alpha))), float(np.sum(beta * (A @ beta))), float(np.sum(beta * (B @ alpha)))
    norm_squared = aa + bb - 2.0 * cross
    norm_tolerance = audit['relative_tolerance'] * max(1.0, abs(aa) + abs(bb) + 2.0 * abs(cross))
    _require(np.isfinite(norm_squared) and norm_squared >= -norm_tolerance,
             'Negative/nonfinite raw-expansion RKHS norm', audit)
    audit['residual_rkhs_norm_squared'] = norm_squared
    audit['residual_rkhs_norm_absolute_tolerance'] = norm_tolerance
    audit['new_fit_plus_regularizer_objective'] = 0.5 * (_norm(alpha) ** 2 + norm_squared)
    arrays = dict(data, Q=Q, rhs=rhs, theta=theta, alpha=alpha, beta=beta, v=v,
                  kkt_error=equation_error, old_residual=old_residual,
                  train_new_residual=new_residual, held_residual=held_residual,
                  claimed_old_scores=claims['old_scores'],
                  claimed_train_new_scores=claims['train_new_scores'],
                  claimed_held_scores=claims['held_scores'], **computed)
    for name, value in arrays.items():
        _finite(value, name, audit)
        if name in ('alpha', 'beta', 'v', 'old_scores', 'train_new_scores', 'held_scores'):
            audit[name + '_class_sum_max_abs'] = float(np.max(np.abs(value.sum(-1)))) if value.size else 0.0
    audit.update(status='POSITIVE_KERNEL_HEAD_CERTIFIED', phase='complete',
                 deployment_coefficient_bytes=int(alpha.nbytes + beta.nbytes + v.nbytes),
                 kkt_matrix_bytes=int(Q.nbytes), certificate_array_bytes=int(sum(value.nbytes for value in arrays.values())))
    return PositiveKernelHeadCertificate(arrays, audit)


def rebuild_positive_kernel_head(*, A, B, D, F, E, M_O, M_N, M_H, R_N):
    """Rebuild independently with a general solve of Q[beta;alpha;v]=[0;R_N;0].

    This is an analysis oracle, not a cheaper training implementation. It runs
    two spectral diagnostics and one dense indefinite solve of order m+p+1.
    No jitter, pseudoinverse, regularization change, or retry is permitted.
    """
    data, audit = _inputs(A=A, B=B, D=D, F=F, E=E, M_O=M_O, M_N=M_N, M_H=M_H, R_N=R_N)
    audit['certificate_mode'] = 'independent_rebuild'
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            _check_positive_kernel(data, audit)
            Q, rhs = _system(data)
            _finite(Q, 'KKT matrix', audit)
            theta = _general_solve(Q, rhs, audit, 'forward_general_kkt_solve')
            m, p = audit['m'], audit['p']
            beta, alpha, v = theta[:m], theta[m:m + p], theta[-1]
            return _certificate(data, audit, alpha=alpha, beta=beta, v=v,
                old_scores=data['M_O'] + data['B'] @ alpha - data['A'] @ beta + v,
                train_new_scores=data['M_N'] + data['D'] @ alpha - data['B'].T @ beta + v,
                held_scores=data['M_H'] + data['E'] @ alpha - data['F'] @ beta + v)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        raise MathCertificateFailure(str(exc), audit) from exc


def certify_positive_kernel_head(*, A, B, D, F, E, M_O, M_N, M_H, R_N,
                                alpha, beta, v, old_scores, train_new_scores, held_scores):
    """Check frozen archived coefficients and scores without any solve/spectrum.

    This verifies the stated KKT/score equations, not authenticity of frozen
    priors or positive definiteness of a caller's raw kernel. Input PSD/SPD and
    lineage must be established separately. Coherently replacing all input and
    output arrays cannot be detected without an independent trusted reference.
    """
    data, audit = _inputs(A=A, B=B, D=D, F=F, E=E, M_O=M_O, M_N=M_N, M_H=M_H, R_N=R_N)
    audit['certificate_mode'] = 'residual_only'
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return _certificate(data, audit, alpha=alpha, beta=beta, v=v,
                old_scores=old_scores, train_new_scores=train_new_scores, held_scores=held_scores)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        raise MathCertificateFailure(str(exc), audit) from exc


def positive_kernel_head_vjp(certificate, G):
    """Complete kernel VJP by independently solving the full KKT adjoint.

    Frobenius contractions use full symmetric A/D gradients; do not double
    their off-diagonal entries again. M_O/M_N/M_H and R_N remain frozen.
    """
    if not isinstance(certificate, PositiveKernelHeadCertificate):
        raise TypeError('A PositiveKernelHeadCertificate is required')
    data = certificate.arrays
    G = _array(G, 'G')
    if G.shape != data['M_H'].shape:
        raise ValueError('G must match all held rows and declared class columns')
    m, p, h, c = (certificate.audit[name] for name in ('m', 'p', 'h', 'class_count'))
    audit = _base_audit(m, p, h, c)
    audit['certificate_mode'] = 'independent_full_kkt_adjoint'
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            g_b = G.sum(axis=0)
            rhs = np.vstack((-data['F'].T @ G, data['E'].T @ G, g_b[None, :]))
            lam = _general_solve(data['Q'], rhs, audit, 'adjoint_general_kkt_solve')
            residual = _relative(data['Q'] @ lam - rhs,
                                 _norm(data['Q']) * _norm(lam), _norm(rhs))
            audit['adjoint_backward_residual'] = residual
            _require(np.isfinite(residual) and residual <= audit['relative_tolerance'],
                     'Full KKT adjoint residual failed', audit)
            Q_bar = -_sym(lam @ data['theta'].T)
            arrays = dict(A=Q_bar[:m, :m],
                          B=-(Q_bar[:m, m:m + p] + Q_bar[m:m + p, :m].T),
                          D=Q_bar[m:m + p, m:m + p],
                          F=-G @ data['beta'].T, E=G @ data['alpha'].T,
                          G=G, g_b=g_b, adjoint_rhs=rhs, lambda_full=lam,
                          Q_bar=Q_bar, theta=data['theta'], Q=data['Q'])
            for name, value in arrays.items():
                _finite(value, name, audit)
            audit.update(status='POSITIVE_KERNEL_VJP_CERTIFIED', phase='complete',
                         g_b_norm=_norm(g_b), vjp_array_bytes=int(sum(value.nbytes for value in arrays.values())))
            return PositiveKernelHeadVJP(arrays, audit)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        raise MathCertificateFailure(str(exc), audit) from exc
