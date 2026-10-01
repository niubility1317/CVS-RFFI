"""Positive-kernel conditional affine ridge and its complete low-rank adjoint.

Only m,p>0 with numerically resolvable SPD raw old Gram A is supported. This
module does not build kernels, fit adapters, establish B lineage or handle
zero/tau0/repeated-old/new0 branches. M and R_N are frozen upstream values.
"""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from scipy.linalg import solve_triangular


EPS = float(np.finfo(np.float64).eps)
ROUNDING_MULTIPLIER = 64.


class NumericalFailure(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = dict(audit)


def _eta(n):
    return ROUNDING_MULTIPLIER*EPS*max(1, n)


def _array(value, name, ndim=2):
    if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.ndim != ndim:
        raise ValueError(name+' must be a NumPy float64 array with ndim='+str(ndim))
    if not np.isfinite(value).all():
        raise ValueError(name+' must be finite')
    return np.array(value, copy=True)


def _readonly(value):
    x = np.asarray(value, dtype=np.float64)
    return np.frombuffer(x.tobytes(), dtype=np.float64).reshape(x.shape)


def _norm(x):
    return float(np.linalg.norm(x))


def _sym(x):
    return .5*(x+x.T)


def _check(condition, message, audit):
    if not condition:
        raise NumericalFailure(message, audit)


def _finite(value, name, audit):
    _check(np.isfinite(value).all(), 'Nonfinite '+name, audit)


def _residual(left, right):
    return _norm(left-right)/max(1., _norm(left)+_norm(right))


def _zero_sum(x):
    return float(np.max(np.abs(x.sum(axis=-1)))) if x.size else 0.


def _counters():
    result = dict(factorization_count=0, completed_factorization_count=0,
        spectral_diagnostic_count=0, optimizer_steps=0, jitter=0.)
    for prefix in ('projection', 'residual', 'residual_adjoint', 'projection_adjoint'):
        for suffix in ('factorization_count', 'triangular_solve_count', 'triangular_rhs_count',
                       'triangular_rhs_element_count', 'triangular_dense_work_unit_count'):
            result[prefix+'_'+suffix] = 0
    return result


def _factor(matrix, prefix, audit):
    audit['phase'] = prefix+'_factorization'
    audit['factorization_count'] += 1
    audit[prefix+'_factorization_count'] += 1
    factor = np.linalg.cholesky(matrix)
    audit['completed_factorization_count'] += 1
    return factor


def _solve(factor, rhs, prefix, audit):
    n, width = rhs.shape
    result = rhs
    for lower in (True, False):
        audit['phase'] = prefix+'_triangular_solve'
        audit[prefix+'_triangular_solve_count'] += 1
        audit[prefix+'_triangular_rhs_count'] += width
        audit[prefix+'_triangular_rhs_element_count'] += n*width
        audit[prefix+'_triangular_dense_work_unit_count'] += n*n*width
        result = solve_triangular(factor if lower else factor.T, result, lower=lower, check_finite=False)
        _finite(result, prefix+' solution', audit)
    return result


def _psd(matrix, name, scale, audit):
    audit['phase'] = name+'_spectral_diagnostic'
    audit['spectral_diagnostic_count'] += 1
    values = np.linalg.eigvalsh(matrix)
    _finite(values, name+' eigenvalues', audit)
    tolerance = _eta(len(matrix))*max(1., scale)
    audit[name+'_minimum_eigenvalue'] = float(values[0])
    audit[name+'_psd_absolute_tolerance'] = tolerance
    _check(values[0] >= -tolerance, name+' is not PSD at the declared rounding scale', audit)
    return values


@dataclass(frozen=True)
class ConditionalAffineExpansion:
    alpha: np.ndarray
    beta: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        a, b, v = (_array(self.alpha, 'alpha'), _array(self.beta, 'beta'), _array(self.v, 'v', 1))
        if not len(a) or not len(b) or a.shape[1] != b.shape[1] or v.shape != (a.shape[1],):
            raise ValueError('Invalid conditional affine expansion shape')
        for name, value in (('alpha', a), ('beta', b), ('v', v)):
            object.__setattr__(self, name, _readonly(value))

    def residual(self, *, k_old, k_new):
        old, new = _array(k_old, 'k_old'), _array(k_new, 'k_new')
        if old.shape != (len(new), len(self.beta)) or new.shape[1] != len(self.alpha):
            raise ValueError('Cross-kernel shape differs from frozen O/N expansion')
        result = new@self.alpha-old@self.beta+self.v
        if not np.isfinite(result).all():
            raise NumericalFailure('Nonfinite deployed residual', dict(phase='deployment', optimizer_steps=0))
        return result

    def score(self, *, k_old, k_new, M):
        prior = _array(M, 'M')
        result = self.residual(k_old=k_old, k_new=k_new)
        if prior.shape != result.shape:
            raise ValueError('Frozen prior must have every declared score column')
        result = prior+result
        if not np.isfinite(result).all():
            raise NumericalFailure('Nonfinite deployed scores', dict(phase='deployment', optimizer_steps=0))
        return result


@dataclass(frozen=True)
class ConditionalAffineState:
    arrays: object
    audit: object
    expansion: ConditionalAffineExpansion

    def __post_init__(self):
        object.__setattr__(self, 'arrays', MappingProxyType({k: _readonly(v) for k, v in self.arrays.items()}))
        object.__setattr__(self, 'audit', MappingProxyType(dict(self.audit)))


@dataclass(frozen=True)
class ConditionalAffineAdjoint:
    arrays: object
    audit: object

    def __post_init__(self):
        object.__setattr__(self, 'arrays', MappingProxyType({k: _readonly(v) for k, v in self.arrays.items()}))
        object.__setattr__(self, 'audit', MappingProxyType(dict(self.audit)))


def fit_conditional_affine(*, A, B, D, F, E, M_O, M_N, M_H, R_N):
    """Equations (6)-(9), with all priors and R_N explicitly frozen.

    A[m,m], B[m,p], D[p,p], F[h,m], E[h,p], M_O[m,C],
    M_N/R_N[p,C], M_H[h,C]. A is the true raw Gram, never expm1.
    """
    data = {name: _array(value, name) for name, value in
            dict(A=A, B=B, D=D, F=F, E=E, M_O=M_O, M_N=M_N, M_H=M_H, R_N=R_N).items()}
    A, B, D, F, E, M_O, M_N, M_H, R_N = (data[k] for k in ('A','B','D','F','E','M_O','M_N','M_H','R_N'))
    m, p, h, c = len(A), len(D), len(F), R_N.shape[1]
    expected = dict(A=(m,m), B=(m,p), D=(p,p), F=(h,m), E=(h,p), M_O=(m,c), M_N=(p,c), M_H=(h,c), R_N=(p,c))
    if min(m,p,c) <= 0 or any(data[k].shape != shape for k, shape in expected.items()):
        raise ValueError('Conditional affine requires m,p,C>0 and consistent block shapes')
    audit = _counters()
    audit.update(m=m, p=p, h=h, class_count=c, phase='input', rounding_multiplier=ROUNDING_MULTIPLIER,
        backward_relative_tolerance=_eta(max(m,p,h,c)), protection_relative_tolerance=np.sqrt(_eta(max(m,p,h,c))),
        frozen_prior=True, frozen_R_N=True, dense_work_scope='n_squared_times_rhs_proxy_not_measured_flops')
    eta = audit['backward_relative_tolerance']; protection = audit['protection_relative_tolerance']
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for name, matrix in (('A', A), ('D', D)):
                audit[name+'_symmetry_residual'] = _residual(matrix, matrix.T)
                _check(audit[name+'_symmetry_residual'] <= eta, name+' is not symmetric', audit)
            A, D = _sym(A), _sym(D)
            audit['phase'] = 'projection_condition'; audit['spectral_diagnostic_count'] += 1
            spectrum = np.linalg.eigvalsh(A); _finite(spectrum, 'old Gram spectrum', audit)
            audit['projection_minimum_eigenvalue'] = float(spectrum[0])
            audit['projection_maximum_eigenvalue'] = float(spectrum[-1])
            rcond = float(spectrum[0]/spectrum[-1]) if spectrum[-1] > 0 else 0.
            audit['projection_reciprocal_condition'] = rcond
            audit['projection_minimum_reciprocal_condition'] = np.sqrt(_eta(m))
            _check(spectrum[0] > 0 and rcond > audit['projection_minimum_reciprocal_condition'],
                'Raw old Gram is non-SPD or numerically ill-conditioned; no jitter allowed', audit)
            chol_A = _factor(A, 'projection', audit)
            combined_rhs = np.column_stack((B, np.ones(m)))
            solution = _solve(chol_A, combined_rhs, 'projection', audit)
            J, z = solution[:, :p], solution[:, p]; s = float(z.sum())
            audit['projection_solve_residual'] = _norm(A@solution-combined_rhs)/max(1., _norm(A)*_norm(solution)+_norm(combined_rhs))
            _check(s > 0 and np.isfinite(s) and audit['projection_solve_residual'] <= eta, 'Projection solve or s failed', audit)
            c_N, c_H = np.ones(p)-B.T@z, np.ones(h)-F@z
            raw_schur = D-B.T@J
            audit['raw_schur_symmetry_residual'] = _residual(raw_schur,raw_schur.T)
            _check(audit['raw_schur_symmetry_residual'] <= eta,'Raw Schur symmetry failed',audit)
            schur = _sym(raw_schur)
            _psd(schur, 'raw_schur', _norm(D)+_norm(B)*_norm(J), audit)
            K = _sym(schur+np.outer(c_N,c_N)/s)
            L = E-F@J+np.outer(c_H,c_N)/s
            _finite(K, 'conditional Gram', audit); _finite(L, 'conditional cross Gram', audit)
            _psd(K, 'conditional_kernel', _norm(D)+_norm(B)*_norm(J)+_norm(c_N)**2/s, audit)
            chol_ridge = _factor(K+np.eye(p), 'residual', audit)
            alpha = _solve(chol_ridge, R_N, 'residual', audit)
            v = c_N@alpha/s; beta = J@alpha+z[:,None]*v
            old_residual = B@alpha-A@beta+v
            new_residual = K@alpha; held_residual = L@alpha
            norm_squared = float(np.sum(alpha*(K@alpha)))
            audit.update(ridge_solve_residual=_norm((K+np.eye(p))@alpha-R_N)/max(1., (_norm(K)+np.sqrt(p))*_norm(alpha)+_norm(R_N)),
                old_residual_max_abs=float(np.max(np.abs(old_residual))), old_residual_norm=_norm(old_residual),
                old_constraint_backward_residual=_norm(old_residual)/max(1., _norm(B)*_norm(alpha)+_norm(A)*_norm(beta)+np.sqrt(m)*_norm(v)),
                coefficient_sum_residual=_residual(beta.sum(0),alpha.sum(0)), residual_norm_squared=norm_squared,
                old_protection_absolute_tolerance=protection*max(1.,_norm(M_O),_norm(R_N)),
                conditional_symmetry_residual=_residual(K,K.T), schur_s=s)
            _check(audit['ridge_solve_residual'] <= eta and audit['old_constraint_backward_residual'] <= eta,
                'Ridge or old-constraint backward residual failed', audit)
            _check(audit['old_residual_norm'] <= audit['old_protection_absolute_tolerance'], 'Old residual protection failed', audit)
            _check(audit['coefficient_sum_residual'] <= protection, 'Raw expansion coefficient sum failed', audit)
            _check(norm_squared >= -eta*max(1.,_norm(K)*_norm(alpha)**2), 'Negative conditional residual norm', audit)
            arrays = dict(A=A,B=B,D=D,F=F,E=E,raw_A=data['A'],raw_D=data['D'],M_O=M_O,M_N=M_N,M_H=M_H,R_N=R_N,
                J=J,z=z,s=np.asarray(s),c_N=c_N,c_H=c_H,K_perp=K,L_perp=L,alpha=alpha,beta=beta,v=v,
                chol_A=chol_A,chol_ridge=chol_ridge,projection_rhs=combined_rhs,
                old_residual=old_residual,old_scores=M_O+old_residual,train_new_scores=M_N+new_residual,
                held_scores=M_H+held_residual)
            for name, value in arrays.items(): _finite(value,name,audit)
            centered = all(_zero_sum(data[name]) <= eta*max(1.,_norm(data[name])) for name in ('M_O','M_N','M_H','R_N'))
            audit['class_centered_inputs'] = centered
            for name in ('alpha','beta','v','old_scores','train_new_scores','held_scores'):
                error = _zero_sum(arrays[name]); audit[name+'_class_sum_max_abs'] = error
                if centered: _check(error <= protection*max(1.,_norm(arrays[name])), 'Class sum failed: '+name,audit)
            expansion = ConditionalAffineExpansion(alpha,beta,v)
            audit['held_expansion_residual'] = _residual(expansion.residual(k_old=F,k_new=E),held_residual)
            _check(audit['held_expansion_residual'] <= protection,'Deployed expansion differs from conditional head',audit)
            audit.update(status='CONDITIONAL_AFFINE_SOLVED',phase='complete',
                deployment_coefficient_bytes=int(alpha.nbytes+beta.nbytes+v.nbytes))
            return ConditionalAffineState(arrays,audit,expansion)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        raise NumericalFailure(str(exc), audit) from exc


def conditional_affine_adjoint(state, G):
    """Equations (14)-(16); gradients for symmetric A/D and full B/F/E.

    M and R_N stay fixed. Frobenius contractions use full symmetric A/D arrays;
    callers must not multiply their off-diagonal upstream values by two again.
    """
    if not isinstance(state, ConditionalAffineState):
        raise TypeError('A solved ConditionalAffineState is required')
    d = state.arrays; G = _array(G, 'G')
    if G.shape != d['M_H'].shape: raise ValueError('G must match held scores [h,C]')
    audit = _counters(); audit.update(phase='adjoint', frozen_prior=True, frozen_R_N=True)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            A,B,F,alpha,beta,z = (d[k] for k in ('A','B','F','alpha','beta','z'))
            s = float(d['s']); p,c = alpha.shape
            T = _solve(d['chol_ridge'],d['L_perp'].T@G,'residual_adjoint',audit)
            rhs = F.T@G; g_b = G.sum(axis=0)
            V = _solve(d['chol_A'],rhs,'projection_adjoint',audit)
            nu = (z@rhs-g_b)/s
            lam_top = V-z[:,None]*nu
            lambda_full = np.vstack((lam_top,nu))
            x_alpha = np.vstack((beta,-d['v']))
            xt_bottom = -(d['c_N']@T)/s
            xt_top = d['J']@T-z[:,None]*xt_bottom
            x_T = np.vstack((xt_top,xt_bottom))
            arrays = dict(A=_sym((lam_top-xt_top)@beta.T),
                B=(xt_top-lam_top)@alpha.T+beta@T.T, D=-_sym(T@alpha.T),
                F=-G@beta.T, E=G@alpha.T, G=G,T=T,Lambda=lambda_full,X_alpha=x_alpha,X_T=x_T,
                g_b=g_b,projection_adjoint_rhs=rhs)
            for name,value in arrays.items(): _finite(value,name,audit)
            eta = state.audit['backward_relative_tolerance']
            audit.update(residual_adjoint_residual=_norm((d['K_perp']+np.eye(p))@T-d['L_perp'].T@G)/
                max(1.,(_norm(d['K_perp'])+np.sqrt(p))*_norm(T)+_norm(d['L_perp'].T@G)),
                projection_adjoint_residual=_norm(A@lam_top+nu-rhs)/max(1.,_norm(A)*_norm(lam_top)+np.sqrt(len(A))*_norm(nu)+_norm(rhs)),
                projection_adjoint_constant_residual=_residual(lam_top.sum(0),g_b),g_b_norm=_norm(g_b))
            _check(audit['residual_adjoint_residual'] <= eta and audit['projection_adjoint_residual'] <= eta
                and audit['projection_adjoint_constant_residual'] <= state.audit['protection_relative_tolerance'],
                'Complete conditional affine adjoint residual failed',audit)
            audit.update(status='CONDITIONAL_AFFINE_ADJOINT_COMPLETE',phase='complete')
            return ConditionalAffineAdjoint(arrays,audit)
    except (np.linalg.LinAlgError,FloatingPointError,OverflowError) as exc:
        raise NumericalFailure(str(exc),audit) from exc
