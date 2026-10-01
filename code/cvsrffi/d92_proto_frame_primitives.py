"""Independent float64 mathematics for the fixed five-coordinate prototype frame.

No data/model I/O, fitting of prototype statistics, finite-difference gradients,
parameter selection, pseudoinverse or jitter.  Jacobians have a final axis of 5.
The caller owns split permissions, frozen old tau/gamma, gate and line search.
"""
from copy import deepcopy
from collections.abc import Mapping
from time import perf_counter
from types import MappingProxyType

import numpy as np

DIMENSION = 160
COORDINATES = 5
KAPPA = 0.25
NORM_FLOOR = 1e-12
_EPS = np.finfo(np.float64).eps
_COUNT_KEYS = (
    'frame_construction_count', 'physical_evaluation_count',
    'factorization_attempts', 'factorizations_completed', 'triangular_calls',
    'triangular_rhs_columns', 'triangular_rhs_elements', 'triangular_dense_work_units',
    'distance_evaluation_count', 'distance_pair_count', 'distance_qr_batch_count',
    'distance_qr_pair_count', 'kernel_evaluation_count', 'kernel_pair_count',
    'jacobian_evaluation_count', 'spectral_check_count', 'eigendecomposition_count',
    'secular_iteration_count', 'secular_evaluation_count',
)
# Counts and elapsed seconds are additive work; result bytes describe each
# returned immutable object, rather than a process high-water mark.
AUDIT_COUNTERS = _COUNT_KEYS
AUDIT_SUM_KEYS = _COUNT_KEYS + ('seconds',)
AUDIT_MAX_KEYS = ()


def _array(value, name, shape=None, ndim=None):
    original = np.asarray(value)
    if original.dtype.kind not in 'fiu':
        raise ValueError(name + ' must be a real numeric array')
    result = np.asarray(original, dtype=np.float64)
    if (ndim is not None and result.ndim != ndim) or (shape is not None and result.shape != shape):
        raise ValueError(name + ' has an invalid shape')
    if not np.isfinite(result).all():
        raise ValueError(name + ' must be finite')
    return result


def _readonly(value):
    array = np.asarray(value)
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze_audit(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_audit(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_audit(item) for item in value)
    return value


def _plain_audit(value):
    if isinstance(value, Mapping):
        return {key: _plain_audit(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_audit(item) for item in value]
    return deepcopy(value)


def _finite(*values):
    if any(not np.isfinite(value).all() for value in values):
        raise FloatingPointError('Nonfinite mathematical result')


def _row_norm(value):
    scale = np.max(np.abs(value), axis=1)
    norm = np.zeros(len(value), dtype=np.float64)
    valid = scale > 0
    reduced = value[valid] / scale[valid, None]
    norm[valid] = scale[valid] * np.sqrt(np.sum(reduced * reduced, axis=1))
    _finite(norm)
    return norm


def _norm(value):
    flat = np.asarray(value).ravel()
    if not len(flat):
        return 0.0
    scale = float(np.max(np.abs(flat)))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.sum((flat / scale) ** 2)))


class PrimitiveFailure(FloatingPointError):
    """Numerical failure with actual attempted work and finite partial arrays."""
    def __init__(self, code, message, audit, arrays):
        super().__init__(message)
        self.code = code
        self._audit = deepcopy(audit)
        self.arrays = MappingProxyType({key: _readonly(value) for key, value in arrays.items()
            if np.asarray(value).dtype.kind in 'fiu' and np.isfinite(value).all()})
        self._audit.update(partial_array_scope='FINITE_NUMERIC_INPUTS_AND_COMPLETED_INTERMEDIATES',
            omitted_nonfinite_array_keys=[key for key, value in arrays.items()
                if np.asarray(value).dtype.kind in 'fiu' and not np.isfinite(value).all()])

    @property
    def audit(self):
        return deepcopy(self._audit)

    def audit_dict(self):
        return self.audit


class _Result:
    def __init__(self, arrays, audit):
        frozen_arrays = MappingProxyType({key: _readonly(value) for key, value in arrays.items() if value is not None})
        metadata = deepcopy(audit)
        metadata['returned_numeric_state_bytes'] = sum(value.nbytes for value in frozen_arrays.values())
        object.__setattr__(self, 'arrays', frozen_arrays)
        object.__setattr__(self, '_audit', _freeze_audit(metadata))

    def __setattr__(self, name, value):
        raise AttributeError('Primitive result objects are immutable')

    @property
    def audit(self):
        return _plain_audit(self._audit)

    def audit_dict(self):
        return self.audit

    def _get(self, key):
        return self.arrays.get(key)


class ProtoFrame(_Result):
    @property
    def Q(self): return self._get('Q')
    @property
    def prototypes(self): return self._get('prototypes')
    @property
    def classes(self): return tuple(self._audit['class_order'])


class TangentResult(_Result):
    @property
    def values(self): return self._get('values')
    @property
    def jacobian(self): return self._get('jacobian')


class BranchGeometry(_Result):
    @property
    def b(self): return self._get('b')
    @property
    def a(self): return self._get('a')
    @property
    def b_jacobian(self): return self._get('b_jacobian')
    @property
    def a_jacobian(self): return self._get('a_jacobian')


class DistanceResult(_Result):
    @property
    def distance(self): return self._get('distance')
    @property
    def jacobian(self): return self._get('jacobian')


class KernelResult(_Result):
    @property
    def kernel(self): return self._get('kernel')
    @property
    def jacobian(self): return self._get('jacobian')
    @property
    def original_distance(self): return self._get('original_distance')
    @property
    def adapted_distance(self): return self._get('adapted_distance')


class RidgeResult(_Result):
    @property
    def alpha(self): return self._get('alpha')
    @property
    def intercept(self): return self._get('intercept')
    @property
    def scores(self): return self._get('scores')
    @property
    def score_jacobian(self): return self._get('score_jacobian')
    @property
    def chol(self): return self._get('chol')


class RMSResult(_Result):
    @property
    def loss(self): return float(self._get('loss'))
    @property
    def gradient(self): return self._get('gradient')
    @property
    def curvature(self): return self._get('curvature')
    @property
    def damped_hessian(self): return self._get('damped_hessian')


class QuadraticStep(_Result):
    @property
    def direction(self): return self._get('direction')
    @property
    def multiplier(self): return float(self._get('multiplier'))


class _Ledger:
    def __init__(self, operation):
        self.start = perf_counter()
        self.data = dict.fromkeys(_COUNT_KEYS, 0)
        self.data.update(operation=operation, process_peak_memory_bytes=None,
            memory_scope='RETURNED_NUMERIC_ARRAYS_ONLY_NOT_PROCESS_PEAK_OR_LAPACK_WORKSPACE',
            work_units_scope='n_squared_times_RHS_COLUMNS_PROXY_NOT_ACTUAL_FLOP')
        self.snapshot = {}

    def __enter__(self): return self

    def __exit__(self, kind, error, traceback):
        if error is not None and isinstance(error, (FloatingPointError, np.linalg.LinAlgError)):
            if isinstance(error, PrimitiveFailure):
                return False
            self.fail('NUMERICAL_FAILURE', str(error))
        return False

    def fail(self, code, message):
        self.data.update(status='TECHNICAL_FAILURE', code=code, seconds=perf_counter() - self.start)
        raise PrimitiveFailure(code, message, self.data, self.snapshot)

    def result(self, result_type, arrays, **metadata):
        _finite(*(value for value in arrays.values() if value is not None))
        self.data.update(metadata, status='COMPLETED', seconds=perf_counter() - self.start)
        return result_type(arrays, self.data)


def build_proto_frame_dictionary(*, prototypes, classes):
    """Six decoded prototype rows; deterministic canonical last-class difference frame."""
    p = _array(prototypes, 'prototypes', (6, DIMENSION))
    if isinstance(classes, str):
        raise ValueError('classes must be a sequence')
    names = tuple(classes)
    if len(names) != 6 or len(set(names)) != 6 or any(type(name) is not str or not name for name in names):
        raise ValueError('Exactly six unique nonempty physical class names required')
    canonical = tuple(sorted(names))
    order = [names.index(name) for name in canonical]
    with _Ledger('build_proto_frame_dictionary') as ledger, np.errstate(over='raise', invalid='raise', divide='raise'):
        ledger.data.update(frame_construction_count=1, physical_evaluation_count=6)
        ordered = p[order]
        norms = _row_norm(ordered)
        unit = np.zeros_like(ordered)
        valid = norms > 0
        unit[valid] = ordered[valid] / norms[valid, None]
        Q = (unit[:5] - unit[5]).T
        ledger.snapshot.update(prototypes=unit, Q=Q)
        return ledger.result(ProtoFrame, dict(prototypes=unit, Q=Q), prototype_class_count=6,
            prototype_dimension=DIMENSION, dictionary_coordinate_count=COORDINATES,
            reference_class=canonical[-1], class_order=list(canonical), zero_prototype_count=int(np.sum(~valid)),
            dictionary_all_zero=bool(not np.any(Q)), normalization='exact_unit_nonzero_zero_stays_zero')


def transform_z_id(*, z_id, Q, theta, jacobian=True):
    """Norm-preserving tangent map (7)/(20); exact identity forward, live zero-theta J."""
    z = _array(z_id, 'z_id', ndim=2)
    if z.shape[1] != DIMENSION:
        raise ValueError('z_id must have 160 columns')
    Q = _array(Q, 'Q', (DIMENSION, COORDINATES))
    theta = _array(theta, 'theta', (COORDINATES,))
    if type(jacobian) is not bool:
        raise ValueError('jacobian must be bool')
    with _Ledger('transform_z_id') as ledger, np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        n = len(z)
        ledger.data.update(physical_evaluation_count=n, jacobian_evaluation_count=int(jacobian))
        rho = _row_norm(z)
        valid = rho > 0
        u = np.zeros_like(z)
        u[valid] = z[valid] / rho[valid, None]
        w = Q @ theta
        t = w[None, :] - u * (u @ w)[:, None]
        t[~valid] = 0
        sigma = np.hypot(KAPPA, _row_norm(t))
        scaled_t = t / sigma[:, None]
        delta = KAPPA * scaled_t
        v = u + delta
        vn = _row_norm(v)
        normalized = np.zeros_like(v)
        normalized[valid] = v[valid] / vn[valid, None]
        output = rho[:, None] * normalized
        if not np.any(theta):
            output = z.copy()
        J = None
        if jacobian:
            projected = Q[None, :, :] - u[:, :, None] * (u @ Q)[:, None, :]
            ddelta = (KAPPA / sigma)[:, None, None] * (projected - scaled_t[:, :, None] *
                np.einsum('ni,nip->np', scaled_t, projected)[:, None, :])
            J = np.zeros((n, DIMENSION, COORDINATES))
            projected_out = ddelta - normalized[:, :, None] * np.einsum('ni,nip->np', normalized, ddelta)[:, None, :]
            J[valid] = (rho[valid] / vn[valid])[:, None, None] * projected_out[valid]
        ledger.snapshot.update(values=output, theta=theta, Q=Q)
        return ledger.result(TangentResult, dict(values=output, jacobian=J), zero_row_count=int(np.sum(~valid)),
            identity_forward=bool(not np.any(theta) or not np.any(Q)), kappa=KAPPA,
            zero_row_jacobian='EXACT_ZERO', nominal_coordinates=COORDINATES)


def _unit_floor(value, derivative=None):
    norms = _row_norm(value)
    denominator = np.maximum(norms, NORM_FLOOR)
    unit = value / denominator[:, None]
    if derivative is None:
        return unit, None
    J = derivative / denominator[:, None, None]
    live = norms > NORM_FLOOR
    J[live] -= unit[live, :, None] * np.einsum('ni,nip->np', unit[live], derivative[live])[:, None, :] / norms[live, None, None]
    boundary = norms == NORM_FLOOR
    if np.any(boundary):
        radial = np.einsum('ni,nip->np', unit[boundary], derivative[boundary])
        scale = np.max(np.abs(derivative[boundary]), axis=1)
        if np.any(np.abs(radial) > 128 * _EPS * DIMENSION * scale):
            raise FloatingPointError('NORMALIZATION_FLOOR_NONDIFFERENTIABLE_DIRECTION')
    return unit, J


def branch_geometry(*, z_id, fft, t_emb, f_emb, pa_local, z_id_jacobian=None):
    """Exactly the original floor-normalized five-branch geometry, without interaction expansion."""
    inputs = [_array(value, name, ndim=2) for name, value in
        zip(('z_id', 'fft', 't_emb', 'f_emb', 'pa_local'), (z_id, fft, t_emb, f_emb, pa_local))]
    n = len(inputs[0])
    if any(value.shape != (n, dim) for value, dim in zip(inputs, (160, 96, 160, 160, 160))):
        raise ValueError('Five physical feature blocks require equal counts and fixed dimensions')
    Jz = None if z_id_jacobian is None else _array(z_id_jacobian, 'z_id_jacobian', (n, 160, 5))
    with _Ledger('branch_geometry') as ledger, np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        ledger.data.update(physical_evaluation_count=n, jacobian_evaluation_count=int(Jz is not None))
        uz, Juz = _unit_floor(inputs[0], Jz)
        others = [_unit_floor(value)[0] for value in inputs[1:]]
        joined = np.concatenate((uz, 4 * others[0]), axis=1)
        Jjoined = None if Jz is None else np.concatenate((Juz, np.zeros((n, 96, 5))), axis=1)
        b, Jb = _unit_floor(joined, Jjoined)
        a = np.concatenate(others[1:], axis=1) / np.sqrt(3.)
        Ja = None if Jz is None else np.zeros((n, 480, 5))
        ledger.snapshot.update(b=b, a=a)
        return ledger.result(BranchGeometry, dict(b=b, a=a, b_jacobian=Jb, a_jacobian=Ja),
            input_dimension=736, implicit_dimension=123616, norm_floor=NORM_FLOOR,
            materialized_interaction_feature_count=0)


def _geometry(value, name):
    if not isinstance(value, BranchGeometry):
        raise ValueError(name + ' must be BranchGeometry')
    return value


def _pair_distance(bi, ai, bj, aj, Ji, Ai, Jj, Aj, ledger):
    db, da = bi - bj, ai - aj
    same = np.all(db == 0, axis=1) & np.all(da == 0, axis=1)
    ledger.data['distance_qr_batch_count'] += 1
    ledger.data['distance_qr_pair_count'] += len(db)
    _, R = np.linalg.qr(np.stack((db, bj), axis=2), mode='reduced')
    product = R @ np.stack((ai, da), axis=1)
    distance = np.sum(db * db, axis=1) + np.sum(da * da, axis=1) + np.sum(product * product, axis=(1, 2))
    distance[same] = 0
    if np.any((~same) & (distance == 0)):
        raise FloatingPointError('DISTANCE_UNDERFLOW_NONIDENTICAL_FEATURES')
    if Ji is None:
        return distance, None
    ddb, dda = Ji - Jj, Ai - Aj
    def dot(x, y): return np.einsum('ni,nip->np', x, y)
    def scalar(x, y): return np.sum(x * y, axis=1)[:, None]
    # Differentiate db outer ai + bj outer da by bilinear inner products.
    interaction = (dot(db, ddb) * scalar(ai, ai) + dot(bj, ddb) * scalar(da, ai)
        + scalar(db, db) * dot(ai, Ai) + scalar(bj, db) * dot(da, Ai)
        + dot(db, Jj) * scalar(ai, da) + dot(bj, Jj) * scalar(da, da)
        + scalar(db, bj) * dot(ai, dda) + scalar(bj, bj) * dot(da, dda))
    derivative = 2 * (dot(db, ddb) + dot(da, dda) + interaction)
    derivative[same] = 0
    return distance, derivative


def interaction_distances(left, right=None, *, jacobian=True):
    """Rank-two QR distances plus exact five-direction derivatives of both ends."""
    left = _geometry(left, 'left')
    symmetric = right is None
    right = left if symmetric else _geometry(right, 'right')
    if type(jacobian) is not bool:
        raise ValueError('jacobian must be bool')
    if jacobian and (left.b_jacobian is None or right.b_jacobian is None):
        raise ValueError('Jacobian requested without both geometry Jacobians')
    with _Ledger('interaction_distances') as ledger, np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        n, m = len(left.b), len(right.b)
        distance = np.zeros((n, m))
        J = np.zeros((n, m, 5)) if jacobian else None
        ledger.snapshot.update(distance=distance)
        ledger.data.update(distance_evaluation_count=1, jacobian_evaluation_count=int(jacobian))
        for i in range(n):
            for start in range(i + 1 if symmetric else 0, m, 256):
                stop = min(start + 256, m)
                count = stop - start
                bi = np.broadcast_to(left.b[i], (count, 256))
                ai = np.broadcast_to(left.a[i], (count, 480))
                Ji = np.broadcast_to(left.b_jacobian[i], (count, 256, 5)) if jacobian else None
                Ai = np.broadcast_to(left.a_jacobian[i], (count, 480, 5)) if jacobian else None
                ledger.data['distance_pair_count'] += count
                d, jd = _pair_distance(bi, ai, right.b[start:stop], right.a[start:stop], Ji, Ai,
                    right.b_jacobian[start:stop] if jacobian else None,
                    right.a_jacobian[start:stop] if jacobian else None, ledger)
                distance[i, start:stop] = d
                if jacobian: J[i, start:stop] = jd
                if symmetric:
                    distance[start:stop, i] = d
                    if jacobian: J[start:stop, i] = jd
        return ledger.result(DistanceResult, dict(distance=distance, jacobian=J), symmetric=symmetric,
            derivative_endpoints='BOTH', materialized_interaction_feature_count=0)


def raw_kernel(*, original_left, adapted_left, tau, gamma, original_right=None, adapted_right=None, jacobian=True):
    """Fixed original half-distance Gaussian; tau0 uses original exact equivalence."""
    original_left, adapted_left = _geometry(original_left, 'original_left'), _geometry(adapted_left, 'adapted_left')
    if (original_right is None) != (adapted_right is None):
        raise ValueError('Both right geometries must be present, or both omitted')
    if original_right is not None:
        original_right, adapted_right = _geometry(original_right, 'original_right'), _geometry(adapted_right, 'adapted_right')
    if len(original_left.b) != len(adapted_left.b) or (original_right is not None and len(original_right.b) != len(adapted_right.b)):
        raise ValueError('Original/adapted physical counts differ')
    if type(jacobian) is not bool:
        raise ValueError('jacobian must be bool')
    for name, value in (('tau', tau), ('gamma', gamma)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value < 0):
            raise ValueError(name + ' must be None or a finite nonnegative scalar')
    if gamma is not None and gamma > 0 and tau is None:
        raise ValueError('Positive gamma requires defined tau')
    if jacobian and gamma is not None and gamma > 0 and tau > 0 and (
            adapted_left.b_jacobian is None or (adapted_right is not None and adapted_right.b_jacobian is None)):
        raise ValueError('Continuous kernel Jacobian requires both adapted geometry Jacobians')
    with _Ledger('raw_kernel') as ledger, np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        def distances(left, right, need_jacobian, prefix):
            try:
                value = interaction_distances(left, right, jacobian=need_jacobian)
            except PrimitiveFailure as failure:
                for key in ('distance_evaluation_count', 'distance_pair_count', 'distance_qr_batch_count', 'distance_qr_pair_count', 'jacobian_evaluation_count'):
                    ledger.data[key] += failure.audit[key]
                ledger.snapshot.update({prefix + '_' + key: value for key, value in failure.arrays.items()})
                ledger.fail(failure.code, str(failure))
            for key in ('distance_evaluation_count', 'distance_pair_count', 'distance_qr_batch_count', 'distance_qr_pair_count'):
                ledger.data[key] += value.audit[key]
            return value
        original = distances(original_left, original_right, False, 'original')
        d0 = original.distance
        ledger.snapshot.update(original_distance=d0)
        adapted_distance = None
        K = np.zeros_like(d0)
        J = np.zeros(d0.shape + (5,)) if jacobian else None
        if gamma is None or gamma == 0:
            status = 'ZERO_KERNEL_NO_CONTINUOUS_INFORMATION'
        elif tau == 0:
            ledger.data.update(kernel_evaluation_count=1, kernel_pair_count=d0.size)
            K = float(gamma) * (d0 == 0)
            status = 'ORIGINAL_EXACT_EQUIVALENCE_PARAMETER_INDEPENDENT'
        else:
            current = distances(adapted_left, adapted_right, jacobian, 'adapted')
            adapted_distance = current.distance
            ledger.data.update(kernel_evaluation_count=1, kernel_pair_count=d0.size, jacobian_evaluation_count=int(jacobian))
            K = float(gamma) * np.exp(-(.5 * d0 + .5 * adapted_distance) / float(tau))
            if jacobian:
                J = -(K[:, :, None] * current.jacobian * .5) / float(tau)
            status = 'FIXED_SCALE_GAUSSIAN'
        ledger.snapshot.update(kernel=K)
        return ledger.result(KernelResult, dict(kernel=K, jacobian=J, original_distance=d0, adapted_distance=adapted_distance),
            tau=tau, gamma=gamma, kernel_gradient_status=status, derivative_endpoints='BOTH',
            adapted_distance_computed=adapted_distance is not None, materialized_interaction_feature_count=0)


def _chol_solve(chol, rhs, ledger):
    """Two true triangular substitutions; NumPy solve would refactor triangular inputs."""
    n, columns = rhs.shape
    value = np.empty_like(rhs)
    ledger.data['triangular_calls'] += 1
    ledger.data['triangular_rhs_columns'] += columns
    ledger.data['triangular_rhs_elements'] += n * columns
    ledger.data['triangular_dense_work_units'] += n * n * columns
    for i in range(n):
        value[i] = (rhs[i] - chol[i, :i] @ value[:i]) / chol[i, i]
    _finite(value)
    ledger.data['triangular_calls'] += 1
    ledger.data['triangular_rhs_columns'] += columns
    ledger.data['triangular_rhs_elements'] += n * columns
    ledger.data['triangular_dense_work_units'] += n * n * columns
    for i in range(n - 1, -1, -1):
        value[i] = (value[i] - chol[i + 1:, i] @ value[i + 1:]) / chol[i, i]
    _finite(value)
    return value


def fit_free_intercept_ridge(*, K, Y, L, K_jacobian=None, L_jacobian=None):
    """Equations (12)/(13), five directions in one RHS group using one factor."""
    K, Y, L = _array(K, 'K', ndim=2), _array(Y, 'Y', ndim=2), _array(L, 'L', ndim=2)
    n, c = Y.shape
    if n == 0 or c == 0 or K.shape != (n, n) or L.shape[1] != n:
        raise ValueError('Nonempty train/classes and compatible K/Y/L required')
    if not np.array_equal(K, K.T):
        raise ValueError('K must be exactly symmetric')
    if (K_jacobian is None) != (L_jacobian is None):
        raise ValueError('Both kernel Jacobians must be supplied or both omitted')
    dK = None if K_jacobian is None else _array(K_jacobian, 'K_jacobian', (n, n, 5))
    dL = None if L_jacobian is None else _array(L_jacobian, 'L_jacobian', L.shape + (5,))
    if dK is not None and not np.array_equal(dK, np.swapaxes(dK, 0, 1)):
        raise ValueError('Each train kernel direction must be exactly symmetric')
    with _Ledger('fit_free_intercept_ridge') as ledger, np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        ledger.snapshot.update(K=K, Y=Y, L=L)
        ledger.data['spectral_check_count'] = 1
        eigenvalues = np.linalg.eigvalsh(K)
        tolerance = 128 * _EPS * n * max(1., float(np.linalg.norm(K, 1)))
        if eigenvalues[0] < -tolerance:
            ledger.fail('NON_PSD_KERNEL', 'Kernel is not PSD within arithmetic tolerance')
        A = K + np.eye(n)
        ledger.data['factorization_attempts'] += 1
        chol = np.linalg.cholesky(A)
        ledger.data['factorizations_completed'] += 1
        ledger.snapshot.update(chol=chol)
        rhs = np.column_stack((Y, np.ones(n)))
        ledger.snapshot.update(combined_rhs=rhs)
        solved = _chol_solve(chol, rhs, ledger)
        F, z = solved[:, :c], solved[:, c]
        s = float(np.sum(z))
        ledger.snapshot.update(F=F, z=z, s=np.asarray(s))
        if not np.isfinite(s) or s <= 0:
            ledger.fail('NONPOSITIVE_INTERCEPT_SCHUR', 'Free-intercept Schur scalar must be positive')
        intercept = np.sum(F, axis=0) / s
        alpha = F - z[:, None] * intercept
        ledger.snapshot.update(alpha=alpha, intercept=intercept)
        scores = L @ alpha + intercept
        ledger.snapshot.update(scores=scores)
        train_scores = K @ alpha + intercept
        ledger.snapshot.update(train_scores=train_scores)
        dF = dz = db = dalpha = dh = jvp_rhs = None
        if dK is not None:
            rhsF = -np.einsum('ijp,jc->icp', dK, F)
            rhsz = -np.einsum('ijp,j->ip', dK, z)
            jvp_rhs = np.column_stack((rhsF.reshape(n, c * 5), rhsz))
            ledger.snapshot.update(jvp_rhs=jvp_rhs)
            derivatives = _chol_solve(chol, jvp_rhs, ledger)
            dF, dz = derivatives[:, :c * 5].reshape(n, c, 5), derivatives[:, c * 5:]
            db = (np.sum(dF, axis=0) - intercept[:, None] * np.sum(dz, axis=0)[None, :]) / s
            dalpha = dF - dz[:, None, :] * intercept[None, :, None] - z[:, None, None] * db[None, :, :]
            dh = np.einsum('hip,ic->hcp', dL, alpha) + np.einsum('hi,icp->hcp', L, dalpha) + db[None, :, :]
            ledger.data['jacobian_evaluation_count'] = 1
        stationarity = A @ alpha + intercept - Y
        scale = max(1., _norm(Y), _norm(A) * _norm(alpha), np.sqrt(n) * _norm(intercept))
        residual = _norm(stationarity) / scale
        mean_residual = _norm(np.sum(alpha, axis=0)) / max(1., _norm(alpha) * np.sqrt(n))
        _finite(np.asarray(scale), np.asarray(residual), np.asarray(mean_residual))
        if max(residual, mean_residual) > 128 * _EPS * max(n, c):
            ledger.fail('RIDGE_RESIDUAL_EXCEEDED', 'Original saddle equations failed arithmetic residual checks')
        arrays = dict(K=K, Y=Y, L=L, chol=chol, combined_rhs=rhs, F=F, z=z, s=np.asarray(s),
            alpha=alpha, intercept=intercept, scores=scores, train_scores=train_scores,
            jvp_rhs=jvp_rhs, F_jacobian=dF, z_jacobian=dz, intercept_jacobian=db,
            alpha_jacobian=dalpha, score_jacobian=dh)
        ledger.snapshot.update(alpha=alpha, intercept=intercept)
        return ledger.result(RidgeResult, arrays, kernel_min_eigenvalue=float(eigenvalues[0]),
            kernel_psd_tolerance=tolerance, normal_equation_residual=residual, intercept_constraint_residual=mean_residual,
            train_count=n, class_count=c, held_count=len(L), jvp_directions=0 if dK is None else 5,
            factor_reused_for_all_directions=dK is not None, intercept_regularized=False,
            unbalanced_target_means_preserved=True)


def class_balanced_rmsce(*, scores, labels, score_jacobian=None):
    """Exact class RMS CE gradient and score-GGN5 without a giant score Hessian."""
    scores = _array(scores, 'scores', ndim=2)
    n, c = scores.shape
    raw_labels = np.asarray(labels)
    if n == 0 or c == 0 or raw_labels.shape != (n,) or raw_labels.dtype.kind not in 'iu' or np.any(raw_labels < 0) or np.any(raw_labels >= c):
        raise ValueError('Nonempty finite scores and legal integer labels required')
    labels = raw_labels.astype(np.int64)
    counts = np.bincount(labels, minlength=c)
    if np.any(counts == 0):
        raise ValueError('Every registered class needs at least one held row for class RMS')
    J = None if score_jacobian is None else _array(score_jacobian, 'score_jacobian', (n, c, 5))
    with _Ledger('class_balanced_rmsce') as ledger, np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        ledger.data.update(physical_evaluation_count=n, jacobian_evaluation_count=int(J is not None))
        shifted = scores - np.max(scores, axis=1)[:, None]
        exponent = np.exp(shifted)
        p = exponent / np.sum(exponent, axis=1)[:, None]
        ce = np.zeros(n)
        if c > 1:
            others = (scores - scores[np.arange(n), labels, None])[np.arange(c)[None, :] != labels[:, None]].reshape(n, c - 1)
            largest = np.max(others, axis=1)
            other_lse = largest + np.log(np.sum(np.exp(others - largest[:, None]), axis=1))
            ce = np.logaddexp(0., other_lse)
            if np.any(ce == 0):
                ledger.snapshot.update(scores=scores, probabilities=p, ce=ce)
                ledger.fail('CE_UNDERFLOW_UNSUPPORTED', 'Finite multiclass CE underflowed to zero; no fabricated derivative')
        residual = p.copy()
        for i, label in enumerate(labels):
            residual[i, label] = -np.sum(p[i, np.arange(c) != label])
        sums = np.bincount(labels, weights=ce, minlength=c)
        means = sums / counts
        largest_mean = float(np.max(means))
        risk = 0. if largest_mean == 0 else largest_mean * float(np.sqrt(np.mean((means / largest_mean) ** 2)))
        if c > 1 and risk == 0:
            ledger.fail('RMSCE_ZERO_UNSUPPORTED', 'Multiclass RMS CE is numerically zero')
        gradient = curvature = hessian = class_J = weights = score_gradient = None
        if J is not None:
            class_J = np.zeros((c, 5))
            each_J = np.einsum('nc,ncp->np', residual, J)
            np.add.at(class_J, labels, each_J)
            class_J /= counts[:, None]
            if c == 1:
                weights = np.zeros(1)
                gradient, curvature = np.zeros(5), np.zeros((5, 5))
            else:
                weights = (means / risk) / c
                gradient = weights @ class_J
                centered_J = J - np.einsum('nc,ncp->np', p, J)[:, None, :]
                sample_weights = weights[labels] / counts[labels]
                first = np.einsum('n,nc,ncp,ncq->pq', sample_weights, p, centered_J, centered_J)
                unit_means = (means / largest_mean) / _norm(means / largest_mean)
                perpendicular = class_J - unit_means[:, None] * (unit_means @ class_J)[None, :]
                scaled = perpendicular / (np.sqrt(c) * np.sqrt(risk))
                curvature = first + scaled.T @ scaled
                curvature = .5 * (curvature + curvature.T)
            hessian = np.eye(5) + curvature
            score_gradient = residual * (weights[labels] / counts[labels])[:, None]
        ledger.snapshot.update(scores=scores, probabilities=p, ce=ce, class_ce_means=means)
        return ledger.result(RMSResult, dict(loss=np.asarray(risk), probabilities=p, ce=ce,
            class_ce_sums=sums, class_ce_counts=counts, class_ce_means=means, class_ce_jacobian=class_J,
            class_weights=weights, score_gradient=score_gradient, gradient=gradient, curvature=curvature, damped_hessian=hessian),
            class_count=c, physical_count=n, unbalanced_class_counts_supported=True,
            damping=1. if J is not None else None, curvature_scope='SCORE_GGN_PLUS_RMS_CURVATURE_NOT_FULL_NONLINEAR_HESSIAN',
            single_class_constant_target=c == 1)


def solve_hard_ball_step(*, gradient, hessian, radius=.5):
    """Unique SPD five-dimensional quadratic ball step, with a monotone secular solve."""
    g, H = _array(gradient, 'gradient', (5,)), _array(hessian, 'hessian', (5, 5))
    if not np.array_equal(H, H.T):
        raise ValueError('hessian must be exactly symmetric')
    if isinstance(radius, (bool, np.bool_)) or not np.isscalar(radius) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be a finite positive scalar')
    radius = float(radius)
    with _Ledger('solve_hard_ball_step') as ledger, np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        ledger.snapshot.update(gradient=g, hessian=H)
        ledger.data['eigendecomposition_count'] = 1
        eigenvalues, vectors = np.linalg.eigh(H)
        if np.any(eigenvalues <= 0):
            ledger.fail('NON_SPD_QUADRATIC', 'Quadratic Hessian is not numerically positive definite')
        projected = vectors.T @ g
        with np.errstate(over='ignore'):
            coefficients = -projected / eigenvalues
            initial_norm = _norm(coefficients) if np.isfinite(coefficients).all() else np.inf
        multiplier = 0.
        active = initial_norm > radius
        if active:
            upper = _norm(g) / radius
            if not np.isfinite(upper) or upper <= 0:
                ledger.fail('NONFINITE_SECULAR_BOUND', 'Finite secular bracket cannot be represented')
            lower = 0.
            tolerance = 128 * _EPS * 5
            for iteration in range(128):
                middle = .5 * lower + .5 * upper
                trial = -projected / (eigenvalues + middle)
                length = _norm(trial)
                ledger.data['secular_iteration_count'] += 1
                ledger.data['secular_evaluation_count'] += 1
                if length > radius:
                    lower = middle
                else:
                    upper = middle
                if abs(length - radius) <= tolerance * radius and length <= radius:
                    break
            else:
                ledger.fail('SECULAR_BUDGET_EXHAUSTED', 'Numerical secular root did not converge')
            multiplier = upper
            coefficients = -projected / (eigenvalues + multiplier)
        direction = vectors @ coefficients
        # Preserve the feasible side despite orthogonal-reconstruction rounding.
        if active:
            for _ in range(16):
                if _norm(direction) <= radius:
                    break
                multiplier = float(np.nextafter(multiplier, np.inf))
                direction = vectors @ (-projected / (eigenvalues + multiplier))
                ledger.data['secular_evaluation_count'] += 1
            else:
                ledger.fail('BALL_FEASIBILITY_UNRESOLVED', 'Feasible-side secular reconstruction failed')
        stationarity = H @ direction + g + multiplier * direction
        scale = max(1., _norm(g), _norm(H) * _norm(direction), multiplier * _norm(direction))
        relative_residual = _norm(stationarity) / scale
        _finite(np.asarray(scale), np.asarray(relative_residual))
        if relative_residual > 128 * _EPS * 5:
            ledger.fail('QUADRATIC_KKT_RESIDUAL', 'Original SPD ball stationarity residual failed')
        ledger.snapshot.update(direction=direction, multiplier=np.asarray(multiplier))
        return ledger.result(QuadraticStep, dict(direction=direction, multiplier=np.asarray(multiplier),
            eigenvalues=eigenvalues, eigenvectors=vectors), active=bool(active), radius=radius,
            direction_norm=_norm(direction), directional_derivative=float(g @ direction),
            quadratic_value=float(g @ direction + .5 * direction @ H @ direction),
            stationarity_relative_residual=relative_residual, zero_update=bool(not np.any(g)),
            numerical_root_tolerance=128 * _EPS * 5, unique_spd_subproblem=True)
