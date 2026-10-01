"""Literal synthetic certificates; no actual features, prototype packets or scores.

The --static-check entry point parses UTF-8 source before importing NumPy/pytest.
Numerical execution belongs to the root agent in the verified project environment.
"""
if __name__ == '__main__':
    import ast
    from pathlib import Path
    import sys
    if sys.argv[1:] != ['--static-check']:
        raise SystemExit('Only --static-check is available as a script')
    root = Path(__file__).resolve().parents[1]
    for relative in ('code/cvsrffi/d92_proto_frame_primitives.py',
                     'tests/test_d92_proto_frame_primitives.py'):
        path = root / relative
        raw = path.read_bytes()
        source = raw.decode('utf-8', errors='strict')
        ast.parse(source, filename=str(path))
        print('STATIC_UTF8_AST_OK', relative, len(raw))
    raise SystemExit(0)

from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from cvsrffi import d92_proto_frame_primitives as primitive


def _frame():
    p = np.zeros((6, 160), dtype=np.float32)
    p[np.arange(6), np.arange(6)] = np.arange(1, 7)
    return primitive.build_proto_frame_dictionary(prototypes=p,
        classes=('c4', 'c1', 'c5', 'c0', 'c3', 'c2'))


def _features(seed, count):
    generator = np.random.default_rng(seed)
    return {name: generator.normal(size=(count, dim)) for name, dim in
        zip(('z_id', 'fft', 't_emb', 'f_emb', 'pa_local'), (160, 96, 160, 160, 160))}


def _adapt(raw, Q, theta, jacobian=True):
    transformed = primitive.transform_z_id(z_id=raw['z_id'], Q=Q, theta=theta, jacobian=jacobian)
    blocks = dict(raw, z_id=transformed.values)
    return primitive.branch_geometry(**blocks, z_id_jacobian=transformed.jacobian)


def _dense_features(geometry):
    # Independent dense oracle, deliberately restricted to tiny synthetic n.
    interaction = np.einsum('ni,nj->nij', geometry.b, geometry.a).reshape(len(geometry.b), 256 * 480)
    return np.concatenate((geometry.b, geometry.a, interaction), axis=1)


def _dense_derivative(geometry):
    interaction = (np.einsum('nip,nj->nijp', geometry.b_jacobian, geometry.a)
        + np.einsum('ni,njp->nijp', geometry.b, geometry.a_jacobian)).reshape(len(geometry.b), 256 * 480, 5)
    return np.concatenate((geometry.b_jacobian, geometry.a_jacobian, interaction), axis=1)


def _target(labels, classes):
    return np.eye(classes)[labels] - 1. / classes


def _ridge_primal(phi, psi, Y):
    # Fit an independent feature-space primal with an unpenalized constant.
    design = np.column_stack((phi, np.ones(len(phi))))
    penalty = np.diag(np.r_[np.ones(phi.shape[1]), 0.])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ Y)
    return np.column_stack((psi, np.ones(len(psi)))) @ coefficients, coefficients[-1]


def _dense_rms_oracle(scores, labels, J):
    n, c = scores.shape
    shifted = scores - scores.max(axis=1)[:, None]
    p = np.exp(shifted)
    p /= p.sum(axis=1)[:, None]
    ce = -np.log(p[np.arange(n), labels])
    counts = np.bincount(labels, minlength=c)
    means = np.bincount(labels, weights=ce, minlength=c) / counts
    risk = np.sqrt(np.mean(means ** 2))
    v = means / (c * risk)
    B = np.zeros((c, n * c))
    W = np.zeros((n * c, n * c))
    for row, label in enumerate(labels):
        residual = p[row] - np.eye(c)[label]
        B[label, row * c:(row + 1) * c] = residual / counts[label]
        block = np.diag(p[row]) - np.outer(p[row], p[row])
        W[row * c:(row + 1) * c, row * c:(row + 1) * c] = v[label] * block / counts[label]
    outer = np.eye(c) / (c * risk) - np.outer(means, means) / (c * c * risk ** 3)
    W += B.T @ outer @ B
    flat_J = J.reshape(n * c, 5)
    return risk, flat_J.T @ B.T @ v, flat_J.T @ W @ flat_J


def test_frame_is_canonical_fixed_difference_and_sealed():
    p = np.zeros((6, 160), dtype=np.float32)
    p[np.arange(6), np.arange(6)] = np.arange(1, 7)
    classes = ('c4', 'c1', 'c5', 'c0', 'c3', 'c2')
    result = primitive.build_proto_frame_dictionary(prototypes=p, classes=classes)
    order = [classes.index(name) for name in sorted(classes)]
    expected = np.eye(6, 160)[order]
    np.testing.assert_array_equal(result.prototypes, expected)
    np.testing.assert_array_equal(result.Q, (expected[:5] - expected[5]).T)
    assert result.classes == tuple(sorted(classes))
    assert result.Q.shape == (160, 5)
    permutation = np.array([4, 0, 5, 2, 1, 3])
    second = primitive.build_proto_frame_dictionary(prototypes=p[permutation],
        classes=tuple(classes[i] for i in permutation))
    np.testing.assert_array_equal(second.Q, result.Q)
    with pytest.raises(AttributeError):
        result.arrays = {}
    with pytest.raises(TypeError):
        result._audit['class_order'] = ()
    with pytest.raises(ValueError):
        result.Q.setflags(write=True)
    changed = result.audit
    changed['class_order'].reverse()
    assert result.classes == tuple(sorted(classes))
    metadata = result.audit_dict()
    assert json.loads(json.dumps(metadata))['class_order'] == list(result.classes)
    assert metadata['frame_construction_count'] == 1
    assert metadata['physical_evaluation_count'] == 6
    assert metadata['returned_numeric_state_bytes'] == result.Q.nbytes + result.prototypes.nbytes
    assert metadata['process_peak_memory_bytes'] is None


def test_frame_zero_and_singular_dictionaries_keep_all_five_coordinates():
    zeros = primitive.build_proto_frame_dictionary(prototypes=np.zeros((6, 160)), classes=list('abcdef'))
    assert zeros.Q.shape == (160, 5)
    assert zeros.audit['zero_prototype_count'] == 6
    assert zeros.audit['dictionary_all_zero']
    p = np.zeros((6, 160))
    p[:5, 0] = 1
    p[5, 1] = 1
    singular = primitive.build_proto_frame_dictionary(prototypes=p, classes=list('abcdef'))
    assert np.linalg.matrix_rank(singular.Q) == 1
    assert singular.Q.shape == (160, 5)
    assert singular.audit['factorization_attempts'] == 0
    assert singular.audit['spectral_check_count'] == 0
    assert singular.audit['eigendecomposition_count'] == 0


@pytest.mark.parametrize('prototypes,classes', [
    (np.zeros((5, 160)), list('abcde')),
    (np.zeros((6, 159)), list('abcdef')),
    (np.full((6, 160), np.nan), list('abcdef')),
    (np.zeros((6, 160)), list('aaaaaa')),
    (np.zeros((6, 160)), [1, 2, 3, 4, 5, 6]),
    (np.zeros((6, 160)), 'abcdef'),
])
def test_frame_rejects_invalid_physical_contract(prototypes, classes):
    with pytest.raises(ValueError):
        primitive.build_proto_frame_dictionary(prototypes=prototypes, classes=classes)


def test_tangent_identity_keeps_live_jacobian_and_zero_rows():
    Q = _frame().Q
    z = _features(810, 4)['z_id']
    z[-1] = 0
    result = primitive.transform_z_id(z_id=z, Q=Q, theta=np.zeros(5))
    np.testing.assert_array_equal(result.values, z)
    assert np.linalg.norm(result.jacobian[:-1]) > 0
    np.testing.assert_array_equal(result.jacobian[-1], np.zeros((160, 5)))
    u = z[:-1] / np.linalg.norm(z[:-1], axis=1)[:, None]
    expected = np.linalg.norm(z[:-1], axis=1)[:, None, None] * (
        Q[None] - u[:, :, None] * (u @ Q)[:, None, :])
    np.testing.assert_allclose(result.jacobian[:-1], expected, rtol=2e-14, atol=2e-14)
    forward = primitive.transform_z_id(z_id=z, Q=Q, theta=np.zeros(5), jacobian=False)
    assert forward.jacobian is None
    assert forward.audit['jacobian_evaluation_count'] == 0


def test_tangent_complete_jacobian_norm_and_bounded_angle():
    Q = _frame().Q
    z = _features(811, 3)['z_id']
    theta = np.array([.03, -.07, .11, .02, -.04])
    result = primitive.transform_z_id(z_id=z, Q=Q, theta=theta)
    np.testing.assert_allclose(np.linalg.norm(result.values, axis=1), np.linalg.norm(z, axis=1), rtol=4e-15)
    np.testing.assert_allclose(np.einsum('ni,nip->np', result.values, result.jacobian), 0, atol=4e-13)
    u = z / np.linalg.norm(z, axis=1)[:, None]
    v = result.values / np.linalg.norm(result.values, axis=1)[:, None]
    cosine = np.sum(u * v, axis=1)
    assert np.all(cosine >= 1 / np.sqrt(1 + primitive.KAPPA ** 2) - 2e-15)
    step = 2e-6
    for coordinate in range(5):
        direction = np.eye(5)[coordinate]
        plus = primitive.transform_z_id(z_id=z, Q=Q, theta=theta + step * direction, jacobian=False)
        minus = primitive.transform_z_id(z_id=z, Q=Q, theta=theta - step * direction, jacobian=False)
        numerical = (plus.values - minus.values) / (2 * step)
        np.testing.assert_allclose(result.jacobian[:, :, coordinate], numerical, rtol=2e-7, atol=1e-9)


def test_tangent_zero_Q_and_representable_small_large_rows():
    z = np.zeros((3, 160))
    z[0, :2] = [1e-250, 2e-250]
    z[1, :2] = [1e250, -2e250]
    result = primitive.transform_z_id(z_id=z, Q=np.zeros((160, 5)), theta=np.ones(5))
    np.testing.assert_allclose(result.values, z, rtol=3e-15, atol=0)
    np.testing.assert_array_equal(result.jacobian, np.zeros((3, 160, 5)))
    assert np.isfinite(result.values).all()
    assert result.audit['zero_row_count'] == 1


def test_branch_geometry_matches_original_and_full_jacobian():
    from cvsrffi import d92_branch_interaction as original
    raw = _features(812, 3)
    geom = primitive.branch_geometry(**raw)
    b, a = original._blocks(**raw)
    np.testing.assert_allclose(geom.b, b, rtol=2e-14, atol=3e-16)
    np.testing.assert_allclose(geom.a, a, rtol=2e-14, atol=3e-16)
    Q, theta = _frame().Q, np.array([.08, -.02, .03, .01, -.05])
    current = _adapt(raw, Q, theta)
    for coordinate in range(5):
        direction = np.eye(5)[coordinate]
        plus = _adapt(raw, Q, theta + 2e-6 * direction, False)
        minus = _adapt(raw, Q, theta - 2e-6 * direction, False)
        np.testing.assert_allclose(current.b_jacobian[:, :, coordinate], (plus.b - minus.b) / 4e-6,
            rtol=2e-7, atol=2e-10)
    np.testing.assert_array_equal(current.a_jacobian, np.zeros((3, 480, 5)))
    assert current.audit['materialized_interaction_feature_count'] == 0


def test_branch_zero_floor_and_nondifferentiable_arbitrary_radial_input():
    raw = {name: np.zeros((2, dim)) for name, dim in
        zip(('z_id', 'fft', 't_emb', 'f_emb', 'pa_local'), (160, 96, 160, 160, 160))}
    raw['z_id'][1, 0] = primitive.NORM_FLOOR
    Q = np.zeros((160, 5)); Q[1, 0] = 1
    transformed = primitive.transform_z_id(z_id=raw['z_id'], Q=Q, theta=np.zeros(5))
    geometry = primitive.branch_geometry(**raw, z_id_jacobian=transformed.jacobian)
    np.testing.assert_array_equal(geometry.b[0], np.zeros(256))
    np.testing.assert_array_equal(geometry.b_jacobian[0], np.zeros((256, 5)))
    assert np.isfinite(geometry.b_jacobian).all()
    arbitrary = np.zeros((2, 160, 5)); arbitrary[1, 0, 0] = 1
    with pytest.raises(primitive.PrimitiveFailure, match='NORMALIZATION_FLOOR_NONDIFFERENTIABLE_DIRECTION') as caught:
        primitive.branch_geometry(**raw, z_id_jacobian=arbitrary)
    assert caught.value.audit['physical_evaluation_count'] == 2
    assert caught.value.audit['jacobian_evaluation_count'] == 1


def test_implicit_interaction_distances_and_both_end_derivatives_have_dense_oracle():
    Q, theta = _frame().Q, np.array([.1, -.03, .04, -.07, .02])
    left, right = _adapt(_features(813, 2), Q, theta), _adapt(_features(814, 3), Q, theta)
    result = primitive.interaction_distances(left, right)
    dense_left, dense_right = _dense_features(left), _dense_features(right)
    difference = dense_left[:, None, :] - dense_right[None, :, :]
    expected = np.sum(difference ** 2, axis=2)
    d_difference = _dense_derivative(left)[:, None] - _dense_derivative(right)[None]
    expected_J = 2 * np.einsum('hni,hnip->hnp', difference, d_difference)
    np.testing.assert_allclose(result.distance, expected, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(result.jacobian, expected_J, rtol=2e-12, atol=2e-14)
    assert result.audit['distance_pair_count'] == 6
    assert result.audit['distance_qr_pair_count'] == 6
    assert result.audit['materialized_interaction_feature_count'] == 0
    left_only = 2 * np.einsum('hni,hip->hnp', difference, _dense_derivative(left))
    assert np.linalg.norm(result.jacobian - left_only) > 1e-5


def test_distances_exact_duplicates_near_duplicates_and_symmetric_work():
    raw = _features(815, 3)
    for value in raw.values(): value[1] = value[0]
    raw['z_id'][2] = raw['z_id'][0]
    raw['z_id'][2, 2] += 1e-7
    for name in ('fft', 't_emb', 'f_emb', 'pa_local'): raw[name][2] = raw[name][0]
    geom = _adapt(raw, _frame().Q, np.array([.01, .02, .03, -.01, -.02]))
    result = primitive.interaction_distances(geom)
    assert result.distance[0, 1] == 0
    assert result.distance[0, 2] > 0
    np.testing.assert_array_equal(result.distance, result.distance.T)
    np.testing.assert_array_equal(result.jacobian, result.jacobian.transpose(1, 0, 2))
    np.testing.assert_array_equal(np.diag(result.distance), np.zeros(3))
    assert result.audit['distance_pair_count'] == 3
    assert result.audit['distance_qr_pair_count'] == 3
    from cvsrffi import d92_branch_local_ridge as original
    np.testing.assert_allclose(result.distance, original._distances(geom.b, geom.a), rtol=2e-12, atol=1e-30)


def test_fixed_half_original_raw_kernel_psd_and_complete_five_directions():
    train, held = _features(816, 3), _features(817, 2)
    Q, theta = _frame().Q, np.array([.05, -.08, .02, .03, .01])
    o_train, o_held = primitive.branch_geometry(**train), primitive.branch_geometry(**held)
    a_train, a_held = _adapt(train, Q, theta), _adapt(held, Q, theta)
    result = primitive.raw_kernel(original_left=o_held, adapted_left=a_held,
        original_right=o_train, adapted_right=a_train, tau=.9, gamma=1.7)
    expected = 1.7 * np.exp(-.5 * (primitive.interaction_distances(o_held, o_train, jacobian=False).distance
        + primitive.interaction_distances(a_held, a_train).distance) / .9)
    np.testing.assert_allclose(result.kernel, expected, rtol=3e-15, atol=0)
    for coordinate in range(5):
        direction = np.eye(5)[coordinate]
        kernels = []
        for sign in (1, -1):
            kernels.append(primitive.raw_kernel(original_left=o_held, adapted_left=_adapt(held, Q, theta + sign * 2e-6 * direction, False),
                original_right=o_train, adapted_right=_adapt(train, Q, theta + sign * 2e-6 * direction, False),
                tau=.9, gamma=1.7, jacobian=False).kernel)
        np.testing.assert_allclose(result.jacobian[:, :, coordinate], (kernels[0] - kernels[1]) / 4e-6,
            rtol=3e-7, atol=1e-11)
    joined = {name: np.concatenate((train[name], held[name])) for name in train}
    full = primitive.raw_kernel(original_left=primitive.branch_geometry(**joined),
        adapted_left=_adapt(joined, Q, theta), tau=.9, gamma=1.7)
    np.testing.assert_array_equal(full.kernel, full.kernel.T)
    assert np.linalg.eigvalsh(full.kernel)[0] >= -1e-13
    assert result.audit['distance_evaluation_count'] == 2
    assert result.audit['distance_pair_count'] == 12
    assert result.audit['kernel_pair_count'] == 6


@pytest.mark.parametrize('gamma,tau', [(None, None), (None, 0.), (0., None), (0., .9)])
def test_zero_kernel_has_no_adapter_gradient_or_hidden_adapted_distance(gamma, tau):
    raw = _features(818, 2)
    original = primitive.branch_geometry(**raw)
    # No adapted Jacobian required in a parameter-independent zero branch.
    result = primitive.raw_kernel(original_left=original, adapted_left=original, tau=tau, gamma=gamma)
    np.testing.assert_array_equal(result.kernel, np.zeros((2, 2)))
    np.testing.assert_array_equal(result.jacobian, np.zeros((2, 2, 5)))
    assert result.adapted_distance is None
    assert result.audit['distance_evaluation_count'] == 1
    assert result.audit['kernel_evaluation_count'] == 0
    assert result.audit['jacobian_evaluation_count'] == 0
    assert result.audit['kernel_gradient_status'] == 'ZERO_KERNEL_NO_CONTINUOUS_INFORMATION'


def test_tau_zero_uses_entire_original_exact_geometry_not_labels_or_thresholds():
    raw = _features(819, 3)
    for value in raw.values(): value[1] = value[0]
    raw['pa_local'][2] = raw['pa_local'][0]
    raw['pa_local'][2, 1] += 1e-7
    for name in ('z_id', 'fft', 't_emb', 'f_emb'): raw[name][2] = raw[name][0]
    original = primitive.branch_geometry(**raw)
    result = primitive.raw_kernel(original_left=original, adapted_left=_adapt(raw, _frame().Q, np.ones(5)), tau=0., gamma=2.)
    expected = np.array([[2., 2., 0.], [2., 2., 0.], [0., 0., 2.]])
    np.testing.assert_array_equal(result.kernel, expected)
    np.testing.assert_array_equal(result.jacobian, np.zeros((3, 3, 5)))
    assert result.adapted_distance is None
    assert result.audit['kernel_gradient_status'] == 'ORIGINAL_EXACT_EQUIVALENCE_PARAMETER_INDEPENDENT'
    with pytest.raises(ValueError):
        primitive.raw_kernel(original_left=original, adapted_left=original, tau=None, gamma=2.)


def test_empty_held_forward_and_single_sample_are_identical_to_batch():
    train, held = _features(820, 3), _features(821, 2)
    Q, theta = _frame().Q, np.array([.06, -.01, .03, .02, -.04])
    o_train, a_train = primitive.branch_geometry(**train), _adapt(train, Q, theta)
    batch = primitive.raw_kernel(original_left=primitive.branch_geometry(**held), adapted_left=_adapt(held, Q, theta),
        original_right=o_train, adapted_right=a_train, tau=1.1, gamma=.7)
    for row in range(2):
        raw = {name: value[row:row + 1] for name, value in held.items()}
        single = primitive.raw_kernel(original_left=primitive.branch_geometry(**raw), adapted_left=_adapt(raw, Q, theta),
            original_right=o_train, adapted_right=a_train, tau=1.1, gamma=.7)
        np.testing.assert_array_equal(single.kernel[0], batch.kernel[row])
        np.testing.assert_array_equal(single.jacobian[0], batch.jacobian[row])
    raw_empty = {name: value[:0] for name, value in held.items()}
    empty = primitive.raw_kernel(original_left=primitive.branch_geometry(**raw_empty),
        adapted_left=_adapt(raw_empty, Q, theta), original_right=o_train, adapted_right=a_train,
        tau=1.1, gamma=.7)
    assert empty.kernel.shape == (0, 3)
    assert empty.jacobian.shape == (0, 3, 5)
    assert empty.audit['distance_pair_count'] == 0
    state = primitive.fit_free_intercept_ridge(K=np.eye(3), Y=np.eye(3), L=empty.kernel,
        K_jacobian=np.zeros((3, 3, 5)), L_jacobian=empty.jacobian)
    assert state.scores.shape == (0, 3)
    assert state.score_jacobian.shape == (0, 3, 5)


def test_free_intercept_ridge_matches_independent_feature_primal_and_saddle():
    generator = np.random.default_rng(822)
    phi, psi = generator.normal(size=(6, 3)), generator.normal(size=(4, 3))
    Y = _target(np.array([0, 0, 0, 1, 2, 2]), 3)
    K, L = phi @ phi.T, psi @ phi.T
    result = primitive.fit_free_intercept_ridge(K=K, Y=Y, L=L)
    expected, b = _ridge_primal(phi, psi, Y)
    np.testing.assert_allclose(result.scores, expected, rtol=3e-13, atol=2e-14)
    np.testing.assert_allclose(result.intercept, b, rtol=3e-13, atol=2e-14)
    saddle = np.block([[K + np.eye(6), np.ones((6, 1))], [np.ones((1, 6)), np.zeros((1, 1))]])
    answer = np.linalg.solve(saddle, np.vstack((Y, np.zeros((1, 3)))))
    np.testing.assert_allclose(result.alpha, answer[:6], rtol=3e-13, atol=2e-14)
    np.testing.assert_allclose(result.intercept, answer[6], rtol=3e-13, atol=2e-14)
    np.testing.assert_allclose(result.alpha.sum(axis=0), 0., atol=1e-14)
    assert result.arrays['s'].shape == ()
    assert result.score_jacobian is None
    assert result.audit['jvp_directions'] == 0


def test_ridge_complete_five_jvp_includes_free_intercept_and_both_cross_ends():
    generator = np.random.default_rng(823)
    phi, psi = generator.normal(size=(5, 3)), generator.normal(size=(3, 3))
    dphi, dpsi = generator.normal(size=(5, 3, 5)), generator.normal(size=(3, 3, 5))
    Y = _target(np.array([0, 0, 0, 1, 2]), 3)
    K, L = phi @ phi.T, psi @ phi.T
    dK = np.einsum('nip,mi->nmp', dphi, phi) + np.einsum('ni,mip->nmp', phi, dphi)
    dL = np.einsum('hip,ni->hnp', dpsi, phi) + np.einsum('hi,nip->hnp', psi, dphi)
    result = primitive.fit_free_intercept_ridge(K=K, Y=Y, L=L, K_jacobian=dK, L_jacobian=dL)
    step = 2e-6
    for coordinate in range(5):
        answers = []
        for sign in (1, -1):
            train = phi + sign * step * dphi[:, :, coordinate]
            held = psi + sign * step * dpsi[:, :, coordinate]
            answers.append(_ridge_primal(train, held, Y))
        numerical = (answers[0][0] - answers[1][0]) / (2 * step)
        db = (answers[0][1] - answers[1][1]) / (2 * step)
        np.testing.assert_allclose(result.score_jacobian[:, :, coordinate], numerical, rtol=3e-7, atol=2e-9)
        np.testing.assert_allclose(result.arrays['intercept_jacobian'][:, coordinate], db, rtol=3e-7, atol=2e-9)
    assert np.linalg.norm(result.arrays['intercept_jacobian']) > 1e-3
    incomplete = np.einsum('hip,ic->hcp', dL, result.alpha) + np.einsum('hi,icp->hcp', L, result.arrays['alpha_jacobian'])
    assert np.linalg.norm(result.score_jacobian - incomplete) > 1e-3


def test_ridge_uses_one_factor_and_real_combined_rhs_work(monkeypatch):
    original = np.linalg.cholesky
    calls = []
    def factor(value):
        calls.append(value.shape)
        return original(value)
    monkeypatch.setattr(np.linalg, 'cholesky', factor)
    n, c, h = 4, 3, 2
    result = primitive.fit_free_intercept_ridge(K=np.eye(n), Y=np.eye(n, c), L=np.zeros((h, n)),
        K_jacobian=np.zeros((n, n, 5)), L_jacobian=np.zeros((h, n, 5)))
    assert calls == [(n, n)]
    audit = result.audit
    assert audit['spectral_check_count'] == 1
    assert audit['factorization_attempts'] == audit['factorizations_completed'] == 1
    assert audit['triangular_calls'] == 4
    assert audit['triangular_rhs_columns'] == 2 * (c + 1) * 6
    assert audit['triangular_rhs_elements'] == n * 2 * (c + 1) * 6
    assert audit['triangular_dense_work_units'] == n * n * 2 * (c + 1) * 6
    assert result.arrays['combined_rhs'].shape == (n, c + 1)
    assert result.arrays['jvp_rhs'].shape == (n, 5 * (c + 1))
    assert audit['returned_numeric_state_bytes'] == sum(value.nbytes for value in result.arrays.values())
    forward = primitive.fit_free_intercept_ridge(K=np.eye(n), Y=np.eye(n, c), L=np.zeros((h, n)))
    assert forward.audit['triangular_calls'] == 2
    assert forward.audit['triangular_rhs_columns'] == 2 * (c + 1)
    assert forward.audit['factorizations_completed'] == 1
    assert forward.audit['jacobian_evaluation_count'] == 0


def test_zero_and_singular_kernels_keep_unbalanced_free_constant_and_n1():
    Y = _target(np.array([0, 0, 0, 1]), 2)
    state = primitive.fit_free_intercept_ridge(K=np.zeros((4, 4)), Y=Y, L=np.zeros((3, 4)),
        K_jacobian=np.zeros((4, 4, 5)), L_jacobian=np.zeros((3, 4, 5)))
    np.testing.assert_allclose(state.intercept, Y.mean(axis=0), atol=0)
    np.testing.assert_array_equal(state.alpha, Y - Y.mean(axis=0))
    np.testing.assert_array_equal(state.scores, np.tile(Y.mean(axis=0), (3, 1)))
    np.testing.assert_array_equal(state.score_jacobian, np.zeros((3, 2, 5)))
    single = primitive.fit_free_intercept_ridge(K=np.array([[7.]]), Y=np.array([[.2, -.3]]), L=np.array([[2.], [0.]]))
    np.testing.assert_allclose(single.alpha, 0., atol=1e-17)
    np.testing.assert_allclose(single.scores, [[.2, -.3], [.2, -.3]], atol=1e-16)
    constant = primitive.fit_free_intercept_ridge(K=np.ones((4, 4)), Y=Y, L=np.ones((2, 4)))
    np.testing.assert_allclose(constant.scores, np.tile(Y.mean(axis=0), (2, 1)), atol=3e-16)


def test_ridge_row_and_class_permutations_keep_the_same_function():
    generator = np.random.default_rng(824)
    phi, psi = generator.normal(size=(5, 3)), generator.normal(size=(2, 3))
    Y = generator.normal(size=(5, 3))
    K, L = phi @ phi.T, psi @ phi.T
    first = primitive.fit_free_intercept_ridge(K=K, Y=Y, L=L)
    rows, columns = np.array([3, 0, 4, 1, 2]), np.array([2, 0, 1])
    permuted = primitive.fit_free_intercept_ridge(K=K[rows][:, rows], Y=Y[rows][:, columns], L=L[:, rows])
    np.testing.assert_allclose(permuted.scores, first.scores[:, columns], rtol=2e-13, atol=2e-14)


def test_ridge_failures_preserve_attempted_work_without_jitter(monkeypatch):
    K = np.diag([1., -1.])
    with pytest.raises(primitive.PrimitiveFailure) as caught:
        primitive.fit_free_intercept_ridge(K=K, Y=np.eye(2), L=np.eye(2))
    assert caught.value.code == 'NON_PSD_KERNEL'
    assert caught.value.audit['spectral_check_count'] == 1
    assert caught.value.audit['factorization_attempts'] == 0
    np.testing.assert_array_equal(caught.value.arrays['K'], K)
    def failed_factor(value):
        raise np.linalg.LinAlgError('synthetic factor failure')
    monkeypatch.setattr(np.linalg, 'cholesky', failed_factor)
    with pytest.raises(primitive.PrimitiveFailure) as second:
        primitive.fit_free_intercept_ridge(K=np.eye(2), Y=np.eye(2), L=np.eye(2))
    assert second.value.audit['factorization_attempts'] == 1
    assert second.value.audit['factorizations_completed'] == 0
    assert second.value.audit['triangular_calls'] == 0
    assert second.value.audit['partial_array_scope'] == 'FINITE_NUMERIC_INPUTS_AND_COMPLETED_INTERMEDIATES'
    assert second.value.audit['omitted_nonfinite_array_keys'] == []
    assert second.value.arrays['Y'].flags.writeable is False
    with pytest.raises(ValueError):
        primitive.fit_free_intercept_ridge(K=np.eye(2), Y=np.eye(2), L=np.eye(2), K_jacobian=np.zeros((2, 2, 5)))


def test_jvp_second_triangle_failure_preserves_completed_forward_and_actual_rhs(monkeypatch):
    n, c = 3, 2
    K = np.eye(n)
    Y = _target(np.array([0, 0, 1]), c)
    L = np.array([[.1, .2, .4], [.5, .3, .1]])
    dK = np.repeat(np.eye(n)[:, :, None], 5, axis=2)
    dL = np.zeros((2, n, 5))
    original_finite = primitive._finite
    validations = []
    def finite(*values):
        if len(values) == 1 and np.asarray(values[0]).shape == (n, 5 * (c + 1)):
            validations.append(np.asarray(values[0]).copy())
            if len(validations) == 2:
                raise FloatingPointError('synthetic JVP second-triangle validation failure')
        return original_finite(*values)
    monkeypatch.setattr(primitive, '_finite', finite)
    with pytest.raises(primitive.PrimitiveFailure) as caught:
        primitive.fit_free_intercept_ridge(K=K, Y=Y, L=L, K_jacobian=dK, L_jacobian=dL)
    assert len(validations) == 2
    failure = caught.value
    assert failure.audit['factorizations_completed'] == 1
    assert failure.audit['triangular_calls'] == 4
    assert failure.audit['triangular_rhs_columns'] == 2 * (c + 1) * 6
    partial = failure.arrays
    np.testing.assert_array_equal(partial['combined_rhs'], np.column_stack((Y, np.ones(n))))
    saddle = np.block([[K + np.eye(n), np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    independent = np.linalg.solve(saddle, np.vstack((Y, np.zeros((1, c)))))
    np.testing.assert_allclose(partial['alpha'], independent[:n], atol=2e-16)
    np.testing.assert_allclose(partial['intercept'], independent[n], atol=2e-16)
    np.testing.assert_allclose(partial['scores'], L @ independent[:n] + independent[n], atol=2e-16)
    np.testing.assert_allclose(partial['train_scores'], K @ independent[:n] + independent[n], atol=2e-16)
    np.testing.assert_allclose((K + np.eye(n)) @ partial['F'], Y, atol=2e-16)
    np.testing.assert_allclose((K + np.eye(n)) @ partial['z'], np.ones(n), atol=2e-16)
    assert partial['s'].shape == ()
    expected_rhs = np.column_stack(((-np.einsum('ijp,jc->icp', dK, partial['F'])).reshape(n, c * 5),
        -np.einsum('ijp,j->ip', dK, partial['z'])))
    np.testing.assert_array_equal(partial['jvp_rhs'], expected_rhs)
    assert 'F_jacobian' not in partial and 'score_jacobian' not in partial
    assert failure.audit['omitted_nonfinite_array_keys'] == []
    for value in partial.values():
        assert np.isfinite(value).all() and not value.flags.writeable


def test_rmsce_unbalanced_classes_matches_independent_dense_score_hessian():
    generator = np.random.default_rng(825)
    scores = generator.normal(size=(7, 3))
    J = generator.normal(size=(7, 3, 5))
    labels = np.array([0, 0, 0, 0, 1, 2, 2])
    result = primitive.class_balanced_rmsce(scores=scores, labels=labels, score_jacobian=J)
    loss, gradient, curvature = _dense_rms_oracle(scores, labels, J)
    np.testing.assert_allclose(result.loss, loss, rtol=3e-15)
    np.testing.assert_allclose(result.gradient, gradient, rtol=3e-14, atol=2e-15)
    np.testing.assert_allclose(result.curvature, curvature, rtol=3e-14, atol=3e-15)
    np.testing.assert_allclose(result.damped_hessian, np.eye(5) + curvature, rtol=3e-14, atol=3e-15)
    assert np.linalg.eigvalsh(result.curvature)[0] >= -1e-14
    assert np.linalg.eigvalsh(result.damped_hessian)[0] >= 1 - 1e-14
    np.testing.assert_array_equal(result.arrays['class_ce_counts'], [4, 1, 2])
    assert abs(result.loss - np.mean(result.arrays['ce'])) > 1e-3
    assert result.audit['curvature_scope'] == 'SCORE_GGN_PLUS_RMS_CURVATURE_NOT_FULL_NONLINEAR_HESSIAN'
    forward = primitive.class_balanced_rmsce(scores=scores, labels=labels)
    assert forward.loss == result.loss
    assert forward.gradient is forward.curvature is forward.damped_hessian is None


def test_rmsce_complete_gradient_and_curvature_finite_difference_in_linear_scores():
    generator = np.random.default_rng(826)
    scores, J = generator.normal(size=(6, 3)), generator.normal(size=(6, 3, 5))
    labels = np.array([0, 0, 0, 1, 2, 2])
    initial = primitive.class_balanced_rmsce(scores=scores, labels=labels, score_jacobian=J)
    step = 2e-5
    for coordinate in range(5):
        plus = primitive.class_balanced_rmsce(scores=scores + step * J[:, :, coordinate], labels=labels, score_jacobian=J)
        minus = primitive.class_balanced_rmsce(scores=scores - step * J[:, :, coordinate], labels=labels, score_jacobian=J)
        np.testing.assert_allclose(initial.gradient[coordinate], (plus.loss - minus.loss) / (2 * step), rtol=2e-7, atol=2e-9)
        np.testing.assert_allclose(initial.curvature[:, coordinate], (plus.gradient - minus.gradient) / (2 * step), rtol=2e-7, atol=2e-9)


def test_rmsce_row_class_permutation_single_class_and_zero_information():
    generator = np.random.default_rng(827)
    scores, J = generator.normal(size=(5, 3)), generator.normal(size=(5, 3, 5))
    labels = np.array([0, 0, 1, 2, 2])
    initial = primitive.class_balanced_rmsce(scores=scores, labels=labels, score_jacobian=J)
    rows, columns = np.array([4, 1, 0, 3, 2]), np.array([2, 0, 1])
    inverse = np.argsort(columns)
    permuted = primitive.class_balanced_rmsce(scores=scores[rows][:, columns], labels=inverse[labels[rows]],
        score_jacobian=J[rows][:, columns])
    np.testing.assert_allclose(permuted.loss, initial.loss, rtol=3e-15)
    np.testing.assert_allclose(permuted.gradient, initial.gradient, rtol=3e-14, atol=2e-15)
    np.testing.assert_allclose(permuted.curvature, initial.curvature, rtol=3e-14, atol=2e-15)
    single = primitive.class_balanced_rmsce(scores=np.array([[2.], [-3.]]), labels=np.zeros(2, dtype=int),
        score_jacobian=np.ones((2, 1, 5)))
    assert single.loss == 0
    np.testing.assert_array_equal(single.gradient, np.zeros(5))
    np.testing.assert_array_equal(single.curvature, np.zeros((5, 5)))
    np.testing.assert_array_equal(single.damped_hessian, np.eye(5))
    zero = primitive.class_balanced_rmsce(scores=scores, labels=labels, score_jacobian=np.zeros((5, 3, 5)))
    np.testing.assert_array_equal(zero.gradient, np.zeros(5))
    np.testing.assert_array_equal(zero.damped_hessian, np.eye(5))


def test_rmsce_extreme_signed_logits_and_explicit_underflow_boundary():
    scores = np.array([[0., -700.], [-700., 0.]])
    J = np.arange(20, dtype=float).reshape(2, 2, 5) / 20
    labels = np.array([0, 1])
    result = primitive.class_balanced_rmsce(scores=scores, labels=labels, score_jacobian=J)
    assert result.loss > 0
    assert np.isfinite(result.gradient).all()
    assert np.isfinite(result.curvature).all()
    assert result.arrays['ce'][0] > 0
    with pytest.raises(primitive.PrimitiveFailure) as caught:
        primitive.class_balanced_rmsce(scores=np.array([[0., -2000.], [-2000., 0.]]), labels=labels, score_jacobian=J)
    assert caught.value.code == 'CE_UNDERFLOW_UNSUPPORTED'
    assert caught.value.audit['physical_evaluation_count'] == 2
    assert caught.value.audit['jacobian_evaluation_count'] == 1
    assert np.isfinite(caught.value.arrays['scores']).all()
    with pytest.raises(ValueError):
        primitive.class_balanced_rmsce(scores=np.zeros((2, 3)), labels=np.array([0, 1]))
    with pytest.raises(ValueError):
        primitive.class_balanced_rmsce(scores=scores, labels=labels.astype(float))


def test_spd_quadratic_ball_interior_zero_and_exact_one_direction_oracle():
    H = np.diag([2., 3., 4., 5., 6.])
    g = np.array([.01, -.02, .03, -.01, .02])
    interior = primitive.solve_hard_ball_step(gradient=g, hessian=H)
    np.testing.assert_allclose(interior.direction, -np.linalg.solve(H, g), atol=2e-16)
    assert interior.multiplier == 0
    assert interior.audit['active'] is False
    assert interior.audit['secular_iteration_count'] == 0
    zero = primitive.solve_hard_ball_step(gradient=np.zeros(5), hessian=H)
    np.testing.assert_array_equal(zero.direction, np.zeros(5))
    assert zero.audit['zero_update'] is True
    g = np.array([4., 0., 0., 0., 0.])
    boundary = primitive.solve_hard_ball_step(gradient=g, hessian=H)
    np.testing.assert_allclose(boundary.direction, [-.5, 0., 0., 0., 0.], atol=4e-14)
    np.testing.assert_allclose(boundary.multiplier, 6., atol=8e-13)
    assert boundary.audit['direction_norm'] <= .5
    assert boundary.audit['active'] is True
    assert boundary.audit['eigendecomposition_count'] == 1
    assert boundary.audit['factorization_attempts'] == 0


def test_spd_ball_rotated_kkt_and_global_quadratic_certificate():
    generator = np.random.default_rng(828)
    matrix = generator.normal(size=(5, 5))
    H = matrix @ matrix.T + np.eye(5)
    g = generator.normal(size=5) * 9
    state = primitive.solve_hard_ball_step(gradient=g, hessian=H)
    d, multiplier = state.direction, state.multiplier
    assert np.linalg.norm(d) <= .5
    assert multiplier > 0
    np.testing.assert_allclose((H + multiplier * np.eye(5)) @ d + g, 0., atol=2e-12)
    assert abs(np.linalg.norm(d) - .5) < 4e-14
    assert float(g @ d) < 0
    assert state.audit['quadratic_value'] < 0
    # KKT suffices for the unique SPD optimum; independently verify the exact
    # objective difference identity for arbitrary feasible physical vectors.
    for _ in range(10):
        x = generator.normal(size=5)
        x *= .49 / np.linalg.norm(x)
        lhs = g @ x + .5 * x @ H @ x - (g @ d + .5 * d @ H @ d)
        rhs = .5 * (x - d) @ H @ (x - d) - multiplier * d @ (x - d)
        np.testing.assert_allclose(lhs, rhs, rtol=3e-14, atol=1e-13)
        assert lhs >= 0


def test_ball_rejects_non_spd_or_nonfinite_controls_without_pseudoinverse():
    with pytest.raises(primitive.PrimitiveFailure) as caught:
        primitive.solve_hard_ball_step(gradient=np.ones(5), hessian=np.diag([1., 1., 1., 1., 0.]))
    assert caught.value.code == 'NON_SPD_QUADRATIC'
    assert caught.value.audit['eigendecomposition_count'] == 1
    assert caught.value.audit['secular_iteration_count'] == 0
    with pytest.raises(ValueError):
        primitive.solve_hard_ball_step(gradient=np.ones(5), hessian=np.eye(5), radius=0)
    H = np.eye(5); H[0, 1] = 1e-12
    with pytest.raises(ValueError):
        primitive.solve_hard_ball_step(gradient=np.ones(5), hessian=H)


def test_complete_736_feature_to_ridge_rms_pipeline_five_coordinate_derivative():
    train, held = _features(829, 5), _features(830, 6)
    Q, theta = _frame().Q, np.array([.04, -.05, .01, -.02, .03])
    train_labels, held_labels = np.array([0, 0, 0, 1, 2]), np.array([0, 0, 0, 1, 2, 2])
    original_train, original_held = primitive.branch_geometry(**train), primitive.branch_geometry(**held)
    def objective(value, need_jacobian):
        current_train = _adapt(train, Q, value, need_jacobian)
        current_held = _adapt(held, Q, value, need_jacobian)
        K = primitive.raw_kernel(original_left=original_train, adapted_left=current_train,
            tau=1.4, gamma=2.3, jacobian=need_jacobian)
        L = primitive.raw_kernel(original_left=original_held, adapted_left=current_held,
            original_right=original_train, adapted_right=current_train,
            tau=1.4, gamma=2.3, jacobian=need_jacobian)
        head = primitive.fit_free_intercept_ridge(K=K.kernel, Y=_target(train_labels, 3), L=L.kernel,
            K_jacobian=K.jacobian, L_jacobian=L.jacobian)
        return primitive.class_balanced_rmsce(scores=head.scores, labels=held_labels,
            score_jacobian=head.score_jacobian)
    initial = objective(theta, True)
    step = 3e-6
    for coordinate in range(5):
        direction = np.eye(5)[coordinate]
        numerical = (objective(theta + step * direction, False).loss
            - objective(theta - step * direction, False).loss) / (2 * step)
        np.testing.assert_allclose(initial.gradient[coordinate], numerical, rtol=2e-5, atol=8e-10)
    step_state = primitive.solve_hard_ball_step(gradient=initial.gradient, hessian=initial.damped_hessian)
    assert step_state.audit['direction_norm'] <= .5
    assert step_state.audit['directional_derivative'] <= 0
    assert initial.audit['damping'] == 1.


def test_raw_kernel_nested_failure_retains_both_attempted_distance_costs(monkeypatch):
    original_distance = primitive.interaction_distances
    calls = []
    def distance(left, right=None, *, jacobian=True):
        calls.append(jacobian)
        if len(calls) == 2:
            audit = dict.fromkeys(primitive.AUDIT_COUNTERS, 0)
            audit.update(distance_evaluation_count=1, distance_pair_count=1,
                distance_qr_batch_count=1, distance_qr_pair_count=1,
                jacobian_evaluation_count=1, seconds=.001)
            raise primitive.PrimitiveFailure('SYNTHETIC_DISTANCE_FAILURE', 'synthetic derivative failure', audit,
                dict(distance=np.ones((2, 2)), bad=np.array([np.nan])))
        return original_distance(left, right, jacobian=jacobian)
    monkeypatch.setattr(primitive, 'interaction_distances', distance)
    raw = _features(831, 2)
    with pytest.raises(primitive.PrimitiveFailure) as caught:
        primitive.raw_kernel(original_left=primitive.branch_geometry(**raw),
            adapted_left=_adapt(raw, _frame().Q, np.ones(5)), tau=1., gamma=1.)
    assert calls == [False, True]
    assert caught.value.code == 'SYNTHETIC_DISTANCE_FAILURE'
    assert caught.value.audit['distance_evaluation_count'] == 2
    assert caught.value.audit['distance_pair_count'] == 2
    assert caught.value.audit['distance_qr_pair_count'] == 2
    assert caught.value.audit['kernel_evaluation_count'] == 0
    assert 'original_distance' in caught.value.arrays
    assert 'adapted_distance' in caught.value.arrays


def test_interrupted_distance_counts_only_attempted_pair_batches(monkeypatch):
    original_qr = np.linalg.qr
    calls = []
    def qr(value, mode='reduced'):
        calls.append(value.shape[0])
        if len(calls) == 2:
            raise np.linalg.LinAlgError('synthetic second distance batch failure')
        return original_qr(value, mode=mode)
    monkeypatch.setattr(np.linalg, 'qr', qr)
    geometry = _adapt(_features(833, 4), _frame().Q, np.zeros(5))
    with pytest.raises(primitive.PrimitiveFailure) as caught:
        primitive.interaction_distances(geometry)
    assert calls == [3, 2]
    audit = caught.value.audit
    assert audit['distance_evaluation_count'] == 1
    assert audit['distance_pair_count'] == audit['distance_qr_pair_count'] == 5
    assert audit['distance_pair_count'] < 4 * 3 // 2
    assert audit['distance_qr_batch_count'] == 2
    partial = caught.value.arrays['distance']
    assert partial.shape == (4, 4)
    assert np.all(partial[0, 1:] > 0)
    assert partial[2, 3] == 0  # Third physical batch was never executed.


def test_all_returned_audits_are_json_safe_actual_work_and_no_peak_claim():
    frame = _frame()
    raw = _features(832, 2)
    transformed = primitive.transform_z_id(z_id=raw['z_id'], Q=frame.Q, theta=np.zeros(5))
    geometry = primitive.branch_geometry(**dict(raw, z_id=transformed.values), z_id_jacobian=transformed.jacobian)
    distances = primitive.interaction_distances(geometry)
    kernel = primitive.raw_kernel(original_left=primitive.branch_geometry(**raw), adapted_left=geometry, tau=1., gamma=1.)
    ridge = primitive.fit_free_intercept_ridge(K=kernel.kernel, Y=np.eye(2), L=kernel.kernel,
        K_jacobian=kernel.jacobian, L_jacobian=kernel.jacobian)
    rms = primitive.class_balanced_rmsce(scores=ridge.scores, labels=np.array([0, 1]), score_jacobian=ridge.score_jacobian)
    quadratic = primitive.solve_hard_ball_step(gradient=rms.gradient, hessian=rms.damped_hessian)
    assert primitive.AUDIT_MAX_KEYS == ()
    assert primitive.AUDIT_SUM_KEYS == primitive.AUDIT_COUNTERS + ('seconds',)
    for state in (frame, transformed, geometry, distances, kernel, ridge, rms, quadratic):
        metadata = state.audit_dict()
        json.dumps(metadata, allow_nan=False)
        assert metadata['process_peak_memory_bytes'] is None
        assert metadata['returned_numeric_state_bytes'] == sum(value.nbytes for value in state.arrays.values())
        assert metadata['seconds'] >= 0
        assert all(type(metadata[key]) is int and metadata[key] >= 0 for key in primitive.AUDIT_COUNTERS)
        for array in state.arrays.values():
            assert not array.flags.writeable
            with pytest.raises(ValueError): array.setflags(write=True)
