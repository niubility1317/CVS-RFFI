"""Synthetic-only tests for the independent full-KKT analysis certificate."""
import ast
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'code'))

from tools.d92_conditional_analysis_math import (  # noqa: E402
    MathCertificateFailure,
    certify_positive_kernel_head,
    positive_kernel_head_vjp,
    rebuild_positive_kernel_head,
)
from cvsrffi.d92_conditional_affine_kernel import (  # noqa: E402
    conditional_affine_adjoint,
    fit_conditional_affine,
)


INPUT_NAMES = ('A', 'B', 'D', 'F', 'E', 'M_O', 'M_N', 'M_H', 'R_N')
CLAIM_NAMES = ('alpha', 'beta', 'v', 'old_scores', 'train_new_scores', 'held_scores')


def _center(x):
    return x - x.mean(axis=-1, keepdims=True)


def _blocks(O, N, H, priors):
    return dict(A=O @ O.T, B=O @ N.T, D=N @ N.T, F=H @ O.T, E=H @ N.T,
                M_O=priors['M_O'], M_N=priors['M_N'], M_H=priors['M_H'], R_N=priors['R_N'])


def _case(seed=203, h=5):
    rng = np.random.default_rng(seed)
    m, p, width, c = 3, 4, 12, 4
    O, N, H = (rng.normal(size=(n, width)) * 0.4 for n in (m, p, h))
    priors = {name: _center(rng.normal(size=(n, c))) for name, n in
              (('M_O', m), ('M_N', p), ('M_H', h), ('R_N', p))}
    G = _center(rng.normal(size=(h, c)))
    return rng, O, N, H, priors, _blocks(O, N, H, priors), G


def _primal(O, N, H, priors):
    """Finite-feature weights, unpenalized constant, equality multipliers.

    This oracle does not construct a conditional kernel or the helper's dual
    system. Unknowns are [w_feature;v;mu_old], not [beta;alpha;v].
    """
    m, width = O.shape
    p, c = priors['R_N'].shape
    ones_p, ones_m = np.ones((p, 1)), np.ones((m, 1))
    primal_matrix = np.block([
        [np.eye(width) + N.T @ N, N.T @ ones_p, O.T],
        [ones_p.T @ N, np.array([[float(p)]]), ones_m.T],
        [O, ones_m, np.zeros((m, m))],
    ])
    primal_rhs = np.vstack((N.T @ priors['R_N'], priors['R_N'].sum(0)[None, :], np.zeros((m, c))))
    solution = np.linalg.solve(primal_matrix, primal_rhs)
    w, v, beta = solution[:width], solution[width], solution[width + 1:]
    alpha = priors['R_N'] - N @ w - v
    return dict(w=w, v=v, alpha=alpha, beta=beta,
                old_scores=priors['M_O'] + O @ w + v,
                train_new_scores=priors['M_N'] + N @ w + v,
                held_scores=priors['M_H'] + H @ w + v,
                residual_rkhs_norm_squared=float(np.sum(w * w)))


def _primal_from_blocks(blocks):
    """Realize perturbed SPD Gram blocks, then use only the feature primal."""
    m = len(blocks['A'])
    gram = np.block([[blocks['A'], blocks['B']], [blocks['B'].T, blocks['D']]])
    train_features = np.linalg.cholesky(gram)
    cross = np.column_stack((blocks['F'], blocks['E']))
    H = np.linalg.solve(train_features, cross.T).T
    priors = {name: blocks[name] for name in ('M_O', 'M_N', 'M_H', 'R_N')}
    return _primal(train_features[:m], train_features[m:], H, priors)


def _claims(certificate):
    return {name: np.array(certificate.arrays[name], copy=True) for name in CLAIM_NAMES}


def _linear_loss(scores, G):
    return float(np.sum(scores * G))


def _difference(function, direction, step=2.0e-6):
    return (function(step, direction) - function(-step, direction)) / (2.0 * step)


def test_rebuild_matches_independent_finite_feature_primal_and_candidate():
    _, O, N, H, priors, blocks, _ = _case()
    certificate = rebuild_positive_kernel_head(**blocks)
    primal = _primal(O, N, H, priors)
    candidate = fit_conditional_affine(**blocks)
    for name in CLAIM_NAMES:
        np.testing.assert_allclose(certificate.arrays[name], primal[name], rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(certificate.arrays[name], candidate.arrays[name], rtol=2e-11, atol=2e-11)
    assert certificate.audit['residual_rkhs_norm_squared'] == pytest.approx(primal['residual_rkhs_norm_squared'], rel=2e-11, abs=2e-11)
    assert certificate.audit['positive_kernel_checked'] is True
    assert certificate.audit['general_solve_count'] == 1
    assert certificate.audit['spectral_diagnostic_count'] == 2
    assert certificate.audit['general_dense_cubic_work_unit_count'] == (len(O) + len(N) + 1) ** 3
    assert certificate.audit['deployment_coefficient_bytes'] == 8 * ((len(O) + len(N)) * 4 + 4)


def test_residual_certificate_has_no_general_solve_or_spectrum():
    *_, blocks, _ = _case()
    rebuilt = rebuild_positive_kernel_head(**blocks)
    result = certify_positive_kernel_head(**blocks, **_claims(rebuilt))
    assert result.audit['certificate_mode'] == 'residual_only'
    assert result.audit['general_solve_count'] == 0
    assert result.audit['spectral_diagnostic_count'] == 0
    assert result.audit['positive_kernel_checked'] is False
    np.testing.assert_allclose(result.arrays['Q'] @ result.arrays['theta'], result.arrays['rhs'], atol=2e-13)


def test_helper_has_no_candidate_dependency_or_conditional_schur_code():
    tree = ast.parse((ROOT / 'tools' / 'd92_conditional_analysis_math.py').read_text(encoding='utf-8'))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {'dataclasses', 'types', 'numpy'}
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert 'fit_conditional_affine' not in called
    assert 'conditional_affine_adjoint' not in called


@pytest.mark.parametrize('name', ['A', 'B', 'D', 'F', 'E'])
def test_full_raw_block_vjp_matches_finite_feature_primal_difference(name):
    rng, _, _, _, _, blocks, G = _case()
    certificate = rebuild_positive_kernel_head(**blocks)
    vjp = positive_kernel_head_vjp(certificate, G)
    direction = rng.normal(size=blocks[name].shape)
    if name in ('A', 'D'):
        direction = 0.5 * (direction + direction.T)
    direction /= np.linalg.norm(direction)

    def loss(step, delta):
        perturbed = {key: value.copy() for key, value in blocks.items()}
        perturbed[name] += step * delta
        return _linear_loss(_primal_from_blocks(perturbed)['held_scores'], G)

    numeric = _difference(loss, direction)
    analytic = float(np.sum(vjp.arrays[name] * direction))
    assert analytic == pytest.approx(numeric, rel=2e-6, abs=2e-7)
    candidate = conditional_affine_adjoint(fit_conditional_affine(**blocks), G)
    np.testing.assert_allclose(vjp.arrays[name], candidate.arrays[name], rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize('endpoint', ['old', 'new', 'held'])
def test_old_new_held_complete_endpoint_vjp_against_primal(endpoint):
    rng, O, N, H, priors, blocks, G = _case()
    vjp = positive_kernel_head_vjp(rebuild_positive_kernel_head(**blocks), G).arrays
    gradients = dict(old=(vjp['A'] + vjp['A'].T) @ O + vjp['B'] @ N + vjp['F'].T @ H,
                     new=(vjp['D'] + vjp['D'].T) @ N + vjp['B'].T @ O + vjp['E'].T @ H,
                     held=vjp['F'] @ O + vjp['E'] @ N)
    features = dict(old=O, new=N, held=H)
    direction = rng.normal(size=features[endpoint].shape)
    direction /= np.linalg.norm(direction)

    def loss(step, delta):
        changed = {name: value.copy() for name, value in features.items()}
        changed[endpoint] += step * delta
        primal = _primal(changed['old'], changed['new'], changed['held'], priors)
        return _linear_loss(primal['held_scores'], G)

    numeric = _difference(loss, direction)
    assert float(np.sum(gradients[endpoint] * direction)) == pytest.approx(numeric, rel=2e-6, abs=2e-7)


def test_nonzero_constant_adjoint_and_omitting_g_b_is_detected():
    rng, _, _, _, _, blocks, G = _case()
    certificate = rebuild_positive_kernel_head(**blocks)
    vjp = positive_kernel_head_vjp(certificate, G)
    assert np.linalg.norm(vjp.arrays['g_b']) > 0.1
    assert np.max(np.abs(G.sum(-1))) < 1e-12  # Row class sum zero does not imply g_b=0.
    wrong_rhs = vjp.arrays['adjoint_rhs'].copy()
    wrong_rhs[-1] = 0.0
    wrong_lambda = np.linalg.solve(certificate.arrays['Q'], wrong_rhs)
    wrong_Q_bar = -0.5 * (wrong_lambda @ certificate.arrays['theta'].T + certificate.arrays['theta'] @ wrong_lambda.T)
    m = len(blocks['A'])
    wrong_A = wrong_Q_bar[:m, :m]
    direction = vjp.arrays['A'] - wrong_A
    assert np.linalg.norm(direction) > 1e-3
    direction /= np.linalg.norm(direction)

    def loss(step, delta):
        perturbed = {key: value.copy() for key, value in blocks.items()}
        perturbed['A'] += step * delta
        return _linear_loss(_primal_from_blocks(perturbed)['held_scores'], G)

    numeric = _difference(loss, direction)
    assert float(np.sum(vjp.arrays['A'] * direction)) == pytest.approx(numeric, rel=2e-6, abs=2e-7)
    assert abs(float(np.sum(wrong_A * direction)) - numeric) > 1e-3


@pytest.mark.parametrize('name', ['alpha', 'beta', 'v'])
def test_tampered_coefficients_fail(name):
    *_, blocks, _ = _case()
    certificate = rebuild_positive_kernel_head(**blocks)
    claims = _claims(certificate)
    claims[name].flat[0] += 1e-4
    with pytest.raises(MathCertificateFailure, match='Failed'):
        certify_positive_kernel_head(**blocks, **claims)


@pytest.mark.parametrize('name', ['M_O', 'M_N', 'M_H', 'R_N'])
def test_tampered_frozen_prior_or_R_N_fails_against_archived_claims(name):
    *_, blocks, _ = _case()
    certificate = rebuild_positive_kernel_head(**blocks)
    changed = {key: value.copy() for key, value in blocks.items()}
    changed[name].flat[0] += 1e-4
    with pytest.raises(MathCertificateFailure, match='Failed'):
        certify_positive_kernel_head(**changed, **_claims(certificate))


@pytest.mark.parametrize('name', ['old_scores', 'train_new_scores', 'held_scores'])
def test_tampered_stored_scores_fail(name):
    *_, blocks, _ = _case()
    certificate = rebuild_positive_kernel_head(**blocks)
    claims = _claims(certificate)
    claims[name].flat[0] += 1e-4
    with pytest.raises(MathCertificateFailure, match='reconstruction'):
        certify_positive_kernel_head(**blocks, **claims)


def test_missing_free_constant_rank_one_head_is_detected():
    *_, blocks, _ = _case()
    A, B, D, F, E, RN = (blocks[name] for name in ('A', 'B', 'D', 'F', 'E', 'R_N'))
    J = np.linalg.solve(A, B)
    alpha = np.linalg.solve(D - B.T @ J + np.eye(len(D)), RN)
    beta, v = J @ alpha, np.zeros(RN.shape[1], dtype=np.float64)
    # This counterfeit protects old points but omits the free-constant direction.
    old_residual = B @ alpha - A @ beta + v
    assert np.linalg.norm(old_residual) < 1e-12
    assert np.linalg.norm(beta.sum(0) - alpha.sum(0)) > 1e-3
    with pytest.raises(MathCertificateFailure, match='Failed'):
        certify_positive_kernel_head(**blocks, alpha=alpha, beta=beta, v=v,
            old_scores=blocks['M_O'] + old_residual,
            train_new_scores=blocks['M_N'] + D @ alpha - B.T @ beta + v,
            held_scores=blocks['M_H'] + E @ alpha - F @ beta + v)


def test_old_constraint_covers_new_class_columns_and_detects_old_only_corruption():
    *_, blocks, _ = _case()
    blocks['M_O'][:, 2:] = 0.0
    blocks['M_N'][:, 2:] = 0.0
    blocks['M_H'][:, 2:] = 0.0
    certificate = rebuild_positive_kernel_head(**blocks)
    np.testing.assert_allclose(certificate.arrays['old_residual'], 0.0, atol=2e-13)
    assert np.linalg.norm(certificate.arrays['alpha'][:, 2:]) > 0.1
    claims = _claims(certificate)
    claims['beta'][0, 2] += 1e-4
    with pytest.raises(MathCertificateFailure, match='Failed'):
        certify_positive_kernel_head(**blocks, **claims)


def test_row_and_class_permutations_commute_with_scores_coefficients_and_vjp():
    rng, O, N, H, priors, blocks, G = _case()
    old_order, new_order, held_order, class_order = (rng.permutation(n) for n in (len(O), len(N), len(H), G.shape[1]))
    perm_priors = dict(M_O=priors['M_O'][old_order][:, class_order],
                      M_N=priors['M_N'][new_order][:, class_order],
                      M_H=priors['M_H'][held_order][:, class_order],
                      R_N=priors['R_N'][new_order][:, class_order])
    perm_blocks = _blocks(O[old_order], N[new_order], H[held_order], perm_priors)
    original = rebuild_positive_kernel_head(**blocks)
    permuted = rebuild_positive_kernel_head(**perm_blocks)
    for name, rows in (('alpha', new_order), ('beta', old_order), ('old_scores', old_order),
                       ('train_new_scores', new_order), ('held_scores', held_order)):
        np.testing.assert_allclose(permuted.arrays[name], original.arrays[name][rows][:, class_order], atol=2e-12)
    np.testing.assert_allclose(permuted.arrays['v'], original.arrays['v'][class_order], atol=2e-12)
    g0 = positive_kernel_head_vjp(original, G).arrays
    gp = positive_kernel_head_vjp(permuted, G[held_order][:, class_order]).arrays
    for name, rows, cols in (('A', old_order, old_order), ('B', old_order, new_order),
                            ('D', new_order, new_order), ('F', held_order, old_order), ('E', held_order, new_order)):
        np.testing.assert_allclose(gp[name], g0[name][rows][:, cols], atol=2e-12)


def test_far_cross_kernel_counterexample_has_nonzero_v_and_changed_decision():
    blocks = dict(A=np.ones((1, 1)), B=np.zeros((1, 1)), D=np.ones((1, 1)),
                  F=np.zeros((1, 1)), E=np.zeros((1, 1)),
                  M_O=np.array([[0.2, -0.2]]), M_N=np.zeros((1, 2)),
                  M_H=np.array([[0.2, -0.2]]), R_N=np.array([[-3.0, 3.0]]))
    certificate = rebuild_positive_kernel_head(**blocks)
    np.testing.assert_allclose(certificate.arrays['v'], [-1.0, 1.0], atol=1e-13)
    np.testing.assert_allclose(certificate.arrays['old_scores'], blocks['M_O'], atol=1e-13)
    np.testing.assert_allclose(certificate.arrays['held_residual'][0], certificate.arrays['v'], atol=1e-13)
    assert np.argmax(blocks['M_H'][0]) == 0
    assert np.argmax(certificate.arrays['held_scores'][0]) == 1


def test_class_sum_records_actual_centered_outputs_and_frozen_buffers():
    *_, blocks, _ = _case()
    certificate = rebuild_positive_kernel_head(**blocks)
    for name in CLAIM_NAMES:
        assert certificate.audit[name + '_class_sum_max_abs'] < 1e-12
        assert not certificate.arrays[name].flags.writeable
    old = certificate.arrays['A'].copy()
    blocks['A'][0, 0] += 1.0
    np.testing.assert_array_equal(certificate.arrays['A'], old)
    with pytest.raises(ValueError):
        certificate.arrays['alpha'][0, 0] = 0.0
    with pytest.raises(TypeError):
        certificate.audit['jitter'] = 1.0


def test_empty_held_rows_have_full_defined_forward_and_zero_vjp():
    *_, blocks, G = _case(h=0)
    certificate = rebuild_positive_kernel_head(**blocks)
    assert certificate.arrays['held_scores'].shape == (0, 4)
    vjp = positive_kernel_head_vjp(certificate, G)
    for name in ('A', 'B', 'D', 'F', 'E', 'g_b'):
        np.testing.assert_array_equal(vjp.arrays[name], np.zeros_like(vjp.arrays[name]))
    assert vjp.audit['general_solve_count'] == 1  # Actual solve remains counted.


@pytest.mark.parametrize('name', INPUT_NAMES)
@pytest.mark.parametrize('kind', ['float32', 'nonfinite', 'wrongshape'])
def test_strict_input_dtype_finite_and_shape_checks(name, kind):
    *_, blocks, _ = _case()
    if kind == 'float32':
        blocks[name] = blocks[name].astype(np.float32)
    elif kind == 'nonfinite':
        blocks[name].flat[0] = np.nan
    else:
        blocks[name] = blocks[name][:, :-1]
    with pytest.raises(ValueError):
        rebuild_positive_kernel_head(**blocks)


@pytest.mark.parametrize('name', CLAIM_NAMES)
def test_strict_claim_shape_dtype_and_finite(name):
    *_, blocks, _ = _case()
    certificate = rebuild_positive_kernel_head(**blocks)
    claims = _claims(certificate)
    claims[name] = claims[name].astype(np.float32)
    with pytest.raises(ValueError):
        certify_positive_kernel_head(**blocks, **claims)


def test_non_spd_non_psd_asymmetry_fail_without_jitter_or_retry():
    *_, blocks, _ = _case()
    for mutation, expected in (('oldzero', 'positive definite'), ('rawnegative', 'not PSD'), ('asymmetry', 'not symmetric')):
        changed = {name: value.copy() for name, value in blocks.items()}
        if mutation == 'oldzero':
            changed['A'][:] = 0.0
        elif mutation == 'rawnegative':
            changed['D'][0, 0] = -1.0
        else:
            changed['A'][0, 1] += 0.1
        with pytest.raises(MathCertificateFailure, match=expected) as caught:
            rebuild_positive_kernel_head(**changed)
        assert caught.value.audit['jitter'] == 0.0
        assert caught.value.audit['general_solve_count'] == 0


def test_wrong_G_shape_dtype_and_nonfinite_fail():
    *_, blocks, G = _case()
    certificate = rebuild_positive_kernel_head(**blocks)
    for wrong in (G[:, :-1], G.astype(np.float32), np.full(G.shape, np.nan)):
        with pytest.raises(ValueError):
            positive_kernel_head_vjp(certificate, wrong)
