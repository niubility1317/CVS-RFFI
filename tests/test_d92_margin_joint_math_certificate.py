"""Synthetic certificate for the margin-joint derivation, not a method implementation.

This file imports no candidate, feature encoder, dataset, or experiment artifact.
The feature-primal oracle and enumerated nonnegative dual are deliberately small.
They certify equations only; they do not prescribe a production QP solver.
"""

from pathlib import Path

# A static-only route: no NumPy import or numerical test runs in this branch.
if __name__ == "__main__":
    import ast

    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    print("AST_UTF8_OK test_d92_margin_joint_math_certificate.py")
    raise SystemExit(0)

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pytest


KKT_ATOL = 2.0e-9
REGULAR_ATOL = 1.0e-8


def _vec(value):
    return np.asarray(value, dtype=np.float64).reshape(-1, order="F")


def _unvec(value, n, classes):
    return np.asarray(value, dtype=np.float64).reshape((n, classes), order="F")


def _sym(value):
    return (value + value.T) * 0.5


def _all_subsets(size):
    for count in range(size + 1):
        yield from combinations(range(size), count)


def _centered_labels(labels, classes):
    return np.eye(classes, dtype=np.float64)[np.asarray(labels)] - 1.0 / classes


def _constraints(prior, labels, old_rows):
    """Construct ALL old true-vs-other rows in column-major score coordinates."""
    n, classes = prior.shape
    rows, lower_bounds, pairs, old_margins = [], [], [], []
    for row in old_rows:
        truth = int(labels[row])
        competitors = [col for col in range(classes) if col != truth]
        margin = min(prior[row, truth] - prior[row, col] for col in competitors)
        old_margins.append(margin)
        for col in competitors:
            difference = np.zeros(n * classes, dtype=np.float64)
            difference[row + n * truth] = 1.0
            difference[row + n * col] = -1.0
            rows.append(difference)
            lower_bounds.append(margin)
            pairs.append((int(row), truth, col))
    return (
        np.asarray(rows), np.asarray(lower_bounds), tuple(pairs),
        np.asarray(old_margins),
    )


def _spd(value):
    """Rank check for a tiny oracle, never a jitter or pseudoinverse fallback."""
    if value.size == 0:
        return True
    eigenvalues = np.linalg.eigvalsh(_sym(value))
    scale = max(1.0, float(np.linalg.norm(value, ord=2)))
    return bool(eigenvalues[0] > 128.0 * np.finfo(np.float64).eps * scale)


def _kkt_feasible(mu, slack):
    return (
        np.min(mu, initial=0.0) >= -KKT_ATOL
        and np.min(slack, initial=0.0) >= -KKT_ATOL
        and np.max(np.abs(mu * slack), initial=0.0) <= KKT_ATOL
    )


def _feature_primal(phi, residual_target, prior, difference, delta):
    """Independent primal QP in [finite-feature weights; free intercept].

    No J, P, kernel coefficient, Schur elimination, or dual Q is used here.
    Dependent active sets are skipped, not regularized.  A minimal independent
    active representation exists for the small polyhedral examples below.
    """
    n, features = phi.shape
    classes = residual_target.shape[1]
    design = np.column_stack((phi, np.ones(n)))
    penalty = np.diag(np.r_[np.ones(features), 0.0])
    hessian = design.T @ design + penalty
    assert _spd(hessian)
    block_hessian = np.kron(np.eye(classes), hessian)
    linear = _vec(design.T @ residual_target)
    score_map = np.kron(np.eye(classes), design)
    inequalities = difference @ score_map
    lower = delta - difference @ _vec(prior)
    q = difference.shape[0]
    solutions = []
    for indices in _all_subsets(q):
        active = np.asarray(indices, dtype=np.int64)
        constraint = inequalities[active]
        if len(active):
            gram = constraint @ np.linalg.solve(block_hessian, constraint.T)
            if not _spd(gram):
                continue
            system = np.block([
                [block_hessian, -constraint.T],
                [constraint, np.zeros((len(active), len(active)))],
            ])
            answer = np.linalg.solve(system, np.r_[linear, lower[active]])
            parameter = answer[:len(linear)]
            active_mu = answer[len(linear):]
        else:
            parameter = np.linalg.solve(block_hessian, linear)
            active_mu = np.empty(0)
        mu = np.zeros(q)
        mu[active] = active_mu
        slack = inequalities @ parameter - lower
        if not _kkt_feasible(mu, slack):
            continue
        theta = _unvec(parameter, features + 1, classes)
        residual = design @ theta
        objective = 0.5 * np.sum((residual - residual_target) ** 2)
        objective += 0.5 * np.sum(theta[:-1] ** 2)
        stationarity = block_hessian @ parameter - linear - inequalities.T @ mu
        np.testing.assert_allclose(stationarity, 0.0, atol=KKT_ATOL, rtol=0.0)
        solutions.append(dict(
            theta=theta, residual=residual, objective=float(objective),
            mu=mu, slack=slack, hessian=hessian,
        ))
    assert solutions, "Synthetic primal oracle found no feasible KKT solution"
    best = min(solutions, key=lambda item: item["objective"])
    # Strict convexity certifies the same feature parameter, even if mu differs.
    for solution in solutions:
        np.testing.assert_allclose(solution["theta"], best["theta"], atol=KKT_ATOL)
    return best


def _feature_equality(phi, residual_target, old_rows):
    """Independent Conditional subset oracle r(O)=0, with the SAME objective."""
    n, features = phi.shape
    classes = residual_target.shape[1]
    design = np.column_stack((phi, np.ones(n)))
    hessian = design.T @ design + np.diag(np.r_[np.ones(features), 0.0])
    block_hessian = np.kron(np.eye(classes), hessian)
    equality = np.kron(np.eye(classes), design[np.asarray(old_rows)])
    assert _spd(equality @ np.linalg.solve(block_hessian, equality.T))
    system = np.block([
        [block_hessian, equality.T],
        [equality, np.zeros((len(equality), len(equality)))],
    ])
    answer = np.linalg.solve(system, np.r_[_vec(design.T @ residual_target),
                                          np.zeros(len(equality))])
    theta = _unvec(answer[:block_hessian.shape[0]], features + 1, classes)
    residual = design @ theta
    objective = 0.5 * np.sum((residual - residual_target) ** 2)
    objective += 0.5 * np.sum(theta[:-1] ** 2)
    np.testing.assert_allclose(residual[np.asarray(old_rows)], 0.0, atol=KKT_ATOL)
    return residual, float(objective)


@dataclass(frozen=True)
class _Head:
    K: np.ndarray
    L: np.ndarray
    R: np.ndarray
    M: np.ndarray
    M_H: np.ndarray
    D: np.ndarray
    delta: np.ndarray
    J: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    z: np.ndarray
    s: float
    s0: np.ndarray
    mu: np.ndarray
    alpha: np.ndarray
    b: np.ndarray
    residual: np.ndarray
    train_scores: np.ndarray
    held_scores: np.ndarray
    slack: np.ndarray
    active: tuple
    objective: float


def _dual_head(K, L, residual_target, prior, difference, delta, prior_held):
    """Small nonnegative dual enumeration, independent of the feature oracle."""
    n, classes = residual_target.shape
    assert K.shape == (n, n) and L.shape[1] == n
    np.testing.assert_allclose(K, K.T, atol=1.0e-14, rtol=0.0)
    assert np.linalg.eigvalsh(K)[0] >= -1.0e-12
    A = np.eye(n) + K
    z = np.linalg.solve(A, np.ones(n))
    s = float(np.sum(z))
    J = np.linalg.solve(A, np.eye(n)) - np.outer(z, z) / s
    P = np.eye(n) - J
    Q = difference @ np.kron(np.eye(classes), P) @ difference.T
    s0 = difference @ _vec(prior + P @ residual_target) - delta
    candidates = []
    for indices in _all_subsets(len(s0)):
        active = np.asarray(indices, dtype=np.int64)
        mu = np.zeros(len(s0))
        if len(active):
            block = Q[np.ix_(active, active)]
            if not _spd(block):
                continue
            mu[active] = np.linalg.solve(block, -s0[active])
        slack = s0 + Q @ mu
        if _kkt_feasible(mu, slack):
            value = 0.5 * mu @ Q @ mu + s0 @ mu
            candidates.append((float(value), mu))
    assert candidates, "Synthetic dual oracle found no feasible KKT solution"
    mu = min(candidates, key=lambda item: item[0])[1]
    E = residual_target + _unvec(difference.T @ mu, n, classes)
    saddle = np.block([[A, np.ones((n, 1))],
                       [np.ones((1, n)), np.zeros((1, 1))]])
    answer = np.linalg.solve(saddle, np.vstack((E, np.zeros((1, classes)))))
    alpha, b = answer[:-1], answer[-1]
    residual = K @ alpha + b
    train_scores = prior + residual
    held_scores = prior_held + L @ alpha + b
    slack = difference @ _vec(train_scores) - delta
    objective = 0.5 * np.sum((residual - residual_target) ** 2)
    objective += 0.5 * np.sum(alpha * (K @ alpha))
    np.testing.assert_allclose(alpha, J @ E, atol=KKT_ATOL)
    np.testing.assert_allclose(b, z @ E / s, atol=KKT_ATOL)
    np.testing.assert_allclose(residual, P @ E, atol=KKT_ATOL)
    np.testing.assert_allclose(np.sum(alpha, axis=0), 0.0, atol=KKT_ATOL)
    assert _kkt_feasible(mu, slack)
    return _Head(
        K=K, L=L, R=residual_target, M=prior, M_H=prior_held,
        D=difference, delta=delta, J=J, P=P, Q=Q, z=z, s=s,
        s0=s0, mu=mu, alpha=alpha, b=b, residual=residual,
        train_scores=train_scores, held_scores=held_scores, slack=slack,
        active=tuple(np.flatnonzero(mu > REGULAR_ATOL)),
        objective=float(objective),
    )


def _regular_vjp(head, G, *, include_intercept=True, include_multiplier=True):
    """Equation (22)/(25), with explicit regular-region rejection.

    The two optional omissions exist ONLY as incorrect negative controls.
    Neither is a permissible derivative for a candidate implementation.
    """
    n, classes = head.R.shape
    active = np.asarray(head.active, dtype=np.int64)
    inactive = np.setdiff1d(np.arange(len(head.mu)), active)
    if len(active):
        block = head.Q[np.ix_(active, active)]
        if not _spd(block):
            raise ValueError("singular active Jacobian: no jitter/pseudoinverse")
        if np.min(head.mu[active]) <= REGULAR_ATOL:
            raise ValueError("active switch: strict complementarity unavailable")
    if len(inactive) and np.min(head.slack[inactive]) <= REGULAR_ATOL:
        raise ValueError("active switch: strict complementarity unavailable")
    A = np.eye(n) + head.K
    g_b = np.sum(G, axis=0)
    if not include_intercept:
        g_b = np.zeros(classes)
    B_G = head.L.T @ G
    t_G = (head.z @ B_G - g_b) / head.s
    T_G = np.linalg.solve(A, B_G) - head.z[:, None] * t_G
    eta = np.zeros(len(active))
    B_eta = np.zeros_like(head.R)
    if len(active) and include_multiplier:
        eta = np.linalg.solve(block, head.D[active] @ _vec(T_G))
        B_eta = _unvec(head.D[active].T @ eta, n, classes)
    W_eta = head.J @ B_eta
    delta_bar = np.zeros_like(head.delta)
    delta_bar[active] = eta
    return dict(
        K=-_sym((T_G + W_eta) @ head.alpha.T),
        L=G @ head.alpha.T,
        R=T_G - head.P @ B_eta,
        M=-B_eta,
        delta=delta_bar,
        M_H=G.copy(), T_G=T_G, t_G=t_G, g_b=g_b,
        eta=eta, W_eta=W_eta,
    )


def _repeat_head(head, **changes):
    inputs = dict(K=head.K, L=head.L, residual_target=head.R, prior=head.M,
                  difference=head.D, delta=head.delta, prior_held=head.M_H)
    inputs.update(changes)
    return _dual_head(**inputs)


def _directional_difference(head, G, plus, minus, epsilon=2.0e-6):
    high = _repeat_head(head, **plus)
    low = _repeat_head(head, **minus)
    assert high.active == low.active == head.active
    # Prove the chosen perturbation did not cross the differentiability boundary.
    _regular_vjp(high, G)
    _regular_vjp(low, G)
    return float(np.sum(G * (high.held_scores - low.held_scores)) / (2 * epsilon))


def _three_class_fixture():
    phi = np.array([[1.0, 0.2, 0.0], [-0.3, 0.8, 0.4],
                    [0.2, -0.5, 0.9], [0.7, 0.1, -0.2]])
    psi = np.array([[0.3, 0.4, -0.2], [-0.1, 0.5, 0.6]])
    prior = np.array([[1.4, -0.5, 0.0], [-0.4, 1.2, 0.0],
                      [0.2, -0.1, 0.0], [0.4, -0.2, 0.0]])
    labels = np.array([0, 1, 2, 2])
    old = np.array([0, 1])
    D, delta, _, _ = _constraints(prior, labels, old)
    R = _centered_labels(labels, 3) - prior
    prior_held = np.array([[0.3, -0.2, 0.0], [0.1, -0.3, 0.0]])
    return phi, psi, prior, R, D, delta, prior_held, old


def _active_fixture():
    # One old constraint with strictly positive multiplier; full K is SPD.
    K = np.array([[1.4, 0.25], [0.25, 0.9]])
    L = np.array([[0.3, 0.8], [-0.2, 0.4]])
    prior = np.array([[2.0, 0.0], [0.7, 0.0]])
    R = _centered_labels(np.array([0, 1]), 2) - prior
    D, delta, _, _ = _constraints(prior, np.array([0, 1]), [0])
    prior_held = np.array([[0.2, 0.0], [-0.1, 0.0]])
    head = _dual_head(K, L, R, prior, D, delta, prior_held)
    assert head.active == (0,) and head.mu[0] > 0.1
    # Every row sums to zero, but the intercept upstream is NONZERO.
    G = np.array([[0.7, -0.7], [0.3, -0.3]])
    return head, G


def _multi_active_fixture():
    # Both true-vs-other rows are active, and Q_II has a nonzero off-diagonal.
    K = np.array([[1.4, 0.25], [0.25, 0.9]])
    L = np.array([[0.3, 0.8], [-0.2, 0.4]])
    M = np.array([[2.0, 0.1, 0.0], [0.7, -0.2, 0.0]])
    labels = np.array([0, 2])
    R = _centered_labels(labels, 3) - M
    D, delta, _, _ = _constraints(M, labels, [0])
    M_H = np.array([[0.2, -0.05, 0.0], [-0.1, 0.1, 0.0]])
    head = _dual_head(K, L, R, M, D, delta, M_H)
    assert head.active == (0, 1) and np.min(head.mu) > 0.1
    assert abs(head.Q[0, 1]) > 0.1
    G = np.array([[0.4, -0.1, -0.3], [0.2, 0.3, -0.5]])
    return head, G


@pytest.mark.parametrize("phi", [
    np.array([[1.0, 0.2], [1.0, 0.2], [-0.3, 0.4]]),
    np.zeros((3, 0)),
    np.array([[0.4, -0.2]]),
])
def test_feature_hessian_strict_convexity_and_affine_response(phi):
    n, features = phi.shape
    X = np.column_stack((phi, np.ones(n)))
    H0 = X.T @ X + np.diag(np.r_[np.ones(features), 0.0])
    assert _spd(H0)
    K = phi @ phi.T
    A = np.eye(n) + K
    z = np.linalg.solve(A, np.ones(n))
    J = np.linalg.solve(A, np.eye(n)) - np.outer(z, z) / np.sum(z)
    P = np.eye(n) - J
    np.testing.assert_allclose(P, X @ np.linalg.solve(H0, X.T), atol=2.0e-12)
    np.testing.assert_allclose(J @ np.ones(n), 0.0, atol=2.0e-12)
    np.testing.assert_allclose(P @ np.ones(n), 1.0, atol=2.0e-12)
    assert np.linalg.eigvalsh(P)[0] >= -2.0e-12
    if n == 1:
        np.testing.assert_allclose(J, 0.0, atol=2.0e-12)
        np.testing.assert_allclose(P, np.ones((1, 1)), atol=2.0e-12)


def test_feature_primal_matches_nonnegative_dual_full_saddle_and_gap():
    phi, psi, M, R, D, delta, M_H, _ = _three_class_fixture()
    primal = _feature_primal(phi, R, M, D, delta)
    head = _dual_head(phi @ phi.T, psi @ phi.T, R, M, D, delta, M_H)
    np.testing.assert_allclose(primal["residual"], head.residual, atol=KKT_ATOL)
    np.testing.assert_allclose(primal["theta"][-1], head.b, atol=KKT_ATOL)
    np.testing.assert_allclose(primal["theta"][:-1], phi.T @ head.alpha,
                               atol=KKT_ATOL)
    np.testing.assert_allclose(M_H + psi @ primal["theta"][:-1]
                               + primal["theta"][-1], head.held_scores,
                               atol=KKT_ATOL)
    assert primal["objective"] == pytest.approx(head.objective, abs=KKT_ATOL)
    # Strong duality and its sign are checked against the independent objective.
    alpha0 = head.J @ R
    b0 = head.z @ R / head.s
    residual0 = head.K @ alpha0 + b0
    J0 = 0.5 * np.sum((residual0 - R) ** 2)
    J0 += 0.5 * np.sum(alpha0 * (head.K @ alpha0))
    dual_value = J0 - head.s0 @ head.mu - 0.5 * head.mu @ head.Q @ head.mu
    assert head.objective == pytest.approx(dual_value, abs=KKT_ATOL)
    assert head.objective - dual_value == pytest.approx(head.mu @ head.slack,
                                                       abs=KKT_ATOL)


def test_zero_residual_feasible_and_conditional_is_same_objective_subset():
    phi, psi, M, R, D, delta, M_H, old = _three_class_fixture()
    h = delta - D @ _vec(M)
    assert np.max(h) <= 0.0
    # Zero g/b is feasible independently of the current kernel.
    assert np.min(D @ _vec(np.zeros_like(M)) - h) >= 0.0
    head = _dual_head(phi @ phi.T, psi @ phi.T, R, M, D, delta, M_H)
    equality_residual, equality_objective = _feature_equality(phi, R, old)
    assert np.min(D @ _vec(M + equality_residual) - delta) >= -KKT_ATOL
    assert head.objective <= equality_objective + KKT_ATOL
    assert head.objective <= 0.5 * np.sum(R ** 2) + KKT_ATOL
    # The comparison concerns this regularized fixed-kernel objective only.


def test_original_positive_zero_negative_margin_and_all_registered_competitors():
    M = np.array([[2.0, 1.0, 0.0, 0.0],
                  [1.0, 1.0, -0.2, 0.0],
                  [2.0, 1.0, 0.0, 0.0]])
    labels = np.array([0, 1, 2])
    D, delta, pairs, margins = _constraints(M, labels, [0, 1, 2])
    np.testing.assert_array_equal(margins, [1.0, 0.0, -2.0])
    assert len(pairs) == 3 * (4 - 1)
    assert all((row, int(labels[row]), 3) in pairs for row in range(3))
    assert np.min(D @ _vec(M) - delta) >= 0.0  # r=0, including d<0.
    f = np.array([[3.0, 2.0, 0.0, 0.0],
                  [2.0, 2.0, -0.2, 0.0],
                  [1.5, 2.0, 0.0, 0.0]])
    assert np.min(D @ _vec(f) - delta) >= 0.0
    assert np.argmax(f[0]) == labels[0]  # Strictly positive minimum margin.
    assert f[1, labels[1]] == np.max(f[1])
    assert np.argmax(f[1]) != labels[1]  # Tie-break is NOT a correctness theorem.
    assert np.argmax(M[2]) == 0 and np.argmax(f[2]) == 1
    assert f[2, 2] - f[2, 1] < M[2, 2] - M[2, 1]
    assert np.min(D @ _vec(M) - np.maximum(delta, 0.0)) < 0.0


def test_multiplier_sign_and_intercept_are_recomputed_jointly():
    head, _ = _active_fixture()
    assert head.s0[0] < 0.0 and head.mu[0] > 0.0
    assert (head.Q @ head.mu)[0] > 0.0
    assert head.slack[0] == pytest.approx(0.0, abs=KKT_ATOL)
    ordinary_b = head.z @ head.R / head.s
    assert np.linalg.norm(head.b - ordinary_b) > 1.0e-3
    phi = np.linalg.cholesky(head.K)
    primal = _feature_primal(phi, head.R, head.M, head.D, head.delta)
    np.testing.assert_allclose(primal["theta"][-1], head.b, atol=KKT_ATOL)


def test_singular_kernel_has_unique_function_not_unique_representer_coefficients():
    phi = np.array([[0.5], [0.5], [-0.2]])  # Duplicate physical observations remain.
    psi = np.array([[0.8], [-0.4]])
    M = np.zeros((3, 2))
    R = _centered_labels(np.array([0, 1, 1]), 2)
    D, delta, _, _ = _constraints(M, np.array([0, 1, 1]), [0, 1])
    head = _dual_head(phi @ phi.T, psi @ phi.T, R, M, D, delta, np.zeros((2, 2)))
    primal = _feature_primal(phi, R, M, D, delta)
    np.testing.assert_allclose(primal["residual"], head.residual, atol=KKT_ATOL)
    np.testing.assert_allclose(primal["theta"][:-1], phi.T @ head.alpha,
                               atol=KKT_ATOL)
    null = np.array([1.0, -1.0, 0.0])
    shift = np.outer(null, np.array([0.4, -0.4]))
    alternative_alpha = head.alpha + shift
    assert np.linalg.norm(alternative_alpha - head.alpha) > 0.1
    np.testing.assert_allclose(head.K @ null, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(head.L @ null, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(head.K @ alternative_alpha + head.b,
                               head.residual, atol=KKT_ATOL)
    np.testing.assert_allclose(head.L @ alternative_alpha + head.b,
                               head.held_scores, atol=KKT_ATOL)
    np.testing.assert_allclose(phi.T @ alternative_alpha, primal["theta"][:-1],
                               atol=KKT_ATOL)
    assert np.sum(alternative_alpha * (head.K @ alternative_alpha)) == pytest.approx(
        np.sum(head.alpha * (head.K @ head.alpha)), abs=KKT_ATOL)
    # An arbitrary inconsistent cross matrix has no function-equivalence guarantee.
    assert np.linalg.norm(np.array([[1.0, 0.0, 0.0]]) @ shift) > 0.1


def test_zero_kernel_nonunique_dual_unique_free_intercept_and_singular_rejection():
    n, classes = 3, 2
    K = np.zeros((n, n))
    M = np.zeros((n, classes))
    R = _centered_labels(np.array([0, 1, 1]), classes)
    D, delta, _, _ = _constraints(M, np.array([0, 1, 1]), [0, 1])
    head = _dual_head(K, np.zeros((2, n)), R, M, D, delta, np.zeros((2, classes)))
    np.testing.assert_allclose(head.P, np.ones((n, n)) / n, atol=1.0e-14)
    np.testing.assert_allclose(head.residual, 0.0, atol=KKT_ATOL)
    np.testing.assert_allclose(head.b, 0.0, atol=KKT_ATOL)
    primal = _feature_primal(np.zeros((n, 1)), R, M, D, delta)
    np.testing.assert_allclose(primal["theta"], 0.0, atol=KKT_ATOL)
    mu1 = np.array([0.5, 0.0])
    mu2 = np.array([1.75, 1.25])
    assert _kkt_feasible(mu1, head.s0 + head.Q @ mu1)
    assert _kkt_feasible(mu2, head.s0 + head.Q @ mu2)
    E1 = R + _unvec(D.T @ mu1, n, classes)
    E2 = R + _unvec(D.T @ mu2, n, classes)
    alpha1, alpha2 = head.J @ E1, head.J @ E2
    assert np.linalg.norm(alpha1 - alpha2) > 1.0
    np.testing.assert_allclose(K @ alpha1, K @ alpha2, atol=1.0e-14)
    np.testing.assert_allclose(head.z @ E1 / head.s, head.z @ E2 / head.s,
                               atol=KKT_ATOL)
    assert np.linalg.matrix_rank(head.Q) == 1
    # Even strictly positive multipliers do not make a dependent active block SPD.
    singular_head = replace(head, mu=mu2, alpha=alpha2, active=(0, 1))
    with pytest.raises(ValueError, match="singular active Jacobian"):
        _regular_vjp(singular_head, np.ones((2, classes)))


@pytest.mark.parametrize("fixture", [_active_fixture, _multi_active_fixture])
def test_regular_full_vjp_real_symmetric_kernel_and_cross_finite_difference(fixture):
    head, G = fixture()
    adjoint = _regular_vjp(head, G)
    dK = np.array([[0.17, -0.11], [-0.11, 0.23]])
    dL = np.array([[0.13, -0.07], [-0.09, 0.16]])
    epsilon = 2.0e-6
    assert _spd(head.K + epsilon * dK) and _spd(head.K - epsilon * dK)
    # SPD train Gram makes each perturbed L a valid finite-feature cross block:
    # Phi=chol(K), Psi=L Phi^{-T}, and held Gram=Psi Psi^T.
    for sign in (-1.0, 1.0):
        K = head.K + sign * epsilon * dK
        L = head.L + sign * epsilon * dL
        phi = np.linalg.cholesky(K)
        psi = np.linalg.solve(phi, L.T).T
        joint_gram = np.block([[K, L.T], [L, psi @ psi.T]])
        assert np.linalg.eigvalsh(joint_gram)[0] >= -2.0e-12
        primal = _feature_primal(phi, head.R, head.M, head.D, head.delta)
        changed = _repeat_head(head, K=K, L=L)
        np.testing.assert_allclose(primal["residual"], changed.residual, atol=KKT_ATOL)
        np.testing.assert_allclose(head.M_H + psi @ primal["theta"][:-1]
                                   + primal["theta"][-1], changed.held_scores,
                                   atol=KKT_ATOL)
    finite = _directional_difference(
        head, G, dict(K=head.K + epsilon * dK, L=head.L + epsilon * dL),
        dict(K=head.K - epsilon * dK, L=head.L - epsilon * dL), epsilon,
    )
    analytic = np.sum(adjoint["K"] * dK) + np.sum(adjoint["L"] * dL)
    assert finite == pytest.approx(analytic, rel=3.0e-6, abs=2.0e-8)
    np.testing.assert_allclose(adjoint["K"], adjoint["K"].T, atol=1.0e-14)
    assert adjoint["K"].shape == head.K.shape
    assert adjoint["L"].shape == head.L.shape


@pytest.mark.parametrize("omission", ["intercept", "multiplier"])
def test_free_intercept_and_multiplier_derivatives_are_both_necessary(omission):
    head, G = _active_fixture()
    good = _regular_vjp(head, G)
    np.testing.assert_allclose(np.sum(G, axis=1), 0.0, atol=1.0e-14)
    assert np.linalg.norm(good["g_b"]) > 1.0
    assert np.linalg.norm(good["W_eta"]) > 1.0e-4
    bad = _regular_vjp(head, G, include_intercept=(omission != "intercept"),
                       include_multiplier=(omission != "multiplier"))
    direction = good["K"] - bad["K"]
    gap = np.linalg.norm(direction)
    assert gap > 1.0e-4
    direction /= gap
    epsilon = 2.0e-6
    finite = _directional_difference(
        head, G, dict(K=head.K + epsilon * direction),
        dict(K=head.K - epsilon * direction), epsilon,
    )
    expected = np.sum(good["K"] * direction)
    incorrect = np.sum(bad["K"] * direction)
    assert finite == pytest.approx(expected, rel=3.0e-6, abs=2.0e-8)
    assert abs(finite - incorrect) > gap * 0.9


def test_regular_vjp_independent_rhs_prior_threshold_and_held_prior_inputs():
    head, G = _active_fixture()
    adjoint = _regular_vjp(head, G)
    dR = np.array([[0.08, -0.03], [-0.04, 0.07]])
    dM = np.array([[-0.05, 0.06], [0.03, -0.02]])
    ddelta = np.array([0.09])
    dM_H = np.array([[0.02, -0.04], [-0.03, 0.05]])
    epsilon = 2.0e-6
    finite = _directional_difference(
        head, G,
        dict(residual_target=head.R + epsilon * dR, prior=head.M + epsilon * dM,
             delta=head.delta + epsilon * ddelta,
             prior_held=head.M_H + epsilon * dM_H),
        dict(residual_target=head.R - epsilon * dR, prior=head.M - epsilon * dM,
             delta=head.delta - epsilon * ddelta,
             prior_held=head.M_H - epsilon * dM_H), epsilon,
    )
    expected = np.sum(adjoint["R"] * dR) + np.sum(adjoint["M"] * dM)
    expected += adjoint["delta"] @ ddelta + np.sum(adjoint["M_H"] * dM_H)
    assert finite == pytest.approx(expected, rel=3.0e-6, abs=2.0e-8)
    # These are independent oracle inputs; the proposed C path freezes them.


def test_residual_target_Y_minus_M_chain_has_the_full_prior_adjoint():
    head, G = _multi_active_fixture()
    adjoint = _regular_vjp(head, G)
    dY = np.array([[0.07, -0.03, -0.04], [-0.02, 0.05, -0.03]])
    dM = np.array([[0.04, -0.06, 0.02], [-0.03, 0.02, 0.01]])
    epsilon = 2.0e-6
    finite = _directional_difference(
        head, G,
        dict(residual_target=head.R + epsilon * (dY - dM),
             prior=head.M + epsilon * dM),
        dict(residual_target=head.R - epsilon * (dY - dM),
             prior=head.M - epsilon * dM), epsilon,
    )
    prior_bar = -(adjoint["T_G"] + adjoint["W_eta"])
    np.testing.assert_allclose(adjoint["M"] - adjoint["R"], prior_bar,
                               atol=2.0e-12)
    expected = np.sum(adjoint["R"] * dY) + np.sum(prior_bar * dM)
    assert finite == pytest.approx(expected, rel=3.0e-6, abs=2.0e-8)
    # delta is deliberately fixed.  Differentiating min(B margins) is outside
    # this candidate's frozen actual-B contract and outside this certificate.


def test_no_active_constraint_reduces_to_full_affine_companion():
    K = np.array([[1.0, 0.2], [0.2, 0.8]])
    L = np.array([[0.3, 0.6]])
    M = np.zeros((2, 2))
    R = np.array([[0.5, -0.5], [0.5, -0.5]])
    D, delta, _, _ = _constraints(M, np.array([0, 0]), [0])
    head = _dual_head(K, L, R, M, D, delta, np.zeros((1, 2)))
    assert head.active == () and np.min(head.slack) > REGULAR_ATOL
    G = np.array([[0.4, -0.4]])
    adjoint = _regular_vjp(head, G)
    assert adjoint["eta"].shape == (0,)
    np.testing.assert_allclose(adjoint["W_eta"], 0.0, atol=1.0e-14)
    saddle = np.block([[np.eye(2) + K, np.ones((2, 1))],
                       [np.ones((1, 2)), np.zeros((1, 1))]])
    answer = np.linalg.solve(saddle, np.vstack((L.T @ G, np.sum(G, axis=0))))
    np.testing.assert_allclose(adjoint["T_G"], answer[:-1], atol=2.0e-12)
    np.testing.assert_allclose(adjoint["t_G"], answer[-1], atol=2.0e-12)
    np.testing.assert_allclose(adjoint["K"], -_sym(answer[:-1] @ head.alpha.T),
                               atol=2.0e-12)


def test_active_switch_does_not_claim_a_unique_regular_jacobian():
    K = np.eye(2)
    M = np.zeros((2, 2))
    R = np.zeros((2, 2))
    D, delta, _, _ = _constraints(M, np.array([0, 1]), [0])
    head = _dual_head(K, np.zeros((1, 2)), R, M, D, delta, np.zeros((1, 2)))
    assert head.mu[0] == pytest.approx(0.0) and head.slack[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="active switch"):
        _regular_vjp(head, np.array([[0.5, -0.5]]))
    with pytest.raises(ValueError, match="active switch"):
        _regular_vjp(replace(head, active=(0,)), np.array([[0.5, -0.5]]))


def test_new_zero_resolving_margin_head_is_not_exact_actual_B_reuse():
    # Balanced zero-class-sum prior: score changes are not just a common offset.
    M = np.array([[0.5, 0.0, -0.5], [-0.5, 0.5, 0.0], [0.0, -0.5, 0.5]])
    labels = np.array([0, 1, 2])
    Y = _centered_labels(labels, 3)
    D, delta, _, _ = _constraints(M, labels, [0, 1, 2])
    R = Y - M
    head = _dual_head(np.eye(3), np.eye(3), R, M, D, delta, M.copy())
    assert np.min(head.slack) >= -KKT_ATOL
    assert np.linalg.norm(head.train_scores - M) > 0.1
    np.testing.assert_allclose(np.sum(head.train_scores, axis=1), 0.0, atol=KKT_ATOL)
    assert head.objective < 0.5 * np.sum(R ** 2) - 1.0e-3
    # Thus a new0 protocol cannot obtain identity by re-solving this QP, even
    # when its optimizer is unique and its old margins are protected.  It must
    # return the actual B object directly; this file has no production router
    # whose object identity/serialization could be tested.


def test_row_and_class_permutations_preserve_the_unique_primal_function():
    phi, psi, M, R, D, delta, M_H, old = _three_class_fixture()
    labels = np.array([0, 1, 2, 2])
    head = _dual_head(phi @ phi.T, psi @ phi.T, R, M, D, delta, M_H)
    row_order = np.array([3, 1, 0, 2])
    class_order = np.array([2, 0, 1])
    class_inverse = np.argsort(class_order)
    new_labels = class_inverse[labels[row_order]]
    new_old = np.flatnonzero(np.isin(row_order, old))
    permuted_M = M[row_order][:, class_order]
    permuted_D, permuted_delta, _, _ = _constraints(permuted_M, new_labels, new_old)
    K = head.K[np.ix_(row_order, row_order)]
    L = head.L[:, row_order]
    other = _dual_head(K, L, R[row_order][:, class_order], permuted_M,
                       permuted_D, permuted_delta, M_H[:, class_order])
    np.testing.assert_allclose(other.train_scores,
                               head.train_scores[row_order][:, class_order],
                               atol=KKT_ATOL)
    np.testing.assert_allclose(other.held_scores, head.held_scores[:, class_order],
                               atol=KKT_ATOL)
    assert other.objective == pytest.approx(head.objective, abs=KKT_ATOL)
