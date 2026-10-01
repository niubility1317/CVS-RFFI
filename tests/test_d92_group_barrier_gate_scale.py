"""Largest declared support shape and independent scalar oracle; synthetic only."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from cvsrffi.d92_group_barrier_gate import (
    fit_group_barrier_gate, group_barrier_gate_vjp, predict_group_barrier_gate,
)


def test_six_old_twenty_new_k20_duplicate_kernel_all_pair_ties():
    # 6*20 old physical rows and 20*20 new physical rows are all retained.
    n, m, classes = 520, 120, 20
    kernel = np.ones((n, n), dtype=np.float64)
    target = np.r_[np.ones(m), np.zeros(n-m)]
    bounds = np.zeros((m, classes), dtype=np.float64)
    state = fit_group_barrier_gate(
        K=kernel, targets=target, old_indices=np.arange(m), lower_bounds=bounds,
        max_newton_iterations=100, max_line_search_trials=64,
        max_factor_buffer_bytes=167772160,
    )
    # With identical physical features the RKHS function is a free constant.
    # Solve its independent 1D stationarity, without the implementation's Newton.
    gap = n * 1e-4
    root = brentq(lambda b: n*expit(b)-m-gap/b, 1e-10, 4., xtol=1e-15)
    L = np.ones((3, n))
    score = predict_group_barrier_gate(state, L=L)
    np.testing.assert_allclose(score, root, rtol=2e-6, atol=1e-10)
    assert state.slacks.shape == (m, classes) and np.all(state.slacks > 0)
    assert len(state.alpha) == n
    assert state.audit['constraint_count'] == m*classes
    # Exact derivative of the scalar equation with respect to EACH tied bound.
    upstream = np.array([.7, -.1, .4])
    adjoint = group_barrier_gate_vjp(state, L=L, G=upstream)
    weight = state.zeta / root**2
    scalar_hessian = n*expit(root)*expit(-root) + gap/root**2
    expected = upstream.sum() * weight/scalar_hessian
    np.testing.assert_allclose(adjoint.gradlower_bounds, expected, rtol=2e-5, atol=1e-11)
    assert state.audit['peak_factor_buffer_bytes'] == 2*n*n*8
