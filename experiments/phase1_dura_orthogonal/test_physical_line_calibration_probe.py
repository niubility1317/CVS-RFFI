import numpy as np
from experiments.phase1_dura_orthogonal.physical_line_calibration_probe import fit_beta, iq_error


def test_fit_recovers_clipped_line_minimum():
    q = np.array([1., 2.])
    g = np.array([.8, 1.6])
    p = np.array([.2, .2])
    assert abs(fit_beta(q, g, p) - .75) < 1e-12
    assert fit_beta(q, -g, p) == 0.
    assert fit_beta(q, 4*g, p) == 1.
    assert fit_beta(q, g, np.ones(2)) == 0.


def test_quadratic_iq_error_matches_direct_calculation():
    e0, q, g = np.array([4.]), np.array([2.]), np.array([1.])
    assert np.allclose(iq_error(e0, q, g, np.array([.5])), [3.5])
