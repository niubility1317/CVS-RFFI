import numpy as np
import pytest

from experiments.pamf_lora_main import common_failure_source as m
from experiments.pamf_lora_main.physical_inverse import real_operator


def test_inverse_uses_same_damping_across_exact_doses():
    source = np.array([1+2j, 2-1j])
    gain = 2.
    signal = gain*source
    noise = np.array([.1+.3j, .4-.1j])
    action = real_operator(lambda x: gain*x, lambda y: gain*y, (2, 2))
    observed = signal+noise
    ridge = 3*np.mean(np.abs(noise)**2)
    solved = m.isolated_inverse(action, observed, signal, noise)
    expected = [gain*(signal+remainder*noise)/(gain**2+ridge) for remainder in (1, .2, 0)]
    np.testing.assert_allclose(solved, expected, rtol=1e-7)


def test_diagnostic_clean_is_identical_for_all_arms(monkeypatch):
    clean = np.arange(6*2*256).reshape(6, 2, 256)
    monkeypatch.setattr(m, 'chirp_windows', lambda value: value)
    for route in m.ROUTES:
        waves, error = m.diagnostic_capture(route, 'clean', (clean, 'source', 'id'))
        assert error == 0
        np.testing.assert_array_equal(waves, np.broadcast_to(clean, (1+len(m.ARMS), *clean.shape)))


def test_cohort_only_includes_clean_correct_shared_failures():
    n = len(m.SOURCE)
    labels = np.zeros(n, dtype=int)
    new = {}
    old = {}
    for route in m.ROUTES:
        rows = {v: np.ones((len(m.MODES), n), dtype=int) for v in m.VIEWS}
        history = {v: np.ones((len(m.HIST_MODES), n), dtype=int) for v in m.VIEWS}
        rows['clean'][0, :-1] = 0
        history['clean'][0, :-1] = 0
        rows[m.VIEWS[-1]][m.MODES.index('noise100'), 1:] = 0
        new[route] = (None, rows)
        old[route] = (None, history)
    old['full'][1][m.VIEWS[-1]][m.HIST_MODES.index('rank_half'), 0] = 0
    res = m.cohort_statistics(new, old, labels, sections=1)
    low = res['residual']['shared_by_view'][m.VIEWS[-1]]
    assert low['n'] == n-2
    assert low['correct']['noise100'] == low['noise_only'] == n-2
    assert res['residual']['local_by_view'][m.VIEWS[-1]]['n'] == n-1
    assert low['by_rx'][m.ARMS.index('noise100')] == [0, 1, 1, 1, 0]

    new['full'][1][m.VIEWS[-1]][0, 0] = 0
    with pytest.raises(ValueError, match='Historical full'):
        m.cohort_statistics(new, old, labels, sections=1)
