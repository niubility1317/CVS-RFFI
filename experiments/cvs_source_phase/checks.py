"""Focused gauge, leakage, availability and synthetic association checks."""
import copy
import json
import numpy as np
from . import design as d
from .core import A, K, SIGNS, closures, observe, phase_audit


def metadata():
    cells = [(t, day, rx) for t in range(6) for day in (1, 2, 3) for rx in d.RX for _ in range(70)]
    return tuple(np.array([v[i] for v in cells]) for i in range(3))


def synthetic(tx, day, rx):
    generator = np.random.default_rng(20261016041)
    n = np.arange(256)
    nuisance = generator.uniform(-np.pi, np.pi, (len(tx), 1)) + generator.uniform(-2, 2, (len(tx), 1)) * K
    beta = np.array([[-4, -2, 0, 2, 4][d.RX.index(int(v))] for v in rx])
    h = np.exp(1j * (beta[:, None] * K**2 / 36 + nuisance))
    z = (h * SIGNS * (1 + 1j)) @ np.exp(2j * np.pi * K[:, None] * n / 20) / np.sqrt(24)
    z *= np.exp(2j * np.pi * 130000 * n / 25000000)[None]
    return np.stack([z.real, z.imag], 1)


def run():
    assert np.linalg.matrix_rank(A) == 10 and not np.any(A.sum(1)) and not np.any(A @ K)
    from experiments.cvs_rff_physics.known_excitation import TONES, SIGNS as old_signs
    assert np.array_equal(K, TONES // 4) and np.array_equal(SIGNS, old_signs)
    rng = np.random.default_rng(20261016043)
    h = rng.normal(size=(8, 12)) + 1j * rng.normal(size=(8, 12))
    gauge = 2 * np.exp(1j * (rng.normal(size=(8, 1)) + rng.normal(size=(8, 1)) * K))
    assert np.max(abs(closures(h) - closures(h * gauge))) < 1e-12
    tx, day, rx = metadata(); x = synthetic(tx, day, rx)
    result = observe(x, (80, 160)); a = phase_audit(result, tx, rx, day)
    assert result['eligible'].all() and a['status'] == 'PASS', a
    assert abs(result['wrapped_CFO_hz'] - 130000).max() < 1e-5
    # Removing phase curvature leaves only per-packet affine nuisance.
    flat = copy.deepcopy(result); flat['value'][:] = 1
    assert phase_audit(flat, tx, rx, day)['status'] == 'FAIL'
    # Held TX AND held day cannot enter their fold's fitted RX means.
    altered = copy.deepcopy(result)
    altered['value'][(tx == 0) | (day == 1)] *= np.exp(.7j)
    b = phase_audit(altered, tx, rx, day)
    for field in ('fit_mean_real', 'fit_mean_imag'):
        assert np.array_equal(a['folds'][0][field], b['folds'][0][field])
    sparse = copy.deepcopy(result); sparse['eligible'][(tx == 0) & (day == 1) & (rx == 1)] = False
    assert not phase_audit(sparse, tx, rx, day)['checks']['coverage']
    bad_repeat = copy.deepcopy(result); bad_repeat['split_cosine'][:] = 0
    assert not phase_audit(bad_repeat, tx, rx, day)['checks']['cycle_repeatability']
    # No information in zero/white-noise inputs qualifies merely because
    # arbitrary angles can be computed.
    assert not observe(np.zeros((4, 2, 256)), (80, 160))['eligible'].any()
    assert observe(rng.normal(size=(32, 2, 256)), (80, 160))['eligible'].mean() < .1
    failures = 0
    for c in (dict(d.config(), target_access=True), dict(d.config(), role='V'), dict(d.config(), count=5580)):
        try: d.validate(c)
        except ValueError: failures += 1
    assert failures == 3
    try: phase_audit(result, tx, np.zeros_like(rx), day)
    except ValueError: failures += 1
    assert failures == 4
    return dict(status='PASS', closure_rank=10, affine_gauge=True, reference_coefficients_match=True,
        synthetic_phase_recovery=True, TX_and_day_exclusion=True, sparse_and_noisy_controls=True,
        protocol_negatives=failures, synthetic_IQ_packets=6336, real_IQ_packets=0,
        network_forwards=0, checkpoint_loads=0)


if __name__ == '__main__':
    print(json.dumps(run()))
