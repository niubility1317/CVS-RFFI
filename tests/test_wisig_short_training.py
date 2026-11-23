import numpy as np
import pytest

from experiments.cvs_rff_physics.known_excitation import template
from experiments.pamf_unrolled_restore.wisig_short_training import READOUTS, readouts


def as_iq(x):
    return np.stack((x.real, x.imag), axis=1).astype(np.float32)


def test_long_lag_unwrap_uses_short_lag_on_dispersive_short_training():
    n = np.arange(256)
    pilot = 0.8j * template(n - 3.2) + 0.25 * template(n - 4.6)
    offsets = np.array([-0.14, -0.092, 0, 0.101, 0.135])
    iq = as_iq(pilot[None] * np.exp(1j * offsets[:, None] * n))
    variants, obs = readouts(iq)
    np.testing.assert_allclose(obs.coarse_rad_per_sample, offsets, atol=3e-8)
    np.testing.assert_allclose(obs.fine_rad_per_sample, offsets, atol=3e-8)
    np.testing.assert_allclose(variants['fine_cfo'], as_iq(np.tile(pilot, (len(iq), 1))), atol=4e-6)
    np.testing.assert_array_equal(variants['direct'], iq)
    assert np.max(obs.effective_variance) < 1e-10
    assert np.min(obs.long_coherence) > 0.9999
    np.testing.assert_allclose(variants['wiener_restore_cfo'][:, :, 160:], iq[:, :, 160:], atol=3e-7)


def test_noise_scale_measures_complex_power_and_clock_slope_retains_sign():
    rng = np.random.default_rng(808)
    n = np.arange(256)
    variance = 0.08
    noise = np.sqrt(variance / 2) * (rng.normal(size=(400, 256)) + 1j * rng.normal(size=(400, 256)))
    _, noisy = readouts(as_iq(template(n) + noise))
    assert noisy.effective_variance.mean() == pytest.approx(variance, rel=0.12)
    drift = np.array([-0.001, 0.001])
    _, obs = readouts(as_iq(template(n * (1 - drift[:, None]))))
    np.testing.assert_allclose(obs.clock_delay_per_sample, drift, atol=2e-5)


def test_degenerate_window_is_finite_and_nonfinite_input_is_rejected():
    zero = np.zeros((2, 2, 256), dtype=np.float32)
    outputs, obs = readouts(zero)
    assert tuple(outputs) == READOUTS
    assert all(np.array_equal(v, zero) and v.dtype == zero.dtype for v in outputs.values())
    assert np.isfinite(obs.response).all()
    zero[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        readouts(zero)
