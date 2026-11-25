import numpy as np

from experiments.pamf_lora_main.w24_factor_audit import affine_phase, fit_chirp_gains


def test_paired_affine_phase_handles_fractional_bin_and_estimated_constant():
    rng = np.random.default_rng(12)
    reference = rng.normal(size=8192) + 1j * rng.normal(size=8192)
    n = np.arange(len(reference))
    rotation = np.exp(1j * (.71 + 2 * np.pi * 217.3 * n / 1e6))
    predicted, frequency = affine_phase(reference * rotation, reference)
    np.testing.assert_allclose(frequency, 217.3, atol=.3)
    np.testing.assert_allclose(predicted, rotation, atol=.02)


def test_chirp_projection_recovers_group_gains_and_bypasses_nulls():
    reference = np.ones(8192, dtype=complex)
    gain = np.exp(1j * np.arange(8)) * np.linspace(.3, 2., 8)
    signal = (reference.reshape(8, 1024) * gain[:, None]).reshape(-1)
    np.testing.assert_allclose(fit_chirp_gains(signal, reference), gain)
    signal = signal.copy()
    signal[:1024] = 0
    assert fit_chirp_gains(signal, reference)[0] == 1
