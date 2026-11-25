import numpy as np

from experiments.pamf_lora_main import w24_factor_audit as audit


def test_noise_sweep_scales_only_added_noise_and_retains_exact_endpoints():
    signal = np.asarray([3 + 2j, -4j], dtype=np.complex64)
    noise = np.asarray([2j, 1 + 1j], dtype=np.complex64)
    received = signal + noise
    mixed = audit.noise_sweep(received, signal, noise)
    assert mixed.dtype == np.complex64
    assert np.array_equal(mixed[0], received)
    assert np.array_equal(mixed[-1], signal)
    assert np.allclose(mixed[1], signal + .75 * noise)
    assert np.allclose(mixed[3], signal + .25 * noise)
