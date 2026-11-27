import numpy as np

from experiments.pamf_unrolled_restore.wisig_full_noise_direction import CENTER, EDGES, interventions, power


def test_known_noise_interventions_match_each_regions_energy_and_keep_clean():
    rng = np.random.default_rng(72)
    shape = (4, 256)
    raw, learned, signal = [
        (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(np.complex64) for _ in range(3)]
    signal[0] = raw[0]
    matched, exact = interventions(raw, learned, signal)
    np.testing.assert_array_equal(matched[0], learned[0])
    np.testing.assert_array_equal(exact[0], learned[0])
    np.testing.assert_array_equal(exact[1:], signal[1:])
    for region in (CENTER, EDGES):
        required = (signal - raw)[1:, region]
        applied = (matched - raw)[1:, region]
        np.testing.assert_allclose(power(applied), power((learned - raw)[1:, region]), rtol=2e-6)
        dot = (applied * required.conj()).sum(-1, keepdims=True)
        np.testing.assert_allclose(dot.real, np.sqrt(power(applied)*power(required)), rtol=2e-6)
        np.testing.assert_allclose(dot.imag, 0., atol=3e-5)
