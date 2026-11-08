import unittest
import numpy as np

from experiments.phase1_dura_orthogonal.spectral_quotient_probe import spectral_features, match_ranks


class SpectralQuotientProbeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7401)
        self.iq = rng.normal(size=(8, 2, 256)).astype(np.float32)

    def test_quotient_is_invariant_to_nonzero_complex_gain(self):
        gain = 0.73 * np.exp(0.8j)
        z = self.iq[:, 0] + 1j * self.iq[:, 1]
        transformed = gain * z
        augmented = np.stack((transformed.real, transformed.imag), axis=1)
        np.testing.assert_allclose(spectral_features(self.iq, 'quotient'),
                                   spectral_features(augmented, 'quotient'),
                                   rtol=2e-6, atol=2e-6)

    def test_identical_source_pairs_retrieve_themselves(self):
        for kind in ('spectrum', 'quotient'):
            f = spectral_features(self.iq, kind)
            np.testing.assert_array_equal(match_ranks(f, f), np.ones(len(f)))

    def test_ties_are_ranked_pessimistically(self):
        features = np.ones((4, 6), dtype=np.complex128)
        np.testing.assert_array_equal(match_ranks(features, features), np.full(4, 4))

    def test_invalid_or_zero_energy_input_is_rejected(self):
        with self.assertRaises(ValueError):
            spectral_features(np.zeros((2, 2, 256)), 'quotient')
        invalid = self.iq.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            spectral_features(invalid, 'spectrum')
        with self.assertRaises(ValueError):
            match_ranks(np.zeros((2, 4)), np.zeros((2, 4)))


if __name__ == '__main__':
    unittest.main()
