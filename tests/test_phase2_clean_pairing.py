import unittest

import numpy as np

from experiments.pamf_phase2_practical.clean_pairing import map_ids, normalize_batch


class CleanPairingTests(unittest.TestCase):
    def test_mapping_preserves_order_and_rejects_duplicate_and_missing_ids(self):
        bank = np.array(['a', 'b', 'c'])
        np.testing.assert_array_equal(map_ids(bank, np.array(['c', 'a'])), [2, 0])
        with self.assertRaises(ValueError):
            map_ids(bank, np.array(['a', 'a']))
        with self.assertRaises(ValueError):
            map_ids(bank, np.array(['x']))
        with self.assertRaises(ValueError):
            map_ids(np.array(['a', 'a']), np.array(['a']))

    def test_matches_original_complex128_normalization(self):
        rng = np.random.default_rng(20261112)
        raw = rng.standard_normal((32, 2, 256)).astype(np.float32)
        observed = normalize_batch(raw)
        for i, item in enumerate(raw):
            z = item[0].astype(np.float64) + 1j * item[1].astype(np.float64)
            z = z / np.sqrt(np.mean(np.abs(z) ** 2))
            expected = np.stack((z.real, z.imag)).astype(np.float32)
            np.testing.assert_array_equal(observed[i], expected)

    def test_rejects_zero_and_nonfinite_input(self):
        raw = np.ones((2, 2, 256), np.float32)
        raw[0] = 0
        with self.assertRaises(ValueError):
            normalize_batch(raw)
        raw[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            normalize_batch(raw)


if __name__ == '__main__':
    unittest.main()
