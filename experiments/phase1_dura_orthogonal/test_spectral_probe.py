import unittest

import torch

from experiments.phase1_dura_orthogonal.spectral_probe import band_energy


class SpectralProbeTest(unittest.TestCase):
    def test_parseval_and_frequency_localization(self):
        t = torch.arange(256, dtype=torch.float32)
        phase = 2*torch.pi*40*t/256
        tone = torch.stack((phase.cos(), phase.sin()))[None]
        clean = torch.ones_like(tone)
        bands = band_energy(tone, clean)
        expected = tone.square().sum((1, 2))/clean.square().sum((1, 2))
        torch.testing.assert_close(bands.sum(-1), expected)
        self.assertEqual(int(bands.argmax(-1).item()), 1)
        self.assertGreater(float(bands[0, 1] / bands.sum()), .999)


if __name__ == '__main__':
    unittest.main()
