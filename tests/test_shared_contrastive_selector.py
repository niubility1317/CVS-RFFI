import unittest

import torch

from experiments.pamf_lora_main.shared_contrastive_selector_run import packet_channels, contrastive_loss


class SharedContrastiveSelectorTest(unittest.TestCase):
    def test_packet_channels_preserve_global_phase_and_amplitude_invariance(self):
        torch.manual_seed(12)
        x = torch.randn(4, 256, dtype=torch.complex64)
        gain = torch.tensor([1+2j, -.5+.7j, .2j, 2-1j])[:, None]
        torch.testing.assert_close(packet_channels(x), packet_channels(x*gain), rtol=1e-4, atol=1e-5)
        self.assertTrue(torch.isfinite(packet_channels(torch.zeros_like(x))).all())

    def test_contrastive_positive_pair_scores_lower_than_unrelated(self):
        z = torch.eye(8)
        self.assertLess(contrastive_loss(z, z).item(), contrastive_loss(z, z.roll(1, 0)).item())

    def test_rejects_short_or_singleton_batch(self):
        with self.assertRaises(ValueError):
            packet_channels(torch.ones(3, 4, dtype=torch.complex64))
        with self.assertRaises(ValueError):
            contrastive_loss(torch.ones(1, 5), torch.ones(1, 5))


if __name__ == '__main__':
    unittest.main()
