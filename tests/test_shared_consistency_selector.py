import unittest

import torch

from experiments.pamf_lora_main.shared_consistency_selector_run import pair_discrimination


class SharedConsistencySelectorTest(unittest.TestCase):
    def test_same_packet_consistency_is_phase_invariant_and_rejects_constant_collapse(self):
        torch.manual_seed(19)
        packets = torch.randn(8, 256, dtype=torch.complex64)
        phase = torch.polar(torch.ones(8), torch.linspace(-2, 2, 8))[:, None]
        score = pair_discrimination(packets, packets * phase)
        self.assertTrue(torch.isfinite(score).all())
        self.assertGreater(score.min().item(), .85)
        collapsed = torch.ones_like(packets)
        torch.testing.assert_close(pair_discrimination(collapsed, collapsed), torch.zeros(8), atol=1e-6, rtol=0)

    def test_requires_two_same_shape_batches(self):
        with self.assertRaises(ValueError):
            pair_discrimination(torch.ones(1, 256, dtype=torch.complex64), torch.ones(1, 256, dtype=torch.complex64))
        with self.assertRaises(ValueError):
            pair_discrimination(torch.ones(2, 256, dtype=torch.complex64), torch.ones(2, 128, dtype=torch.complex64))


if __name__ == '__main__':
    unittest.main()
