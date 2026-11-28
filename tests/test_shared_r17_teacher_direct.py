import unittest

import torch

from experiments.pamf_lora_main.shared_dura_realgain import SharedTwoPriorGainR17


class R17AnchoredGainTest(unittest.TestCase):
    def test_zero_correction_starts_exactly_at_r17_for_both_packet_shapes(self):
        class Identity:
            def __call__(self, x): return x
            def adjoint(self, x): return x
        for width in (32, 1024):
            model = SharedTwoPriorGainR17(width)
            received = torch.randn(2, 8*width, dtype=torch.complex64)
            real, r17 = received*2, received*.4
            torch.testing.assert_close(model(received, real, r17, Identity()), r17, atol=0, rtol=0)

    def test_nonzero_gain_uses_real_gain_direction(self):
        class Identity:
            def __call__(self, x): return x
            def adjoint(self, x): return x
        model = SharedTwoPriorGainR17(32)
        with torch.no_grad():
            model.net[-1].bias[0] = .3
        r17 = torch.zeros(2, 256, dtype=torch.complex64)
        real = torch.randn_like(r17)
        self.assertGreater(model(real, real, r17, Identity()).abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
