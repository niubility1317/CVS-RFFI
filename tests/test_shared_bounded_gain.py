import unittest

import torch

from experiments.pamf_lora_main.shared_dura_realgain import SharedBoundedGain


class BoundedGainTest(unittest.TestCase):
    def test_initial_output_matches_the_configured_source_anchor(self):
        class Identity:
            def __call__(self, x): return x
            def adjoint(self, x): return x
        y = torch.randn(3, 256, dtype=torch.complex64)
        real, r17 = 2*y, .5*y
        for anchor_real in (True, False):
            model = SharedBoundedGain(32, anchor_real=anchor_real)
            expected = real if anchor_real else r17
            torch.testing.assert_close(model(y, real, r17, Identity()), expected, atol=0, rtol=0)

    def test_packet_correction_cannot_exceed_fixed_fraction(self):
        class Identity:
            def __call__(self, x): return x
            def adjoint(self, x): return x
        y = torch.randn(3, 256, dtype=torch.complex64)
        real, r17 = 2*y, .5*y
        model = SharedBoundedGain(32, anchor_real=False)
        with torch.no_grad():
            model.net[-1].bias.fill_(5.)
        out = model(y, real, r17, Identity())
        delta_rms = (out-r17).abs().square().mean(-1).sqrt()
        anchor_rms = r17.abs().square().mean(-1).sqrt()
        self.assertTrue((delta_rms <= .010001*anchor_rms).all())


if __name__ == '__main__':
    unittest.main()
