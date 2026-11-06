import unittest
import torch

from experiments.phase1_dura_innovation_probe.metrics import oracle_line_probe


class OracleLineProbeTest(unittest.TestCase):
    def test_signed_positive_negative_and_zero_direction(self):
        anchor = torch.ones(3, 2, 256)
        innovation = torch.stack((torch.ones(2, 256), torch.ones(2, 256), torch.zeros(2, 256)))
        clean = anchor.clone()
        clean[0] *= 1.25
        clean[1] *= .75
        result = oracle_line_probe(anchor, clean, innovation)
        torch.testing.assert_close(result['optimal_alpha'], torch.tensor([.25, -.25, 0.]))
        self.assertEqual(float(result['positive_error'][0]), 0.)
        self.assertEqual(float(result['signed_error'][1]), 0.)
        self.assertGreater(float(result['positive_error'][1]), 0.)
        self.assertEqual(float(result['signed_error'][2]), 0.)

    def test_bound_and_sample_independence(self):
        anchor = torch.ones(2, 2, 256)
        clean = anchor * 2
        innovation = torch.ones_like(anchor)
        result = oracle_line_probe(anchor, clean, innovation)
        self.assertTrue(result['positive_clipped'].all())
        self.assertTrue(torch.isfinite(result['signed_error']).all())
        solo = oracle_line_probe(anchor[:1], clean[:1], innovation[:1])
        torch.testing.assert_close(solo['signed_error'], result['signed_error'][:1])
        with self.assertRaises(ValueError):
            oracle_line_probe(anchor, clean[:1], innovation)


if __name__ == '__main__':
    unittest.main()
