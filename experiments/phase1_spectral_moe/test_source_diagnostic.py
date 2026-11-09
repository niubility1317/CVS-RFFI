import unittest
import torch
from experiments.phase1_spectral_moe.source_diagnostic import frequency_oracle

class FrequencyOracleTest(unittest.TestCase):
    def test_oracle_bounded_by_each_input(self):
        torch.manual_seed(12)
        clean = torch.randn(3, 2, 256)
        experts = tuple(clean + s*torch.randn_like(clean) for s in (0.1, 0.3, 1.0))
        oracle, choices = frequency_oracle(experts, clean)
        self.assertEqual(tuple(choices.shape), (3, 256))
        error = (oracle-clean).square().sum((1, 2))
        for x in experts:
            self.assertTrue(torch.all(error <= (x-clean).square().sum((1, 2)) + 1e-4))
        single, _ = frequency_oracle(tuple(x[:1] for x in experts), clean[:1])
        torch.testing.assert_close(single, oracle[:1], rtol=1e-5, atol=1e-6)

    def test_identical_candidates_preserved(self):
        torch.manual_seed(13)
        clean = torch.randn(2, 2, 256)
        value, choices = frequency_oracle((clean, clean, clean), clean)
        torch.testing.assert_close(value, clean, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.all(choices == 0))

if __name__ == '__main__': unittest.main()
