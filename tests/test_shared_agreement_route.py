import unittest

import torch

from experiments.pamf_lora_main.shared_agreement_route import select_by_agreement


class SharedAgreementRouteTest(unittest.TestCase):
    def test_raw_agreement_overrides_confidence_only_when_unique(self):
        real = torch.tensor([[1.], [2.], [3.]])
        r17 = torch.tensor([[4.], [5.], [6.]])
        real_logits = torch.tensor([[5., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        r17_logits = torch.tensor([[0., 1., 0.], [7., 0., 0.], [0., 0., 2.]])
        raw_logits = torch.tensor([[0., 1., 0.], [0., 1., 0.], [1., 0., 0.]])
        result = select_by_agreement(real, r17, real_logits, r17_logits, raw_logits)
        torch.testing.assert_close(result, torch.tensor([[4.], [2.], [6.]]))

    def test_raw_logits_must_be_finite_and_match_shape(self):
        x = torch.ones(2, 3)
        a = torch.zeros(2, 4)
        with self.assertRaises(ValueError):
            select_by_agreement(x, x, a, a, torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            select_by_agreement(x, x, a, a, torch.full((2, 4), float('nan')))


if __name__ == '__main__':
    unittest.main()
