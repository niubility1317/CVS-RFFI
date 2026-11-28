import unittest

import numpy as np
import torch

from experiments.pamf_lora_main.shared_confidence_route import prediction_classes, select_by_confidence


class SharedConfidenceRouteTest(unittest.TestCase):
    def test_each_packet_uses_its_own_confidence_and_tie_selects_r17(self):
        real = torch.tensor([[1., 2.], [3., 4.]])
        r17 = torch.tensor([[5., 6.], [7., 8.]])
        real_logits = torch.tensor([[8., 0.], [1., 0.]])
        r17_logits = torch.tensor([[1., 0.], [1., 0.]])
        output = select_by_confidence(real, r17, real_logits, r17_logits)
        torch.testing.assert_close(output, torch.stack((real[0], r17[1])))

    def test_rejects_nonfinite_and_mismatched_logits(self):
        x = torch.ones(2, 256)
        with self.assertRaises(ValueError):
            select_by_confidence(x, x, torch.tensor([[float('nan'), 0.], [0., 1.]]), torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            select_by_confidence(x, x, torch.zeros(2, 3), torch.zeros(2, 2))

    def test_predictions_are_plain_safe_integer_arrays(self):
        result = prediction_classes(torch.tensor([[1., 4.], [5., 2.]]))
        self.assertEqual(result.dtype, np.dtype('int16'))
        self.assertEqual(result.tolist(), [1, 0])


if __name__ == '__main__':
    unittest.main()
