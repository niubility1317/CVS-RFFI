import unittest

import torch

from experiments.pamf_lora_main.shared_teacher_selector_run import preferred_expert


class SharedTeacherSelectorTest(unittest.TestCase):
    def test_preference_uses_clean_teacher_distribution_without_true_labels(self):
        clean = torch.tensor([[4., 0.], [0., 4.]])
        real = torch.tensor([[3., 0.], [3., 0.]])
        r17 = torch.tensor([[0., 3.], [0., 3.]])
        self.assertEqual(preferred_expert(clean, real, r17).tolist(), [0., 1.])

    def test_invalid_or_nonfinite_logits_are_rejected(self):
        good = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            preferred_expert(good, good, torch.zeros(2, 4))
        with self.assertRaises(ValueError):
            preferred_expert(good, good, torch.full((2, 3), float('nan')))


if __name__ == '__main__':
    unittest.main()
