import unittest

import torch

from experiments.pamf_lora_main.shared_teacher_mixture_run import grid_preference
from experiments.pamf_lora_main.shared_dura_realgain import SharedPacketMixture


class TeacherMixtureTest(unittest.TestCase):
    def test_teacher_grid_targets_nearest_clean_distribution_and_ties_choose_r17(self):
        clean = torch.tensor([[4., 0.], [0., 4.]])
        candidates = torch.tensor([[[0., 4.], [4., 0.]], [[4., 0.], [4., 0.]],
                                   [[0., 4.], [0., 4.]], [[0., 4.], [4., 0.]],
                                   [[0., 4.], [4., 0.]]])
        self.assertEqual(grid_preference(clean, candidates).tolist(), [.25, .5])
        self.assertEqual(grid_preference(clean, clean[None].expand(5, -1, -1)).tolist(), [0., 0.])

    def test_initial_mixture_is_bounded_and_uses_both_experts(self):
        class Identity:
            def __call__(self, x): return x
            def adjoint(self, x): return x
        model = SharedPacketMixture(32)
        y = torch.randn(2, 256, dtype=torch.complex64)
        a, b = y*2, y*.5
        torch.testing.assert_close(model(y, a, b, Identity()), .5*(a+b))

    def test_invalid_teacher_logits_fail(self):
        with self.assertRaises(ValueError):
            grid_preference(torch.zeros(2, 3), torch.full((5, 2, 3), float('nan')))


if __name__ == '__main__':
    unittest.main()
