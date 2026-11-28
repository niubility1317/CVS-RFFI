import unittest

import torch

from experiments.pamf_lora_main.shared_matched_mixture_run import mse_weight
from experiments.pamf_lora_main import shared_matched_mixture_run as control
from experiments.pamf_lora_main import shared_teacher_mixture_run as teacher


class MatchedMixtureTest(unittest.TestCase):
    def test_same_seed_and_budget(self):
        self.assertEqual((control.SEED, control.STEPS), (teacher.SEED, teacher.STEPS))

    def test_mse_target_uses_same_five_weights_and_tie_break(self):
        clean = torch.tensor([[1.+0j, 1.+0j], [3.+0j, 3.+0j], [-1.+0j, -1.+0j]])
        real, r17 = torch.full_like(clean, 2), torch.zeros_like(clean)
        torch.testing.assert_close(mse_weight(clean, real, r17), torch.tensor([.5, 1., 0.]))
        torch.testing.assert_close(mse_weight(clean, r17, r17), torch.zeros(3))

    def test_nonfinite_input_is_rejected(self):
        z = torch.ones(2, 3, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            mse_weight(z*float('nan'), z, z)


if __name__ == '__main__':
    unittest.main()
