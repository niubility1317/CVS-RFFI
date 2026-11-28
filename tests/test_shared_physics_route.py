import unittest

import torch

from experiments.pamf_lora_main.shared_physics_route import select_by_physics


class SharedPhysicsRouteTest(unittest.TestCase):
    def test_forward_residual_selects_the_more_consistent_whole_packet(self):
        observed = torch.tensor([[2.+0j, 4.+0j], [2.+0j, 4.+0j]])
        real = torch.tensor([[1.+0j, 2.+0j], [0.+0j, 0.+0j]])
        r17 = torch.tensor([[0.+0j, 0.+0j], [1.+0j, 2.+0j]])
        result = select_by_physics(observed, real, r17, lambda z: 2*z)
        torch.testing.assert_close(result, torch.stack((real[0], r17[1])))

    def test_tie_selects_r17_and_nonfinite_is_rejected(self):
        y = torch.ones(2, 3, dtype=torch.complex64)
        torch.testing.assert_close(select_by_physics(y, y, y, lambda z: z), y)
        with self.assertRaises(ValueError):
            select_by_physics(y, y, y, lambda z: z*float('nan'))


if __name__ == '__main__':
    unittest.main()
