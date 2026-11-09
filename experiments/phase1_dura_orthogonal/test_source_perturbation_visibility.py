import unittest
import torch

from experiments.phase1_dura_orthogonal.source_perturbation_visibility import relative_energy_db


class VisibilityTest(unittest.TestCase):
    def test_equal_and_scaled_energy(self):
        residual = torch.ones(2, 256, dtype=torch.complex64)
        effects = torch.ones(2, 4, 256, dtype=torch.complex64)
        torch.testing.assert_close(relative_energy_db(effects, residual), torch.zeros(2, 4))
        torch.testing.assert_close(relative_energy_db(2*effects, residual),
                                   torch.full((2, 4), 10*torch.log10(torch.tensor(4.)).item()))

    def test_zero_nonfinite_and_shape_rejected(self):
        residual = torch.ones(2, 256, dtype=torch.complex64)
        effects = torch.ones(2, 4, 256, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            relative_energy_db(effects[:1], residual)
        with self.assertRaises(ValueError):
            relative_energy_db(effects, torch.zeros_like(residual))
        with self.assertRaises(ValueError):
            relative_energy_db(torch.zeros_like(effects), residual)
        effects[0, 0, 0] = complex(float('nan'), 0.)
        with self.assertRaises(FloatingPointError):
            relative_energy_db(effects, residual)


if __name__ == '__main__':
    unittest.main()
