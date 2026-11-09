import unittest
import torch
from experiments.phase1_cumulant_response10.train import scaled_perturbation, EPSILON, SCALE
from experiments.phase1_dura_orthogonal.counterfactual_response import choose_perturbation

class ResponseScaleTest(unittest.TestCase):
    def test_perturbations_preserve_direction_and_have_ten_percent_rms(self):
        torch.manual_seed(3)
        clean=torch.randn(8,2,256)
        delta=scaled_perturbation(clean,0)
        torch.testing.assert_close(delta,SCALE*choose_perturbation(clean,0),rtol=0,atol=0)
        rms=delta.abs().square().mean(-1).sqrt()
        z=torch.complex(clean[:,0],clean[:,1])
        reference=z.abs().square().mean(-1).sqrt()
        torch.testing.assert_close(rms/reference,torch.full_like(reference,EPSILON),rtol=1e-5,atol=1e-6)
        self.assertAlmostEqual(EPSILON,.1)

if __name__ == '__main__': unittest.main()
