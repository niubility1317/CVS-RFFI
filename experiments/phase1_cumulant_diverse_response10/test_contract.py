import unittest
import torch
from experiments.phase1_cumulant_diverse_response10.train import SCHEDULE, schedule_contract, EPSILON, scaled_perturbation

class CombinedContractTest(unittest.TestCase):
    def test_schedule_and_response_scale(self):
        self.assertEqual(schedule_contract(),SCHEDULE)
        self.assertEqual(len(SCHEDULE),20)
        self.assertEqual(sorted(x for row in SCHEDULE for x in row),sorted(list(range(1,31))*2))
        torch.manual_seed(7);x=torch.randn(4,2,256);d=scaled_perturbation(x,0)
        z=torch.complex(x[:,0],x[:,1]);ratio=d.abs().square().mean(-1).sqrt()/z.abs().square().mean(-1).sqrt()
        torch.testing.assert_close(ratio,torch.full_like(ratio,EPSILON),rtol=1e-5,atol=1e-6)
        self.assertAlmostEqual(EPSILON,.1)

if __name__ == '__main__':unittest.main()
