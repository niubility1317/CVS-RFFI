import unittest
import torch
from experiments.phase1_cumulant_context.source_geometry import standardize,features
from experiments.phase1_dura_orthogonal.source_retrieval_geometry import retrieval

class SourceGeometryTest(unittest.TestCase):
    def test_identity_retrieval_and_shapes(self):
        torch.manual_seed(9)
        x=torch.randn(8,2,256)
        power,cum=features(x)
        self.assertEqual(tuple(power.shape),(8,32))
        self.assertEqual(tuple(cum.shape),(8,21))
        for value in (power,cum):
            mean=value.mean(0);scale=value.std(0,unbiased=False).clamp_min(.001)
            z=standardize(value,mean,scale)
            result=retrieval(z,z,torch.zeros(z.shape[1]),torch.ones(z.shape[1]))
            self.assertEqual(result['top1'],1.)
        with self.assertRaises(ValueError):standardize(power,torch.zeros(5),torch.ones(5))

if __name__=='__main__':unittest.main()
