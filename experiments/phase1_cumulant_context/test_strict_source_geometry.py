import unittest
import torch
from experiments.phase1_cumulant_context.model import cumulant_features
from experiments.phase1_cumulant_context.strict_source_geometry import strict_features

class StrictGeometryTest(unittest.TestCase):
    def test_zero_lag_and_global_gain(self):
        torch.manual_seed(4);x=torch.randn(4,2,256)
        strict=strict_features(x);approx=cumulant_features(x)
        torch.testing.assert_close(strict[:,:3],approx[:,:3],rtol=1e-5,atol=1e-6)
        z=torch.complex(x[:,0],x[:,1])*complex(1.2,.7)+complex(.4,-.2)
        y=torch.stack((z.real,z.imag),1)
        torch.testing.assert_close(strict,strict_features(y),rtol=1e-4,atol=1e-5)
    def test_nonzero_lag_can_differ(self):
        t=torch.linspace(-2.,2.,256)
        x=torch.stack((t,t.square()),0).unsqueeze(0)
        self.assertGreater((strict_features(x)[:,3:]-cumulant_features(x)[:,3:]).abs().max().item(),1e-5)
        self.assertTrue(torch.isfinite(strict_features(torch.ones_like(x))).all())

if __name__=='__main__':unittest.main()
