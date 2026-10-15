"""Independent analytic checks for source-risk displacement projection."""
import unittest
from unittest.mock import patch
import torch
from .geometry import project

class Geometry(unittest.TestCase):
    def check(self,u,a,expected):
        result=project(torch.tensor(u,dtype=torch.float64),torch.tensor(a,dtype=torch.float64))
        self.assertTrue(torch.allclose(result['projected'],torch.tensor(expected,dtype=torch.float64),atol=1e-8,rtol=1e-8))
        return result
    def test_no_conflict(self):self.check([-1.,-2.],[[1.,0.],[0.,1.]],[-1.,-2.])
    def test_single_halfspace(self):self.check([2.,3.],[[1.,0.]],[0.,3.])
    def test_two_conflicts(self):self.check([2.,3.,-4.],[[1.,0.,0.],[0.,1.,0.]],[0.,0.,-4.])
    def test_opposing_constraints(self):self.check([2.,3.],[[1.,0.],[-1.,0.]],[0.,3.])
    def test_duplicate_constraints(self):self.check([2.,3.],[[1.,0.],[2.,0.]],[0.,3.])
    def test_zero_increment(self):self.check([0.,0.],[[1.,0.]],[0.,0.])
    def test_projection_does_not_require_tensor_numpy_bridge(self):
        # N607 torch2.1 / numpy2 produces incompatible bridge arrays.
        with patch.object(torch.Tensor,'numpy',side_effect=TypeError('incompatible NumPy bridge')):
            self.check([2.,3.],[[1.,0.]],[0.,3.])
    def test_scale_invariance(self):
        a=self.check([2.,3.],[[1.,0.],[1.,1.]],[-.5,.5])
        b=self.check([20.,30.],[[100.,0.],[.001,.001]],[-5.,5.])
        self.assertTrue(torch.allclose(a['projected']*10,b['projected'],atol=1e-8))
    def test_invalid(self):
        cases=[([1.,2.],[[0.,0.]]),([float('nan'),2.],[[1.,0.]]),([1.,2.],[[1.,float('inf')]])]
        for u,a in cases:
            with self.assertRaises(ValueError):project(torch.tensor(u),torch.tensor(a))

if __name__=='__main__':unittest.main(verbosity=2)
