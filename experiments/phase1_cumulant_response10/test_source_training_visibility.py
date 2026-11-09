import unittest
import numpy as np
import torch
from experiments.phase1_cumulant_response10.source_training_visibility import relative_db, cell_summary

class SourceVisibilityTest(unittest.TestCase):
    def test_fixed_action_scale_is_twenty_db(self):
        x=torch.ones(3,256,dtype=torch.complex64)
        base=relative_db(x,x)
        scaled=relative_db(10*x,x)
        torch.testing.assert_close(scaled-base,torch.full_like(base,20.),rtol=0,atol=1e-5)
        v=cell_summary(np.array([-45.,-30.,-10.]))
        self.assertEqual(v['below_minus20_at_1pct'],2)
        self.assertEqual(v['below_minus20_at_10pct'],1)

    def test_server_tensor_conversion(self):
        x=torch.tensor([1.,2.],dtype=torch.float32)
        y=np.from_dlpack(x.cpu().contiguous()).copy()
        np.testing.assert_array_equal(y,np.array([1.,2.],dtype=np.float32))

    def test_zero_nonfinite_or_mismatched_rejected(self):
        x=torch.ones(1,256,dtype=torch.complex64)
        with self.assertRaises(ValueError): relative_db(x,torch.zeros_like(x))
        with self.assertRaises(FloatingPointError): relative_db(x*float('nan'),x)
        with self.assertRaises(ValueError): relative_db(x,x[:,:128])
        with self.assertRaises(ValueError): cell_summary(np.array([]))

if __name__ == '__main__': unittest.main()
