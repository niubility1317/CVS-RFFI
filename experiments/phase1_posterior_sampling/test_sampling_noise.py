import gc
import unittest
from unittest.mock import patch
import numpy as np
import torch
from sampling_noise import source_noise, tensor_array

class SamplingNoiseTests(unittest.TestCase):
 def test_seed_values_and_order_are_preserved_without_from_numpy(self):
  ids=[11,29]; expected_z=[]; expected_e=[]
  for i in ids:
   rng=np.random.default_rng(2026165054+i)
   expected_z.append(rng.standard_normal((6,512)).astype(np.float32))
   expected_e.append(rng.standard_normal((6,512)).astype(np.float32))
  with patch.object(torch,"from_numpy",side_effect=TypeError("expected np.ndarray (got numpy.ndarray)")):
   z,e=source_noise(ids,6,512)
   reverse_z,reverse_e=source_noise(ids[::-1],6,512)
  gc.collect()
  np.testing.assert_array_equal(np.from_dlpack(z),np.stack(expected_z))
  np.testing.assert_array_equal(np.from_dlpack(e),np.stack(expected_e))
  self.assertTrue(torch.equal(z,reverse_z.flip(0)))
  self.assertTrue(torch.equal(e,reverse_e.flip(0)))

 def test_tensor_export_uses_native_numpy_without_tensor_numpy(self):
  x=torch.tensor([[1.,2.,3.],[4.,5.,6.]]).T
  with patch.object(torch.Tensor,"numpy",side_effect=TypeError("incompatible NumPy runtime")):
   exported=tensor_array(x)
  np.testing.assert_array_equal(exported,np.array([[1.,4.],[2.,5.],[3.,6.]],dtype=np.float32))
  self.assertTrue(np.array_equal(exported,exported.copy()))

 def test_boolean_export_supports_older_dlpack(self):
  x=torch.tensor([True,False,True]);real=np.from_dlpack
  def reject_bool(t):
   if t.dtype==torch.bool:raise RuntimeError("Bool type is not supported by dlpack")
   return real(t)
  with patch.object(np,"from_dlpack",side_effect=reject_bool):exported=tensor_array(x)
  self.assertEqual(exported.dtype,np.bool_)
  np.testing.assert_array_equal(exported,np.array([True,False,True]))

if __name__=="__main__": unittest.main()
