import importlib,importlib.util,unittest
import numpy as np

class SourceGateTest(unittest.TestCase):
 def test_fixed_gate_selects_complete_raw_or_restored_record(self):
  self.assertIsNotNone(importlib.util.find_spec('derive_gated_source'));route=importlib.import_module('derive_gated_source').route
  a=np.array([[.99,1,2,.02,3,.03,4,.04],[.98,2,1,.12,4,.13,5,.14],[.97,3,0,.22,1,.23,2,.24]],dtype=np.float32);out=route(a,np.array([.79,.80,.81],dtype=np.float32))
  np.testing.assert_array_equal(out[:2,4],a[:2,2]);np.testing.assert_array_equal(out[:2,5],a[:2,3]);np.testing.assert_array_equal(out[:2,6],a[:2,2]);np.testing.assert_array_equal(out[:2,7],a[:2,3]);np.testing.assert_array_equal(out[2],a[2]);np.testing.assert_array_equal(a[:,4],np.array([3,4,1],dtype=np.float32))
if __name__=='__main__':unittest.main()
