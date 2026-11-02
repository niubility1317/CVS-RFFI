import importlib,importlib.util,unittest
import numpy as np
class PairComparisonTest(unittest.TestCase):
 def test_both_models_share_the_fixed_clean_guard(self):
  self.assertIsNotNone(importlib.util.find_spec('compare_scaled_source'));f=importlib.import_module('compare_scaled_source').route_pair
  small=np.array([[.99,1,2,.1,3,.2,4,.3],[.98,2,1,.4,2,.5,3,.6]],dtype=np.float32);large=small.copy();large[:,6]=[5,4];large[:,7]=[.7,.8];r=f(small,large,np.array([.8,.81],dtype=np.float32));np.testing.assert_array_equal(r[:,3],np.array([2,3],dtype=np.float32));np.testing.assert_array_equal(r[:,4],np.array([2,4],dtype=np.float32));np.testing.assert_allclose(r[:,5],[.1,.6]);np.testing.assert_allclose(r[:,6],[.1,.8])
if __name__=='__main__':unittest.main()
