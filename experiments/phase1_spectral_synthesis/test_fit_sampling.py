import importlib,importlib.util,unittest
import numpy as np
class FitSamplingTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('fit_sampling'));self.select=importlib.import_module('fit_sampling').select_fit
 def test_scaled_fit_contains_original_and_excludes_physical_prior(self):
  training=np.arange(20000);prior=training[:8192];small=self.select(training,prior,1024);large=self.select(training,prior,8192);expected=np.sort(np.random.default_rng(2026135055).choice(training[8192:],size=1024,replace=False));np.testing.assert_array_equal(small,expected);self.assertEqual(len(np.unique(large)),8192);self.assertTrue(np.isin(small,large).all());self.assertFalse(np.isin(large,prior).any());np.testing.assert_array_equal(large,self.select(training,prior,8192))
 def test_invalid_fit_size_is_rejected(self):
  with self.assertRaises(ValueError):self.select(np.arange(20000),np.arange(8192),1025)
if __name__=='__main__':unittest.main()
