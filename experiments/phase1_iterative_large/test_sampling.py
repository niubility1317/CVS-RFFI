import unittest
import numpy as np
from experiments.phase1_spectral_synthesis.fit_sampling import select_fit
from experiments.phase1_iterative_large.sampling import select_large_fit

class LargeSourceSamplingTest(unittest.TestCase):
 def test_nested_deterministic_and_excludes_prior(self):
  training=np.arange(50000,dtype=np.int64);prior=np.arange(8192,dtype=np.int64)
  original=select_fit(training,prior,8192);expanded=select_large_fit(training,prior,32768)
  self.assertEqual(len(expanded),32768);self.assertEqual(len(np.unique(expanded)),32768)
  self.assertTrue(np.isin(original,expanded).all());self.assertFalse(np.intersect1d(expanded,prior).size)
  self.assertTrue(np.array_equal(expanded,select_large_fit(training,prior,32768)))
 def test_rejects_ineligible_size_and_insufficient_pool(self):
  with self.assertRaises(ValueError):select_large_fit(np.arange(50000),np.arange(8192),8192)
  with self.assertRaises(ValueError):select_large_fit(np.arange(20000),np.arange(8192),32768)

if __name__=='__main__':unittest.main()
