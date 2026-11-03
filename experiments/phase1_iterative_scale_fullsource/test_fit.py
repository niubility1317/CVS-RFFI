import unittest
import numpy as np
from experiments.phase1_spectral_synthesis.fit_sampling import select_fit
from experiments.phase1_iterative_large.sampling import select_large_fit
from experiments.phase1_iterative_scale_fullsource.replay import verify_nested_fit,model_for_schema
from experiments.phase1_iterative_timefreq.model import IterativeTimeFreq

class ScaleContractTest(unittest.TestCase):
 def setUp(self):
  self.training=np.arange(50000,dtype=np.int64);self.prior=np.arange(8192,dtype=np.int64);self.holdout=np.arange(50000,51000,dtype=np.int64)
  self.small=select_fit(self.training,self.prior,8192);self.large=select_large_fit(self.training,self.prior,32768)
 def test_exact_nested_approved_fits(self):
  verify_nested_fit(self.small,self.large,self.training,self.prior,self.holdout)
  self.assertIs(type(model_for_schema('phase1_iterative_timefreq_v1')),IterativeTimeFreq);self.assertIs(type(model_for_schema('phase1_iterative_large_v1')),IterativeTimeFreq)
  with self.assertRaises(ValueError):model_for_schema('unknown')
 def test_rejects_duplicate_changed_prior_and_holdout(self):
  bad=self.large.copy();bad[-1]=bad[-2]
  with self.assertRaises(ValueError):verify_nested_fit(self.small,bad,self.training,self.prior,self.holdout)
  bad=self.large.copy();bad[-1]=self.holdout[0]
  with self.assertRaises(ValueError):verify_nested_fit(self.small,bad,self.training,self.prior,self.holdout)
  bad=self.large.copy();bad[-1]=self.prior[0]
  with self.assertRaises(ValueError):verify_nested_fit(self.small,bad,self.training,self.prior,self.holdout)
  bad=self.small.copy();bad[0]=self.holdout[0]
  with self.assertRaises(ValueError):verify_nested_fit(bad,self.large,self.training,self.prior,self.holdout)

if __name__=='__main__':unittest.main()
