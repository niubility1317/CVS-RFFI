import unittest
import numpy as np
from score_source_noise import noise_bins,pair_counts

class NoiseBinTest(unittest.TestCase):
 def test_fixed_edges_are_left_closed(self):
  x=np.array([-7.,-6.,-5.,-4.,-3.,-2.,-1.,1.]);self.assertEqual(noise_bins(x).tolist(),[0,1,2,3,4,5,6,6])
 def test_paired_repair_and_damage(self):
  target=np.array([0,1,2,1]);base=np.array([1,1,0,0]);pred=np.array([0,0,2,0]);self.assertEqual(pair_counts(pred,base,target),(2,2,1))
if __name__=='__main__':unittest.main()
