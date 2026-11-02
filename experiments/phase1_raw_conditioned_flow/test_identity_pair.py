import unittest
import numpy as np
from compare_identity_source import paired

class PairTest(unittest.TestCase):
 def test_paired_repairs_and_harms_exclude_low_confidence(self):
  old=np.array([[.99,0,0,.1,1,.2],[.97,1,0,.2,1,.3],[.5,2,2,.3,2,.2]])
  new=old.copy();new[:,4]=[0,0,1];new[:,5]=[.1,.4,.3];m=paired(old,new);self.assertEqual(m['trusted_count'],2);self.assertEqual(m['old_correct'],1);self.assertEqual(m['new_correct'],1);self.assertEqual(m['repairs'],1);self.assertEqual(m['damages'],1);self.assertAlmostEqual(m['old_iq_error'],.7/3)
 def test_unmatched_raw_is_rejected(self):
  old=np.array([[.99,0,0,.1,1,.2]]);new=old.copy();new[0,2]=1
  with self.assertRaises(ValueError):paired(old,new)
if __name__=='__main__':unittest.main()
