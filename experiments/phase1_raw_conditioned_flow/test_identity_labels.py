import unittest
import numpy as np
from score_identity_labels import label_stats

class LabelStatsTest(unittest.TestCase):
 def test_true_identity_repairs_and_harms_are_paired(self):
  truth=np.array([0,1,2,1]);raw=np.array([0,0,2,1]);old=np.array([1,1,2,0]);new=np.array([0,1,0,1]);x=label_stats(truth,raw,old,new)
  self.assertEqual((x['count'],x['raw_correct'],x['old_correct'],x['new_correct']),(4,3,2,3));self.assertEqual((x['repairs_vs_old'],x['damages_vs_old']),(2,1));self.assertEqual((x['new_repairs_vs_raw'],x['new_damages_vs_raw']),(1,1))
if __name__=='__main__':unittest.main()
