import unittest
import numpy as np
from score_source import summarize

class SourceScoreTest(unittest.TestCase):
 def test_paired_repairs_and_damage_only_on_confident_pseudo_targets(self):
  v=np.array([[.99,0,0,.1,1,.2],[.96,1,0,.3,1,.2],[.5,2,2,.2,0,.4],[.97,2,2,.4,2,.1]])
  m=summarize(v);self.assertEqual(m['count'],4);self.assertEqual(m['trusted_count'],3);self.assertEqual(m['raw_correct'],2);self.assertEqual(m['flow_correct'],2);self.assertEqual(m['repairs'],1);self.assertEqual(m['damages'],1);self.assertAlmostEqual(m['flow_aligned_iq_error'],.225)
if __name__=='__main__':unittest.main()
