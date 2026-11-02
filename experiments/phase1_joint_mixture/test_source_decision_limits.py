import unittest
import numpy as np
from source_decision_limits import fit_mapping,apply_mapping,pattern_upper

class DecisionLimitTest(unittest.TestCase):
 def test_training_majority_and_unseen_fallback_use_actual_candidates(self):
  c=np.array([[0,1,2,2,2],[0,1,2,2,2],[0,1,2,2,2],[5,4,3,3,3]]);g=np.ones(4,dtype=bool);y=np.array([1,1,0,4]);m=fit_mapping(c[:3],g[:3],y[:3],False);out,unseen=apply_mapping(m,c,g,False);self.assertEqual(out.tolist(),[1,1,1,4]);self.assertEqual(unseen.tolist(),[False,False,False,True])
 def test_truth_aided_pattern_ceiling_reflects_conflicting_labels(self):
  c=np.array([[0,1,2,2,2],[0,1,2,2,2],[0,1,2,2,2]]);g=np.array([0,0,1],dtype=bool);y=np.array([0,0,1]);self.assertEqual(pattern_upper(c,g,y,False),2);self.assertEqual(pattern_upper(c,g,y,True),3)
if __name__=='__main__':unittest.main()
