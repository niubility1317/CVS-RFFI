import unittest,numpy as np
try:from labeled_counts import summarize
except ImportError:summarize=None
class LabeledCountsTests(unittest.TestCase):
 def test_paired_repairs_and_damages(self):
  self.assertIsNotNone(summarize);truth=np.array([0,1,2,0]);pred=np.array([[0,1,0,1],[1,1,1,1],[1,1,2,2],[0,0,2,0]]);r=summarize(pred,truth,np.array([True,True,True,True]));self.assertEqual(r['count'],4);self.assertEqual(r['dura']['correct'],3);self.assertEqual(r['blind']['correct'],3);self.assertEqual(r['blind']['repairs'],1);self.assertEqual(r['blind']['damages'],1)
 def test_empty_rejected(self):
  with self.assertRaises(ValueError):summarize(np.zeros((1,4),dtype=int),np.zeros(1,dtype=int),np.zeros(1,dtype=bool))
if __name__=='__main__':unittest.main()
