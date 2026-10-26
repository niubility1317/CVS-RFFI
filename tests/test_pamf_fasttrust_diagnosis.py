import unittest
import numpy as np
from experiments.pamf_u_history.fasttrust_diagnosis import mass_set,pair_purity,view_gradient
from experiments.pamf_u_history.fasttrust_risk import risk_threshold,leave_receiver_out


class DiagnosisTest(unittest.TestCase):
    def test_mass_set_includes_crossing_class(self):
        p=np.array([[.9,.06,.01,.01,.01,.01],[.95,.01,.01,.01,.01,.01]])
        s=mass_set(p)
        self.assertEqual(s.sum(1).tolist(),[2,1])
        self.assertTrue(s[0,0] and s[0,1])

    def test_cross_receiver_pair_purity_is_integer_exact(self):
        r=pair_purity(np.array([0,0,0]),np.ones(3,dtype=bool),np.array([1,3,3]),np.array([0,0,1]))
        self.assertEqual(r['positive_pairs'],2)
        self.assertEqual(r['false_positive_pairs'],1)

    def test_correct_hard_label_is_oracle_but_smoothing_can_oppose_it(self):
        r=view_gradient(np.array([[.999,.0002,.0002,.0002,.0002,.0002]]),np.array([0]),np.array([0]),np.array([True]))
        self.assertTrue(r['hard_equals_oracle_on_correct'])
        self.assertEqual(r['LS01_vs_oracle_conflicts'],1)

    def test_risk_threshold_cannot_split_equal_scores(self):
        threshold=risk_threshold(np.ones(4),np.array([True,True,False,True]),np.ones(4,dtype=bool),np.array([1,3,1,3]),precision=1.,min_per_rx=1)
        self.assertIsNone(threshold)

    def test_held_receiver_label_does_not_change_its_threshold(self):
        rx=np.repeat([1,3,4],4);score=np.tile([.99,.98,.97,.96],3)
        safe=np.ones(12,dtype=bool);eligible=np.ones(12,dtype=bool)
        before,folds=leave_receiver_out(score,safe,eligible,rx,min_per_rx=1)
        safe[rx==1]=False
        after,new=leave_receiver_out(score,safe,eligible,rx,min_per_rx=1)
        self.assertEqual(folds[0]['threshold'],new[0]['threshold'])
        self.assertTrue(np.array_equal(before[rx==1],after[rx==1]))


if __name__=='__main__':unittest.main()
