import unittest
import numpy as np
import torch
from experiments.phase1_local_teacher_fullsource.replay import gate_numpy
from experiments.phase1_local_teacher_fullsource.logic import match_ids, pair_counts
from experiments.phase1_local_teacher_fullsource.audit import count_sets

class SourcePairTest(unittest.TestCase):
    def test_match_unsorted_physical_indices(self):
        found=match_ids(np.array([8,2,5]),np.array([5,8]))
        np.testing.assert_array_equal(found,np.array([2,0]))
    def test_rejects_missing_or_duplicate(self):
        with self.assertRaises(ValueError):match_ids(np.array([1,1]),np.array([1]))
        with self.assertRaises(ValueError):match_ids(np.array([1,2]),np.array([3]))
        with self.assertRaises(ValueError):match_ids(np.array([1,2]),np.array([1,1]))
    def test_pair_counts_on_explicit_eligibility(self):
        truth=np.array([0,0,1,1]);a=np.array([0,1,1,0]);b=np.array([1,0,1,1])
        result=pair_counts(truth,a,b,np.array([True,True,True,False]))
        self.assertEqual(result,dict(count=3,baseline_correct=2,candidate_correct=2,repairs=1,damages=1))


    def test_independent_set_recount_matches_vector_counts(self):
        rng=np.random.default_rng(810)
        target=rng.integers(0,6,200)
        first=rng.integers(0,6,200)
        second=rng.integers(0,6,200)
        mask=rng.random(200)>.45
        self.assertEqual(count_sets(target,first,second,mask),pair_counts(target,first,second,mask))


    def test_boolean_gate_crosses_numpy_boundary(self):
        active=torch.tensor([True,False,True])
        np.testing.assert_array_equal(gate_numpy(active),np.array([1,0,1]))

if __name__=='__main__':unittest.main()
