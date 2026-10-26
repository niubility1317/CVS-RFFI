import unittest
import numpy as np
from experiments.pamf_channel_pl.raw_anchor_analysis import certify_pair


class RawAnchorPolicyTests(unittest.TestCase):
    def logits(self,labels):
        p=np.full((len(labels),6),.002);p[np.arange(len(labels)),labels]=.99
        return np.log(p)

    def test_agreement_only_adds_original_hard_samples(self):
        r=self.logits([0,1,2,3]);a=self.logits([0,1,4,3])
        result=certify_pair(r,a,np.array([.90,.99,.90,.94]))
        np.testing.assert_array_equal(result['accepted'],[True,False,False,True])
        np.testing.assert_array_equal(result['label'],[0,1,2,3])

    def test_low_anchor_confidence_is_rejected(self):
        r=self.logits([1]);a=np.zeros((1,6))
        self.assertFalse(certify_pair(r,a,np.array([.8]))['accepted'][0])

    def test_class_and_batch_permutation_equivariance(self):
        r=self.logits([0,3,5]);old=np.array([.8,.9,.94]);p=np.array([2,5,1,4,0,3])
        a=certify_pair(r,r,old);b=certify_pair(r[:,p],r[:,p],old)
        np.testing.assert_array_equal(a['accepted'],b['accepted']);np.testing.assert_array_equal(p[b['label']],a['label'])
        for i in range(3):self.assertEqual(bool(certify_pair(r[i:i+1],r[i:i+1],old[i:i+1])['accepted'][0]),bool(a['accepted'][i]))

    def test_invalid_shapes_or_truth_handles_rejected(self):
        r=self.logits([1])
        for a,old in [(r[:,:5],np.array([.8])),(r,np.array([1.2])),(r*np.nan,np.array([.8]))]:
            with self.assertRaises(ValueError):certify_pair(r,a,old)
        with self.assertRaises(TypeError):certify_pair(r,r,np.array([.8]),truth=np.array([1]))


if __name__=='__main__':unittest.main()
