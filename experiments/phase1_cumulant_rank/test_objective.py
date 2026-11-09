import unittest
import torch
from experiments.phase1_cumulant_rank.objective import power_features,raw_features,normalize_features,rank_loss
from experiments.phase1_dura_orthogonal.source_retrieval_geometry import descriptors

class RankObjectiveTest(unittest.TestCase):
    def test_power_matches_source_diagnostic(self):
        torch.manual_seed(11);x=torch.randn(8,2,256)
        torch.testing.assert_close(power_features(x),descriptors(x)[0])
        power,cum=raw_features(x)
        stats=[(z.mean(0),z.std(0,unbiased=False).clamp_min(.001)) for z in (power,cum)]
        self.assertEqual(tuple(normalize_features((power,cum),stats).shape),(8,53))

    def test_duplicate_source_records_are_positives(self):
        q=torch.tensor([[0.,0.],[.1,.2],[2.,2.]],requires_grad=True)
        r=torch.tensor([[0.,0.],[.1,.2],[2.,2.]])
        unique=rank_loss(q,r,torch.tensor([0,1,2]))
        all_same=rank_loss(q,r,torch.tensor([0,0,0]))
        self.assertGreater(unique.item(),0)
        self.assertEqual(all_same.item(),0.)
        unique.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(q.grad.abs().sum().item(),0.)

    def test_simultaneous_permutation_is_invariant(self):
        torch.manual_seed(12);q=torch.randn(5,7);r=torch.randn(5,7);ids=torch.tensor([1,2,3,1,4])
        permutation=torch.tensor([4,2,0,3,1])
        torch.testing.assert_close(rank_loss(q,r,ids),rank_loss(q[permutation],r[permutation],ids[permutation]))

if __name__=='__main__':unittest.main()
