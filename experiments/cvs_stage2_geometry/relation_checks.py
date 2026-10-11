import unittest
import numpy as np
import torch
from .relation_audit import (solve_box,bounded_oracle,components,component_evidence,
                            shared_prediction,transfer_evidence,split_group_means)


class RelationChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_box_matches_separable_optimum(self):
        matrix=np.diag([1.,2.,.5]);target=np.array([3.,-3.,.2])
        x,diag=solve_box(matrix,target)
        np.testing.assert_allclose(x.tolist(),[1.,-1.,.4],atol=1e-10)
        self.assertLess(diag['convex_gap_upper_bound'],1e-10)
        self.assertEqual(diag['active_coordinates'],2)

    def test_rank_deficient_box_is_verified(self):
        matrix=np.array([[1.,1.,0.],[2.,2.,0.],[0.,0.,0.]])
        x,diag=solve_box(matrix,np.array([3.,6.,0.]))
        self.assertTrue(diag['success']);self.assertLessEqual(float(x.abs().max()),1.)
        self.assertAlmostEqual(float((torch.tensor(matrix.tolist(),dtype=torch.float64)@x)[0]),2.)
        self.assertLess(diag['gap_relative_to_zero'],1e-10)

    def test_component_partition_and_negative_cross_are_preserved(self):
        g=torch.Generator().manual_seed(4)
        a=torch.randn(6,3,5,8,generator=g,dtype=torch.float64)
        b=-a
        evidence=component_evidence((a+b)/2,a,b)
        self.assertEqual(evidence['total_rx_centered_energy'],0.)
        self.assertTrue(all(v['half_cross_energy']<0 for v in evidence['components'].values()))
        parts,centered=components(a)
        self.assertTrue(torch.allclose(sum(parts.values()),centered))

    @staticmethod
    def cells():
        groups=[];blocks=[];means=[]
        for tx in range(6):
            for day in range(3):
                ids=[];bid=len(blocks)
                for rx in range(3):
                    ids.append(len(groups));groups.append(dict(tx=tx,day=day,rx=rx))
                    means.append([tx+day,rx-1.,2*(rx-1.)])
                blocks.append(dict(tx=tx,day=day,group_ids=ids,block_id=bid))
        return torch.tensor(means,dtype=torch.float64),groups,blocks

    def test_held_tx_and_day_values_do_not_fit_shared_predictor(self):
        means,groups,blocks=self.cells()
        first,_=shared_prediction(means,groups,blocks,[2,3,4,5],[1,2],False)
        altered=means.clone()
        for i,g in enumerate(groups):
            if g['tx']<2 or g['day']==0:altered[i]*=10000
        second,_=shared_prediction(altered,groups,blocks,[2,3,4,5],[1,2],False)
        self.assertTrue(torch.equal(first,second))

    def test_known_stable_rx_transfers_across_tx_and_day(self):
        means,groups,blocks=self.cells()
        rows=transfer_evidence(means,means,means,groups,blocks)
        self.assertEqual(len(rows),90)
        for r in rows:
            if r['scale']==1.:self.assertLess(r['mse_ratio'],1e-20)

    def test_split_is_disjoint_reproducible_and_complete(self):
        cache=dict(ids=['packet'+str(i) for i in range(20)],
                   parent_feature=torch.arange(60,dtype=torch.float64).reshape(20,3))
        groups=[dict(group_id=0,indices=list(range(10))),dict(group_id=1,indices=list(range(10,20)))]
        a,b,splits=split_group_means(cache,groups)
        aa,bb,other=split_group_means(cache,groups)
        self.assertTrue(torch.equal(a,aa));self.assertTrue(torch.equal(b,bb));self.assertEqual(splits,other)
        for group,s in zip(groups,splits):
            self.assertFalse(set(s['half_a_indices'])&set(s['half_b_indices']))
            self.assertEqual(set(s['half_a_indices']+s['half_b_indices']),set(group['indices']))
            self.assertTrue(torch.allclose((a[group['group_id']]+b[group['group_id']])/2,
                cache['parent_feature'][group['indices']].mean(0)))


if __name__=='__main__':unittest.main()
