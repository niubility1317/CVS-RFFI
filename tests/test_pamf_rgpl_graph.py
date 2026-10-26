import unittest
import torch
from experiments.pamf_channel_pl.rgpl_graph import geometry,propagate,exclude_own_RX,certify,extend


class RGPLGraphTests(unittest.TestCase):
    def data(self):
        torch.manual_seed(9);z=torch.randn(12,8,dtype=torch.float64);rx=torch.tensor([1,3,4]*4)
        labels=torch.tensor([0,0,0,1,1,1,-1,-1,-1,-1,-1,-1]);roles=(labels<0).long()
        return z,rx,labels,roles

    def test_exact_mutual_edges_have_no_same_RX_and_symmetric_normalization(self):
        z,rx,y,roles=self.data();g=geometry(z,rx,[1,3,4],k=3,block=4);ij=g['S'].indices()
        self.assertTrue((rx[ij[0]]!=rx[ij[1]]).all());torch.testing.assert_close(g['S'].to_dense(),g['S'].to_dense().T)
        h=geometry(z,rx,[1,3,4],k=3,block=7);torch.testing.assert_close(g['S'].to_dense(),h['S'].to_dense())

    def test_source_diffusion_matches_linear_solution_and_hides_U_labels(self):
        z,rx,y,roles=self.data();g=geometry(z,rx,[1,3,4],k=3);p=propagate(g,y,roles,2)
        Y=torch.zeros(12,6,dtype=torch.float64)
        for i in range(6):Y[i,[1,3,4].index(int(rx[i]))*2+int(y[i])]=1.
        expected=torch.linalg.solve(torch.eye(12,dtype=torch.float64)-.9*g['S'].to_dense(),.1*Y)
        torch.testing.assert_close(p['Q'].reshape(12,6),expected,atol=1e-5,rtol=1e-5)
        bad=y.clone();bad[-1]=0
        with self.assertRaises(ValueError):propagate(g,bad,roles,2)
        badrole=roles.clone();badrole[-1]=2
        with self.assertRaises(ValueError):propagate(g,y,badrole,2)
        swapped=y.clone();swapped[y>=0]=1-y[y>=0]
        q=propagate(g,swapped,roles,2);torch.testing.assert_close(q['Q'],p['Q'].flip(-1))

    def test_own_RX_seed_mass_cannot_certify_and_high_raw_confidence_keeps_original_PL(self):
        rx=torch.tensor([1,1]);Q=torch.zeros(2,3,2);Q[:,0,0]=100;Q[:,1:,1]=1
        p,_,_=exclude_own_RX(Q,rx,[1,3,4]);torch.testing.assert_close(p,torch.tensor([[0.,1.],[0.,1.]]))
        c=certify(Q,Q,rx,[1,3,4],torch.tensor([.8,.96]));self.assertEqual(c['accepted'].tolist(),[True,False]);self.assertEqual(c['label'].tolist(),[1,1])
        one=certify(Q[:,:1],Q[:,:1],rx,[1],torch.tensor([.8,.8]));self.assertFalse(one['accepted'].any())
        with self.assertRaises(ValueError):exclude_own_RX(Q,torch.tensor([9,1]),[1,3,4])
        tied=Q.clone();tied[:,1]=torch.tensor([1e-8,1e-8])
        a=certify(tied,tied,rx,[1,3,4],torch.tensor([.8,.8]));self.assertFalse(a['accepted'].any())
        b=certify(tied.flip(-1),tied.flip(-1),rx,[1,3,4],torch.tensor([.8,.8]));self.assertTrue(torch.equal(a['accepted'],b['accepted']))

    def test_inductive_queries_do_not_update_graph_and_are_batch_independent(self):
        z,rx,y,roles=self.data();g=geometry(z,rx,[1,3,4],k=3);p=propagate(g,y,roles,2)
        before=g['S'].to_dense().clone();qbefore=p['Q'].clone()
        a=extend(g,p,z[:3],rx[:3]);b=torch.cat([extend(g,p,z[i:i+1],rx[i:i+1]) for i in range(3)])
        torch.testing.assert_close(a,b);torch.testing.assert_close(g['S'].to_dense(),before);torch.testing.assert_close(p['Q'],qbefore)
        with self.assertRaises(ValueError):extend(g,p,z[:1],torch.tensor([9]))


if __name__=='__main__':unittest.main()
