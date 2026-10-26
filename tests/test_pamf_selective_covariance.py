import unittest
import torch
from experiments.pamf_u_history.selective_covariance import correlation,consistency,MaskStatistics


class SelectiveCovarianceTest(unittest.TestCase):
    def test_centering_scaling_and_short_sequence_are_finite(self):
        h=torch.randn(3,4,8);c=correlation(h)
        self.assertTrue(torch.allclose(c,correlation(3*h+7),atol=1e-5))
        self.assertTrue(torch.isfinite(correlation(torch.zeros(3,4,2))).all())
        self.assertTrue(torch.allclose(c,c.transpose(1,2)))

    def test_only_selected_correlation_entries_have_direct_loss_gradient(self):
        raw=torch.randn(2,3,3,requires_grad=True);aug=torch.randn(2,3,3,requires_grad=True)
        mask=torch.zeros(3,3,dtype=torch.bool);mask[0,2]=True
        loss=consistency(raw,aug,mask);loss.backward()
        self.assertTrue(torch.equal(raw.grad[:,~mask],torch.zeros_like(raw.grad[:,~mask])))
        self.assertTrue(torch.allclose(raw.grad,-aug.grad))

    def test_known_RX_balance_and_no_foreign_receiver(self):
        raw=torch.zeros(3,2,2);a=raw.clone();a[:,0,1]=torch.tensor([1.,1.,3.]);m=torch.tensor([[False,True],[False,False]])
        self.assertAlmostEqual(float(consistency(raw,a,m,receivers=torch.tensor([1,1,3]),source_rxs=[1,3])),5.)
        with self.assertRaises(ValueError):consistency(raw,a,m,receivers=torch.tensor([1,1,8]),source_rxs=[1,3])

    def test_mask_fit_rejects_U_or_validation(self):
        s=MaskStatistics(3);c=torch.zeros(2,3,3);y=torch.zeros(2,dtype=torch.long);r=torch.ones(2,dtype=torch.long)
        for role in ('U_s','V'):
            with self.assertRaises(ValueError):s.update(c,c[None],y,r,r,role=role)

    def test_fixed_sparse_mask_and_same_size_random(self):
        s=MaskStatistics(5);raw=[];aug=[];ys=[];rs=[];ds=[]
        for y in range(6):
            for r in (1,3,4,6,8):
                for d in (1,2,3):
                    for repeat in range(2):
                        c=torch.zeros(5,5);c[0,1]=y*.1;c[1,0]=y*.1;c[0,2]=repeat*.01;c[2,0]=repeat*.01
                        a=c.clone();a[0,2]+=.5;a[2,0]+=.5;a[0,3]+=.3;a[3,0]+=.3;a[1,4]+=.2;a[4,1]+=.2
                        raw.append(c);aug.append(a);ys.append(y);rs.append(r);ds.append(d)
        s.update(torch.stack(raw),torch.stack(aug)[None],torch.tensor(ys),torch.tensor(rs),torch.tensor(ds),role='L_s')
        masks,info=s.masks();self.assertEqual(int(masks['selective'].sum()),1)
        self.assertEqual(int(masks['random'].sum()),1);self.assertEqual(int(masks['full'].sum()),10)
        self.assertTrue(masks['selective'][0,2]);self.assertFalse(masks['selective'][0,1])
        self.assertEqual(info['samples'],180)


if __name__=='__main__':unittest.main()
