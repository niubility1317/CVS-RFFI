import unittest
import torch
from experiments.pamf_u_history.receiver_trust import gaussian_view,instance_alignment,cross_receiver_relation,hard_view_trust


class ReceiverTrustTest(unittest.TestCase):
    def test_hard_trust_requires_three_views_and_never_fills(self):
        p=torch.full((3,8,6),.002);p[:,:,0]=.99
        p[1,0]=torch.tensor([.002,.99,.002,.002,.002,.002])
        rx=torch.tensor([1,3,1,3,1,3,1,3]);labels,w,stats=hard_view_trust(p,rx)
        self.assertEqual(stats['eligible'],7);self.assertLessEqual(int((w>0).sum()),2)
        self.assertEqual(float(w[0]),0.)
        p.fill_(1/6);_,w,stats=hard_view_trust(p,rx)
        self.assertEqual(stats['selected'],0);self.assertEqual(float(w.sum()),0.)

    def values(self):
        h=torch.tensor([[1.,0.],[.9,.1],[0.,1.],[.1,.9]],requires_grad=True)
        return h,dict(rx=torch.tensor([1,3,1,3]),source_rxs=[1,3],labeled=torch.tensor([True,False,True,False]),
            visible_tx=torch.tensor([0,-1,1,-1]),pseudo=torch.tensor([5,0,5,1]),trust=torch.tensor([0.,1.,0.,1.]))

    def test_L_uses_truth_independent_of_pseudo_and_trust(self):
        h,k=self.values();loss,stats=cross_receiver_relation(h,**k)
        self.assertEqual(stats['anchors'],4);self.assertEqual(stats['positive_pairs'],4)
        self.assertTrue(torch.isfinite(loss))

    def test_U_scope_keeps_L_reference_gradient_zero(self):
        h,k=self.values();loss,_=cross_receiver_relation(h,**k,scope='U');loss.backward()
        self.assertTrue(torch.equal(h.grad[k['labeled']],torch.zeros_like(h.grad[k['labeled']])))
        self.assertGreater(float(h.grad[~k['labeled']].norm()),0)

    def test_scope_changes_L_gradient_only_not_scalar_or_U_feature_gradient(self):
        h,k=self.values();a,sa=cross_receiver_relation(h,**k,scope='U');a.backward();ga=h.grad.clone();h.grad=None
        b,sb=cross_receiver_relation(h,**k,scope='LU');b.backward()
        self.assertTrue(torch.equal(a,b));self.assertEqual(sa['anchors'],sb['anchors'])
        self.assertTrue(torch.allclose(ga[~k['labeled']],h.grad[~k['labeled']],atol=1e-7))

    def test_reject_visible_U_truth_and_foreign_RX(self):
        h,k=self.values();k['visible_tx'][1]=0
        with self.assertRaises(ValueError):cross_receiver_relation(h,**k)
        k['visible_tx'][1]=-1;k['rx'][1]=8
        with self.assertRaises(ValueError):cross_receiver_relation(h,**k)

    def test_no_trust_is_graph_safe_and_cannot_invent_cross_RX_positive(self):
        h,k=self.values();k['trust'].zero_();loss,stats=cross_receiver_relation(h,**k)
        loss.backward();self.assertEqual(stats['anchors'],0)
        self.assertTrue(torch.equal(h.grad,torch.zeros_like(h)))

    def test_same_sample_alignment_uses_no_other_TX_negative(self):
        reference=torch.randn(4,8,requires_grad=True);views=reference.detach()[None].repeat(2,1,1).requires_grad_(True)
        loss=instance_alignment(views,reference);loss.backward()
        self.assertLess(abs(float(loss.detach())),1e-6);self.assertIsNone(reference.grad)

    def test_gaussian_snr_and_independent_views(self):
        x=torch.randn(4,2,256);g=torch.Generator().manual_seed(17)
        a=gaussian_view(x,snr_db=35.,generator=g);b=gaussian_view(x,snr_db=35.,generator=g)
        snr=20*torch.log10(x.flatten(1).norm(dim=1)/(a-x).flatten(1).norm(dim=1))
        self.assertTrue(torch.allclose(snr,torch.full_like(snr,35),atol=1e-4))
        self.assertFalse(torch.equal(a,b))


if __name__=='__main__':unittest.main()
