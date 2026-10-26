import unittest
import torch
from experiments.pamf_channel_pl.restoration_bridge import relation_loss


class RestorationBridgeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.a=torch.randn(8,160,requires_grad=True)
        self.b=torch.randn(8,160,requires_grad=True)
        self.q=torch.nn.functional.one_hot(torch.tensor([0,0,1,1,2,2,0,1]),6).float().requires_grad_()
        self.rx=torch.tensor([3,4,3,4,3,4,1,1]);self.mask=self.rx!=1

    def test_pair_counts_and_full_batch_normalization(self):
        separate,sd=relation_loss(self.a,self.b,self.q,self.mask,self.rx,'separate')
        bridge,bd=relation_loss(self.a,self.b,self.q,self.mask,self.rx,'bridge')
        self.assertEqual(sd['valid_anchors'],12)
        for k in ('accepted_packets','valid_anchors','positive_pairs','negative_pairs'):
            self.assertEqual(sd[k],bd[k])
        small,_=relation_loss(self.a[:6],self.b[:6],self.q[:6],self.mask[:6],self.rx[:6],'bridge')
        torch.testing.assert_close(bridge,small*.75)
        self.assertGreater(float(separate),0);self.assertGreater(float(bridge),0)

    def test_hidden_records_and_teacher_have_no_gradient(self):
        value,_=relation_loss(self.a,self.b,self.q,self.mask,self.rx,'bridge');value.backward()
        self.assertIsNone(self.q.grad)
        for v in (self.a,self.b):
            self.assertTrue(torch.isfinite(v.grad).all());self.assertGreater(float(v.grad[:6].norm()),0)
            self.assertEqual(float(v.grad[6:].abs().sum()),0)
        a=self.a.detach().clone();b=self.b.detach().clone();a[6:]=1e6;b[6:]=-1e6
        other,_=relation_loss(a,b,self.q,self.mask,self.rx,'bridge')
        torch.testing.assert_close(value,other)

    def test_same_view_bridge_matches_separate_and_permutation(self):
        v,_=relation_loss(self.a,self.a,self.q,self.mask,self.rx,'separate')
        w,_=relation_loss(self.a,self.a,self.q,self.mask,self.rx,'bridge');torch.testing.assert_close(v,w)
        ix=torch.tensor([5,0,7,2,1,6,3,4])
        x,_=relation_loss(self.a[ix],self.a[ix],self.q[ix],self.mask[ix],self.rx[ix],'bridge')
        torch.testing.assert_close(v,x)

    def test_no_cross_receiver_or_no_negative_yields_finite_zero(self):
        for rx,q,mask in [(torch.full((8,),3),self.q,self.mask),
                          (self.rx,torch.nn.functional.one_hot(torch.zeros(8,dtype=torch.long),6).float(),self.mask),
                          (self.rx,self.q,torch.zeros(8,dtype=torch.bool))]:
            v,d=relation_loss(self.a,self.b,q,mask,rx,'bridge')
            self.assertEqual(float(v),0);self.assertEqual(d['valid_anchors'],0)

    def test_foreign_mode_and_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):relation_loss(self.a,self.b,self.q,self.mask,self.rx,'all')
        a=self.a.detach().clone();a[0,0]=float('nan')
        with self.assertRaises(ValueError):relation_loss(a,self.b,self.q,self.mask,self.rx,'bridge')


if __name__=='__main__':unittest.main()
