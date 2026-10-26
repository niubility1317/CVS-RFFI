import unittest
import torch
from experiments.pamf_channel_pl.asymmetric_hg import terms,extra,METHODS


class AsymmetricHGTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.h=torch.randn(8,160,requires_grad=True);self.g=torch.randn(8,160,requires_grad=True)
        self.head=torch.nn.Linear(160,6,bias=False)
        self.q=torch.nn.functional.one_hot(torch.tensor([0,0,1,1,2,2,0,1]),6).float()
        self.rx=torch.tensor([3,4,3,4,3,4,1,1]);self.mask=self.rx!=1
    def call(self,method):
        h=dict(z_id=self.h,tx_logits=self.head(self.h));g=dict(z_id=self.g,tx_logits=self.head(self.g))
        return terms(h,g,self.q,self.mask,self.rx,method)
    def test_detachment_changes_only_restored_feature_gradient(self):
        a,da=self.call('HJoint');b,db=self.call('HAnchor');torch.testing.assert_close(a,b)
        ah,ag=torch.autograd.grad(a,[self.h,self.g],retain_graph=True)
        bh,bg=torch.autograd.grad(b,[self.h,self.g],retain_graph=True,allow_unused=True)
        torch.testing.assert_close(ah,bh);self.assertGreater(float(ag.detach().norm()),0);self.assertIsNone(bg)
        for key in ('accepted_packets','valid_anchors','positive_pairs','negative_pairs'):
            self.assertEqual(da[key],db[key])
        self.assertIsNone(torch.autograd.grad(b,self.head.weight,allow_unused=True)[0])
    def test_both_task_is_mean_of_h_and_g_with_same_relation(self):
        h,_=self.call('HJoint');g,_=self.call('GJoint');both,d=self.call('BothJoint')
        torch.testing.assert_close(both,.5*(h+g));self.assertEqual(d['CE_H_accepted'],6);self.assertEqual(d['CE_G_accepted'],6)
        for v in (h,g,both):self.assertIsNone(torch.autograd.grad(v,self.head.weight,retain_graph=True,allow_unused=True)[0])
    def test_masked_records_have_no_gradient_and_frozen_targets_required(self):
        v,_=self.call('BothJoint');hh,gg=torch.autograd.grad(v,[self.h,self.g])
        self.assertEqual(float(hh[6:].abs().sum()+gg[6:].abs().sum()),0)
        self.q.requires_grad_()
        with self.assertRaises(ValueError):self.call('HJoint')
        with self.assertRaises(ValueError):self.call('other')
    def test_actual_two_forward_roles(self):
        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__();self.features=torch.nn.Linear(512,160);self.head=torch.nn.Linear(160,6,bias=False);self.calls=0
            def forward_identity_only(self,x,domain_labels=None):
                self.calls+=1;z=self.features(x.flatten(1));return dict(z_id=z,tx_logits=self.head(z))
        for method in METHODS:
            s=Student();v,d=extra(s,[torch.randn(8,2,256),torch.randn(8,2,256)],torch.zeros(8,dtype=torch.long),self.q,self.mask,self.rx,method,True)
            v.backward();self.assertEqual(s.calls,2);self.assertGreater(float(s.features.weight.grad.norm()),0);self.assertIsNone(s.head.weight.grad)
            self.assertEqual(d['G_feature_gradient_norm']==0,method=='HAnchor')


if __name__=='__main__':unittest.main()
