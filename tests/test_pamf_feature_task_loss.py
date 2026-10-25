import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_channel_pl.feature_task_loss import task_loss


class FeatureTaskLossTests(unittest.TestCase):
    def test_shield_keeps_exact_backbone_gradient_without_head_gradient(self):
        torch.manual_seed(41)
        backbone=nn.Linear(4,5).double()
        head=nn.Sequential(nn.LayerNorm(5),nn.Linear(5,3)).double()
        x=torch.randn(7,4,dtype=torch.double)
        q=torch.randn(7,3,dtype=torch.double).softmax(1)
        mask=torch.tensor([True,False,True,True,False,False,True])
        z=backbone(x);logits=head(z)
        expected=(F.cross_entropy(logits,q.argmax(1),reduction='none')*mask).mean()
        parameters=list(backbone.parameters())+list(head.parameters())
        joint=torch.autograd.grad(expected,parameters,retain_graph=True)
        loss=task_loss(z,logits,q,mask,shield_head=True)
        actual=torch.autograd.grad(loss,parameters,allow_unused=True)
        self.assertEqual(float(loss.detach()),float(expected.detach()))
        for a,b in zip(actual[:2],joint[:2]):
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        self.assertTrue(all(v is None for v in actual[2:]))
        self.assertTrue(all(p.requires_grad for p in parameters))

    def test_joint_loss_and_empty_mask_preserve_full_U_mean(self):
        torch.manual_seed(42)
        z=torch.randn(4,5,dtype=torch.double,requires_grad=True)
        head=nn.Linear(5,3).double();q=torch.tensor([[1.,0.,0.]]*4,dtype=torch.double)
        logits=head(z);mask=torch.tensor([True,False,False,False])
        actual=task_loss(z,logits,q,mask,False)
        torch.testing.assert_close(actual,F.cross_entropy(logits[:1],q[:1].argmax(1))/4)
        self.assertGreater(float(torch.autograd.grad(actual,head.weight,retain_graph=True)[0].norm()),0.)
        zero=task_loss(z,logits,q,torch.zeros(4,dtype=torch.bool),True)
        self.assertEqual(float(zero.detach()),0.)
        values=torch.autograd.grad(zero,[z,head.weight],allow_unused=True)
        self.assertEqual(float(values[0].norm()),0.)
        self.assertIsNone(values[1])

    def test_rejects_detached_or_foreign_feature_graph_and_mutable_targets(self):
        z=torch.randn(3,4,requires_grad=True);head=nn.Linear(4,3)
        q=torch.tensor([[1.,0.,0.]]*3);mask=torch.ones(3,dtype=torch.bool)
        with self.assertRaises(ValueError):task_loss(z,head(z.detach()),q,mask,True)
        with self.assertRaises(ValueError):task_loss(z.detach(),head(z),q,mask,True)
        with self.assertRaises(ValueError):task_loss(z,head(z),q.requires_grad_(),mask,True)
        with self.assertRaises(ValueError):task_loss(z,head(z),q.detach(),mask.float(),True)
        with self.assertRaises(ValueError):task_loss(z,head(z),q.detach()*2,mask,True)


if __name__=='__main__':
    unittest.main()
