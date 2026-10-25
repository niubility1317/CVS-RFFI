import unittest
import torch
from torch.nn import functional as F
from experiments.pamf_channel_pl.receiver_reference import relation_loss
from experiments.pamf_channel_pl.task_scaled_reference import task_scaled_loss,MAX_SCALE


class TaskScaledReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23);self.z=torch.randn(8,160,requires_grad=True)
        self.w=torch.nn.Parameter(F.normalize(torch.randn(6,160),dim=1))
        self.refs=F.normalize(torch.randn(6,5,160),dim=2)
        self.target=F.one_hot(torch.tensor([0,1,2,3,4,5,0,1]),6).float()
        self.rx=torch.tensor([1,3,4,6,8,1,3,4]);self.rxs=(1,3,4,6,8)
        self.mask=torch.tensor([True,False,True,True,False,True,True,True])

    def logits(self):return 10*F.linear(F.normalize(self.z,dim=1),F.normalize(self.w,dim=1))

    def test_same_head_and_reference_reduce_to_original_without_new_head_gradient(self):
        refs=self.w.detach()[:,None,:].expand(-1,5,-1).clone()
        logits=self.logits();base=relation_loss(self.z,refs,self.rx,self.rxs,self.target,self.mask)
        loss,metrics=task_scaled_loss(self.z,logits,refs,self.rx,self.rxs,self.target,self.mask)
        self.assertTrue(torch.allclose(base,loss,atol=1e-7,rtol=1e-5))
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        loss.backward();self.assertIsNone(self.w.grad);self.assertTrue(torch.isfinite(self.z.grad).all())

    def test_reference_direction_retained_and_magnitude_bounded_by_identity_task(self):
        logits=self.logits();base=relation_loss(self.z,self.refs,self.rx,self.rxs,self.target,self.mask)
        hard=(F.cross_entropy(logits,self.target.argmax(1),reduction='none')*self.mask).mean()
        gr=torch.autograd.grad(base,self.z,retain_graph=True)[0];gh=torch.autograd.grad(hard,self.z,retain_graph=True)[0]
        loss,metrics=task_scaled_loss(self.z,logits,self.refs,self.rx,self.rxs,self.target,self.mask)
        self.assertIsNone(self.w.grad);loss.backward();actual=self.z.grad
        self.assertTrue(torch.all(actual.norm(dim=1)<=gh.norm(dim=1)+1e-7))
        self.assertTrue(torch.allclose(F.cosine_similarity(actual[self.mask],gr[self.mask]),torch.ones(int(self.mask.sum())),atol=1e-5))
        self.assertTrue(torch.equal(actual[~self.mask],torch.zeros_like(actual[~self.mask])))
        self.assertLessEqual(float(metrics['task_scale_max']),MAX_SCALE);self.assertIsNone(self.w.grad)

    def test_zero_acceptance_gives_zero_loss_and_zero_gradient(self):
        mask=torch.zeros(8,dtype=torch.bool)
        loss,metrics=task_scaled_loss(self.z,self.logits(),self.refs,self.rx,self.rxs,self.target,mask)
        self.assertEqual(float(loss.detach()),0.);loss.backward();self.assertEqual(float(self.z.grad.norm()),0.)
        self.assertEqual(float(metrics['task_scale_sum']),0.);self.assertIsNone(self.w.grad)


if __name__=='__main__':unittest.main()
