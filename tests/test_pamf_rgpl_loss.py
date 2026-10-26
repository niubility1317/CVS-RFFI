import unittest
import torch
from torch.nn import functional as F

from experiments.pamf_channel_pl.rgpl_loss import unlabeled_loss


class GraphLossTests(unittest.TestCase):
    def setUp(self):
        self.raw=torch.tensor([[2.,-1.,.5],[-.2,1.1,.4],[.1,.2,1.4],[1.3,.4,-.1]],requires_grad=True)
        self.pseudo=torch.tensor([0,1,2,0]);self.accepted=torch.tensor([True,False,True,False])
        self.graph=torch.tensor([1,2,0,1]);self.certified=torch.tensor([False,True,False,True])

    def test_graph_raw_replaces_label_and_falls_back_to_ordinary_PL(self):
        out=unlabeled_loss(self.raw,self.pseudo,self.accepted,self.graph,self.certified,rho=0.)
        expected=(F.cross_entropy(self.raw[:1],self.pseudo[:1],reduction='sum')+
            F.cross_entropy(self.raw[1:2],self.graph[1:2],reduction='sum')+
            F.cross_entropy(self.raw[2:3],self.pseudo[2:3],reduction='sum')+
            F.cross_entropy(self.raw[3:],self.graph[3:],reduction='sum'))/4
        self.assertTrue(torch.allclose(out,expected))
        out.backward();self.assertTrue(torch.isfinite(self.raw.grad).all())
        self.assertLess(self.raw.grad[1,self.graph[1]],0.)

    def test_two_raw_views_preserve_total_CE_dose_and_gradients(self):
        extra=self.raw[self.certified].detach().clone().requires_grad_()
        a=unlabeled_loss(self.raw,self.pseudo,self.accepted,self.graph,self.certified,rho=.25,extra_logits=extra)
        b=unlabeled_loss(self.raw,self.pseudo,self.accepted,self.graph,self.certified,rho=0.)
        self.assertTrue(torch.allclose(a,b))
        ga,ge=torch.autograd.grad(a,(self.raw,extra));gb=torch.autograd.grad(b,self.raw)[0]
        combined=ga.clone();combined[self.certified]+=ge
        self.assertTrue(torch.allclose(combined,gb))

    def test_rejected_non_graph_records_have_zero_loss_and_no_denominator_inflation(self):
        certified=torch.zeros(4,dtype=torch.bool)
        result=unlabeled_loss(self.raw,self.pseudo,self.accepted,self.graph,certified,rho=.25,extra_logits=self.raw[:0])
        expected=F.cross_entropy(self.raw,self.pseudo,reduction='none').mul(self.accepted).sum()/4
        self.assertTrue(torch.allclose(result,expected))
        grad=torch.autograd.grad(result,self.raw)[0]
        self.assertTrue(torch.equal(grad[~self.accepted],torch.zeros_like(grad[~self.accepted])))

    def test_recovery_view_only_changes_certified_second_term(self):
        extra=torch.tensor([[1.,2.,-1.],[.2,.9,.4]],requires_grad=True)
        got=unlabeled_loss(self.raw,self.pseudo,self.accepted,self.graph,self.certified,rho=.25,extra_logits=extra)
        raw=F.cross_entropy(self.raw,torch.where(self.certified,self.graph,self.pseudo),reduction='none')
        weights=torch.where(self.certified,.75,self.accepted.float())
        expected=(raw.mul(weights).sum()+.25*F.cross_entropy(extra,self.graph[self.certified],reduction='sum'))/4
        self.assertTrue(torch.allclose(got,expected))
        self.assertGreater(float(torch.autograd.grad(got,extra)[0].norm()),0.)

    def test_invalid_shapes_labels_or_dose_are_rejected(self):
        cases=[dict(rho=1.1),dict(rho=.25),dict(rho=0.,extra_logits=self.raw[:1]),
            dict(rho=.25,extra_logits=self.raw[:1]),dict(rho=0.,certified=self.certified.float()),
            dict(rho=0.,graph_label=torch.tensor([1,-1,0,1]))]
        for change in cases:
            args=dict(raw_logits=self.raw,pseudo=self.pseudo,accepted=self.accepted,graph_label=self.graph,certified=self.certified)
            args.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):unlabeled_loss(**args)


if __name__=='__main__':unittest.main()
