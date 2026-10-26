import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from experiments.pamf_channel_pl.receiver_balance_objective import bounded_mean_one,group_indices,ReceiverFrequency,semantic_reallocation,ReceiverBalancedObjective
from experiments.pamf_channel_pl.clean_h_objective import CleanHObjective,physical_mask
from experiments.pamf_ssl_research.hard_pl_objective import masked_pair,mechanism_loss


class ReceiverBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_bounded_positive_mass_and_detached_extreme_ratios(self):
        for values in ([1.,1.,1.],[.01,.1,1.,10.,100.],[1e-6,1e6],[3.]):
            r=torch.tensor(values,dtype=torch.double,requires_grad=True);w=bounded_mean_one(r)
            self.assertFalse(w.requires_grad);self.assertGreaterEqual(float(w.min()),.5);self.assertLessEqual(float(w.max()),2.)
            self.assertAlmostEqual(float(w.mean()),1.,places=10)
        for bad in ([0.,1.],[-1.,2.],[float('nan'),1.]):
            with self.assertRaises(ValueError):bounded_mean_one(torch.tensor(bad))
        self.assertEqual(len(bounded_mean_one(torch.empty(0))),0)

    def test_visible_receiver_and_pseudo_class_grouping_rejects_foreign_RX(self):
        rx=torch.tensor([1,3,1,8]);pseudo=torch.tensor([0,3,5,2]);allowed=[1,3,4,6,8]
        self.assertTrue(torch.equal(group_indices(rx,pseudo,allowed,'RX'),torch.tensor([0,1,0,4])))
        self.assertTrue(torch.equal(group_indices(rx,pseudo,allowed,'RX_TX'),torch.tensor([0,9,5,26])))
        with self.assertRaises(ValueError):group_indices(torch.tensor([9]),torch.tensor([0]),allowed,'RX')
        with self.assertRaises(ValueError):group_indices(torch.tensor([1]),torch.tensor([-1]),allowed,'RX_TX')

    def test_frequency_uses_accepted_only_and_neutral_rejected_rows(self):
        f=ReceiverFrequency([1,3],torch.tensor([1,1]),'RX')
        rx=torch.tensor([1,1,1,3,3]);pseudo=torch.tensor([0,1,2,3,4]);a=torch.tensor([1,1,1,0,0],dtype=torch.bool)
        w=f.weights(rx,pseudo,a)
        self.assertTrue(torch.equal(w[~a],torch.ones(2,dtype=w.dtype)));self.assertAlmostEqual(float(w[a].mean()),1.,places=10)
        self.assertTrue(torch.allclose(f.frequency,torch.tensor([.505,.495],dtype=torch.double)))
        before=f.frequency.clone();w=f.weights(rx,pseudo,torch.zeros(5,dtype=torch.bool))
        self.assertTrue(torch.equal(before,f.frequency));self.assertTrue(torch.equal(w,torch.ones_like(w)))
        self.assertEqual(f.updates,1);self.assertEqual(f.calls,2);self.assertEqual(f.seen,10);self.assertEqual(f.accepted,3)

    def test_class_and_receiver_permutations_preserve_query_weights(self):
        counts=torch.arange(1,13,dtype=torch.double);rx=torch.tensor([2,7,2,7,2,7,7,2]);pseudo=torch.tensor([0,1,1,2,3,4,5,0]);accepted=torch.tensor([1,1,1,1,1,1,0,0],dtype=torch.bool)
        f=ReceiverFrequency([2,7],counts,'RX_TX');first=f.weights(rx,pseudo,accepted)
        perm=torch.tensor([3,2,5,0,4,1]);new_counts=torch.empty_like(counts).reshape(2,6);new_counts[:,perm]=counts.reshape(2,6)
        other=ReceiverFrequency([7,2],new_counts.flip(0).reshape(-1),'RX_TX')
        self.assertTrue(torch.allclose(first,other.weights(rx,perm[pseudo],accepted),atol=1e-12))

    def test_joint_coverage_differs_even_with_balanced_class_and_RX_marginals(self):
        rx=torch.tensor([1,1,1,1,3,3,3,3]);pseudo=torch.tensor([0,0,0,1,0,1,1,1]);a=torch.ones(8,dtype=torch.bool)
        marginal=ReceiverFrequency([1,3],torch.ones(2),'RX').weights(rx,pseudo,a)
        joint=ReceiverFrequency([1,3],torch.ones(12),'RX_TX').weights(rx,pseudo,a)
        self.assertTrue(torch.allclose(marginal,torch.ones_like(marginal),atol=1e-10));self.assertGreater(float(joint[3]),float(joint[0]))
        self.assertGreater(float(joint[4]),float(joint[5]));self.assertAlmostEqual(float(joint.mean()),1.,places=10)

    def test_scalar_and_gradients_match_independent_per_packet_weighted_CE_RDrop(self):
        torch.manual_seed(13);n=8;z=[torch.randn(n,6,dtype=torch.double,requires_grad=True) for _ in range(6)]
        y=torch.arange(n)%6;a=torch.tensor([1,0,1,1,1,0,1,1],dtype=torch.bool);physical=torch.arange(n)%2==0
        w=torch.ones(n,dtype=torch.double);w[a]=torch.tensor([.6,.8,1.,1.2,1.4,1.],dtype=torch.double)
        delta,before,after=semantic_reallocation(z,y,a,physical,w)
        original,_=mechanism_loss(*z,y,a,physical,z[0].sum()*0.)
        expected=z[0].sum()*0.
        for i in range(n):
            pair=lambda u,v,k:masked_pair(z[u][i:i+1],z[v][i:i+1],y[i:i+1],a[i:i+1],k)[0]
            expected+=w[i]*(.5*pair(0,1,1.)+.5*pair(2,3,0. if physical[i] else 1.)+.25*pair(4,5,1.))/(1.25*n)
        self.assertTrue(torch.allclose(before,original,atol=1e-12));self.assertTrue(torch.allclose(after,expected,atol=1e-12))
        first=torch.autograd.grad(original+delta,z,retain_graph=True);second=torch.autograd.grad(expected,z)
        self.assertTrue(all(torch.allclose(x,y,atol=1e-12) for x,y in zip(first,second)))

    def test_uniform_and_all_rejected_are_exactly_original_and_partial_batch_supported(self):
        for n,rejected in ((124,False),(8,True)):
            z=[torch.randn(n,6,requires_grad=True) for _ in range(6)];y=torch.arange(n)%6;a=torch.full((n,),not rejected,dtype=torch.bool)
            delta,before,after=semantic_reallocation(z,y,a,torch.arange(n)%2==0,torch.ones(n,dtype=torch.double))
            self.assertEqual(float(delta.detach()),0.);self.assertEqual(float((after-before).detach()),0.)
            if rejected:self.assertEqual(float(before.detach()),0.)
        with self.assertRaises(ValueError):semantic_reallocation(z,y,a,torch.ones(n,dtype=torch.bool),torch.ones(n))

    def objective(self):
        obj=ReceiverBalancedObjective.__new__(ReceiverBalancedObjective);obj.c=dict(source_rxs=[1,3],step_offset=4440)
        obj.frequency=ReceiverFrequency([1,3],torch.tensor([1,1]),'RX');obj._rb_logits=obj._rb_current=None
        obj.rb_captures=obj.rb_packets=0;obj.rb_applied_mass_error=0.;obj.settings=dict(weight=6.);obj.metrics={}
        return obj

    def test_actual_loss_capture_updates_frequency_once_and_leaves_feature_auxiliary_gradient(self):
        obj=self.objective();n=8;student=SimpleNamespace(_feature_l_steps=1)
        pseudo=torch.arange(n)%6;a=torch.tensor([1,1,0,1,1,1,0,1],dtype=torch.bool);meta=dict(rx_i=torch.tensor([1,1,1,1,1,3,3,3]),tx_label_visible=torch.zeros(n,dtype=torch.bool))
        z=[torch.randn(n,6,dtype=torch.double,requires_grad=True) for _ in range(6)];feature_aux=torch.tensor(.8,dtype=torch.double,requires_grad=True);physical=physical_mask(n,4441,'cpu')
        def forward(this,student,x,*args):return z[x]
        # An input proxy supplies the real device field without adding forwards.
        class Input(int):device=torch.device('cpu')
        def parent(this,student,device):
            for k in range(6):this.unlabeled_student_forward(student,Input(k),None,pseudo,a,meta)
            this.pending_gradient_loss=6*mechanism_loss(*z,pseudo,a,physical,feature_aux)[0]
            return z[0].sum()*0.
        with patch.object(CleanHObjective,'unlabeled_student_forward',forward),patch.object(CleanHObjective,'loss',parent):obj.loss(student,'cpu')
        self.assertEqual(obj.frequency.calls,1);self.assertEqual(obj.rb_captures,6);self.assertEqual(obj.rb_packets,48);self.assertIsNone(obj._rb_logits)
        self.assertAlmostEqual(float(torch.autograd.grad(obj.pending_gradient_loss,feature_aux)[0]),6*.1/1.25,places=12)

    def test_labeled_or_outside_context_cannot_enter_new_unlabeled_capture(self):
        obj=self.objective();student=SimpleNamespace();x=torch.zeros(4,2,256);p=torch.zeros(4,dtype=torch.long);a=torch.ones(4,dtype=torch.bool)
        meta=dict(rx_i=torch.ones(4,dtype=torch.long),tx_label_visible=torch.zeros(4,dtype=torch.bool))
        with self.assertRaises(ValueError):obj.unlabeled_student_forward(student,x,None,p,a,meta)
        obj._rb_logits=[];meta['tx_label_visible'][0]=True
        with self.assertRaises(ValueError):obj.unlabeled_student_forward(student,x,None,p,a,meta)


if __name__=='__main__':unittest.main()
