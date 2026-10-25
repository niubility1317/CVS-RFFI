import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.pamf_channel_pl.receiver_reference import build_references,relation_loss,decide_shared_dose,metadata_long,VARIANTS


class ReceiverReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.refs=torch.nn.functional.normalize(torch.randn(3,3,160),dim=2)
        self.features=torch.randn(4,160,requires_grad=True)
        self.q=torch.tensor([[.8,.1,.1]]*4);self.mask=torch.ones(4,dtype=torch.bool)
        self.rx=torch.tensor([1]*4);self.rxs=(1,3,8)

    def loss(self,refs=None,q=None,mask=None):
        return relation_loss(self.features,self.refs if refs is None else refs,self.rx,self.rxs,
            self.q if q is None else q,self.mask if mask is None else mask)

    def test_own_receiver_reference_cannot_influence_loss(self):
        changed=self.refs.clone();changed[:,0]*=-1
        self.assertTrue(torch.equal(self.loss(),self.loss(refs=changed)))
        changed[:,1]*=-1
        self.assertGreater(abs(float(self.loss()-self.loss(refs=changed))),1e-4)

    def test_identity_and_receiver_permutations_preserve_rule(self):
        p=torch.tensor([2,0,1])
        self.assertTrue(torch.allclose(self.loss(),self.loss(refs=self.refs[p],q=self.q[:,p])))
        other=relation_loss(self.features,self.refs[:,[2,0,1]],self.rx,(8,1,3),self.q,self.mask)
        self.assertTrue(torch.allclose(self.loss(),other))

    def test_source_roles_and_missing_references_are_rejected(self):
        features=torch.randn(6,160);labels=torch.tensor([0,0,0,1,1,1]);rx=torch.tensor([1,3,8]*2)
        refs,counts=build_references(features,labels,rx,self.rxs,2,'L_s')
        self.assertEqual(tuple(refs.shape),(2,3,160));self.assertEqual(int(counts.sum()),6)
        for role in ['V','U_s','target']:
            with self.assertRaises(ValueError):build_references(features,labels,rx,self.rxs,2,role)
        with self.assertRaises(ValueError):build_references(features[:-1],labels[:-1],rx[:-1],self.rxs,2,'L_s')
        with self.assertRaises(ValueError):relation_loss(self.features,self.refs,torch.tensor([9]*4),self.rxs,self.q,self.mask)

    def test_full_U_mean_detached_targets_and_finite_backbone_gradient(self):
        same=self.features[0:1].expand(4,-1)
        full=relation_loss(same,self.refs,self.rx,self.rxs,self.q,self.mask)
        half=relation_loss(same,self.refs,self.rx,self.rxs,self.q,torch.tensor([True,True,False,False]))
        self.assertTrue(torch.allclose(half,full/2))
        empty=self.loss(mask=torch.zeros(4,dtype=torch.bool));self.assertEqual(float(empty),0.)
        self.loss().backward();self.assertTrue(torch.isfinite(self.features.grad).all());self.assertGreater(float(self.features.grad.norm()),0.)
        with self.assertRaises(ValueError):self.loss(q=self.q.clone().requires_grad_())
        with self.assertRaises(ValueError):self.loss(refs=self.refs.clone().requires_grad_())

    def test_one_dose_limits_every_variant_and_rejects_partial_or_nonfinite(self):
        row=dict(primary_norm=2.,**{v+'_norm':float(i+1) for i,v in enumerate(VARIANTS)},**{v+'_cosine':-.1 for v in VARIANTS})
        result=decide_shared_dose([row]*12)
        self.assertAlmostEqual(result['selected_weight'],.2/3)
        self.assertTrue(all(v<=.1 for v in result['weighted_median_norm_ratio'].values()))
        with self.assertRaises(ValueError):decide_shared_dose([row]*11)
        with self.assertRaises(ValueError):decide_shared_dose([dict(row,primary_norm=float('nan'))]*12)

    def test_numpy_metadata_without_a_torch_numpy_bridge(self):
        with patch('torch.as_tensor',side_effect=RuntimeError('NumPy unavailable')),patch('torch.from_numpy',side_effect=RuntimeError('NumPy unavailable')):
            labels=metadata_long(np.asarray([0,0,0,1,1,1],dtype=np.int64))
            rx=metadata_long(np.asarray([1,3,8]*2,dtype=np.int64))
            refs,counts=build_references(torch.randn(6,160),labels,rx,self.rxs,2,'L_s')
            self.assertEqual(tuple(refs.shape),(2,3,160));self.assertEqual(int(counts.sum()),6)


if __name__=='__main__':unittest.main()
