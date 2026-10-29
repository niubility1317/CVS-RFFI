"""Pure tensor objectives and the limited phase/gain training view."""
import importlib
import unittest

import torch
from torch.nn import functional as F


class StarRepresentationLossTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_representation_loss')

    def test_labeled_two_view_CE_and_raw_teacher_KD(self):
        m=self.module();g=torch.Generator().manual_seed(12)
        a=torch.randn(8,6,generator=g,requires_grad=True);b=torch.randn(8,6,generator=g,requires_grad=True)
        teacher=torch.randn(8,6,generator=g,requires_grad=True);y=torch.arange(8)%6
        result=m.labeled(a,b,y,teacher)
        expected=.5*(F.cross_entropy(a,y)+F.cross_entropy(b,y))
        kd=4*F.kl_div((a/2).log_softmax(1),(teacher.detach()/2).softmax(1),reduction='batchmean')
        torch.testing.assert_close(result['ce'],expected);torch.testing.assert_close(result['kd'],kd)
        result['total'].backward();self.assertIsNone(teacher.grad)
        self.assertGreater(float(a.grad.norm()),0);self.assertGreater(float(b.grad.norm()),0)

    def test_uniform_OE_uses_both_complete_six_class_views(self):
        m=self.module();a=torch.randn(10,6,requires_grad=True);b=torch.randn(10,6,requires_grad=True)
        loss=m.uniform_oe(a,b)
        expected=-.5*(a.log_softmax(1).mean()+b.log_softmax(1).mean())-torch.log(torch.tensor(6.))
        torch.testing.assert_close(loss,expected);loss.backward()
        self.assertGreater(float(a.grad.norm()),0);self.assertGreater(float(b.grad.norm()),0)
        self.assertAlmostEqual(float(m.uniform_oe(torch.zeros(4,6),torch.zeros(4,6))),0.,places=6)

    def test_vicreg_official_sample_variance_covariance_raw_scale(self):
        m=self.module();g=torch.Generator().manual_seed(33)
        a=torch.randn(8,160,generator=g,requires_grad=True);b=(a.detach()+.2*torch.randn(8,160,generator=g)).requires_grad_()
        r=m.vicreg(a,b);ac=a-a.mean(0);bc=b-b.mean(0)
        std=.5*(F.relu(1-torch.sqrt(a.var(0)+1e-4)).mean()+F.relu(1-torch.sqrt(b.var(0)+1e-4)).mean())
        ca=ac.T@ac/7;cb=bc.T@bc/7;mask=~torch.eye(160,dtype=torch.bool)
        cov=(ca[mask].square().sum()+cb[mask].square().sum())/160
        torch.testing.assert_close(r['std'],std);torch.testing.assert_close(r['cov'],cov)
        torch.testing.assert_close(r['total'],25*F.mse_loss(a,b)+25*std+cov)
        r['total'].backward();self.assertGreater(float(a.grad.norm()),0);self.assertGreater(float(b.grad.norm()),0)
        # No hidden unit normalization: rescaling changes the raw covariance.
        self.assertGreater(float(m.vicreg(a.detach()*3,b.detach()*3)['cov']),float(cov.detach())*10)

    def test_phase_gain_is_packetwise_global_and_preserves_relative_waveform(self):
        m=self.module();x=torch.randn(12,2,256,generator=torch.Generator().manual_seed(8))
        generator=torch.Generator().manual_seed(9);before=x.clone();v=m.phase_gain_view(x,generator)
        z=torch.complex(x[:,0],x[:,1]);w=torch.complex(v[:,0],v[:,1]);ratio=w/z
        torch.testing.assert_close(ratio,ratio[:,:1].expand_as(ratio),atol=3e-5,rtol=3e-5)
        gain=ratio[:,0].abs();self.assertTrue(bool(((gain>=.95)&(gain<=1.05)).all()))
        self.assertFalse(torch.equal(x,v));torch.testing.assert_close(x,before)
        torch.testing.assert_close(v,m.phase_gain_view(x,torch.Generator().manual_seed(9)))

    def test_shape_nonfinite_labels_and_batch_size_negatives(self):
        m=self.module()
        for value in (torch.ones(8,5),torch.full((8,6),float('nan'))):
            with self.subTest(logits=tuple(value.shape)),self.assertRaises(ValueError):m.uniform_oe(value,value)
        with self.assertRaises(ValueError):m.vicreg(torch.ones(1,160),torch.ones(1,160))
        with self.assertRaises(ValueError):m.vicreg(torch.ones(4,159),torch.ones(4,159))
        with self.assertRaises(ValueError):m.labeled(torch.ones(4,6),torch.ones(4,6),torch.tensor([0,1,2,6]),torch.ones(4,6))
        with self.assertRaises(ValueError):m.phase_gain_view(torch.ones(4,2,128),torch.Generator())


if __name__=='__main__':unittest.main()
