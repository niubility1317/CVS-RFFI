import unittest
import torch
from experiments.pamf_channel_pl.repl_rx_moments import LabeledReceiverMoments


class ReceiverMomentTests(unittest.TestCase):
    def moments(self):
        means=torch.zeros(6,5,160)
        for c in range(6):
            means[c,:,c]=.5
            for r in range(5):means[c,r,10+r]=.2
        return LabeledReceiverMoments(means,torch.full((6,5),210))

    def inputs(self):
        h=torch.randn(2,160,requires_grad=True)
        return h,torch.tensor([0,0]),torch.tensor([True,True]),torch.tensor([1,3]),torch.tensor([1,0]),torch.tensor([True,True])

    def test_projection_preserves_L_class_mean_directions(self):
        m=self.moments();offset=m.protected_offsets('cpu',torch.float32)
        center=m.means.mean(1);center=center-center.mean(0)
        self.assertLess(float((offset@center.T).abs().max()),1e-7)
        self.assertGreater(float(offset.norm()),0)

    def test_U_partner_features_receive_no_cross_packet_gradient(self):
        m=self.moments();args=self.inputs();value=m.extrapolate(*args)
        value[0,10].backward()
        self.assertGreater(float(args[0].grad[0].norm()),0)
        self.assertEqual(float(args[0].grad[1].norm()),0)
        self.assertFalse(m.means.requires_grad)

    def test_partial_L_cells_update_without_imputing_from_U(self):
        m=self.moments();before=m.means.clone();h=torch.ones(2,160,requires_grad=True)
        m.observe(h,torch.tensor([0,0]),torch.tensor([1,1]),torch.ones(2,dtype=torch.bool))
        self.assertFalse(torch.equal(before[0,0],m.means[0,0]));self.assertTrue(torch.equal(before[1:],m.means[1:]))
        self.assertEqual(m.updates,1);self.assertEqual(m.exposures,2);self.assertEqual(int(m.counts[0,0]),212)
        self.assertFalse(m.means.requires_grad)

    def test_V_U_and_hidden_labels_cannot_update_moments(self):
        m=self.moments();h=torch.ones(2,160);y=torch.tensor([0,0]);rx=torch.tensor([1,3]);visible=torch.ones(2,dtype=torch.bool)
        for role in ('V','U_s'):
            with self.assertRaises(ValueError):m.observe(h,y,rx,visible,role=role)
        with self.assertRaises(ValueError):m.observe(h,torch.tensor([-1,0]),rx,visible)
        with self.assertRaises(ValueError):m.observe(h,y,rx,torch.zeros(2,dtype=torch.bool))

    def test_pair_must_retain_confidence_pseudo_identity_and_cross_RX(self):
        m=self.moments();args=list(self.inputs());args[1]=torch.tensor([0,1])
        with self.assertRaises(ValueError):m.extrapolate(*args)
        args=list(self.inputs());args[3]=torch.tensor([1,1])
        with self.assertRaises(ValueError):m.extrapolate(*args)
        args=list(self.inputs());args[2]=torch.tensor([True,False])
        with self.assertRaises(ValueError):m.extrapolate(*args)

    def test_state_resumes_own_L_statistics_and_rejects_RX_permutation(self):
        m=self.moments();m.observe(torch.ones(2,160),torch.tensor([0,0]),torch.tensor([1,3]),torch.ones(2,dtype=torch.bool))
        state=m.state_dict();other=self.moments();other.load_state_dict(state)
        self.assertTrue(torch.equal(m.means,other.means));self.assertEqual(other.updates,1);self.assertEqual(other.exposures,2)
        with self.assertRaises(ValueError):other.load_state_dict(dict(state,receivers=(3,1,4,6,8)))


if __name__=='__main__':unittest.main()
