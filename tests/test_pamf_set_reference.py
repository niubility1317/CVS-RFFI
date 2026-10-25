import unittest
import torch
from torch.nn import functional as F
from experiments.pamf_channel_pl.set_reference import build_bank,set_logits,mean_logits
from experiments.pamf_channel_pl.receiver_reference import build_references,relation_loss


class SetReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12);self.rxs=(1,3);self.labels=torch.tensor([0,0,0,0,1,1,1,1])
        self.rx=torch.tensor([1,1,3,3,1,1,3,3]);self.z=torch.randn(8,160,requires_grad=True)

    def test_hidden_role_and_incomplete_or_foreign_support_rejected(self):
        for role in ('U_s','V','target'):
            with self.assertRaises(ValueError):build_bank(self.z,self.labels,self.rx,self.rxs,2,role)
        with self.assertRaises(ValueError):build_bank(self.z[:4],self.labels[:4],self.rx[:4],self.rxs,2,'L_s')
        rx=self.rx.clone();rx[0]=9
        with self.assertRaises(ValueError):build_bank(self.z,self.labels,rx,self.rxs,2,'L_s')

    def test_identical_support_reduces_to_original_mean_relation(self):
        feature=F.normalize(torch.randn(2,160),dim=1);x=feature[self.labels]
        bank,valid=build_bank(x,self.labels,self.rx,self.rxs,2,'L_s')
        refs,_=build_references(x,self.labels,self.rx,self.rxs,2,'L_s')
        q=F.one_hot(self.labels,2).float();mask=torch.ones(8,dtype=torch.bool)
        logits=set_logits(self.z,bank,valid,self.rx,self.rxs,.1)
        self.assertTrue(torch.allclose(logits,mean_logits(self.z,refs,self.rx,self.rxs),atol=2e-6))
        self.assertTrue(torch.allclose(F.cross_entropy(logits,self.labels),relation_loss(self.z,refs,self.rx,self.rxs,q,mask),atol=2e-6))

    def test_multiple_identity_modes_survive_mean_collapse(self):
        a=torch.zeros(160);a[0]=1
        b=torch.zeros(160);b[0]=-.8;b[1]=.6
        other=torch.zeros(160);other[0]=.8;other[2]=.6
        x=torch.stack([a,b,a,b,other,other,other,other])
        bank,valid=build_bank(x,self.labels,self.rx,self.rxs,2,'L_s');refs,_=build_references(x,self.labels,self.rx,self.rxs,2,'L_s')
        query=a[None].requires_grad_(True);rx=torch.tensor([1])
        self.assertEqual(int(mean_logits(query,refs,rx,self.rxs).argmax(1)),1)
        score=set_logits(query,bank,valid,rx,self.rxs,.1);self.assertEqual(int(score.argmax(1)),0)
        F.cross_entropy(score,torch.tensor([0])).backward();self.assertTrue(torch.isfinite(query.grad).all())
        self.assertFalse(bank.requires_grad);self.assertIsNone(bank.grad)

    def test_support_duplication_and_own_RX_do_not_change_scores(self):
        bank,valid=build_bank(self.z,self.labels,self.rx,self.rxs,2,'L_s');query=self.z[:2];rx=self.rx[:2]
        expected=set_logits(query,bank,valid,rx,self.rxs,.05)
        twice=bank.repeat_interleave(2,dim=2);twice_valid=valid.repeat_interleave(2,dim=2)
        self.assertTrue(torch.allclose(expected,set_logits(query,twice,twice_valid,rx,self.rxs,.05),atol=2e-6))
        changed=bank.clone();changed[:,0]=torch.randn_like(changed[:,0])
        self.assertTrue(torch.allclose(expected,set_logits(query,changed,valid,rx,self.rxs,.05),atol=2e-6))
        with self.assertRaises(ValueError):set_logits(query,bank,valid,torch.tensor([9,9]),self.rxs,.1)


if __name__=='__main__':unittest.main()
