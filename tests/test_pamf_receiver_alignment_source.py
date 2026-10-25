import unittest
import torch
from torch.nn import functional as F
from experiments.pamf_channel_pl.receiver_alignment_source import feature_gradients


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(F.normalize(torch.randn(6,160),dim=1),requires_grad=False)
    def classify_features(self,z):return 10*F.linear(F.normalize(z,dim=1),self.weight)


class FrozenAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19);self.m=Reader().eval();self.z=torch.randn(8,160)
        self.refs=self.m.weight[:,None,:].expand(-1,5,-1).clone().detach();self.rx=torch.tensor([1,3,4,6,8,1,3,4])
        self.q=torch.softmax(torch.randn(8,6),dim=1)

    def test_matching_reference_and_real_head_have_equal_soft_feature_gradients(self):
        v=feature_gradients(self.m,self.z,self.refs,self.rx,(1,3,4,6,8),self.q)
        self.assertTrue(torch.allclose(v['soft_cosine'],torch.ones(8),atol=1e-5,rtol=1e-5))
        self.assertTrue(torch.allclose(v['soft_gradient_norm'],v['reference_gradient_norm'],atol=1e-6,rtol=1e-5))
        self.assertTrue(all(p.grad is None for p in self.m.parameters()))

    def test_visible_true_onehot_and_soft_identity_losses_have_equal_gradients(self):
        q=F.one_hot(torch.tensor([0,1,2,3,4,5,0,1]),6).float()
        v=feature_gradients(self.m,self.z,self.refs,self.rx,(1,3,4,6,8),q)
        self.assertTrue(torch.allclose(v['soft_gradient_norm'],v['hard_gradient_norm'],atol=1e-7,rtol=1e-5))
        self.assertTrue(torch.isfinite(v['hard_cosine']).all())
        self.assertTrue(all(p.grad is None for p in self.m.parameters()))

    def test_trainable_models_or_targets_rejected(self):
        with self.assertRaises(ValueError):feature_gradients(self.m.train(),self.z,self.refs,self.rx,(1,3,4,6,8),self.q)
        self.m.eval();self.m.weight.requires_grad_(True)
        with self.assertRaises(ValueError):feature_gradients(self.m,self.z,self.refs,self.rx,(1,3,4,6,8),self.q)
        self.m.weight.requires_grad_(False)
        with self.assertRaises(ValueError):feature_gradients(self.m,self.z,self.refs,self.rx,(1,3,4,6,8),self.q.clone().requires_grad_())


if __name__=='__main__':unittest.main()
