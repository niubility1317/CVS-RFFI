import importlib
import unittest
import torch
from torch import nn
from experiments.phase1_dura_feedback.model import pack_context
from experiments.phase1_dura_blendgate.model import align_to_anchor


class FakeExpert(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(1.0))
    def forward(self,x,initial,lv):return self.scale*torch.roll(x,3,-1)


def rotate(x,angle):
    c,s=torch.cos(angle),torch.sin(angle)
    return torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),dim=1)


class TripleBlendTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(77)
        self.x=torch.randn(3,2,256);self.anchor=self.x+.2*torch.randn_like(self.x)
        self.features=torch.randn(3,160);self.logits=torch.randn(3,6);self.lv=torch.full((3,),-2.)
        self.packed=pack_context(self.anchor,self.features,self.logits)

    def model(self):
        try:return importlib.import_module('experiments.phase1_dura_tripleblend.model').TripleBlend(FakeExpert())
        except ModuleNotFoundError as exc:
            if exc.name=='experiments.phase1_dura_tripleblend.model':self.fail('three-input waveform fusion is not implemented')
            raise

    def test_zero_output_is_dura_and_expert_is_frozen(self):
        m=self.model();self.assertTrue(torch.equal(m(self.x,self.packed,self.lv),self.anchor))
        self.assertTrue(all(not p.requires_grad for p in m.expert.parameters()))

    def test_raw_branch_can_modify_waveform_without_spectral_branch(self):
        m=self.model().eval()
        with torch.no_grad():
            m.gate_output.bias[0]=.4
            y=m(self.x,self.packed,self.lv)
            raw=align_to_anchor(self.x,self.anchor,preserve_edges=True)
            expected=self.anchor+.35*torch.tanh(torch.tensor(.4))*(raw-self.anchor)
        torch.testing.assert_close(y,expected,rtol=3e-5,atol=3e-5)

    def test_spectral_branch_can_rotate_complex_difference(self):
        m=self.model().eval()
        with torch.no_grad():
            m.gate_output.bias[3]=.3
            y=m(self.x,self.packed,self.lv);expert=m.expert(self.x,self.x,self.lv)
            diff=align_to_anchor(expert,self.anchor,preserve_edges=True)-self.anchor
            expected=self.anchor+.35*torch.tanh(torch.tensor(.3))*torch.stack((-diff[:,1],diff[:,0]),dim=1)
        torch.testing.assert_close(y,expected,rtol=3e-5,atol=3e-5)

    def test_fixed_condition_phase_rotation_and_batch_independence(self):
        m=self.model().eval()
        with torch.no_grad():
            m.gate_output.weight.normal_(0,.02);y=m(self.x,self.packed,self.lv);angle=torch.tensor(.6)
            packed=pack_context(rotate(self.anchor,angle),self.features,self.logits)
            got=m(rotate(self.x,angle),packed,self.lv);one=m(self.x[:1],self.packed[:1],self.lv[:1])
        torch.testing.assert_close(got,rotate(y,angle),rtol=5e-5,atol=5e-5)
        torch.testing.assert_close(one,y[:1],rtol=5e-5,atol=5e-5)

    def test_condition_is_detached_while_gate_remains_trainable(self):
        m=self.model();packed=self.packed.detach().requires_grad_();y=m(self.x,packed,self.lv)
        (y-self.x).square().mean().backward()
        self.assertGreater(float(m.gate_output.weight.grad.abs().sum()),0)
        self.assertTrue(torch.all(packed.grad[:,2]==0))
        self.assertTrue(all(p.grad is None for p in m.expert.parameters()))


if __name__=='__main__':unittest.main()
