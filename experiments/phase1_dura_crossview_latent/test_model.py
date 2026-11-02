import math
import unittest
import torch
from experiments.phase1_dura_crossview_latent.model import CrossViewDURA
from experiments.phase1_invariant_latent.objective import latent_loss

def rotate(x,angle):
    c,s=math.cos(angle),math.sin(angle)
    return torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),dim=1)

class CrossViewDURATest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(118)
        self.x=torch.randn(3,2,256);self.anchor=torch.randn(3,2,256);self.lv=torch.randn(3)
    def test_zero_initialized_output_equals_anchor(self):
        m=CrossViewDURA().eval()
        torch.testing.assert_close(m(self.x,self.anchor,self.lv),self.anchor,rtol=0,atol=0)
    def test_phase_invariant_representation_and_equivariant_output(self):
        m=CrossViewDURA().eval()
        with torch.no_grad():
            m.gate_output.weight.normal_(0,.02);m.gate_output.bias.normal_(0,.02)
            y=m(self.x,self.anchor,self.lv);z=m.last_latent.clone()
            r=m(rotate(self.x,.69),rotate(self.anchor,.69),self.lv);rz=m.last_latent.clone()
        torch.testing.assert_close(r,rotate(y,.69),rtol=3e-5,atol=3e-5)
        torch.testing.assert_close(rz,z,rtol=3e-5,atol=3e-5)
    def test_batch_independent_inference(self):
        m=CrossViewDURA().eval()
        with torch.no_grad():
            m.gate_output.weight.normal_(0,.02)
            y=m(self.x,self.anchor,self.lv);one=m(self.x[:1],self.anchor[:1],self.lv[:1])
        torch.testing.assert_close(one,y[:1],rtol=2e-5,atol=2e-5)
    def test_rejects_nonfinite(self):
        self.x[0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):CrossViewDURA()(self.x,self.anchor,self.lv)
    def test_paired_latent_groups_and_noncollapse_term(self):
        torch.manual_seed(73)
        base=torch.randn(8,64)
        groups=torch.tensor([4,4,4,4,8,8,8,8]*3)
        same=base.repeat(3,1).requires_grad_()
        loss,terms=latent_loss(same,groups)
        self.assertLess(terms['invariance'].item(),1e-12)
        self.assertTrue(torch.isfinite(loss))
        collapse=torch.zeros(24,64,requires_grad=True)
        _,ct=latent_loss(collapse,groups)
        self.assertGreater(ct['variance'].item(),.9)
        bad=groups.clone();bad[8]=8
        with self.assertRaises(ValueError):latent_loss(same,bad)
    def test_both_latent_and_decoder_receive_gradients(self):
        m=CrossViewDURA().train();x=self.x.repeat(8,1,1);a=self.anchor.repeat(8,1,1);lv=self.lv.repeat(8)
        groups=torch.tensor([4,4,4,4,8,8,8,8]*3)
        out=m(x,a,lv);aux,_=latent_loss(m.last_latent,groups)
        (out.square().mean()+aux).backward()
        self.assertGreater(m.gate_output.weight.grad.abs().sum().item(),0)
        self.assertGreater(m.latent.weight.grad.abs().sum().item(),0)

if __name__=='__main__':unittest.main()
